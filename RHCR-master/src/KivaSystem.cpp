#include "KivaSystem.h"
#include "WHCAStar.h"
#include "ECBS.h"
#include "LRAStar.h"
#include "PBS.h"

KivaSystem::KivaSystem(const KivaGrid& G, MAPFSolver& solver) : BasicSystem(G, solver), G(G) {}

KivaSystem::~KivaSystem() {}

void KivaSystem::initialize_zones() {
    zone_endpoints.resize(6);
    zone_task_batches.resize(6);  // per zone
    endpoint_to_zone.clear();

    for (int id : G.endpoints) {
        int x = id % G.cols;
        int y = id / G.cols;

        int zone = 0;
        if (y < G.rows / 2) {
            if (x < G.cols / 3) zone = 0;
            else if (x < 2 * G.cols / 3) zone = 1;
            else zone = 2;
        } else {
            if (x < G.cols / 3) zone = 3;
            else if (x < 2 * G.cols / 3) zone = 4;
            else zone = 5;
        }

        zone_endpoints[zone].push_back(id);
        endpoint_to_zone[id] = zone;
    }
}

void KivaSystem::generate_zone_task_batch() {
    const int MAX_TASKS = 750;
    std::vector<double> zone_weights = {0.17, 0.17, 0.17, 0.16, 0.17, 0.16};
    int total = 0;

    for (int z = 0; z < 6; z++) {
        int count = static_cast<int>(zone_weights[z] * MAX_TASKS);
        const auto& candidates = zone_endpoints[z];
        for (int i = 0; i < count && !candidates.empty(); i++) {
            int goal = candidates[rand() % candidates.size()];
            zone_task_batches[z].push(goal);
            total++;
           // std::cout << "Goal " << goal << "in zone"<<z<<std::endl;
        }
    }
   // std::cout << "[Batch] Added " << total << " new tasks to zones\n";
}


void KivaSystem::initialize() {
    initialize_solvers();

    starts.resize(num_of_drives);
    goal_locations.resize(num_of_drives);
    paths.resize(num_of_drives);
    finished_tasks.resize(num_of_drives);
    agent_zone.resize(num_of_drives);

    bool succ = load_records();
    if (!succ) {
        timestep = 0;
        succ = load_locations();
        if (!succ) {
            std::cout << "Randomly generating initial locations\n";
            initialize_start_locations();
        }
    }
}

void KivaSystem::initialize_start_locations() {
    std::vector<int> zone_count(6, 0);
    auto pickups = G.pickup_locations;  // make a local copy to shuffle

    if (pickups.size() < num_of_drives) {
        std::cerr << "Not enough pickup locations for all agents!" << std::endl;
        return;
    }

    // Shuffle pickups randomly using a random device
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pickups.begin(), pickups.end(), g);

    for (int k = 0; k < num_of_drives; ++k) {
        int home = pickups[k % pickups.size()];  // Now randomized

        starts[k] = State(home, 0, consider_rotation ? rand() % 4 : -1);
        paths[k].push_back(starts[k]);
        finished_tasks[k].emplace_back(home, 0);

        // Assign zone with least agents so far
        int min_zone = 0;
        for (int z = 1; z < 6; ++z)
            if (zone_count[z] < zone_count[min_zone]) min_zone = z;

        agent_zone[k] = min_zone;
        zone_count[min_zone]++;
        
       // std::cout << "Agent " << k << "   starts at " << home << " and is assigned to zone " << min_zone << std::endl;
    }
}

vector<int> get_greedy_pickup_order(int start, const vector<int>& pickups, const BasicGraph& G) {
    vector<int> order;
    vector<bool> visited(pickups.size(), false);
    int current = start;

    for (int i = 0; i < pickups.size(); ++i) {
        int best_idx = -1;
        int min_dist = INT_MAX;

        for (int j = 0; j < pickups.size(); ++j) {
            if (!visited[j]) {
                int dist = G.get_Manhattan_distance(current, pickups[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = j;
                }
            }
        }

        if (best_idx != -1) {
            visited[best_idx] = true;
            order.push_back(pickups[best_idx]);
            current = pickups[best_idx];
        }
    }

    return order;
}

void KivaSystem::initialize_goal_locations(int capacity) {
    if (hold_endpoints || useDummyPaths) return;
 

    std::random_device rd;
    std::mt19937 g(rd());

    for (int k = 0; k < num_of_drives; ++k) {
        int zone = agent_zone[k];

        std::vector<int> raw_pickups;
        for (int i = 0; i < capacity && !zone_task_batches[zone].empty(); ++i) {
            raw_pickups.push_back(zone_task_batches[zone].front());
            zone_task_batches[zone].pop();
        }

        if (raw_pickups.empty()) {
            std::cout << "No task to allot for agent " << k << std::endl;
            continue;
        }

        // Compute optimal order using Held-Karp TSP
        std::vector<int> ordered_pickups = get_greedy_pickup_order(starts[k].location, raw_pickups, G);

        // Add pickups in order
        for (int goal : ordered_pickups) {
            goal_locations[k].emplace_back(goal, 0);
        }

        // Add one dropoff at end
        if (!G.dropoff_locations.empty()) {
            std::uniform_int_distribution<> dist(0, G.dropoff_locations.size() - 1);
            int dropoff = G.dropoff_locations[dist(g)];
            goal_locations[k].emplace_back(dropoff, 0);
            std::cout << "[Init Goal] Agent " << k << ": Pickups " ;
            for(auto it : ordered_pickups){std::cout<<it<<" ";} 
            cout<< " → Dropoff " << dropoff << std::endl;
        } else {
            std::cerr << "No dropoff locations defined in the graph!" << std::endl;
        }
    }
}


void KivaSystem::update_goal_locations(int capacity) {
    if (!LRA_called)
        new_agents.clear();

    if (hold_endpoints) {
        unordered_map<int, int> held_locations; 

        for (int k = 0; k < num_of_drives; k++) {
            int curr = paths[k][timestep].location;

            if (goal_locations[k].empty()) {
                int zone = agent_zone[k];
                if (!zone_task_batches[zone].empty()) {
                    int next = zone_task_batches[zone].front();
                    zone_task_batches[zone].pop();
                    goal_locations[k].emplace_back(next, 0);
                    held_endpoints.insert(next);
                }
            }

            if (!goal_locations[k].empty() &&
                paths[k].back().location == goal_locations[k].back().first &&
                paths[k].back().timestep >= goal_locations[k].back().second) {
                // Agent has reached its goal
                int agent = k;
                int loc = goal_locations[k].back().first;
                auto it = held_locations.find(loc);
                while (it != held_locations.end()) {
                    int removed_agent = it->second;
                    new_agents.remove(removed_agent);
                    held_locations[loc] = agent;
                    agent = removed_agent;
                    loc = paths[agent][timestep].location;
                    it = held_locations.find(loc);
                }
                held_locations[loc] = agent;
            } else if (!goal_locations[k].empty()) {
                int goal_loc = goal_locations[k].back().first;
                if (held_locations.find(goal_loc) == held_locations.end()) {
                    held_locations[goal_loc] = k;
                    new_agents.emplace_back(k);
                } else {
                    // Conflict - hold start location instead
                    int agent = k;
                    int loc = curr;
                    auto it = held_locations.find(loc);
                    while (it != held_locations.end()) {
                        int removed_agent = it->second;
                        new_agents.remove(removed_agent);
                        held_locations[loc] = agent;
                        agent = removed_agent;
                        loc = paths[agent][timestep].location;
                        it = held_locations.find(loc);
                    }
                    held_locations[loc] = agent;
                }
            }
        }
    } else {
        for (int k = 0; k < num_of_drives; k++) {
            int curr = paths[k][timestep].location;
    
            if (useDummyPaths) {
                if (goal_locations[k].empty()) {
                    goal_locations[k].emplace_back(G.agent_home_locations[k], 0);
                }
    
                if (goal_locations[k].size() == 1) {
                    int zone = agent_zone[k];
                    if (!zone_task_batches[zone].empty()) {
                        int next = zone_task_batches[zone].front();
                        zone_task_batches[zone].pop();
                        goal_locations[k].emplace(goal_locations[k].begin(), next, 0);
                        new_agents.emplace_back(k);
                    }
                }
            } else {
                if (goal_locations[k].empty() ||
                    (goal_locations[k].size() == 1 &&
                     paths[k].back().location == goal_locations[k].back().first &&
                     paths[k].back().timestep >= goal_locations[k].back().second)) {
    
                    int zone = agent_zone[k];
                    std::vector<int> new_pickups;
                     
    
                    for (int i = 0; i < capacity && !zone_task_batches[zone].empty(); ++i) {
                        int next = zone_task_batches[zone].front();
                        zone_task_batches[zone].pop();
                        new_pickups.push_back(next);
                    }
    
                    if (!new_pickups.empty()) {
                        std::vector<int> ordered = get_greedy_pickup_order(curr, new_pickups, G);
    
                        // Add pickups
                        for (int id : ordered)
                            goal_locations[k].emplace_back(id, 0);
    
                        // Add one dropoff
                        if (!G.dropoff_locations.empty()) {
                            int drop = G.dropoff_locations[k % G.dropoff_locations.size()];
                            goal_locations[k].emplace_back(drop, 0);
                        }
    
                        new_agents.emplace_back(k);
                    }
                }
            }
        }
    }
}
    



void KivaSystem::simulate(int simulation_time) {
    std::cout << "*** Simulating " << seed << " ***\n";
    this->simulation_time = simulation_time;

    initialize_zones();
    generate_zone_task_batch();
    initialize();
    initialize_goal_locations(3);

    for (; timestep < simulation_time; timestep += simulation_window) {
        std::cout << "Timestep " << timestep << std::endl;

        if (timestep && timestep % 300 == 0) {
            generate_zone_task_batch();
        }

        update_start_locations();
        update_goal_locations(3);
        solve();

        auto new_finished = move();
        std::cout << new_finished.size() << " tasks completed\n";

        for (auto [id, loc, t] : new_finished) {
            finished_tasks[id].emplace_back(loc, t);
            num_of_tasks++;
            if (hold_endpoints) held_endpoints.erase(loc);
        }

        if (congested()) {
            std::cout << "***** Too many traffic jams ***\n";
            break;
        }
    }

    update_start_locations();
    std::cout << "\nDone!\n";
    save_results();
}