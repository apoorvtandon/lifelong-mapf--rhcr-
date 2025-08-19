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
    shelf_is_inbound.resize(G.rows * G.cols, false); // if it's a vector

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
    auto pickups = G.agent_home_locations;  // make a local copy to shuffle

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

vector<int> KivaSystem::get_greedy_pickup_order(int start, const vector<int>& pickups, const BasicGraph& G) const
{
    if (pickups.empty()) return {};

    vector<int> order;
    order.reserve(pickups.size());
    vector<char> used(pickups.size(), 0);

    int current = start;
    for (size_t step = 0; step < pickups.size(); ++step) {
        int best_idx = -1;
        int bestd = INT_MAX;
        for (size_t j = 0; j < pickups.size(); ++j) {
            if (used[j]) continue;
            int d = G.get_Manhattan_distance(current, pickups[j]);
            if (d < bestd) {
                bestd = d;
                best_idx = static_cast<int>(j);
            }
        }
        if (best_idx == -1 || bestd == INT_MAX) break;
        used[best_idx] = 1;
        order.push_back(pickups[best_idx]);
        current = pickups[best_idx];
    }

    bool improved = true;
    int max_passes = 20;
    int passes = 0;
    while (improved && passes < max_passes) {
        improved = false;
        int n = (int)order.size();
        if (n < 2) break;
        for (int i = 0; i < n - 1 && !improved; ++i) {
            for (int k = i + 1; k < n && !improved; ++k) {
                int prev = (i == 0) ? start : order[i - 1];
                int a = order[i];
                int b = order[k];
                int next = (k == n - 1) ? -1 : order[k + 1];

                int before = 0;
                int after = 0;

                before += G.get_Manhattan_distance(prev, a);
                if (next != -1) before += G.get_Manhattan_distance(b, next);

                after += G.get_Manhattan_distance(prev, b);
                if (next != -1) after += G.get_Manhattan_distance(a, next);

                int delta = after - before;
                if (delta < 0) {
                    std::reverse(order.begin() + i, order.begin() + k + 1);
                    improved = true;
                }
            }
        }
        ++passes;
    }

    return order;
}


// Helper: count inbound vs outbound tasks in a given list of endpoints
std::pair<int,int> KivaSystem::count_inbound_outbound(const std::vector<int>& tasks) const {
    int inbound_count = 0, outbound_count = 0;
    for (int t : tasks) {
        if (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), t) != G.pickup_locations.end())
            inbound_count++;
        else
            outbound_count++;
    }
    return {inbound_count, outbound_count};
}
 
std::vector<int> KivaSystem::queue_to_vector(int zone) const {
    vector<int> res;
    std::queue<int> copy = zone_task_batches[zone];
    while (!copy.empty()) {
        res.push_back(copy.front());
        copy.pop();
    }
    return res;
}


bool KivaSystem::remove_task_from_zone_queue(int zone, int task_id) {
    bool removed = false;
    std::queue<int> tmp;
    while (!zone_task_batches[zone].empty()) {
        int v = zone_task_batches[zone].front();
        zone_task_batches[zone].pop();
        if (!removed && v == task_id) {
            removed = true;
            continue; // skip
        }
        tmp.push(v);
    }
    // swap back
    zone_task_batches[zone].swap(tmp);
    return removed;
}

std::vector<int> KivaSystem::get_candidates_near_path(const std::vector<int>& path_nodes,
                                                      const std::vector<int>& candidates,
                                                      const KivaGrid& G, int threshold) const {
    std::vector<int> res;
    for (int c : candidates) {
        for (int p : path_nodes) {
            if (G.get_Manhattan_distance(c, p) <= threshold) {
                res.push_back(c);
                break;
            }
        }
    }
    return res;
}


void KivaSystem::append_dropoff_if_needed(int k, std::mt19937 g) {
 
    if (G.dropoff_locations.empty()) return;
    if (goal_locations[k].empty()) return;
    std::uniform_int_distribution<> d(0, (int)G.dropoff_locations.size() - 1);
    int drop = G.dropoff_locations[d(g)];
    goal_locations[k].emplace_back(drop, 0);
}


 void KivaSystem::initialize_goal_locations(int capacity) {
    new_agents.clear();
    goal_locations.resize(num_of_drives);

    for (int k = 0; k < num_of_drives; ++k) {
        goal_locations[k].clear();

        int zone = agent_zone[k];
        std::vector<int> initial_pickups;

         for (int i = 0; i < capacity && !zone_task_batches[zone].empty(); ++i) {
            int next = zone_task_batches[zone].front();
            zone_task_batches[zone].pop();
            initial_pickups.push_back(next);
        }

         bool inbound_mode = true;
        if (!G.pickup_locations.empty() && !G.dropoff_locations.empty())
            inbound_mode = (rand() % 2 == 0);
        else if (!G.pickup_locations.empty())
            inbound_mode = true;
        else
            inbound_mode = false;

        int start_loc = paths[k][0].location;  

        if (inbound_mode) {
             if (!G.pickup_locations.empty()) {
                int pickup = G.pickup_locations[k % G.pickup_locations.size()];
                goal_locations[k].emplace_back(pickup, 0);
            }

            auto ordered = get_greedy_pickup_order(start_loc, initial_pickups, G);
            for (int s : ordered) {
                goal_locations[k].emplace_back(s, 0);
                shelf_is_inbound[s] = true;
            }

             std::vector<int> remaining = queue_to_vector(zone);
            std::vector<int> path_nodes;
            for (auto &gp : goal_locations[k]) path_nodes.push_back(gp.first);

            auto near_candidates = get_candidates_near_path(path_nodes, remaining, G, 1);
            auto [inbound_count, outbound_count] = count_inbound_outbound(remaining);
            auto opportunistic_order = get_greedy_pickup_order(path_nodes.back(), near_candidates, G);
            int planned_load = (int)ordered.size();

            for (int oc : opportunistic_order) {
                if (planned_load >= capacity) break;
                bool is_outbound_task = (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), oc) == G.pickup_locations.end());
                if (is_outbound_task && outbound_count <= inbound_count) continue;

                if (remove_task_from_zone_queue(zone, oc)) {
                    goal_locations[k].emplace_back(oc, 0);
                    shelf_is_inbound[oc] = true;
                    planned_load++;
                }
            }

            if (!G.dropoff_locations.empty())
                append_dropoff_if_needed(k, std::mt19937(rand()));

        } else {
             auto ordered = get_greedy_pickup_order(start_loc, initial_pickups, G);
            for (int s : ordered) {
                goal_locations[k].emplace_back(s, 0);
                shelf_is_inbound[s] = false;
            }

            if (!G.dropoff_locations.empty()) {
                int drop = G.dropoff_locations[k % G.dropoff_locations.size()];
                goal_locations[k].emplace_back(drop, 0);
            }
        }

        new_agents.emplace_back(k);
    }
}



void KivaSystem::update_goal_locations(int capacity) {
    std::random_device rd;
    std::mt19937 g(rd());

    if (!LRA_called)
        new_agents.clear();

    if (hold_endpoints) {
        unordered_map<int, int> held_locations;

        for (int k = 0; k < num_of_drives; ++k) {
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

                bool new_mode = (rand() % 2 == 0);
                for (auto &gp : goal_locations[k]) {
                    int shelf = gp.first;
                    if (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), shelf) == G.pickup_locations.end() &&
                        std::find(G.dropoff_locations.begin(), G.dropoff_locations.end(), shelf) == G.dropoff_locations.end()) {
                        shelf_is_inbound[shelf] = new_mode;
                    }
                }
            } else if (!goal_locations[k].empty()) {
                int goal_loc = goal_locations[k].back().first;
                if (held_locations.find(goal_loc) == held_locations.end()) {
                    held_locations[goal_loc] = k;
                    new_agents.emplace_back(k);
                } else {
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
        for (int k = 0; k < num_of_drives; ++k) {
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

            } else {  // Regular mode
                bool needs_new_goals = goal_locations[k].empty() ||
                                       (goal_locations[k].size() == 1 &&
                                        paths[k].back().location == goal_locations[k].back().first &&
                                        paths[k].back().timestep >= goal_locations[k].back().second);

                if (!needs_new_goals) continue;

                int zone = agent_zone[k];
                std::vector<int> new_pickups;
                for (int i = 0; i < capacity && !zone_task_batches[zone].empty(); ++i) {
                    int next = zone_task_batches[zone].front();
                    zone_task_batches[zone].pop();
                    new_pickups.push_back(next);
                }
                if (new_pickups.empty()) continue;

                // Randomly decide inbound or outbound mode
                bool inbound_mode = true;
                if (!G.pickup_locations.empty() && !G.dropoff_locations.empty())
                    inbound_mode = (rand() % 2 == 0);
                else if (!G.pickup_locations.empty())
                    inbound_mode = true;
                else
                    inbound_mode = false;

                goal_locations[k].clear();

                if (inbound_mode) {
                    // Inbound: pickup -> shelves
                    if (!G.pickup_locations.empty()) {
                        int pickup = G.pickup_locations[k % G.pickup_locations.size()];
                        goal_locations[k].emplace_back(pickup, 0);
                    }

                    auto ordered = get_greedy_pickup_order(curr, new_pickups, G);
                    for (int s : ordered) {
                        goal_locations[k].emplace_back(s, 0);
                        shelf_is_inbound[s] = true;
                    }

                    std::vector<int> remaining = queue_to_vector(zone);
                    std::vector<int> path_nodes;
                    for (auto &gp : goal_locations[k]) path_nodes.push_back(gp.first);
                    auto near_candidates = get_candidates_near_path(path_nodes, remaining, G, 1);
                    auto [inbound_count, outbound_count] = count_inbound_outbound(remaining);
                    auto opportunistic_order = get_greedy_pickup_order(path_nodes.back(), near_candidates, G);
                    int planned_load = (int)ordered.size();

                    for (int oc : opportunistic_order) {
                        if (planned_load >= capacity) break;
                        bool is_outbound_task = (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), oc) == G.pickup_locations.end());
                        if (is_outbound_task && outbound_count <= inbound_count) continue;

                        if (remove_task_from_zone_queue(zone, oc)) {
                            goal_locations[k].emplace_back(oc, 0);
                            shelf_is_inbound[oc] = true;
                            planned_load++;
                        }
                    }

                    if (!G.dropoff_locations.empty())
                        append_dropoff_if_needed(k, g);

                } else {
                       
                    auto ordered = get_greedy_pickup_order(curr, new_pickups, G);
                    for (int s : ordered) {
                        goal_locations[k].emplace_back(s, 0);
                        shelf_is_inbound[s] = false;
                    }

                    if (!G.dropoff_locations.empty()) {
                        int drop = G.dropoff_locations[k % G.dropoff_locations.size()];
                        goal_locations[k].emplace_back(drop, 0);
                    }
                }
                new_agents.emplace_back(k);
            }
        }
    }
}



void KivaSystem::simulate(int simulation_time) {
    std::cout << "*** Simulating " << seed << " ***\n";
    this->simulation_time = simulation_time;
    int capacity = 5;
    initialize_zones();
    generate_zone_task_batch();
    initialize();
    initialize_goal_locations(capacity);
    
    for (; timestep < simulation_time; timestep += simulation_window) {
        std::cout << "Timestep " << timestep << std::endl;

        if (timestep && timestep % 300 == 0) {
            generate_zone_task_batch();
        }

        update_start_locations();
        update_goal_locations(capacity);
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
