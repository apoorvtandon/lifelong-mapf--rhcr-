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

// Remove a single task id from the zone queue (if present)
// returns true if removed
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

// Return candidates within threshold distance of any node along path_nodes
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

// Append a dropoff location if route contains any outbound picks (we use presence of any shelf appended after a pickup to indicate outbound content)
void KivaSystem::append_dropoff_if_needed(int k, std::mt19937& g) {
    // If route contains any node but we have at least one shelf (non-pickup) that we expect to drop, append a dropoff.
    // Simpler heuristic: always append a dropoff if dropoff locations exist and the last stops contain shelves (not pickup station).
    if (G.dropoff_locations.empty()) return;
    if (goal_locations[k].empty()) return;
    // assume last nodes are shelves; append one dropoff
    std::uniform_int_distribution<> d(0, (int)G.dropoff_locations.size() - 1);
    int drop = G.dropoff_locations[d(g)];
    goal_locations[k].emplace_back(drop, 0);
}


// -------------------- Replaced initialize_goal_locations --------------------
void KivaSystem::initialize_goal_locations(int capacity) {
    if (hold_endpoints || useDummyPaths) return;

    std::random_device rd;
    std::mt19937 g(rd());

   

    for (int k = 0; k < num_of_drives; ++k) {
        int zone = agent_zone[k];

        // 1) Take up to 'capacity' tasks (temporarily) from zone queue as main candidates
        std::vector<int> main_candidates;
        for (int i = 0; i < capacity && !zone_task_batches[zone].empty(); ++i) {
            main_candidates.push_back(zone_task_batches[zone].front());
            zone_task_batches[zone].pop();
        }

        if (main_candidates.empty()) {
            // nothing for this agent now
            // leave goal_locations[k] empty
            continue;
        }

         std::vector<int> ordered_main = get_greedy_pickup_order(starts[k].location, main_candidates, G);

         bool inbound_mode = true;
        if (!G.pickup_locations.empty() && !G.dropoff_locations.empty()) {
            // 50-50 if both exist
            inbound_mode = (rand() % 2 == 0);
        } else if (!G.pickup_locations.empty()) {
            inbound_mode = true;
        } else {
            inbound_mode = false;
        }

        // 4) Build route: inbound => pickup station then shelves; outbound => shelves then dropoff
        if (inbound_mode) {
            // pickup location first
            if (!G.pickup_locations.empty()) {
                std::uniform_int_distribution<> dist_pick(0, (int)G.pickup_locations.size() - 1);
                int pickup = G.pickup_locations[dist_pick(g)];
                goal_locations[k].emplace_back(pickup, 0);
            }

            // append the ordered main shelves
            for (int s : ordered_main) {
                goal_locations[k].emplace_back(s, 0);
            }

            // current planned load is ordered_main.size()
            int planned_load = (int)ordered_main.size();

            // 5) Opportunistic step: look at the remaining tasks in the zone queue and pick those near the path
            // Convert remaining zone queue to vector
            std::vector<int> remaining = queue_to_vector(zone);

            // gather candidates near the current planned path (pickup + ordered_main)
            std::vector<int> path_nodes;
            for (auto &gp : goal_locations[k]) path_nodes.push_back(gp.first);
            // threshold 1 (direct neighbor). you can tune this number to allow larger detours.
            std::vector<int> near_candidates = get_candidates_near_path(path_nodes, remaining, G, 1);

            // order near_candidates greedily starting from last planned node (if any), else start
           // Count inbound/outbound tasks in remaining queue
            auto [inbound_count, outbound_count] = count_inbound_outbound(remaining);

            int last_node = path_nodes.empty() ? starts[k].location : path_nodes.back();
            std::vector<int> opportunistic_order = get_greedy_pickup_order(last_node, near_candidates, G);

            for (int oc : opportunistic_order) {
                if (planned_load >= capacity) break;

                bool is_outbound_task = (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), oc) == G.pickup_locations.end());
                
                // If outbound task, only allow if outbound_count > inbound_count
                if (is_outbound_task && outbound_count <= inbound_count) continue;

                bool removed = remove_task_from_zone_queue(zone, oc);
                if (!removed) continue;

                goal_locations[k].emplace_back(oc, 0);
                planned_load++;
            }


            // 6) If we picked any outbound shelves (opportunistic picks), or if user wants dropoff after inbound mixes,
            // append a dropoff at the end so outbound items can be delivered.
            // Heuristic: append dropoff if we collected fewer than capacity but there exist dropoff locations and we added any opportunistic picks.
            if (!G.dropoff_locations.empty()) {
                // detect opportunistic picks by checking if any of the last nodes are not in ordered_main
                bool has_opportunistic = false;
                std::unordered_set<int> mainSet(ordered_main.begin(), ordered_main.end());
                for (auto &gp : goal_locations[k]) {
                    if (mainSet.find(gp.first) == mainSet.end()) {
                        // This node is not one of the ordered_main shelves; it might be pickup station or opportunistic shelf
                        // If it's not the pickup station, treat it as opportunistic shelf
                        if (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), gp.first) == G.pickup_locations.end())
                            has_opportunistic = true;
                    }
                }
                if (has_opportunistic) {
                    append_dropoff_if_needed(k, g);
                }
            }
        } else {
            // Outbound-first: start from current location -> shelves -> dropoff
            std::vector<int> ordered_out = get_greedy_pickup_order(starts[k].location, ordered_main, G);
            for (int s : ordered_out) goal_locations[k].emplace_back(s, 0);

            if (!G.dropoff_locations.empty()) {
                std::uniform_int_distribution<> dist_drop(0, (int)G.dropoff_locations.size() - 1);
                int drop = G.dropoff_locations[dist_drop(g)];
                goal_locations[k].emplace_back(drop, 0);
            }
            // Per user instruction "not reverse", we do not opportunistically add inbound picks while outbound.
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
                        bool is_inbound = false;
                        // decide mode adaptively: prefer inbound if pickups exist, else outbound
                        if (!G.pickup_locations.empty() && !G.dropoff_locations.empty()) {
                            is_inbound = (rand() % 2 == 0);
                        } else if (!G.pickup_locations.empty()) {
                            is_inbound = true;
                        } else {
                            is_inbound = false;
                        }

                        if (is_inbound) {
                            // Inbound agent: pickup location -> shelves
                            if (!G.pickup_locations.empty()) {
                                int pickup = G.pickup_locations[k % G.pickup_locations.size()];
                                goal_locations[k].emplace_back(pickup, 0);
                                std::vector<int> ordered = get_greedy_pickup_order(pickup, new_pickups, G);
                                for (int id : ordered)
                                    goal_locations[k].emplace_back(id, 0);
                                new_agents.emplace_back(k);

                                // opportunistic: check remaining queue for near-path outbound shelves
                                std::vector<int> remaining = queue_to_vector(zone);
                                std::vector<int> path_nodes;
                                for (auto &gp : goal_locations[k]) path_nodes.push_back(gp.first);
                                std::vector<int> near_candidates = get_candidates_near_path(path_nodes, remaining, G, 1);
                                auto [inbound_count, outbound_count] = count_inbound_outbound(remaining);
                                std::vector<int> opportunistic_order = get_greedy_pickup_order(path_nodes.back(), near_candidates, G);
                                int planned_load = (int)ordered.size();

                                for (int oc : opportunistic_order) {
                                    if (planned_load >= capacity) break;

                                    bool is_outbound_task = (std::find(G.pickup_locations.begin(), G.pickup_locations.end(), oc) == G.pickup_locations.end());
                                    if (is_outbound_task && outbound_count <= inbound_count) continue;

                                    bool removed = remove_task_from_zone_queue(zone, oc);
                                    if (!removed) continue;

                                    goal_locations[k].emplace_back(oc, 0);
                                    planned_load++;
                                }

                                if (!G.dropoff_locations.empty()) {
                                    // append dropoff if opportunistic picks exist
                                    append_dropoff_if_needed(k, *(new std::mt19937(rand())));
                                }
                            }
                        } else {
                            // Outbound agent: shelves -> dropoff
                            std::vector<int> ordered = get_greedy_pickup_order(curr, new_pickups, G);
                            for (int id : ordered)
                                goal_locations[k].emplace_back(id, 0);
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
