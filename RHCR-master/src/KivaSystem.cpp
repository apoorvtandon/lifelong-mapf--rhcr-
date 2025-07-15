#include "KivaSystem.h"
#include "WHCAStar.h"
#include "ECBS.h"
#include "LRAStar.h"
#include "PBS.h"

KivaSystem::KivaSystem(const KivaGrid& G, MAPFSolver& solver): BasicSystem(G, solver), G(G) {}

KivaSystem::~KivaSystem() {}

void KivaSystem::initialize_zones() {
    zone_endpoints.resize(6);
    endpoint_to_zone.clear();

    for (int id : G.endpoints) {
        int x = id % G.cols;
        int y = id / G.cols;

        int zone = 0;
        if (y < G.rows / 2) {
            if (x < G.cols / 3) zone = 0;           // left-top
            else if (x < 2 * G.cols / 3) zone = 1;   // center-top
            else zone = 2;                           // right-top
        } else {
            if (x < G.cols / 3) zone = 3;           // left-bottom
            else if (x < 2 * G.cols / 3) zone = 4;  // center-bottom
            else zone = 5;                          // right-bottom
        }

        zone_endpoints[zone].push_back(id);
        endpoint_to_zone[id] = zone;
    }
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
            std::cout << "Randomly generating initial locations" << std::endl;
            initialize_start_locations();
            initialize_goal_locations();
        }
    }
}

void KivaSystem::initialize_start_locations() {
    starts.resize(num_of_drives);
    paths.resize(num_of_drives);
    finished_tasks.resize(num_of_drives);
    agent_zone.resize(num_of_drives);

    std::vector<int> zone_agent_count(6, 0); // 6 zones

    for (int k = 0; k < num_of_drives; k++) {
        int home_location = G.agent_home_locations[k];

        starts[k] = State(home_location, 0, consider_rotation ? rand() % 4 : -1);
        paths[k].emplace_back(starts[k]);
        finished_tasks[k].emplace_back(home_location, 0);

        int best_zone = 0;
        for (int z = 1; z < 6; z++) {
            if (zone_agent_count[z] < zone_agent_count[best_zone])
                best_zone = z;
        }

        agent_zone[k] = best_zone;
        zone_agent_count[best_zone]++;

        std::cout << "Agent " << k << " starts at " << home_location
                  << " and assigned to zone " << best_zone << std::endl;
    }
}

void KivaSystem::initialize_goal_locations() {
    if (hold_endpoints || useDummyPaths) return;

    for (int k = 0; k < num_of_drives; k++) {
        int zone = agent_zone[k];
        const std::vector<int>& candidates = zone_endpoints[zone];

        if (candidates.empty()) {
            std::cerr << "Zone " << zone << " has no endpoints!" << std::endl;
            continue;
        }

        int goal = candidates[rand() % candidates.size()];
        goal_locations[k].emplace_back(goal, 0);

        std::cout << "[Init Goal] Agent " << k << " got goal " << goal << " in zone " << zone << std::endl;
    }
}

void KivaSystem::update_goal_locations() {
    if (hold_endpoints || useDummyPaths) return;

    for (int k = 0; k < num_of_drives; k++) {
        int curr = paths[k][timestep].location;
        pair<int, int> goal = goal_locations[k].empty() ? make_pair(curr, 0) : goal_locations[k].back();
        double min_timesteps = G.get_Manhattan_distance(goal.first, curr);

        while (min_timesteps <= simulation_window) {
            int zone = agent_zone[k];
            const std::vector<int>& candidates = zone_endpoints[zone];

            if (candidates.empty()) {
                std::cerr << "Zone " << zone << " has no endpoints!" << std::endl;
                break;
            }

            int next_loc;
            do {
                next_loc = candidates[rand() % candidates.size()];
            } while (next_loc == goal.first);

            pair<int, int> next = make_pair(next_loc, 0);
            goal_locations[k].emplace_back(next);
            min_timesteps += G.get_Manhattan_distance(next.first, goal.first);
            goal = next;

         //   std::cout << "[Update Goal] Agent " << k << " got new goal " << next_loc << " in zone " << zone << std::endl;
        }
    }
}

void KivaSystem::simulate(int simulation_time) {
    std::cout << "*** Simulating " << seed << " ***" << std::endl;
    this->simulation_time = simulation_time;

    initialize_zones();
    initialize();
    for (; timestep < simulation_time; timestep += simulation_window) {
        std::cout << "Timestep " << timestep << std::endl;

        update_start_locations();
        update_goal_locations();

        solve();

        auto new_finished_tasks = move();
        std::cout << new_finished_tasks.size() << " tasks has been finished" << std::endl;

        for (auto task : new_finished_tasks) {
            int id, loc, t;
            std::tie(id, loc, t) = task;
            finished_tasks[id].emplace_back(loc, t);
            num_of_tasks++;
            if (hold_endpoints)
                held_endpoints.erase(loc);
        }

        if (congested()) {
            std::cout << "***** Too many traffic jams ***" << std::endl;
            break;
        }
    }

    update_start_locations();
    std::cout << std::endl << "Done!" << std::endl;
    save_results();
}