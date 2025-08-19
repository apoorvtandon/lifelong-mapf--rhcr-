#pragma once
#include "BasicSystem.h"
#include "KivaGraph.h"

class KivaSystem : public BasicSystem
{
public:
    KivaSystem(const KivaGrid& G, MAPFSolver& solver);
    ~KivaSystem();

    void simulate(int simulation_time);

private:
    // Core warehouse layout and graph
    const KivaGrid& G;
    vector<int> shelf_is_inbound;
    // Zone handling
    std::vector<std::vector<int>> zone_endpoints;       // endpoints per zone
    std::unordered_map<int, int> endpoint_to_zone;      // endpoint -> zone
    std::vector<int> agent_zone;                        // agent -> zone mapping
    std::vector<std::queue<int>> zone_task_batches;     // zone -> tasks queue

    // Agent state
    std::vector<int> current_load;                      // how many items agent is carrying
    std::vector<std::vector<int>> cargo;                // per agent cargo item IDs
    std::vector<bool> is_inbound_agent;                 // optional flag

    // Parameters
    unordered_set<int> held_endpoints;
    const int PICKUP_BATCH_SIZE = 3;

    // Initial setup
    void initialize();
    void initialize_start_locations();
    void initialize_zones();
    void generate_zone_task_batch();
    std::pair<int,int> count_inbound_outbound(const std::vector<int>& tasks) const;

 
    void initialize_goal_locations(int capacity);
    void update_goal_locations(int capacity);

 
    std::vector<int> queue_to_vector(int zone) const;  
    bool remove_task_from_zone_queue(int zone, int task_id);  
    std::vector<int> get_candidates_near_path(const std::vector<int>& path_nodes,
                                              const std::vector<int>& candidates,
                                              const KivaGrid& G, int threshold) const;  
    void append_dropoff_if_needed(int agent_id, std::mt19937 g); 

    // Greedy ordering helper
    std::vector<int> get_greedy_pickup_order(int start,
                                             const std::vector<int>& pickups,
                                             const BasicGraph& G) const;
};
