#pragma once
#include "BasicSystem.h"
#include "KivaGraph.h"

class KivaSystem :
	public BasicSystem
{
public:
	KivaSystem(const KivaGrid& G, MAPFSolver& solver);
	~KivaSystem();

	void simulate(int simulation_time);


private:
	// Add to KivaSystem.h (in class declaration)
	std::vector<bool> is_inbound_agent; // true = inbound, false = outbound
	std::vector<std::queue<std::vector<int>>> inbound_tasks;  // [zone] c shelves
	std::vector<std::queue<std::vector<int>>> outbound_tasks; // [zone] c shelves

	const KivaGrid& G;
	const int PICKUP_BATCH_SIZE = 3;
	unordered_set<int> held_endpoints;
	std::vector<std::vector<int>> zone_endpoints;
	std::unordered_map<int, int> endpoint_to_zone;
	std::vector<int> agent_zone;
	std::vector<std::queue<int>> zone_task_batches;
	void initialize();
	void initialize_start_locations();
	void initialize_goal_locations(int capacity );
	void update_goal_locations(int capacity);
	void initialize_zones();
	void generate_zone_task_batch();
};
