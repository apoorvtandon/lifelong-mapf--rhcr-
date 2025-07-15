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
	const KivaGrid& G;
	unordered_set<int> held_endpoints;
	std::vector<std::vector<int>> zone_endpoints;
	std::unordered_map<int, int> endpoint_to_zone;
	std::vector<int> agent_zone;

	void initialize();
	void initialize_start_locations();
	void initialize_goal_locations();
	void update_goal_locations();
	void initialize_zones();
};
