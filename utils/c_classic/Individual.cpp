#include "Individual.h" 

void Individual::evaluateCompleteCost()
{
	myCostSol = CostSol();
	for (int r = 0; r < params->nbVehicles; r++)
	{
		if (!chromR[r].empty())
		{
			double distance = params->timeCost[0][chromR[r][0]];
			double load = params->cli[chromR[r][0]].demand;
			double service = params->cli[chromR[r][0]].serviceDuration;
			predecessors[chromR[r][0]] = 0;
			for (int i = 1; i < (int)chromR[r].size(); i++)
			{
				distance += params->timeCost[chromR[r][i-1]][chromR[r][i]];
				load += params->cli[chromR[r][i]].demand;
				service += params->cli[chromR[r][i]].serviceDuration;
				predecessors[chromR[r][i]] = chromR[r][i-1];
				successors[chromR[r][i-1]] = chromR[r][i];
			}
			successors[chromR[r][chromR[r].size()-1]] = 0;
			distance += params->timeCost[chromR[r][chromR[r].size()-1]][0];
			myCostSol.distance += distance;
			myCostSol.nbRoutes++;
			if (load > params->vehicleCapacity) myCostSol.capacityExcess += load - params->vehicleCapacity;
			if (distance + service > params->durationLimit) myCostSol.durationExcess += distance + service - params->durationLimit;
		}
	}

	myCostSol.penalizedCost = myCostSol.distance + myCostSol.capacityExcess*params->penaltyCapacity + myCostSol.durationExcess*params->penaltyDuration;
	isFeasible = (myCostSol.capacityExcess < MY_EPSILON && myCostSol.durationExcess < MY_EPSILON);
}

void Individual::removeProximity(Individual * indiv)
{
	auto it = indivsPerProximity.begin();
	while (it->second != indiv) ++it;
	indivsPerProximity.erase(it);
}

double Individual::brokenPairsDistance(Individual * indiv2)
{
	int differences = 0;
	for (int j = 1; j <= params->nbClients; j++)
	{
		if (successors[j] != indiv2->successors[j] && successors[j] != indiv2->predecessors[j]) differences++;
		if (predecessors[j] == 0 && indiv2->predecessors[j] != 0 && indiv2->successors[j] != 0) differences++;
	}
	return (double)differences/(double)params->nbClients;
}

double Individual::averageBrokenPairsDistanceClosest(int nbClosest) 
{
	double result = 0 ;
	int maxSize = std::min<int>(nbClosest, indivsPerProximity.size());
	auto it = indivsPerProximity.begin();
	for (int i=0 ; i < maxSize; i++)
	{
		result += it->first ;
		++it ;
	}
	return result/(double)maxSize ;
}

void Individual::exportCVRPLibFormat(std::string fileName, int show_info)
{
	if (show_info) {
		std::cout << "----- WRITING SOLUTION WITH VALUE " << myCostSol.penalizedCost << " IN : " << fileName << std::endl;
	}
	std::ofstream myfile(fileName);
	if (myfile.is_open())
	{
		for (int k = 0; k < params->nbVehicles; k++)
		{
			if (!chromR[k].empty())
			{
				myfile << "Route #" << k+1 << ":"; // Route IDs start at 1 in the file format
				for (int i : chromR[k]) myfile << " " << i;
				myfile << std::endl;
			}
		}
		myfile << "Cost " << myCostSol.penalizedCost << std::endl;
		myfile << "Time " << (double)clock()/(double)CLOCKS_PER_SEC << std::endl;
	}
	else std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
}

bool Individual::readCVRPLibFormat(std::string fileName, std::vector<std::vector<int>> & readSolution, double & readCost)
{
	readSolution.clear();
	std::ifstream inputFile(fileName);
	if (inputFile.is_open())
	{
		std::string inputString;
		inputFile >> inputString;
		// Loops as long as the first line keyword is "Route"
		for (int r = 0; inputString == "Route" ; r++) 
		{
			readSolution.push_back(std::vector<int>());
			inputFile >> inputString;
			getline(inputFile, inputString);
			std::stringstream ss(inputString);
			int inputCustomer;
			while (ss >> inputCustomer) // Loops as long as there is an integer to read
				readSolution[r].push_back(inputCustomer);
			inputFile >> inputString;
		}
		if (inputString == "Cost")
		{
			inputFile >> readCost;
			return true;
		}
		else std::cout << "----- UNEXPECTED WORD IN SOLUTION FORMAT: " << inputString << std::endl;
	}
	else std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
	return false;
}

Individual::Individual(
	Params * params, short *tour, int tour_length
) : params(params)
{
	successors = std::vector <int>(params->nbClients + 1);
	predecessors = std::vector <int>(params->nbClients + 1);
	chromR = std::vector < std::vector <int> >(params->nbVehicles);
	//std::vector<std::vector<int>> chromR;
	//chromR.reserve(params->nbVehicles);  // optional but good

	chromT = std::vector <int>(params->nbClients);

	//std::fprintf(stderr, "[DEBUG] Individual ctor: tour_length=%d, nbClients=%d, nbVehicles=%d\n",
	//	tour_length, params->nbClients, params->nbVehicles);
	//std::fprintf(stderr, "[DEBUG] tour values: ");
	//for (int i = 0; i < tour_length; i++) std::fprintf(stderr, "%d ", (int)tour[i]);
	//std::fprintf(stderr, "\n");

	std::vector<int> currentRoute;
	int suc_idx = 0;
	int pre_idx = 0;
	int chromR_idx = 0;
	int chromT_idx = 0;
	successors[suc_idx] = 0;
	predecessors[pre_idx] = 0;

	for (int i = 1; i < tour_length; i++){
		if (tour[i] < 1){
			if (!currentRoute.empty()){
				if (chromR_idx >= params->nbVehicles)
					std::fprintf(stderr, "[DEBUG] WARNING: chromR_idx=%d >= nbVehicles=%d at i=%d\n",
						chromR_idx, params->nbVehicles, i);
				if (chromR_idx < params->nbVehicles)
					chromR[chromR_idx] = currentRoute;
				chromR_idx ++;
				currentRoute.clear();
			}
		}
		else{
			if (tour[i] > params->nbClients)
				std::fprintf(stderr, "[DEBUG] WARNING: tour[%d]=%d > nbClients=%d\n",
					i, (int)tour[i], params->nbClients);
			currentRoute.push_back(tour[i]);
			if (chromT_idx >= params->nbClients)
				std::fprintf(stderr, "[DEBUG] WARNING: chromT_idx=%d >= nbClients=%d at i=%d\n",
					chromT_idx, params->nbClients, i);
			if (chromT_idx < params->nbClients)
				chromT[chromT_idx] = tour[i];
			chromT_idx ++;
			// Avoid reading past the end of the tour array
			int next = (i + 1 < tour_length) ? tour[i + 1] : 0;
			successors[tour[i]] = next;
			predecessors[tour[i]] = tour[i-1];
		}

	}
	//std::fprintf(stderr, "[DEBUG] Done: chromR_idx=%d, chromT_idx=%d\n", chromR_idx, chromT_idx);

}

Individual::Individual()
{
	myCostSol.penalizedCost = 1.e30;
}