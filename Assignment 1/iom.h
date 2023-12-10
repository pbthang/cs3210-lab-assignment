#include <vector>
using namespace std;

int iom(int nThreads, int nGenerations, vector<vector<int>>& startWorld, int nRows, int nCols, int nInvasions, vector<int> invasionTimes, vector<vector<vector<int>>> invasionPlans);

void updateWorld(vector<vector<int>>& world, int nRows, int nCols, bool isInvaded, vector<vector<int>> invasionPlan, int &deathsCount);

bool hasAny(int row, int col, vector<vector<int>>& world);

vector<int> getSurroundingCells(int row, int col, vector<vector<int>>& world, int nRows, int nCols);

bool isInFight(vector<int> surroundingCells, int cellValue);

bool isUnderpopulation(vector<int> surroundingCells, int cellValue);

bool isOverpopulation(vector<int> surroundingCells, int cellValue);

int getReproduction(vector<int> surroundingCells);
