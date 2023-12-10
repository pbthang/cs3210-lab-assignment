#include "iom.h"
#include <iostream>
#include <omp.h>

using namespace std;

int iom(int nThreads, int nGenerations, vector<vector<int>> &startWorld, int nRows, int nCols, int nInvasions, vector<int> invasionTimes, vector<vector<vector<int>>> invasionPlans)
{
    int deathToll = 0;
    int invasionIndex = 0;
    vector<vector<int>> world = startWorld;

    // Set the number of threads to use
    omp_set_num_threads(nThreads);

    // go through every generation
    for (int i = 0; i < nGenerations; i++)
    {

        // update the world

        // invasion happen
        if (invasionIndex < nInvasions && i == invasionTimes[invasionIndex] - 1)
        {
            updateWorld(world, nRows, nCols, true, invasionPlans[invasionIndex], deathToll);
            invasionIndex++;
        }
        else
        {
            updateWorld(world, nRows, nCols, false, startWorld, deathToll);
        }
    }

    return deathToll;
}

// go through every cell in the world and update the value
void updateWorld(vector<vector<int>> &world, int nRows, int nCols, bool isInvaded, vector<vector<int>> invasionPlan, int &deathToll)
{
    vector<vector<int>> newWorld = world;

#pragma omp parallel for reduction(+ : deathToll) collapse(2)
    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            int cellValue = world[row][col];
            vector<int> surroundingCells = getSurroundingCells(row, col, world, nRows, nCols);

            // invasion happen
            if (isInvaded)
            {
                if (cellValue == 0)
                {
                    newWorld[row][col] = getReproduction(surroundingCells);
                }

                else
                {
                    if (isInFight(surroundingCells, cellValue) && invasionPlan[row][col] == 0)
                    {
                        newWorld[row][col] = 0;
                        deathToll++;
                    }
                    // over or under population
                    else if (isOverpopulation(surroundingCells, cellValue) || isUnderpopulation(surroundingCells, cellValue))
                    {
                        newWorld[row][col] = 0;
                    }
                }
                if (invasionPlan[row][col] != 0)
                {
                    if (world[row][col] != 0)
                    {
                        deathToll++;
                    }
                    newWorld[row][col] = invasionPlan[row][col];
                }
            }

            else
            {
                // cell has no number living on it
                if (cellValue == 0)
                {
                    // reproduction
                    newWorld[row][col] = getReproduction(surroundingCells);
                }

                // cell has a number living on it
                else
                {
                    // is fighting
                    if (isInFight(surroundingCells, cellValue))
                    {
                        newWorld[row][col] = 0;
                        deathToll++;
                    }

                    // over or under population
                    else if (isOverpopulation(surroundingCells, cellValue) || isUnderpopulation(surroundingCells, cellValue))
                    {
                        newWorld[row][col] = 0;
                    }
                }
            }
        }
    }
    world = newWorld;

    // print the deaths count
    // cout<< "Deaths: " << deathToll << endl;
}

// check if the cell has any number living on
bool hasAny(int row, int col, vector<vector<int>> &world)
{
    return world[row][col] != 0;
}

// return the surrounding cells value in an array
vector<int> getSurroundingCells(int row, int col, vector<vector<int>> &world, int nRows, int nCols)
{
    vector<int> surroundingCells;
    int rowAbove = row - 1;
    int rowBelow = row + 1;
    int colLeft = col - 1;
    int colRight = col + 1;

    // check if the cell is on the edge of the world
    if (row == 0)
    {
        rowAbove = nRows - 1;
    }
    if (row == nRows - 1)
    {
        rowBelow = 0;
    }
    if (col == 0)
    {
        colLeft = nCols - 1;
    }
    if (col == nCols - 1)
    {
        colRight = 0;
    }

    // add the surrounding cells to the array
    surroundingCells.push_back(world[rowAbove][colLeft]);
    surroundingCells.push_back(world[rowAbove][col]);
    surroundingCells.push_back(world[rowAbove][colRight]);
    surroundingCells.push_back(world[row][colLeft]);
    surroundingCells.push_back(world[row][colRight]);
    surroundingCells.push_back(world[rowBelow][colLeft]);
    surroundingCells.push_back(world[rowBelow][col]);
    surroundingCells.push_back(world[rowBelow][colRight]);

    return surroundingCells;
}

// check if the cell is in fight (at least 1 different value in the surrounding cells)
bool isInFight(vector<int> surroundingCells, int cellValue)
{
    for (int i = 0; i < (int)surroundingCells.size(); i++)
    {
        if (surroundingCells[i] != 0 && surroundingCells[i] != cellValue)
        {
            return true;
        }
    }
    return false;
}

// check if the cell is underpopulation
bool isUnderpopulation(vector<int> surroundingCells, int cellValue)
{
    int count = 0;
    for (int i = 0; i < (int)surroundingCells.size(); i++)
    {
        if (surroundingCells[i] == cellValue)
        {
            count++;
        }
    }
    if (count < 2)
    {
        return true;
    }
    return false;
}

// check if the cell is overpopulation
bool isOverpopulation(vector<int> surroundingCells, int cellValue)
{
    int count = 0;
    for (int i = 0; i < (int)surroundingCells.size(); i++)
    {
        if (surroundingCells[i] == cellValue)
        {
            count++;
        }
    }
    if (count > 3)
    {
        return true;
    }
    return false;
}

// check if the cell can be reproducted (tie-break included)
int getReproduction(vector<int> surroundingCells)
{
    // {1, 2, 3, 4, 5, 6, 7, 8, 9}
    vector<int> countSurroundingCells = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < (int)surroundingCells.size(); i++)
    {
        if (surroundingCells[i] != 0)
        {
            countSurroundingCells[surroundingCells[i] - 1]++;
        }
    }

    // go through the surrounding cells count and return the highest number with count == 3
    int maxIndex = 0;
    for (int i = 0; i < 9; i++)
    {
        if (countSurroundingCells[i] == 3)
        {
            maxIndex = i + 1;
        }
    }

    // reproduction is possible if the max count is exactly 3
    return maxIndex;
}
