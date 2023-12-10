#pragma once

#include "tasks.hpp"
#include <queue>
#include <vector>
#include <algorithm>

using std::vector, std::queue;

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params);

queue<task_t> get_initial_tasks(params_t &params);
