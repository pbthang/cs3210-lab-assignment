#include "runner.hpp"
#include <fstream>

/* HEADER FILE START */

#include "tasks.hpp"
#include <queue>
#include <vector>
#include <iostream>
#include <memory>

using std::vector, std::queue, std::unique_ptr;

vector<task_t> get_initial_tasks(params_t &params);

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params);

void execute_tasks(metric_t &stats, vector<task_t> &tasks, vector<task_t> &new_tasks);

void get_task_dist(int queue_size, int num_procs, vector<int> &send_counts, vector<int> &displs);

void get_task_displs(vector<int> &send_counts, vector<int> &displs);

void print_tasks(std::string &&label, vector<task_t> &arr);

void print_arr(std::string &&label, vector<int> &arr);

/* HEADER FILE END */

/*
 * ==========================================
 * | [START] OK TO MODIFY THIS FILE [START] |
 * ==========================================
 */
const task_t EMPTY_TASK = {0, -1, TaskType::BITONIC, 0, 0, 0, {}, {}};

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params)
{
    MPI_Datatype task_t_type;
    MPI_Type_contiguous(sizeof(task_t), MPI_BYTE, &task_t_type);
    MPI_Type_commit(&task_t_type);

    if (rank == 0) // Master
    {
        vector<task_t> task_queue = get_initial_tasks(params);

        while (true)
        {
            // print_tasks("task queue", task_queue);
            // Broadcast task queue size
            int queue_size = task_queue.size();
            MPI_Bcast(&queue_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (queue_size == 0)
            {
                break;
            }

            // Scatterv tasks from the master to all processes
            vector<int> send_counts(num_procs);
            vector<int> displs(num_procs);
            get_task_dist(queue_size, num_procs, send_counts, displs);
            // print_arr("send_counts", send_counts);
            // print_arr("displs", displs);

            vector<task_t> local_tasks(send_counts[rank]);
            MPI_Scatterv(task_queue.data(), send_counts.data(), displs.data(), task_t_type, local_tasks.data(), send_counts[rank], task_t_type, 0, MPI_COMM_WORLD);
            // print_tasks("local tasks", local_tasks);

            // Execute tasks
            vector<task_t> new_tasks;
            execute_tasks(stats, local_tasks, new_tasks);

            // print_tasks("new tasks", new_tasks);

            // AllGather new task counts from all processes
            int num_new_tasks = new_tasks.size();
            vector<int> new_task_counts(num_procs);
            MPI_Allgather(&num_new_tasks, 1, MPI_INT, new_task_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // print_arr("new_task_counts", new_task_counts);

            // Gather new tasks from all processes
            vector<int> new_task_displs(num_procs);
            get_task_displs(new_task_counts, new_task_displs);
            // print_arr("new_task_displs", new_task_displs);
            task_queue.resize(new_task_displs[num_procs - 1] + new_task_counts[num_procs - 1]);
            MPI_Gatherv(new_tasks.data(), num_new_tasks, task_t_type, task_queue.data(), new_task_counts.data(), new_task_displs.data(), task_t_type, 0, MPI_COMM_WORLD);

            // print_tasks("all new tasks", task_queue);
        }
    }
    else // Workers
    {
        while (true)
        {
            vector<task_t> task_queue;

            // Broadcase task queue size
            int queue_size;
            MPI_Bcast(&queue_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (queue_size == 0)
            {
                break;
            }
            // Scatterv tasks from the master to all processes
            vector<int> send_counts(num_procs);
            vector<int> displs(num_procs);
            get_task_dist(queue_size, num_procs, send_counts, displs);
            // print_arr("send_counts", send_counts);
            // print_arr("displs", displs);

            vector<task_t> local_tasks(send_counts[rank]);
            MPI_Scatterv(task_queue.data(), send_counts.data(), displs.data(), task_t_type, local_tasks.data(), send_counts[rank], task_t_type, 0, MPI_COMM_WORLD);
            // print_tasks("local tasks", local_tasks);

            // Execute tasks
            vector<task_t> new_tasks;
            execute_tasks(stats, local_tasks, new_tasks);

            // print_tasks("new tasks", new_tasks);

            // AllGather new task counts from all processes
            int num_new_tasks = new_tasks.size();
            vector<int> new_task_counts(num_procs);
            MPI_Allgather(&num_new_tasks, 1, MPI_INT, new_task_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // Gather new tasks from all processes
            vector<int> new_task_displs(num_procs);
            get_task_displs(new_task_counts, new_task_displs);
            vector<task_t> new_task_buffer(new_task_displs[num_procs - 1] + new_task_counts[num_procs - 1]);
            MPI_Gatherv(new_tasks.data(), num_new_tasks, task_t_type, new_task_buffer.data(), new_task_counts.data(), new_task_displs.data(), task_t_type, 0, MPI_COMM_WORLD);
        }
    }
    std::cout << "Rank " << rank << " completed" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

vector<task_t> get_initial_tasks(params_t &params)
{
    std::vector<task_t> tasks;
    std::ifstream istrm(params.input_path, std::ios::binary);
    // Read initial tasks
    int count;
    istrm >> count;

    for (int i = 0; i < count; ++i)
    {
        task_t task;
        int type;
        istrm >> type >> task.arg_seed;

        task.type = static_cast<TaskType>(type);
        task.id = task.arg_seed;
        task.gen = 0;
        tasks.push_back(task);
    }
    return tasks;
}

void execute_tasks(metric_t &stats, vector<task_t> &tasks, vector<task_t> &new_tasks)
{
    new_tasks.clear();
    vector<task_t> task_buffer(Nmax);
    int num_new_tasks = 0;
    for (int i = 0; i < tasks.size(); ++i)
    {
        execute_task(stats, tasks[i], num_new_tasks, task_buffer);
        // concatenate new tasks
        new_tasks.insert(new_tasks.end(), task_buffer.begin(), task_buffer.begin() + num_new_tasks);
    }
}

void get_task_dist(int queue_size, int num_procs, vector<int> &send_counts, vector<int> &displs)
{
    int quotient = queue_size / num_procs;
    int remainder = queue_size % num_procs;
    int offset = 0;

    for (int i = 0; i < num_procs; ++i)
    {
        send_counts[i] = i < remainder ? quotient + 1 : quotient;
        displs[i] = offset;
        offset += send_counts[i];
    }
}

void get_task_displs(vector<int> &send_counts, vector<int> &displs)
{
    int offset = 0;
    for (int i = 0; i < send_counts.size(); ++i)
    {
        displs[i] = offset;
        offset += send_counts[i];
    }
}

void print_tasks(std::string &&label, vector<task_t> &arr)
{
    int rank;
    int numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    std::cout << "Rank [" << rank << "/" << numProcs << "] " << label << "[" << arr.size() << "]:\t\t";
    for (int i = 0; i < arr.size(); ++i)
    {
        std::cout << "g" << arr[i].gen << "." << (int)arr[i].type << "#" << arr[i].id << " ";
    }
    std::cout << std::endl;
}

void print_arr(std::string &&label, vector<int> &arr)
{
    int rank;
    int numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    std::cout << "Rank [" << rank << "/" << numProcs << "] " << label << ":\t\t";
    for (int i = 0; i < arr.size(); ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
