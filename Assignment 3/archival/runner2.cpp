#include "runner.hpp"
#include <fstream>

#include "tasks.hpp"
#include <queue>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>

void print_tasks(std::string &&label, task_t *arr, int size);

void print_task(std::string &&label, task_t &task);

void print_arr(std::string &&label, vector<int> &arr);

/*
 * ==========================================
 * | [START] OK TO MODIFY THIS FILE [START] |
 * ==========================================
 */
#define TERM_TAG 4
#define SIG_TERM -1
#define SIG_CONT 0

const task_t INVALID_TASK = {0, -1, TaskType::BITONIC, 0, 0, 0, {}, {}};

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params)
{
    MPI_Datatype TASK_T_TYPE;
    MPI_Type_contiguous(sizeof(task_t), MPI_BYTE, &TASK_T_TYPE);
    MPI_Type_commit(&TASK_T_TYPE);

    int MASTER = 0;
    int N_WORKERS = num_procs - 1;

    if (rank == MASTER) // Master
    {
        queue<task_t> task_queue = get_initial_tasks(params);
        vector<MPI_Request> proc_requests(N_WORKERS, MPI_REQUEST_NULL);
        vector<bool> availability(N_WORKERS, true);
        vector<task_t *> desc_tasks(num_procs - 1, nullptr);

        while (true)
        {
            // get unavailable processes
            vector<MPI_Request> unavailable_requests;
            unavailable_requests.reserve(N_WORKERS);
            vector<int> unavailable_worker_idxs;
            unavailable_worker_idxs.reserve(N_WORKERS);
            for (int i = 0; i < N_WORKERS; ++i)
            {
                if (!availability[i])
                {
                    unavailable_requests.push_back(proc_requests[i]);
                    unavailable_worker_idxs.push_back(i);
                }
            }

            // wait some unavailable processes
            int outcount = 0;
            vector<int> indices(num_procs);
            MPI_Waitsome(unavailable_requests.size(), unavailable_requests.data(), &outcount, indices.data(), MPI_STATUSES_IGNORE);

            // update availability and put new tasks into task_queue
            for (int i = 0; i < outcount; ++i)
            {
                int idx_in_unavailable = indices[i];
                int idx_in_worker = unavailable_worker_idxs[idx_in_unavailable];
                availability[idx_in_worker] = true;

                task_t *new_tasks = desc_tasks[idx_in_worker];

                for (int j = 0; j < Nmax; ++j)
                {
                    if (new_tasks[j].gen <= 0)
                    {
                        break;
                    }
                    task_queue.push(new_tasks[j]);
                }
            }

            int queue_size = task_queue.size();

            // get worker processes based on availability and number of tasks in queue
            vector<int> worker_procs;
            int num_workers = std::min(queue_size, N_WORKERS);
            for (int i = 0; i < num_workers; ++i)
            {
                if (availability[i])
                {
                    worker_procs.push_back(i);
                    availability[i] = false;
                }
            }

            // send signal to available processes, terminate if no more tasks and all processes are available
            if (task_queue.empty() && std::all_of(availability.begin(), availability.end(), [](bool v)
                                                  { return v; }))
            { // terminate
                for (int i = 0; i < N_WORKERS; ++i)
                {
                    int worker_rank = i + 1;
                    int term = SIG_TERM;
                    MPI_Send(&term, 1, MPI_INT, worker_rank, TERM_TAG, MPI_COMM_WORLD);
                }
                break;
            }
            else
            { // continue
                for (int i = 0; i < worker_procs.size(); ++i)
                {
                    int proc = worker_procs[i];
                    int worker_rank = proc + 1;
                    int cont = SIG_CONT;
                    MPI_Send(&cont, 1, MPI_INT, worker_rank, TERM_TAG, MPI_COMM_WORLD);
                }
            }

            // send 1 task to each available process
            for (int i = 0; i < worker_procs.size(); ++i)
            {
                if (task_queue.empty())
                {
                    break;
                }
                int proc = worker_procs[i];
                int worker_rank = proc + 1;
                task_t task = task_queue.front();
                task_queue.pop();
                MPI_Send(&task, 1, TASK_T_TYPE, worker_rank, 0, MPI_COMM_WORLD);
            }

            // non-blocking receive new tasks from available processes
            for (int i = 0; i < worker_procs.size(); ++i)
            {
                int proc = worker_procs[i];
                int worker_rank = proc + 1;
                desc_tasks[proc] = new task_t[Nmax];

                MPI_Irecv(desc_tasks[proc], Nmax, TASK_T_TYPE, worker_rank, 0, MPI_COMM_WORLD, &proc_requests[proc]);
            }
        }
    }
    else // Workers
    {
        while (true)
        {
            int signal;
            MPI_Recv(&signal, 1, MPI_INT, 0, TERM_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (signal == SIG_TERM)
            {
                break;
            }

            // receive 1 task from master
            task_t task;
            MPI_Recv(&task, 1, TASK_T_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // execute task
            int num_new_tasks = 0;
            std::vector<task_t> task_buffer(Nmax, INVALID_TASK);
            execute_task(stats, task, num_new_tasks, task_buffer);

            // send new tasks to master
            MPI_Send(task_buffer.data(), Nmax, TASK_T_TYPE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

/* =========================================== */
/* ============ UTILITY FUNCTIONS ============ */
/* =========================================== */

queue<task_t> get_initial_tasks(params_t &params)
{
    std::queue<task_t> tasks;
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
        tasks.push(task);
    }
    return tasks;
}

void print_tasks(std::string &&label, task_t *arr, int size)
{
    int rank;
    int numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    std::cout << "Rank [" << rank << "/" << numProcs << "] " << label << "[" << size << "]:\t\t";
    for (int i = 0; i < size; ++i)
    {
        std::cout << "g" << arr[i].gen << "." << (int)arr[i].type << "#" << arr[i].id << "/" << arr[i].arg_seed << " ";
    }
    std::cout << std::endl;
}

void print_task(std::string &&label, task_t &task)
{
    int rank;
    int numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    std::cout << "Rank [" << rank << "/" << numProcs << "] " << label << ":\t\t";
    std::cout << "g" << task.gen << "." << (int)task.type << "#" << task.id << "/" << task.arg_seed << " ";
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
