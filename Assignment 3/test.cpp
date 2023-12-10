#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

typedef struct
{
    int num_tasks;
    vector<long> task_ids;
} desc_tasks_t;

int main()
{
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype DESC_TASKS_TYPE;
    MPI_Type_contiguous(sizeof(int) + 5 * sizeof(long), MPI_BYTE, &DESC_TASKS_TYPE);
    MPI_Type_commit(&DESC_TASKS_TYPE);

    printf("Hello world from rank %d of %d\n", rank, size);

    if (rank == 0)
    {
        vector<MPI_Request> requests(1);
        int data = 42;

        MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &requests[0]);

        int outcount = 0;
        int indices[1];
        printf("1 outcount: %d\n", outcount);
        MPI_Waitsome(1, requests.data(), &outcount, indices, MPI_STATUSES_IGNORE);

        printf("outcount: %d\n", outcount);
        printf("indices[0]: %d\n", indices[0]);
    }
    else
    {
        int data;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Received data %d\n", data);
    }

    MPI_Finalize();

    return 0;
}
