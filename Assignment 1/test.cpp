#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <random>

using namespace std;

int main()
{
    long long sum = 0LL;
    int n = 10000;
    int m = 10000;
    int nThreads = 8;

#pragma omp parallel num_threads(nThreads)
    {
        // cout << "Thread " << omp_get_thread_num() << " starting" << endl;
#pragma omp for collapse(2) reduction(+ : sum)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                // sleep for a random float time between 0 and 1 seconds
                // float randNum = (float)rand() / (5*RAND_MAX);
                // cout << "Thread " << omp_get_thread_num() << " sleeping for " << randNum << " seconds" << endl;
                // sleep(randNum / RAND_MAX);
                // #pragma omp atomic
                sum += 1LL;
            }
        }
    }
    cout << "Sum: " << sum << endl;
    return 0;
}