#include "omp_qrd.h"
#include <omp.h>
#include <math.h>


void omp_qrd(struct Matrices* m)
{
    size_t i, j, k;
    data_t nrm = 0, red1 = 0;

    size_t n = m->dim;
    data_t* A = m->A;
    data_t* Q = m->Q;
    data_t* R = m->R;


    for(k = 0; k < n; k++)
    {
        nrm = 0;

        // Performs a reduction on the variable nrm.
        #pragma omp parallel for schedule(static) reduction(+: nrm) \
                    default(none) shared(A) firstprivate(k, n) private(i)
        for(i = 0; i < n; i++)
            nrm += A[i * n + k] * A[i * n + k];

        R[k * n + k] = sqrt(nrm);

        /* Computes the Q matrix elements using parallelization and
         * vectorization using SIMD units (if available).
         */
        #pragma omp parallel for simd schedule(static) \
                    default(none) shared(A, Q, R) firstprivate(k, n) private(i)
        for(i = 0; i < n; i++)
            Q[i * n + k] = A[i * n + k] / R[k * n + k];

        // Distributes the workload using dynamic load balancing.
        #pragma omp parallel for schedule(dynamic, 1) \
                    default(none) shared(A, Q, R) firstprivate(k, n) private(j, i, red1)
        for(j = k + 1; j < n; j++)
        {
            red1 = 0;

            for(i = 0; i < n; i++)
                red1 += Q[i * n + k] * A[i * n + j];
            
            R[k * n + j] = red1;


            for (i = 0; i < n; i++) {
                A[i * n + j] -= Q[i * n + k] * R[k * n + j];
            }
        }
    }
}


void omp_transposed_qrd(struct Matrices* m)
{
    size_t i, j, k;
    data_t nrm = 0, red1 = 0;

    size_t n = m->dim;
    data_t* A = m->A;
    data_t* Q = m->Q;
    data_t* R = m->R;

    for(k = 0; k < n; k++)
    {
        nrm = 0;

        // Performs a reduction on the variable nrm.
        #pragma omp parallel for schedule(static) reduction(+: nrm) \
                    default(none) shared(A) firstprivate(k, n) private(i)
        for(i = 0; i < n; i++)
            nrm += A[k * n + i] * A[k * n + i];   

        R[k * n + k] = sqrt(nrm);

        /* Computes the Q matrix elements using parallelization and
         * vectorization using SIMD units (if available).
         */
        #pragma omp parallel for simd schedule(static) \
                    default(none) shared(A, Q, R) firstprivate(k, n) private(i)
        for(i = 0; i < n; i++)
            Q[k * n + i] = A[k * n + i] / R[k * n + k];

        // Distributes the workload using dynamic load balancing.
        #pragma omp parallel for schedule(dynamic, 1) \
                    default(none) shared(A, Q, R) firstprivate(k, n) private(j, i, red1)
        for(j = k + 1; j < n; j++)
        {
            red1 = 0;

            for(i = 0; i < n; i++)
                red1 += Q[k * n + i] * A[j * n + i];
            
            R[k * n + j] = red1;

            for(i = 0; i < n; i++) 
                A[j * n + i] -= Q[k * n + i] * R[k * n + j];
        }
    }
}


void omp_target_qrd(struct Matrices* m)
{
    size_t i, j, k;
    data_t nrm = 0, red1 = 0;

    size_t n = m->dim;
    data_t* A = m->A;
    data_t* Q = m->Q;
    data_t* R = m->R;

    // Creates the teams and maps the data to the target.
    #pragma omp target teams \
                map(tofrom: A[0:n*n], Q[0:n*n], R[0:n*n]) \
                map(to: n, i, j, k, nrm, red1)
    for (k = 0; k < n; k++)
    {
        nrm = 0;

        /* Distributes the work between the teams (high-level parallelism) 
         * and between the threads of a team (low-level parallelism), 
         * performs the reduction on the variable nrm.
         */
        #pragma omp distribute parallel for reduction(+: nrm)
        for (i = 0; i < n; i++)
            nrm += A[k * n + i] * A[k * n + i];

        R[k * n + k] = sqrt(nrm);

        /* Distributes the work between the teams (high-level parallelism)
         * and between the threads of a team (low-level parallelism).
         */
        #pragma omp distribute parallel for
        for (i = 0; i < n; i++)
            Q[k * n + i] = A[k * n + i] / R[k * n + k];

        // Distributes the work (high-level parallelism) between the teams.
        #pragma omp distribute
        for (j = k + 1; j < n; j++)
        {
            red1 = 0;

            /* Distributes the work between the threads of a team (low-level parallelism),
             * performs the reduction on the variable red1.
             */
            #pragma omp parallel for reduction(+: red1)
            for (i = 0; i < n; i++)
                red1 += Q[k * n + i] * A[j * n + i];

            R[k * n + j] = red1;

            // Distributes the work between the threads of a team (low-level parallelism).
            #pragma omp parallel for 
            for (i = 0; i < n; i++)
                A[j * n + i] -= Q[k * n + i] * R[k * n + j];
        }
    }
}
