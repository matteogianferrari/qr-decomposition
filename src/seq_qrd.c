#include "seq_qrd.h"
#include <math.h>


void seq_qrd(struct Matrices* m)
{
    size_t i, j, k;
    data_t nrm = 0, temp = 0;

    size_t n = m->dim;
    data_t* A = m->A;
    data_t* Q = m->Q;
    data_t* R = m->R;

    for(k = 0; k < n; k++)
    {
        // Perfoms the root squared sum.
        nrm = 0;

        for(i = 0; i < n; i++)
            nrm += A[i * n + k] * A[i * n + k];

        R[k * n + k] = sqrt(nrm);

        // Computes the Q matrix elements.
        for(i = 0; i < n; i++)
            Q[i * n + k] = A[i * n + k] / R[k * n + k];

        for(j = k + 1; j < n; j++)
        {
            // Perfoms a reduction on the R matrix elements.
            R[k * n + j] = 0;
            for(i = 0; i < n; i++)
                R[k * n + j] += Q[i * n + k] * A[i * n + j];

            // Updates the A matrix for the next iteration.
            for(i = 0; i < n; i++)
                A[i * n + j] -= Q[i * n + k] * R[k * n + j];
        }
    }
}



void seq_transposed_qrd(struct Matrices* m)
{
    size_t i, j, k;
    data_t nrm = 0, temp = 0;

    size_t n = m->dim;
    data_t* A = m->A;
    data_t* Q = m->Q;
    data_t* R = m->R;

    for(k = 0; k < n; k++)
    {
        // Perfoms the root squared sum.
        nrm = 0;

        for(i = 0; i < n; i++)
            nrm += A[k * n + i] * A[k * n + i];    

        R[k * n + k] = sqrt(nrm);

        // Computes the Q matrix elements.
        for(i = 0; i < n; i++)
            Q[k * n + i] = A[k * n + i] / R[k * n + k];

        for(j = k + 1; j < n; j++)
        {
            // Perfoms a reduction on the R matrix elements.
            R[k * n + j] = 0;
            for(i = 0; i < n; i++)
                R[k * n + j] += Q[k * n + i] * A[j * n + i];

            // Updates the A matrix for the next iteration.
            for(i = 0; i < n; i++)
                A[j * n + i] -= Q[k * n + i] * R[k * n + j];
        }
    }
}