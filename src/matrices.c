#include "matrices.h"
#include <omp.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>


#define TOLERANCE 1e-9  /* Tolerance used in the assert function.*/


void mallocMatrices(struct Matrices* m, size_t dim)
{
    m->dim = dim;

    m->A = (data_t*)malloc(dim * dim * sizeof(data_t));
    m->Q = (data_t*)malloc(dim * dim * sizeof(data_t));
    m->R = (data_t*)malloc(dim * dim * sizeof(data_t));

    if(!m->A || !m->Q || !m->R)
    {
        exit(1);
    }
}


void cudaMallocMatrices(struct Matrices* m, size_t dim)
{
    m->dim = dim;

    cudaMallocHost((void**)&(m->A), dim * dim * sizeof(data_t));
    cudaMallocHost((void**)&(m->Q), dim * dim * sizeof(data_t));
    cudaMallocHost((void**)&(m->R), dim * dim * sizeof(data_t));

    if(!m->A || !m->Q || !m->R)
    {
        exit(1);
    }
}


void cudaMallocManagedMatrices(struct Matrices* m, size_t dim)
{
    m->dim = dim;

    cudaMallocManaged((void**)&(m->A), dim * dim * sizeof(data_t), cudaMemAttachGlobal);
    cudaMallocManaged((void**)&(m->Q), dim * dim * sizeof(data_t), cudaMemAttachGlobal);
    cudaMallocManaged((void**)&(m->R), dim * dim * sizeof(data_t), cudaMemAttachGlobal);

    if(!m->A || !m->Q || !m->R)
    {
        exit(1);
    }
}


void initMatrices(struct Matrices* m)
{
    size_t i, j;
    size_t n = m->dim;
    
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
        {
            m->A[i * n + j] = ((data_t)i * j + 1);
            m->Q[i * n + j] = (i == j);
            m->R[i * n + j] = 0;
        }
}


void transposeMatrices(struct Matrices* m)
{
    size_t i, j;
    size_t n = m->dim;
    data_t tempA, tempQ;

    // Parallelizes the outer loop between the CPU cores.
    #pragma omp parallel for schedule(static)
    for(i = 0; i < n; i++) 
    {
        for(j = 0; j < n; j++)
        {
            if(i < j)
            {
                tempA = m->A[i * n + j];
                m->A[i * n + j] = m->A[j * n + i];
                m->A[j * n + i] = tempA;

                tempQ = m->Q[i * n + j];
                m->Q[i * n + j] = m->Q[j * n + i];
                m->Q[j * n + i] = tempQ;
            }
        }
    }
}


void printMatrices(struct Matrices* m, FILE* outFile)
{
    size_t i, j;
    size_t n = m->dim;

    fprintf(outFile, "Matrix A:\n");
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
            fprintf(outFile, "%.2f ", m->A[i * n + j]);

        fprintf(outFile, "\n");
    }

    fprintf(outFile, "\nMatrix Q:\n");
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
            fprintf(outFile, "%.2f ", m->Q[i * n + j]);

        fprintf(outFile, "\n");
    }

    fprintf(outFile, "\nMatrix R:\n");
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
            fprintf(outFile, "%.2f ", m->R[i * n + j]);

        fprintf(outFile, "\n");
    }
}


void printQRDecomposition(struct Matrices* m, FILE* outFile)
{
    size_t i, j;
    size_t n = m->dim;

    fprintf(outFile, "\nMatrix Q:\n");
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
            fprintf(outFile, "%.2f ", m->Q[i * n + j]);

        fprintf(outFile, "\n");
    }

    fprintf(outFile, "\nMatrix R:\n");
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
            fprintf(outFile, "%.2f ", m->R[i * n + j]);

        fprintf(outFile, "\n");
    }
}


void assertMatrices(struct Matrices* matrices1, struct Matrices* matrices2)
{
    if (matrices1->dim != matrices2->dim) {
        printf("Dimensions do not match.\n");
        exit(1);
    }

    size_t i, j, k;
    size_t n = matrices1->dim;

    /* Parallelizes the 2 outer-most loops by collapsing them and
     * by splitting the computation between the CPU cores.
     */
    #pragma omp parallel for schedule(static) collapse(2)
    for (i = 0; i < n; i++) 
        for (j = 0; j < n; j++) 
            for (k = 0; k < n; k++)
            {
                matrices1->A[i * n + j] += matrices1->Q[i * n + k] * matrices1->R[k * n + j];
                matrices2->A[i * n + j] += matrices2->Q[i * n + k] * matrices2->R[k * n + j];
            }

    /* Parallelizes the 2 loops by collapsing them and
     * by splitting the computation between the CPU cores.
     */
    #pragma omp parallel for schedule(static) collapse(2)
    for (i = 0; i < n; i++) 
        for (j = 0; j < n; j++) 
            if (fabs(matrices1->A[i * n + j] - matrices2->A[i * n + j]) > TOLERANCE)
            {
                printf("Matrices are not equal at element (%zu, %zu).\n", i, j);
                exit(1);
            }

    printf("All matrices are equal within the specified tolerance.\n");
}


void freeMatrices(struct Matrices* m)
{
    free(m->A);
    free(m->Q);
    free(m->R);
}


void cudaFreeMatrices(struct Matrices* m)
{
    cudaFree(m->A);
    cudaFree(m->Q);
    cudaFree(m->R);
}