/**
 * @file    matrices.h
 * @author  Matteo Gianferrari (https://github.com/matteogianferrari)
 *
 * @brief   This file contains the definition of the struct Matrices and
 *          its related functions.
 * 
 * @version 0.1
 * @date    2024-01-01
 */

#ifndef MATRICES_H
#define MATRICES_H

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float data_t;  /* Data type used for a single value in the matrices.*/

/**
 * @struct  Matrices
 * @brief   Struct that contains a triplet of square matrices.
 */
struct Matrices
{
    data_t* A; 
    data_t* Q;
    data_t* R;

    size_t dim;
};

/**
 * @fn      mallocMatrices
 * @brief   Allocates the memory required by the 3 matrices.
 * 
 * @details The matrices are allocated in their vectorized form.
 * 
 * @param   m Pointer to struct Matrices.
 * @param   dim Size of the square matrices.
 */
void mallocMatrices(struct Matrices* m, size_t dim);

/**
 * @fn      cudaMallocMatrices
 * @brief   Allocates the memory required by the 3 matrices.
 * 
 * @details The matrices are allocated in their vectorized form.
 *          @n This function allocates the memory on the non-paged side
 *          of the host DRAM. This is required to execute the CUDA kernels
 *          using the pinned memory method.
 * 
 * @param   m Pointer to struct Matrices.
 * @param   dim Size of the square matrices.
 */
void cudaMallocMatrices(struct Matrices* m , size_t dim);

/**
 * @fn      cudaMallocManagedMatrices
 * @brief   Allocates the memory required by the 3 matrices.
 * 
 * @details The matrices are allocated in their vectorized form.
 *          @n This function allocates the memory in the Unified Virtual Memory
 *          (UVM), which is a logical view of the system memory.
 *          This is required to execute the CUDA kernels using the UVM method.
 * 
 * @param   m Pointer to struct Matrices.
 * @param   dim Size of the square matrices.
 */
void cudaMallocManagedMatrices(struct Matrices* m, size_t dim);

/**
 * @fn      initMatrices
 * @brief   Initialises the values in the 3 matrices.
 * 
 * @details The matrix A is initialised with values that follow this rule: A[i][j] = i * j + 1.
 *          @n The matrix Q is initialised as an identity matrix.
 *          @n The matrix R is initialised as a zero matrix.
 * 
 * @param   m Pointer to struct Matrices. 
 */
void initMatrices(struct Matrices* m);

/**
 * @fn      transposeMatrices
 * @brief   Transposes the A and Q matrices.
 * 
 * @details This function uses OpenMP CPU parallelization to transpose the matrices.
 *          @n The matrix R doesn't require to be transposed because the access pattern is row-wise.
 *          @n The matrix A and Q require to be transposed because their access pattern is column-wise
 *          (by transposing the matrices the cache misses are reduced, leading to increased performance).
 * 
 * @param   m Pointer to struct Matrices. 
 */
void transposeMatrices(struct Matrices* m);

/**
 * @fn      printMatrices
 * @brief   Prints the 3 matrices on the specified output file.
 * 
 * @param   m Pointer to struct Matrices. 
 * @param   outFile Pointer to the output file.
 */
void printMatrices(struct Matrices* m, FILE* outFile);

/**
 * @fn      printQRDecomposition
 * @brief   Prints the Q and R matrices on the specified output file.
 * 
 * @param   m Pointer to struct Matrices. 
 * @param   outFile Pointer to the output file.
 */
void printQRDecomposition(struct Matrices* m, FILE* outFile);

/**
 * @fn      assertMatrices
 * @brief   Asserts the correctness of the QR decomposition.
 * 
 * @details This function was created for debug and testing purposes.
 *          @n This function should be called with a correct triplets of matrices
 *          as 1 of the 2 inputs, to test the correctness of the second one.
 *          @n To assert the correctness of the values, the matrix A is computed
 *          by multiplying the Q and R matrices (for both triplets), and then the
 *          values in every cells of both A matrices are compared (the absolute
 *          difference between the 2 values must not exceed a specified tolerance).
 * 
 * @param   m1 Pointer to a struct Matrices. 
 * @param   m2 Pointer to a struct Matrices. 
 */
void assertMatrices(struct Matrices* m1, struct Matrices* m2);

/**
 * @fn      freeMatrices
 * @brief   Frees the memory allocated for the 3 matrices.
 * 
 * @param   m Pointer to struct Matrices.
 */
void freeMatrices(struct Matrices* m);

/**
 * @fn      cudaFreeMatrices
 * @brief   Frees the memory allocated for the 3 matrices.
 * 
 * @details This function frees the memory allocated with the functions
 *          "cudaMallocHost" and "cudaMallocManaged".
 * 
 * @param   m Pointer to struct Matrices.
 */
void cudaFreeMatrices(struct Matrices* m);

#ifdef __cplusplus
}
#endif

#endif  //MATRICES_H