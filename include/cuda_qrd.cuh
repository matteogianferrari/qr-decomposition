/**
 * @file    cuda_qrd.cuh
 * @author  Matteo Gianferrari (https://github.com/matteogianferrari)
 *
 * @brief   This file contains the definition of the functions that
 *          compute the Gram-Schmidt QR decomposition, using
 *          CUDA parallelization techniques.
 * 
 * @version 0.1
 * @date    2024-01-01
 */

#ifndef CUDA_QRD_CUH
#define CUDA_QRD_CUH

#include "matrices.h"
#include <cuda_runtime.h>
  
/* These block sizes are optimal values obtained from a performance analysis.
 * They may not be optimal for the user specific hardware configuration.
 */
#define BLOCK_SIZE1D 768
#define BLOCK_SIZE2D 27 

/* Provides a convenient way to check for errors after each CUDA function call in a program.
 * The gpuErrchk macro is used to wrap CUDA function calls, and if an error occurs,
 * it prints an error message and optionally exits the program. This can be helpful for debugging
 * and ensuring proper error handling in CUDA applications.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/**
 * @fn      cuda_pageable_qrd
 * @brief   Performs the QR decomposition using the GPU architecture.
 * 
 * @details This function uses the pageable memory model.
 * 
 * @param   m Pointer to struct Matrices.
 */
void cuda_pageable_qrd(struct Matrices* m);

/**
 * @fn      cuda_pinned_qrd
 * @brief   Performs the QR decomposition using the GPU architecture.
 * 
 * @details This function uses the pinned memory model.
 * 
 * @param   m Pointer to struct Matrices.
 */
void cuda_pinned_qrd(struct Matrices* m);

/**
 * @fn      cuda_uvm_qrd
 * @brief   Performs the QR decomposition using the GPU architecture.
 * 
 * @details This function uses the UVM memory model.
 * 
 * @param   m Pointer to struct Matrices.
 */
void cuda_uvm_qrd(struct Matrices* m);

/**
 * @fn      computeNorm
 * @brief   CUDA kernel that computes the root squared sum.
 * 
 * @details This function implement the reduction using an advanced technique 
 *          on the shared memory.
 *          @n For more info on the implementation see:
 *          @n https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 * 
 * @param   A Pointer to the A matrix. 
 * @param   R Pointer to the R matrix.
 * @param   k Index of the current iteration.
 * @param   n Size of the square matrix.
 */
__global__ void computeNorm(data_t* __restrict__ A, data_t* __restrict__ R, size_t k, size_t n);

/**
 * @fn      computeQMatrix
 * @brief   CUDA kernel that computes the values in a row of the Q matrix.
 * 
 * @param   A Pointer to the A matrix. 
 * @param   Q Pointer to the Q matrix.
 * @param   R Pointer to the R matrix.
 * @param   k Index of the current iteration.
 * @param   n Size of the square matrix.
 */
__global__ void computeQMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n);

/**
 * @fn      computeRMatrix
 * @brief   CUDA kernel that computes the values in the R matrix.
 * 
 * @details This function implement the reduction using an advanced technique 
 *          on the shared memory.
 *          @n For more info on the implementation see:
 *          @n https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 * 
 * @param   A Pointer to the A matrix. 
 * @param   Q Pointer to the Q matrix.
 * @param   R Pointer to the R matrix.
 * @param   k Index of the current iteration.
 * @param   n Size of the square matrix.
 */
__global__ void computeRMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n);

/**
 * @fn      computeAMatrix
 * @brief   CUDA kernel that computes the values in the A matrix.
 * 
 * @param   A Pointer to the A matrix. 
 * @param   Q Pointer to the Q matrix.
 * @param   R Pointer to the R matrix.
 * @param   k Index of the current iteration.
 * @param   n Size of the square matrix.
 */
__global__ void computeAMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n);

#endif  //CUDA_QRD_CUH