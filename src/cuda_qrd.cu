#include "cuda_qrd.cuh"


void cuda_pageable_qrd(struct Matrices* m)
{
    data_t* dA;
    data_t* dQ;
    data_t* dR;

    // CUDA malloc.
    cudaMalloc((void**)&dA, m->dim * m->dim * sizeof(data_t));
    cudaMalloc((void**)&dQ, m->dim * m->dim * sizeof(data_t));
    cudaMalloc((void**)&dR, m->dim * m->dim * sizeof(data_t));

    // CUDA memory copy from host to device.
    cudaMemcpy(dA, m->A, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dQ, m->Q, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dR, m->R, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);

    // Sets the dimension of block sizes and grid sizes for the kernels.
    dim3 blockSize1D {BLOCK_SIZE1D};
    dim3 gridSize1D {((unsigned int)m->dim + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D};
    dim3 blockSize2D {BLOCK_SIZE2D, BLOCK_SIZE2D};
    dim3 gridSize2D {((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, ((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D};


    size_t k, n = m->dim;

    // QR decomposition.
    for(k = 0; k < n; k++)
    {
        computeNorm<<<gridSize1D, blockSize1D>>>(dA, dR, k, n);

        computeQMatrix<<<gridSize1D, blockSize1D>>>(dA, dQ, dR, k, n);

        computeRMatrix<<<gridSize2D, blockSize2D>>>(dA, dQ, dR, k, n);

        computeAMatrix<<<gridSize2D, blockSize2D>>>(dA, dQ, dR, k, n);
    }


    // CUDA memory copy from device to host.
    cudaMemcpy(m->A, dA, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->Q, dQ, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->R, dR, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);

    // CUDA free.
    cudaFree(dA);
    cudaFree(dQ);
    cudaFree(dR);
}


void cuda_pinned_qrd(struct Matrices* m)
{
    data_t* dA;
    data_t* dQ;
    data_t* dR;
 
    // CUDA malloc.
    cudaMalloc((void**)&dA, m->dim * m->dim * sizeof(data_t));
    cudaMalloc((void**)&dQ, m->dim * m->dim * sizeof(data_t));
    cudaMalloc((void**)&dR, m->dim * m->dim * sizeof(data_t));

    // CUDA memory copy from host to device.
    cudaMemcpy(dA, m->A, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dQ, m->Q, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dR, m->R, m->dim * m->dim * sizeof(data_t), cudaMemcpyHostToDevice);

    // Sets the dimension of block sizes and grid sizes for the kernels.
    dim3 blockSize1D {BLOCK_SIZE1D};
    dim3 gridSize1D {((unsigned int)m->dim + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D};
    dim3 blockSize2D {BLOCK_SIZE2D, BLOCK_SIZE2D};
    dim3 gridSize2D {((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, ((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D};


    size_t k, n = m->dim;

    // QR decomposition.
    for(k = 0; k < n; k++)
    {
        computeNorm<<<gridSize1D, blockSize1D>>>(dA, dR, k, n);

        computeQMatrix<<<gridSize1D, blockSize1D>>>(dA, dQ, dR, k, n);

        computeRMatrix<<<gridSize2D, blockSize2D>>>(dA, dQ, dR, k, n);

        computeAMatrix<<<gridSize2D, blockSize2D>>>(dA, dQ, dR, k, n);
    }


    // CUDA memory copy from device to host.
    cudaMemcpy(m->A, dA, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->Q, dQ, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->R, dR, m->dim * m->dim * sizeof(data_t), cudaMemcpyDeviceToHost);

    // CUDA free.  
    cudaFree(dA);
    cudaFree(dQ);
    cudaFree(dR);
}


void cuda_uvm_qrd(struct Matrices* m)
{
    // Sets the dimension of block sizes and grid sizes for the kernels.
    dim3 blockSize1D {BLOCK_SIZE1D};
    dim3 gridSize1D {((unsigned int)m->dim + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D};
    dim3 blockSize2D {BLOCK_SIZE2D, BLOCK_SIZE2D};
    dim3 gridSize2D {((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, ((unsigned int)m->dim + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D};


    size_t k, n = m->dim;
    
    // QR decomposition.
    for(k = 0; k < n; k++)
    {
        computeNorm<<<gridSize1D, blockSize1D>>>(m->A, m->R, k, n);

        computeQMatrix<<<gridSize1D, blockSize1D>>>(m->A, m->Q, m->R, k, n);
        
        computeRMatrix<<<gridSize2D, blockSize2D>>>(m->A, m->Q, m->R, k, n);
        
        computeAMatrix<<<gridSize2D, blockSize2D>>>(m->A, m->Q, m->R, k, n);
    }

    // Synchronize the CPU and the GPU.
    cudaDeviceSynchronize();
}


__global__ void computeNorm(data_t* __restrict__ A, data_t* __restrict__ R, size_t k, size_t n)
{
    __shared__ data_t arr[BLOCK_SIZE1D];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tId = threadIdx.x;

    // Used to run only the correct number of thread suited for the input.
    if(i < n)
    {
        arr[tId] = A[k * n + i] * A[k * n + i];

        __syncthreads();

        // Parallel reduction in shared memory.
        // (https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
        for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if(tId < stride)
            {
                arr[tId] += arr[tId + stride];
            }

            __syncthreads();
        }

        // Only the master thread of the warp updates the R value.
        if(tId == 0)
        {
            R[k * n + k] = sqrt(arr[0]);
        }
    }
}


__global__ void computeQMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Used to run only the correct number of thread suited for the input.
    if(i < n)
    {
        Q[k * n + i] = A[k * n + i] / R[k * n + k];
    }
}


__global__ void computeRMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tId = threadIdx.y;

    __shared__ data_t arr[BLOCK_SIZE2D];

    // Used to run only the correct number of thread suited for the input.
    if(j > k && j < n && i < n)
    {
        arr[tId] = Q[k * n + i] * A[j * n + i];

        __syncthreads();

        // Parallel reduction in shared memory.
        // (https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
        for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1)
        {
            if (tId < stride)
            {
                arr[tId] += arr[tId + stride];
            }

            __syncthreads();
        }

        // Only the master thread of the warp updates the R value.
        if (tId == 0)
        {
            R[k * n + j] = arr[0];
        }
    }
}


__global__ void computeAMatrix(data_t* __restrict__ A, data_t* __restrict__ Q, data_t* __restrict__ R, size_t k, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Used to run only the correct number of thread suited for the input.
    if(j > k && j < n && i < n)
    {
        // Each thread updates one element of the matrix A.
        atomicAdd(&A[j * n + i], -(Q[k * n + i] * R[k * n + j]));
    }
}
