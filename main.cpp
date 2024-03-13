#include "matrices.h"
#include "seq_qrd.h"
#include "omp_qrd.h"
#include "cuda_qrd.cuh"
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>


/**
 * @fn      displayUsage
 * @brief   Shows on the prompt the program usage.
 * 
 * @param   programName Name of the program.
 */
void displayUsage(const char* programName);

/**
 * @fn      executeQRDecomposition
 * @brief   Executes the QR decomposition using the specified method.
 * 
 * @param   method Specifies the methos to use for the QR decomposition. 
 * @param   dim Specifies the square matrices dimension.
 * @param   enablePrint Flag used to enable/disable the print the decomposition.
 */
void executeQRDecomposition(size_t method, size_t dim, bool enablePrint);


int main(int argc, char** argv)
{
    if(argc == 2 && strcmp(argv[1], "--help") == 0) {
        displayUsage(argv[0]);
        return 0;
    }

    int dimension = 0;
    int method = 0;
    bool enablePrint = false;
    int flag;

    // Parse command line arguments with getopt
    while((flag = getopt(argc, argv, "d:m:ph")) != -1)
    {
        switch (flag)
        {
            case 'd': // Set matrix dimension
                dimension = std::atoi(optarg);
                break;

            case 'm': // Set method
                method = std::atoi(optarg);
                break;

            case 'p': // Enable printing
                enablePrint = true;
                break;

            case 'h': // Display help
                displayUsage(argv[0]);
                return 0;

            case '?': // Handle unknown options
                if(optopt == 'd' || optopt == 'm')
                    std::cerr << "Option -" << static_cast<char>(optopt) << " requires an argument." << std::endl;
                else if(isprint(optopt))
                    std::cerr << "Unknown option `-" << static_cast<char>(optopt) << "`." << std::endl;
                else
                    std::cerr << "Unknown option character `\\x" << std::hex << optopt << "`." << std::endl;
                
                return 1;

            default:
                abort();
        }
    }

    if(dimension <= 0 || method < 0 || method > 7)
    {
        displayUsage(argv[0]);
        return 1;
    }

    // Continue with the execution
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Method: " << method << std::endl;
    if(enablePrint)
        std::cout << "Printing is enabled." << std::endl;
    else
        std::cout << "Printing is disabled." << std::endl;


    executeQRDecomposition(method, dimension, enablePrint);

    return 0;
}


void displayUsage(const char* programName)
{
    std::cerr << "Usage: " << programName << " -d <dimension> -m <method> [-p]\n" << std::endl;
    std::cerr << "--help: Shows the program usage.\n" << std::endl;

    std::cerr << "-d: Specifies the square matrix dimension.\n" << std::endl;

    std::cerr << "-m: Specifies the available computation methods:" << std::endl;
    std::cerr << "\t0 -> Sequential CPU QR Decomposition." << std::endl;
    std::cerr << "\t1 -> Sequential transposed CPU QR Decomposition." << std::endl;
    std::cerr << "\t2 -> OpenMP CPU QR Decomposition." << std::endl;
    std::cerr << "\t3 -> OpenMP transposed CPU QR Decomposition." << std::endl;
    std::cerr << "\t4 -> OpenMP transposed GPU QR Decomposition." << std::endl;
    std::cerr << "\t5 -> CUDA transposed pageable QR Decomposition." << std::endl;
    std::cerr << "\t6 -> CUDA transposed pinned QR Decomposition." << std::endl;
    std::cerr << "\t7 -> CUDA transposed UVM QR Decomposition.\n" << std::endl;

    std::cerr << "-p: Prints the QR Decomposition on the standard output." << std::endl;
}


void executeQRDecomposition(size_t method, size_t dim, bool enablePrint)
{
    struct timespec rt[2];
    double wt;

    struct Matrices m;

    switch(method)
    {
        case 0:     /* Sequential CPU QR Decomposition.*/    
            mallocMatrices(&m, dim);
            initMatrices(&m);
            
            clock_gettime(CLOCK_REALTIME, rt + 0);
            seq_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("Sequential CPU QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 1:     /* Sequential transposed CPU QR Decomposition.*/    
            mallocMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            seq_transposed_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("Sequential transposed CPU QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 2:     /* OpenMP CPU QR Decomposition.*/
            mallocMatrices(&m, dim);
            initMatrices(&m);
    
            clock_gettime(CLOCK_REALTIME, rt + 0);
            omp_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("OpenMP CPU QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 3:     /* OpenMP transposed CPU QR Decomposition.*/    
            mallocMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            omp_transposed_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("OpenMP transposed CPU QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 4:     /* OpenMP transposed GPU QR Decomposition.*/    
            mallocMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            omp_target_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("OpenMP transposed GPU QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 5:     /* CUDA transposed pageable QR Decomposition.*/    
            mallocMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            cuda_pageable_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            freeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("CUDA transposed pageable QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 6:     /* CUDA transposed pinned QR Decomposition.*/    
            cudaMallocMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            cuda_pinned_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            cudaFreeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("CUDA transposed pinned QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        case 7:     /* CUDA transposed UVM QR Decomposition.*/    
            cudaMallocManagedMatrices(&m, dim);
            initMatrices(&m);    
            transposeMatrices(&m);

            clock_gettime(CLOCK_REALTIME, rt + 0);
            cuda_uvm_qrd(&m);
            clock_gettime(CLOCK_REALTIME, rt + 1);

            transposeMatrices(&m);
            if(enablePrint)
                printQRDecomposition(&m, stdout);

            cudaFreeMatrices(&m);

            wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
            printf("CUDA transposed UVM QR Decomposition on %dx%d matrix: %9.6f sec\n", dim, dim, wt);

            break;

        default:
            printf("Invalid method number!\n");
            break;
    } 
}