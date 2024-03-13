/**
 * @file    omp_qrd.h
 * @author  Matteo Gianferrari (https://github.com/matteogianferrari)
 *
 * @brief   This file contains the definition of the functions that
 *          compute the Gram-Schmidt QR decomposition, using
 *          OpenMP parallelization techniques.
 * 
 * @version 0.1
 * @date    2024-01-01
 */

#ifndef OMP_QRD_H
#define OMP_QRD_H

#include "matrices.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn      omp_qrd
 * @brief   Performs the QR decomposition using multiple CPU cores.
 * 
 * @param   m Pointer to struct Matrices.
 */
void omp_qrd(struct Matrices* m);

/**
 * @fn      omp_transposed_qrd
 * @brief   Performs the QR decomposition using multiple CPU cores.
 * 
 * @details The A and Q matrices must be pre-transposed before
 *          passing them as an input to this function.
 * 
 * @param   m Pointer to struct Matrices.
 */
void omp_transposed_qrd(struct Matrices* m);

/**
 * @fn      omp_target_qrd
 * @brief   Performs the QR decomposition using target offloading to the GPU.
 * 
 * @details The A and Q matrices must be pre-transposed before
 *          passing them as an input to this function.
 * 
 * @param   m Pointer to struct Matrices.
 */
void omp_target_qrd(struct Matrices* m);

#ifdef __cplusplus
}
#endif

#endif  //OMP_QRD_H