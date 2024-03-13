/**
 * @file    seq_qrd.h
 * @author  Matteo Gianferrari (https://github.com/matteogianferrari)
 *
 * @brief   This file contains the definition of the functions that
 *          compute the Gram-Schmidt QR decomposition, using only 
 *          1 core of the CPU.
 * 
 * @version 0.1
 * @date    2024-01-01
 */

#ifndef SEQ_QRD_H
#define SEQ_QRD_H

#include "matrices.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn      seq_qrd
 * @brief   Performs the QR decomposition using 1 CPU core.
 * 
 * @param   m Pointer to struct Matrices.
 */
void seq_qrd(struct Matrices* m);

/**
 * @fn      seq_transposed_qrd
 * @brief   Performs the QR decomposition using 1 CPU core.
 * 
 * @details The A and Q matrices must be pre-transposed before
 *          passing them as an input to this function.
 * 
 * @param   m Pointer to struct Matrices.
 */
void seq_transposed_qrd(struct Matrices* m);

#ifdef __cplusplus
}
#endif

#endif  //SEQ_QRD_H