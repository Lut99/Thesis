/* FUNCTIONS.h
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:58 PM
 * Last edited:
 *   28/04/2020, 19:20:16
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file describes several functions, like activation or loss functions,
 *   used in the NeuralNetwork library. This is the header file.
**/

#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include "Matrix.h"


/***** ACTIVATION FUNCTIONS *****/

/* Implements the sigmoid function working on an entire matrix. Note that this operation is performed in-place. */
matrix* sigmoid(matrix* z);

/* Implements a hyperbolic tangent activation function working on an entire matrix. Note that this operation is performed in-place. */
matrix* hyperbolic_tangent(matrix* z);



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

/* Implements the derivative of the sigmoid function working on an entire matrix. Note that this operation is performed in-place. */
matrix* dydx_sigmoid(const matrix* z);

/* Implements the derivative of a hyperbolic tangent activation function working on an entire matrix. Note that this operation is performed in-place. */
matrix* dydx_hyperbolic_tangent(const matrix* z);

#endif