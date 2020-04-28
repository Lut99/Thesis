/* FUNCTIONS.h
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:58 PM
 * Last edited:
 *   28/04/2020, 14:03:10
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

/* Implements the simplest of activation functions: f(x) = x. */
matrix* simple(matrix* z);



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

/* Implements the derivative of the sigmoid function working on an entire matrix. Note that this operation is performed in-place. */
matrix* dydx_sigmoid(const matrix* z);

/* Implements the derivative of a hyperbolic tangent activation function working on an entire matrix. Note that this operation is performed in-place. */
matrix* dydx_hyperbolic_tangent(const matrix* z);



/***** COST FUNCTIONS *****/

/* Implements the Mean Squared Error as cost function. Both matrices should be matrices of the same length. */
double mean_squared_error(const matrix* output, const matrix* expected);

/* Implements the categorical cross entropy cost function. Both matrices should be matrices of the same length. */
double categorical_cross_entropy(const matrix* output, const matrix* expected);



/***** COST FUNCTIONS (PARTIAL) DERIVATIVES *****/

/* Implements the partial derivative of the mean squared error cost function that returns the change in weights for given deltas of the next layer and given outputs of this layer. */
double dydx_mean_squared_error(const matrix* output, const matrix* expected);

/* Implements the partial derivative of the categorical cross entropy cost function that returns the change in weights for given deltas of the next layer and given outputs of this layer. */
double dydx_categorical_cross_entropy(const matrix* output, const matrix* expected);

#endif