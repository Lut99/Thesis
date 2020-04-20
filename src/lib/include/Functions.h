/* FUNCTIONS.h
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:58 PM
 * Last edited:
 *   20/04/2020, 17:24:55
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



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

/* Implements the derivative of the sigmoid function working on an entire matrix. Note that this operation is performed in-place. */
matrix* dydx_sigmoid(matrix* z);



/***** COST FUNCTIONS *****/

/* Implements the Mean Squared Error as cost function. Both matrices should be vectors of the same length, and if they aren't, a negative double (-1) is returned and an error to stderr is printed. Note that for efficiency, the first matrix is used as buffer for the in-between values and will be overwritten. */
double mean_squared_error(matrix* output, const matrix* expected);

/* Implements some other cost function as cost function. Both matrices should be vectors of the same length, and if they aren't, a negative double (-1) is returned and an error to stderr is printed. Note that for efficiency, the first matrix is used as buffer for the in-between values and will be overwritten. */
double other_cost_func(matrix* output, const matrix* expected);



/***** COST FUNCTIONS (PARTIAL) DERIVATIVES *****/

/* Implements the partial derivative of some other cost function that returns the change in weights for given deltas of the next layer and given outputs of this layer. */
matrix* dydx_other_cost_func(const matrix* deltas, const matrix* output);

#endif