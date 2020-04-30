/* FUNCTIONS.h
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:58 PM
 * Last edited:
 *   30/04/2020, 16:31:09
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

/* Implements the sigmoid function. Note that this operation is performed in-place. */
double sigmoid(double z);

/* Implements a hyperbolic tangent activation function. Note that this operation is performed in-place. */
double hyperbolic_tangent(double z);



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

/* Implements the derivative of the sigmoid function. Note that this operation is performed in-place. */
double dydx_sigmoid(double z);

/* Implements the derivative of a hyperbolic tangent activation function. Note that this operation is performed in-place. */
double dydx_hyperbolic_tangent(double z);

#endif