/* FUNCTIONS.h
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:58 PM
 * Last edited:
 *   4/18/2020, 11:31:38 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file describes several functions, like activation or loss functions,
 *   used in the NeuralNetwork library. This is the header file.
**/

#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

/* Implements the sigmoid function working on a scalar. */
double sigmoid(double z);
/* Implements the derivative of the sigmoid function working on a scalar. */
double dydx_sigmoid(double z);

#endif