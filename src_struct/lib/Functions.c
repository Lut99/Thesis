/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   01/05/2020, 16:35:14
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file describes several functions, like activation or loss functions,
 *   used in the NeuralNetwork library.
**/

#include <stdio.h>
#include <math.h>

#include "Functions.h"


/***** ACTIVATION FUNCTIONS *****/

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double hyperbolic_tangent(double z) {
    return (tanh(z) + 1) / 2;
}



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

double dydx_sigmoid(double z) {
    return z * (1 - z);

    // // Compute sigmoid z
    // double sz = sigmoid(z);
    
    // // Return sigmoid(z) * (1 - sigmoid(z))
    // return sz * (1 - sz);
}

double dydx_hyperbolic_tangent(double z) {
    // Compute tanh(z)
    double tanh_z = tanh(z);

    // Return (1 - tanh(z)^2) / 2
    return (1 - (tanh_z * tanh_z)) / 2;
}

