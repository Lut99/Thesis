/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   4/18/2020, 11:32:27 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file describes several functions, like activation or loss functions,
 *   used in the NeuralNetwork library.
**/

#include "math.h"

#include "Functions.h"


double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double dydx_sigmoid(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}
