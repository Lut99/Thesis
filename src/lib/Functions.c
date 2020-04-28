/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   28/04/2020, 19:20:07
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

matrix* sigmoid(matrix* z) {
    // Compute exp(-z)
    matrix_exp_inplace(matrix_mul_c_inplace(z, -1));
    // Compute 1 / (1 + exp(-z))
    matrix_inv_inplace(matrix_add_c_inplace(z, 1));

    return z;
}

matrix* hyperbolic_tangent(matrix* z) {
    // Compute tanh(z) + 1
    matrix_add_c_inplace(matrix_tanh_inplace(z), 1);
    // Compute (tanh(z) + 1) / 2
    matrix_mul_c_inplace(z, 0.5);

    return z;
}



/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

matrix* dydx_sigmoid(const matrix* z) {
    // Compute sigmoid z
    matrix* sz = sigmoid(copy_matrix_new(z));
    // Compute sigmoid(z) * (1 - sigmoid(z))
    matrix* sz2 = copy_matrix_new(sz);
    matrix* result = matrix_mul_inplace(sz, matrix_sub2_c_inplace(1, sz2));
    
    // Clean the extra matrix
    destroy_matrix(sz2);
    
    return result;
}

matrix* dydx_hyperbolic_tangent(const matrix* z) {
    // Compute tanh(z)^2
    matrix* tanh_z2 = matrix_square_inplace(matrix_tanh(z));
    // Compute (1 - tanh(z)^2) / 2
    matrix_mul_c_inplace(matrix_sub2_c_inplace(1, tanh_z2), 0.5);

    return tanh_z2;
}

