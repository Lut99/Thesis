/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   4/19/2020, 11:33:53 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file describes several functions, like activation or loss functions,
 *   used in the NeuralNetwork library.
**/

#include "math.h"

#include "Functions.h"


matrix* sigmoid(matrix* z) {
    // Compute exp(-z)
    matrix* exp_mz = matrix_exp_inplace(matrix_mul_c_inplace(z, -1));
    // Compute 1 / (1 + exp(-z))
    matrix* result = matrix_inv_inplace(matrix_add_c_inplace(exp_mz, 1));

    return result;
}

matrix* dydx_sigmoid(matrix* z) {
    // Compute sigmoid z
    matrix* sz = sigmoid(z);
    // Compute sigmoid(z) * (1 - sigmoid(z))
    matrix* sz2 = copy_matrix_new(sz);
    matrix* result = matrix_mul_inplace(sz, matrix_sub2_c_inplace(1, sz2));
    
    // Clean the extra matrix
    destroy_matrix(sz2);
    
    return result;
}
