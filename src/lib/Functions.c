/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   20/04/2020, 14:19:33
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
    matrix* exp_mz = matrix_exp_inplace(matrix_mul_c_inplace(z, -1));
    // Compute 1 / (1 + exp(-z))
    matrix* result = matrix_inv_inplace(matrix_add_c_inplace(exp_mz, 1));

    return result;
}





/***** ACTIVATION FUNCTIONS DERIVATIVES *****/

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





/***** COST FUNCTIONS *****/

double mean_squared_error(matrix* output, const matrix* expected) {
    // Sanity check to make sure the matrices have the correct size
    if (output->rows != expected->rows || output->cols != 1 || expected->cols != 1) {
        fprintf(stderr, "ERROR: mean_squared_error: matrix output (%ldx%ld) and expected (%ldx%ld) do not have the same sizes\n",
                output->rows,
                output->cols,
                expected->rows,
                expected->cols);
        return -1;
    }

    // Subtract one matrix from the other
    matrix_sub_inplace(output, expected);
    // Square the values
    matrix_square_inplace(output);
    // Sum all values
    double sum = matrix_sum(output);
    // Return the result normalised
    return sum / expected->rows;
}