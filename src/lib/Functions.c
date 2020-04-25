/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   4/25/2020, 11:52:03 PM
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

matrix* simple(matrix* z) {
    return z;
}

/* Code from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ */
matrix* softmax(matrix* z) {
    // Compute z - np.max(z)
    matrix_sub1_c_inplace(z, matrix_max(z));
    // Compute exp(z - np.max(z))
    matrix_exp_inplace(z);
    // Compute sum(exp(z - np.max(z)))
    double s = matrix_sum(z);
    // Compute exp(z - np.max(z)) / sum(exp(z - np.max(z)))
    matrix_mul_c_inplace(z, 1 / s);

    return z;
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

matrix* dydx_hyperbolic_tangent(matrix* z) {
    // Compute tanh(z)^2
    matrix_square_inplace(matrix_tanh_inplace(z));
    // Compute (1 - tanh(z)^2) / 2
    matrix_mul_c_inplace(matrix_sub2_c_inplace(1, z), 0.5);

    return z;
}



/***** COST FUNCTIONS *****/

double mean_squared_error(const matrix* output, const matrix* expected) {
    // Sanity check to make sure the matrices have the correct size
    if (output->rows != expected->rows || output->cols != expected->cols) {
        fprintf(stderr, "ERROR: mean_squared_error: matrix output (%ldx%ld) and expected (%ldx%ld) do not have the same sizes\n",
                output->rows,
                output->cols,
                expected->rows,
                expected->cols);
        return -1;
    }

    // Subtract one matrix from the other
    matrix* error = matrix_sub(output, expected);
    // Square the values
    matrix_square_inplace(error);
    // Sum all values
    double sum = matrix_sum(error);
    // Clean the new matrix
    destroy_matrix(error);
    // Return the result normalised
    return sum / expected->rows;
}

double other_cost_func (const matrix* output, const matrix* expected) {
    // Sanity check to make sure the matrices have the correct size
    if (output->rows != expected->rows || output->cols != expected->cols) {
        fprintf(stderr, "ERROR: other_cost_func: matrix output (%ldx%ld) and expected (%ldx%ld) do not have the same sizes\n",
                output->rows,
                output->cols,
                expected->rows,
                expected->cols);
        return -1;
    }
    
    // Compute ln(output)
    matrix* output_ln = matrix_ln(output);
    // Compute ln(1 - output)
    matrix* output_ln_m1 = matrix_ln_inplace(matrix_sub2_c(1, output));
    // Compute 1 - expected
    matrix* expected_m1 = matrix_sub2_c(1, expected);

    // Compute expected * ln(output)
    matrix* term = matrix_mul(expected, output_ln);
    // Compute (1 - expected) * ln(1 - output)
    matrix* term2 = matrix_mul_inplace(expected_m1, output_ln_m1);
    // Compute expected * ln(output) + (1 - expected) * ln(1 - output)
    matrix_add_inplace(term, term2);
    // Take the sum of this term
    double cost = matrix_sum(term);
    
    // Cleanup
    destroy_matrix(output_ln_m1);
    destroy_matrix(output_ln);
    destroy_matrix(term);
    destroy_matrix(term2);

    // Return the negative cost
    return -cost;
}



/***** COST FUNCTIONS (PARTIAL) DERIVATIVES *****/

matrix* dydx_other_cost_func(const matrix* deltas, const matrix* output) {
    // Take note that: deltas has shape 'n_nodes_next_layer x samples'
    //                 output has shape 'n_nodes_this_layer x samples'
    // Returning matrix should be: 'n_nodes_next_layer x n_nodes_this_layer'
    matrix* output_T = matrix_transpose(output);
    matrix* d_weights = matrix_matmul(deltas, output_T);

    // Cleanup
    destroy_matrix(output_T);
    
    // Return the new matrix
    return d_weights;
}
