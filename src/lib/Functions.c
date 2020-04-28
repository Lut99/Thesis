/* FUNCTIONS.c
 *   by Tim Müller
 *
 * Created:
 *   4/18/2020, 11:19:37 PM
 * Last edited:
 *   28/04/2020, 14:48:28
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

// matrix* softmax(matrix* z) {
//     // Find the max of z
//     double max = matrix_max(z);
//     // Find the sum while computing the exp of each element
//     double sum = 0;
//     for (size_t i = 0; i < z->cols; i++) {
//         z->data[i] = exp(z->data[i] - max);
//         sum += z->data[i];
//     }
//     // Return the weighted answer
//     return matrix_mul_c_inplace(z, 1 / sum);
// }



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

matrix* dydx_softmax(const matrix* z) {
    // Find the max
    double max = matrix_max(z);
    for (size_t i = 0; i < z->cols; i++) {
        
    }
}



/***** COST FUNCTIONS *****/

double mean_squared_error(const matrix* output, const matrix* expected) {
    // Subtract one matrix from the other (output - expected)
    matrix* error = matrix_sub(output, expected);
    // Square the values ((output - expected) ^ 2)
    matrix_square_inplace(error);
    // Sum all values (sum((output - expected) ^ 2))
    double sum = matrix_sum(error);
    // Clean the new matrix
    destroy_matrix(error);
    // Return the result normalised (1 / n * sum((output - expected) ^ 2))
    return sum / 2;// / expected->rows;
}

double categorical_cross_entropy(const matrix* output, const matrix* expected) {
    // Compute the log of the output
    matrix* ln_output = matrix_ln(output);
    // Compute expected * ln(output)
    matrix* error = matrix_mul_inplace(ln_output, expected);
    // Return the negative sum
    double sum = matrix_sum(error);
    // Clean the new matrix
    destroy_matrix(error);
    return -sum;
}



/***** COST FUNCTIONS (PARTIAL) DERIVATIVES *****/

double dydx_mean_squared_error(const matrix* output, const matrix* expected) {
    // Compute (exp - output)
    matrix* error = matrix_sub(output, expected);
    // Compute the sum
    double sum = matrix_sum(error);
    // Clean the matrix
    destroy_matrix(error);
    // Return the result normalised
    return /*(-2 / expected->rows) **/ sum;
}

double dydx_categorical_cross_entropy(const matrix* output, const matrix* expected) {
    // Compute the log of the output
    matrix* ln_output = matrix_inv(output);
    // Compute expected * ln(output)
    matrix* error = matrix_mul_inplace(ln_output, expected);
    // Return the negative sum
    double sum = matrix_sum(error);
    // Clean the new matrix
    destroy_matrix(error);
    return -sum;
}
