/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   27/04/2020, 23:05:02
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a neural network using a matrix-based
 *   implementation (using Matrix.c) rather than an object-oriented-based
 *   implementation. Any special functions used (such as activation or loss
 *   functions) are defined in Functions.c.
**/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "NeuralNetwork.h"

#define WEIGHTS_MIN -3.0
#define WEIGHTS_MAX 3.0
#define BIAS_MIN -3.0
#define BIAS_MAX 3.0
#define BIAS 1.0


/***** HELPER FUNCTIONS *****/

/* Creates and initialises a weights matrix of given proportions. */
matrix* initialise_weights(size_t input_size, size_t output_size) {
    // Create a new matrix of the proper dimensions
    matrix* to_ret = create_empty_matrix(output_size, input_size);

    // Set each value to a random one in the range WEIGHTS_MIN (inclusive) and WEIGHTS_MAX (exclusive)
    for (size_t i = 0; i < to_ret->rows * to_ret->cols; i++) {
        to_ret->data[i] = (double)rand()/RAND_MAX * (WEIGHTS_MAX - WEIGHTS_MIN) + WEIGHTS_MIN;
    }

    // Return the weights
    return to_ret;
}

/* Creates and initialises a bias matrix for the number of given nodes. */
matrix* initialise_biases(size_t n_nodes) {
    // Create a new matrix of the proper dimensions
    matrix* to_ret = create_empty_matrix(n_nodes, 1);

    // Set each value to a random one in the range BIAS_MIN (inclusive) and BIAS_MAX (exclusive)
    for (size_t i = 0; i < to_ret->rows * to_ret->cols; i++) {
        to_ret->data[i] = BIAS;//(double)rand()/RAND_MAX * (BIAS_MAX - BIAS_MIN) + BIAS_MIN;
    }

    // Return the weights
    return to_ret;
}



/***** MEMORY MANAGEMENT *****/

neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t hidden_nodes[n_hidden_layers], size_t output_nodes) {
    // Start by seeding the pseudo-random number generator
    srand(time(NULL));

    // Create a new neural net object
    neural_net* to_ret = malloc(sizeof(neural_net));

    // Store the total number of layers
    to_ret->n_layers = n_hidden_layers + 2;
    to_ret->n_biases = n_hidden_layers;
    to_ret->n_weights = to_ret->n_layers - 1;

    // Store the number of nodes per layer
    to_ret->nodes_per_layer = malloc(sizeof(size_t) * to_ret->n_layers);
    for (size_t i = 0; i < n_hidden_layers; i++) {
        to_ret->nodes_per_layer[i + 1] = hidden_nodes[i];
    }
    // Also add the input and output layers
    to_ret->nodes_per_layer[0] = input_nodes;
    to_ret->nodes_per_layer[to_ret->n_layers - 1] = output_nodes;

    // Initialise the biases to the set BIAS value
    to_ret->biases = malloc(sizeof(double) * to_ret->n_biases);
    for (size_t i = 0; i < to_ret->n_biases; i++) {
        to_ret->biases[i] = BIAS;
    }
    
    // Initialise the biases and weights of the neural network randomly
    to_ret->weights = malloc(sizeof(matrix*) * to_ret->n_weights);
    for (size_t i = 0; i < to_ret->n_weights; i++) {
        to_ret->weights[i] = initialise_weights(to_ret->nodes_per_layer[i], to_ret->nodes_per_layer[i + 1]);
    }

    // Done, return
    return to_ret;
}

void destroy_nn(neural_net* nn) {
    free(nn->nodes_per_layer);
    for (size_t i = 0; i < nn->n_weights; i++) {
        destroy_matrix(nn->weights[i]);
    }
    free(nn->biases);
    free(nn->weights);
    free(nn);
}



/***** NEURAL NETWORK OPERATIONS *****/

#include "stdio.h"
void nn_activate_all(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* inputs, matrix* (*activation_func)(matrix* z)) {
    // Copy the input matrix to be sure we do not deallocate it
    matrix* inputs2 = copy_matrix_new(inputs);

    // Iterate over each layer to feedforward through the network
    for (size_t i = 0; i < nn->n_weights; i++) {
        // Store the output with bias in the list
        outputs[i] = inputs2;
        // Then, compute the input values for the nodes in this layer
        matrix* z = matrix_matmul(nn->weights[i], inputs2);
        // Add the bias if we're in a hidden layer
        if (i > 0) {
            matrix_add_c_inplace(z, nn->biases[i - 1]);
        }
        // Apply the activation function
        activation_func(z);
        // Set z as the new one
        inputs2 = z;
    }
    // Add the output from the final layer in the outputs
    outputs[nn->n_layers - 1] = inputs2;
}

matrix* nn_activate(neural_net* nn, const matrix* inputs, matrix* (*activation_func)(matrix* z)) {
    // Prepare the buffer for all the outputs
    matrix* outputs[nn->n_layers];

    // Let the activation run through the matrix
    nn_activate_all(nn, outputs, inputs, activation_func);

    // Destroy all outputs except the last one
    for (size_t i = 0; i < nn->n_layers - 1; i++) {
        destroy_matrix(outputs[i]);
    } 

    // Return the last output
    return outputs[nn->n_layers - 1];
}

void nn_backpropagate(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate, matrix* (*dydx_act)(const matrix*), double (*dydx_cost)(const matrix*, const matrix*)) {
    // Compute the error on the output layer
    // Compute the activation of this layer
    matrix* in = matrix_matmul(nn->weights[nn->n_weights - 1], outputs[nn->n_weights - 1]);
    matrix* act = dydx_act(in);
    // Compute the activation times the derivative of the cost function
    matrix* error = matrix_mul_c_inplace(act, dydx_cost(outputs[nn->n_layers - 1], expected));
    destroy_matrix(in);

    // Loop backwards through all the layers
    for (size_t i = nn->n_layers - 2; i <= nn->n_layers - 2; i--) {
        // Compute the weights deltas with the old error
        matrix* output_T = matrix_transpose(outputs[i]);
        matrix* d_weights = matrix_matmul(error, output_T);

        if (i > 0) {
            // Compute the delta bias for this layer
            double d_bias = matrix_sum(error);

            // Get the input through the derivative of the activation function. Don't forget to add the bias.
            in = matrix_matmul(nn->weights[i - 1], outputs[i - 1]);
            matrix_add_c_inplace(act, nn->biases[i]);
            act = dydx_act(in);

            // Compute the new error based on the input
            matrix* weights_T = matrix_transpose(nn->weights[i]);
            matrix* weighted_err = matrix_matmul(weights_T, error);
            destroy_matrix(error);
            error = matrix_mul_inplace(weighted_err, act);

            // Update the bias matrix
            nn->biases[i] -= learning_rate * d_bias;

            // Cleanup
            destroy_matrix(in);
            destroy_matrix(act);
            destroy_matrix(weights_T);
        }



        // Update the weights of this layer
        matrix_sub_inplace(nn->weights[i], matrix_mul_c_inplace(d_weights, learning_rate));

        // Cleanup
        destroy_matrix(output_T);
        destroy_matrix(d_weights);
    }

    // Cleanup the error
    destroy_matrix(error);
}



double* nn_train_costs(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*), double (*cost)(const matrix*, const matrix*), double (*dydx_cost)(const matrix*, const matrix*)) {
    // Allocate the list for the costs
    double* costs = malloc(sizeof(double) * n_iterations);

    // Perform the training
    matrix* outputs[nn->n_layers];
    for (size_t i = 0; i < n_iterations; i++) {
        // First, perform a forward pass through the network
        nn_activate_all(nn, outputs, inputs, act);

        // Compute the cost
        costs[i] = (*cost)(outputs[nn->n_layers - 1], expected);

        // Print the cost once every 100 iterations
        if (i % 100 == 0) {
            printf("    (Iter %lu) Cost: %.2f\n", i, costs[i]);
        }
        
        // Perform a backpropagation
        nn_backpropagate(nn, outputs, expected, learning_rate, dydx_act, dydx_cost);

        // Destroy all matrices
        for (size_t i = 0; i < nn->n_layers; i++) {
            destroy_matrix(outputs[i]);
        }
    }

    // Return the costs list
    return costs;
}

void nn_train(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*), double (*dydx_cost)(const matrix*, const matrix*)) {
    // Perform the training
    matrix* outputs[nn->n_layers];
    for (size_t i = 0; i < n_iterations; i++) {
        // First, perform a forward pass through the network
        nn_activate_all(nn, outputs, inputs, act);
        
        // Perform a backpropagation
        nn_backpropagate(nn, outputs, expected, learning_rate, dydx_act, dydx_cost);

        // Destroy all matrices
        for (size_t i = 0; i < nn->n_layers; i++) {
            destroy_matrix(outputs[i]);
        }
    }
}



/***** USEFUL TOOLS *****/

matrix* nn_flatten_results(matrix* outputs) {
    double highest_indices[outputs->cols];
    double highest_values[outputs->cols];
    // Set highest values to -inf
    for (size_t i = 0; i < outputs->cols; i++) {
        highest_indices[i] = 0;
        highest_values[i] = -INFINITY;
    }

    // First pass: find the highest values with matching index
    for (size_t y = 0; y < outputs->rows; y++) {
        for (size_t x = 0; x < outputs->cols; x++) {
            double data = outputs->data[y * outputs->cols + x];
            if (data > highest_values[x]) {
                highest_indices[x] = y;
                highest_values[x] = data;
            }
        }
    }

    // Now do another pass where all values are set to zero (except the highest ones)
    for (size_t y = 0; y < outputs->rows; y++) {
        for (size_t x = 0; x < outputs->cols; x++) {
            if (y == highest_indices[x]) {
                outputs->data[y * outputs->cols + x] = 1;
            } else {
                outputs->data[y * outputs->cols + x] = 0;
            }
        }
    }

    // Return the matrix for chaining
    return outputs;
}
