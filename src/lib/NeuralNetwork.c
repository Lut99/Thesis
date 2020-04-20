/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   20/04/2020, 23:23:27
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

#include "NeuralNetwork.h"

#define WEIGHTS_MIN -3.0
#define WEIGHTS_MAX 3.0
#define BIAS 1.0


/***** HELPER FUNCTIONS *****/

/* Creates and initialises a weights matrix of given proportions. Note that it adds the space for a bias node. */
matrix* initialise_weights(size_t input_size, size_t output_size) {
    // Create a new matrix of the proper dimensions
    matrix* to_ret = create_empty_matrix(output_size, input_size + 1);

    // Set each value to a random one in the range WEIGHTS_MIN (inclusive) and WEIGHTS_MAX (exclusive)
    srand(time(NULL));
    for (size_t i = 0; i < to_ret->rows * to_ret->cols; i++) {
        to_ret->data[i] = (double)rand()/RAND_MAX * (WEIGHTS_MAX - WEIGHTS_MIN) + WEIGHTS_MIN;
    }

    // Return the weights
    return to_ret;
}

/* Adds a bias node at the start of each column to the given matrix. The bias value is set by the BIAS macro. Note that this function deallocates the given pointer. */
matrix* add_bias(matrix* v) {
    matrix* to_ret = create_empty_matrix(v->rows + 1, v->cols);

    // First, set all biases to the top row
    for (size_t i = 0; i < v->cols; i++) {
        to_ret->data[i] = BIAS;
    }
    // Then, copy the rest of the data
    for (size_t i = 0; i < v->rows * v->cols; i++) {
        to_ret->data[v->cols + i] = v->data[i];
    }

    // Destroy the old vector
    destroy_matrix(v);

    // Return
    return to_ret;
}



/***** MEMORY MANAGEMENT *****/

neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t hidden_nodes[n_hidden_layers], size_t output_nodes) {
    // Create a new neural net object
    neural_net* to_ret = malloc(sizeof(neural_net));

    // Store the total number of layers
    to_ret->n_layers = n_hidden_layers + 2;

    // Store the number of nodes per layer
    to_ret->nodes_per_layer = malloc(sizeof(size_t) * to_ret->n_layers);
    for (size_t i = 0; i < n_hidden_layers; i++) {
        to_ret->nodes_per_layer[i + 1] = hidden_nodes[i];
    }
    // Also add the input and output layers
    to_ret->nodes_per_layer[0] = input_nodes;
    to_ret->nodes_per_layer[to_ret->n_layers - 1] = output_nodes;

    // Initialise the weights of the neural network randomly
    to_ret->n_weights = to_ret->n_layers - 1;
    to_ret->weights = malloc(sizeof(matrix*) * (to_ret->n_weights));
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
    free(nn->weights);
    free(nn);
}



/***** NEURAL NETWORK OPERATIONS *****/

void nn_activate_all(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* inputs, matrix* (*activation_func)(matrix* z)) {
    // Copy the input matrix to be sure we do not deallocate it
    matrix* inputs2 = copy_matrix_new(inputs);

    // Iterate over each layer to feedforward through the network
    for (size_t i = 0; i < nn->n_weights; i++) {
        // // Store the output without bias in the list
        // outputs[i] = copy_matrix_new(inputs2);
        // First, add a bias node to the input of this layer
        inputs2 = add_bias(inputs2);
        // Store the output with bias in the list
        outputs[i] = inputs2;
        // Then, compute the input values for the nodes in this layer
        matrix* z = matrix_matmul(nn->weights[i], inputs2);
        // Apply the activation function
        activation_func(z);
        // // Cleanup the old inputs2
        // destroy_matrix(inputs2);
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

matrix* nn_classify(neural_net* nn, const matrix* inputs, matrix* (*activation_func)(matrix* z)) {
    // Call the activation function
    matrix* outputs = nn_activate(nn, inputs, activation_func);

    // Create the return vector
    matrix* to_ret = create_empty_matrix(outputs->cols, 1);
    // Create the buffer that stores the highest value of each column so far
    double highest[outputs->cols];
    for (size_t i = 0; i < outputs->cols; i++) {
        highest[i] = 0;
    }

    // Loop through each row and note if that node is the best for each column
    for (size_t y = 0; y < outputs->rows; y++) {
        for (size_t x = 0; x < outputs->cols; x++) {
            // Check if this one is the best of the columns
            double val = outputs->data[y * outputs->cols + x];
            if (val > highest[x]) {
                highest[x] = val;
                to_ret->data[x] = y;
            }
        }
    }

    // Cleanup the outputs matrix
    destroy_matrix(outputs);

    // Return the classifications
    return to_ret;
}

void nn_backpropagate(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate, matrix* (*dxdy_cost_func)(const matrix* deltas, const matrix* output)) {
    // Compute the deltas for the output layer
    matrix* deltas = matrix_sub(outputs[nn->n_layers - 1], expected);

    // Loop through all weights to update them
    // Note that we use a funny loop ending here, but since there is no negative we stop if it overflows to something larger than this
    for (size_t i = nn->n_layers - 2; i <= nn->n_layers - 2; i++) {
        // Compute the delta weights for this layer
        matrix* d_weights = dxdy_cost_func(deltas, outputs[i]);
        // Add that to the correct weight matrix (weights = weights - learning_rate * d_weights)
        matrix_sub_inplace(nn->weights[i], matrix_mul_c_inplace(d_weights, learning_rate));

        // Compute the new deltas
        // Compute 1 - output
        matrix* output_m1 = matrix_sub2_c(1, outputs[i]);
        // Compute (1 - output) * output
        matrix* term1 = matrix_mul_inplace(output_m1, outputs[i]);
        // Compute dot(deltas, weights)
        matrix* weights_i_T = matrix_transpose(nn->weights[i]);
        matrix* term2 = matrix_matmul(weights_i_T, deltas);
        // Cleanup the old deltas before we overwrite it
        destroy_matrix(deltas);
        // Set deltas to (1 - output) * output * dot(deltas, weights)
        deltas = matrix_mul_inplace(term1, term2);

        // Cleanup
        destroy_matrix(d_weights);
        destroy_matrix(weights_i_T);
        destroy_matrix(term2);
    }

    // Done
}

double* nn_train_costs(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act_func)(matrix*), double (*cost_func)(const matrix*, const matrix*), matrix* (*dxdy_cost_func)(const matrix*, const matrix*)) {
    // Allocate the list for the costs
    double* costs = malloc(sizeof(double) * n_iterations);

    // Perform the training
    for (size_t i = 0; i < n_iterations; i++) {
        // First, perform a forward pass through the network
        matrix* outputs[nn->n_layers];
        nn_activate_all(nn, outputs, inputs, act_func);

        // Compute the cost
        costs[i] = (*cost_func)(outputs[nn->n_layers - 1], expected);
        
        // Perform a backpropagation
        nn_backpropagate(nn, outputs, expected, learning_rate, dxdy_cost_func);
    }

    // Return the costs list
    return costs;
}

void nn_train(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act_func)(matrix*), matrix* (*dxdy_cost_func)(const matrix*, const matrix*)) {
    // Perform the training
    for (size_t i = 0; i < n_iterations; i++) {
        // First, perform a forward pass through the network
        matrix* outputs[nn->n_layers];
        nn_activate_all(nn, outputs, inputs, act_func);
        
        // Perform a backpropagation
        nn_backpropagate(nn, outputs, expected, learning_rate, dxdy_cost_func);
    }
}
