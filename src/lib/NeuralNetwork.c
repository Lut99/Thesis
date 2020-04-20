/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   20/04/2020, 13:10:58
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
    for (size_t i = 0; i < input_size * output_size; i++) {
        to_ret->data[i] = (double)rand()/RAND_MAX * (WEIGHTS_MAX - WEIGHTS_MIN) + WEIGHTS_MIN;
    }

    // Return the weights
    return to_ret;
}

/* Adds a bias node to the given n x 1 matrix (vector). The bias value is set by the BIAS macro. Note that this function deallocates the given pointer. */
matrix* add_bias(matrix* v) {
    matrix* to_ret = create_empty_matrix(v->rows + 1, 1);

    // Add the bias node, then copy all the data
    to_ret->data[0] = BIAS;
    for (size_t i = 0; i < v->rows; i++) {
        to_ret->data[i + 1] = v->data[i];
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
    for (size_t i = 1; i < to_ret->n_layers - 1; i++) {
        to_ret->nodes_per_layer[i] = hidden_nodes[i];
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

void nn_activate(neural_net* nn, matrix* output, const matrix* input, matrix* (*activation_func)(matrix* z)) {
    // Copy the input matrix to be sure we do not deallocate it
    matrix* input2 = copy_matrix_new(input);

    // Iterate over each layer to feedforward through the network
    for (size_t i = 0; i < nn->n_weights; i++) {
        // First, add a bias node to the input of this layer
        input2 = add_bias(input2);
        // Then, compute the input values for the nodes in this layer
        matrix* z = matrix_matmul(nn->weights[i], input2);
        // Apply the activation function
        activation_func(z);
        // Deallocate the old input matrix
        destroy_matrix(input2);
        // Set z as the new one
        input2 = z;
    }

    // Copy the output to the output matrix
    copy_matrix(output, input2);
    
    // Cleanup
    destroy_matrix(input2);
}

void nn_train_pass(neural_net* nn, const matrix* input, const matrix* expected, matrix* (*activation_func)(matrix* z), double (*loss_func)(matrix* output, matrix* expected)) {
    // First, perform a forward pass with the given inputs
    matrix* output = create_empty_matrix(nn->nodes_per_layer[nn->n_layers - 1], 1);
    nn_activate(nn, output, input, activation_func);

    // Second, compute the cost of the inputs
    double cost = (*loss_func)(output, expected);
}
