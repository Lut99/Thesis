/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   4/19/2020, 12:05:57 AM
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

#include "Functions.h"
#include "NeuralNetwork.h"

#define WEIGHTS_MIN -3
#define WEIGHTS_MAX 3


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
    for (size_t i = 0; i < nn->n_weights) {
        free(nn->weights[i]);
    }
    free(nn->weights);
    free(nn);
}



/***** NEURAL NETWORK OPERATIONS *****/

void nn_activate(neural_net* nn, matrix* output, const matrix* input, double (*activation_func)(double z)) {
    // TODO
}
