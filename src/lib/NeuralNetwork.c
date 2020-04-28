/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   28/04/2020, 21:29:20
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
#define ITERATION_STOP_MARGIN 0.000001


/***** HELPER FUNCTIONS *****/

/* Creates and initialises a bias matrix for the number of given nodes. */
array* initialize_biases(size_t n_nodes) {
    // Create a new matrix of the proper dimensions
    array* to_ret = create_empty_array(n_nodes);

    // Set each value to a random one in the range BIAS_MIN (inclusive) and BIAS_MAX (exclusive)
    for (size_t i = 0; i < to_ret->size; i++) {
        to_ret->d[i] = (double)rand()/RAND_MAX * (BIAS_MAX - BIAS_MIN) + BIAS_MIN;
    }

    // Return
    return to_ret;
}

/* Creates and initialises a weights matrix of given proportions. */
matrix* initialize_weights(size_t input_size, size_t output_size) {
    // Create a new matrix of the proper dimensions
    matrix* to_ret = create_empty_matrix(input_size, output_size);

    // Set each value to a random one in the range WEIGHTS_MIN (inclusive) and WEIGHTS_MAX (exclusive)
    for (size_t i = 0; i < to_ret->rows * to_ret->cols; i++) {
        to_ret->data[i] = (double)rand()/RAND_MAX * (WEIGHTS_MAX - WEIGHTS_MIN) + WEIGHTS_MIN;
    }

    // Return
    return to_ret;
}



/***** MEMORY MANAGEMENT *****/

neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t hidden_nodes[n_hidden_layers], size_t output_nodes) {
    // Start by seeding the pseudo-random number generator
    srand(time(NULL));

    // Create a new neural net object
    neural_net* to_ret = malloc(sizeof(neural_net));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate struct (%lu bytes).\n",
                sizeof(neural_net));
        return NULL;
    }

    // Allocate the required lists for the neural network
    to_ret->nodes_per_layer = malloc(sizeof(size_t) * to_ret->n_layers);
    to_ret->biases = malloc(sizeof(array*) * to_ret->n_weights);
    to_ret->weights = malloc(sizeof(matrix*) * to_ret->n_weights);
    if (to_ret->nodes_per_layer == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate nodes list (%lu bytes).\n",
                sizeof(size_t) * to_ret->n_layers);
        return NULL;
    } else if (to_ret->biases == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate biases list (%lu bytes).\n",
                sizeof(array*) * to_ret->n_weights);
        return NULL;
    } else if (to_ret->weights == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate weights list (%lu bytes).\n",
                sizeof(matrix*) * to_ret->n_weights);
        return NULL;
    }

    // Fill in the size values
    to_ret->n_layers = n_hidden_layers + 2;
    to_ret->n_weights = to_ret->n_layers - 1;

    // Store the number of nodes per layer
    for (size_t i = 0; i < n_hidden_layers; i++) {
        to_ret->nodes_per_layer[i + 1] = hidden_nodes[i];
    }
    // Also add the input and output layers
    to_ret->nodes_per_layer[0] = input_nodes;
    to_ret->nodes_per_layer[to_ret->n_layers - 1] = output_nodes;
    
    // Initialise the biases and weights of the neural network randomly
    for (size_t i = 0; i < to_ret->n_weights; i++) {
        to_ret->biases[i] = initialize_biases(to_ret->nodes_per_layer[i + 1]);
        if (to_ret->biases[i] == NULL) {
            fprintf(stderr, "ERROR: create_nn: could not initialize bias (%lu/%lu).\n",
                    i + 1, to_ret->n_weights);
            return NULL;
        }
        to_ret->weights[i] = initialize_weights(to_ret->nodes_per_layer[i], to_ret->nodes_per_layer[i + 1]);
        if (to_ret->weights[i] == NULL) {
            fprintf(stderr, "ERROR: create_nn: could not initialize weight (%lu/%lu).\n",
                    i + 1, to_ret->n_weights)
            return NULL;
        }
    }

    // Done, return
    return to_ret;
}

void destroy_nn(neural_net* nn) {
    free(nn->nodes_per_layer);
    for (size_t i = 0; i < nn->n_weights; i++) {
        destroy_array(nn->biases[i]);
        destroy_matrix(nn->weights[i]);
    }
    free(nn->biases);
    free(nn->weights);
    free(nn);
}



/***** NEURAL NETWORK OPERATIONS *****/

void nn_activate_all(neural_net* nn, array* outputs, const matrix* inputs, matrix* (*act)(matrix*)) {
    // Copy the input matrix to be sure we do not deallocate it
    outputs[0] = copy_create_array(inputs);

    // Iterate over each layer to feedforward through the network
    for (size_t l = 1; l < nn->n_layers; l++) {
        create_array[i]

        // Compute the activation for each node on this layer
        for (size_t n = 0; n < nn->nodes_per_layer[l]; n++) {
            
        }

        // Then, compute the input values for the nodes in this layer
        matrix* z = matrix_matmul(outputs, nn->weights[i]);

        // Add the bias if we're in a hidden layer
        matrix_add_inplace(z, nn->biases[i]);

        // Apply the activation function
        act(z);

        // Set z as the new one
        outputs[i] = z;
    }
    // Add the output from the final layer in the outputs
    outputs[nn->n_layers - 1] = inputs2;
}

matrix* nn_activate(neural_net* nn, const matrix* inputs, matrix* (*act)(matrix*)) {
    // Prepare the output matrix
    matrix* to_ret = create_empty_matrix(inputs->rows, nn->nodes_per_layer[nn->n_layers - 1]);

    // Prepare a hacky matrix to reference one row only
    matrix* input = malloc(sizeof(matrix));
    input->rows = 1;
    input->cols = inputs->cols;

    // Prepare the buffer for all the intermediate outputs
    matrix* outputs[nn->n_layers];

    // Loop through all samples
    for (size_t y = 0; y < inputs->rows; y++) {
        // Use the hacky matrix to select only the current row
        input->data = inputs->data + y * inputs->cols;

        // Let the activation run through the matrix
        nn_activate_all(nn, outputs, input, act);

        // Copy the last output to the relevant row in the output matrix
        for (size_t x = 0; x < outputs[nn->n_layers - 1]->cols; x++) {
            to_ret->data[y * to_ret->cols + x] = outputs[nn->n_layers - 1]->data[x];
        }

        // Destroy all outputs
        for (size_t i = 0; i < nn->n_layers; i++) {
            destroy_matrix(outputs[i]);
        }
    }

    // Free the hacky matrix
    free(input);

    // Return the last output
    return to_ret;
}

void nn_backpropagate(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate, matrix* (*dydx_act)(const matrix*)) {
    // Compute the deltas at the output layer first
    matrix* error = matrix_sub(expected, outputs[nn->n_layers - 1]);
    matrix* deltas = matrix_mul_inplace(dydx_act(outputs[nn->n_layers - 1]), error);
    destroy_matrix(error);

    // For all other layers, update the weights and the biases. Only compute a new error for all hidden layers.
    for (size_t i = nn->n_layers - 1; i > 0; i--) {
        // Compute the d_weights and d_bias
        matrix* d_bias = matrix_mul_c(deltas, learning_rate);
        matrix* output_T = matrix_transpose(outputs[i - 1]);
        matrix* d_weights = matrix_mul_c_inplace(matrix_matmul(output_T, deltas), learning_rate);

        // Compute a new deltas
        matrix* weight_T = matrix_transpose(nn->weights[i - 1]);
        matrix* error = matrix_matmul(deltas, weight_T);
        destroy_matrix(deltas);
        deltas = matrix_mul_inplace(dydx_act(outputs[i - 1]), error);

        // Update the bias and the weights
        matrix_add_inplace(nn->biases[i - 1], d_bias);
        matrix_add_inplace(nn->weights[i - 1], d_weights);

        // Cleanup
        destroy_matrix(d_bias);
        destroy_matrix(output_T);
        destroy_matrix(d_weights);
        destroy_matrix(weight_T);
        destroy_matrix(error);
    }

    // Destroy the deltas
    destroy_matrix(deltas);
}



matrix* nn_train_costs(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t max_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*)) {
    // Allocate the list for the costs
    matrix* costs = create_empty_matrix(1, max_iterations);

    // Create some hacky matrix objects that will be used to quickly reference single rows
    matrix* input = malloc(sizeof(matrix));
    input->rows = 1;
    input->cols = inputs->cols;
    matrix* input_gold = malloc(sizeof(matrix));
    input_gold->rows = 1;
    input_gold->cols = expected->cols;

    // Perform the training
    matrix* outputs[nn->n_layers];
    for (size_t i = 0; i < max_iterations; i++) {
        costs->data[i] = 0;

        for (size_t j = 0; j < inputs->rows; j++) {
            // Assign the current row to the hacky matrices
            input->data = inputs->data + j * inputs->cols;
            input_gold->data = expected->data + j * expected->cols;

            // First, perform a forward pass through the network
            nn_activate_all(nn, outputs, input, act);

            // Compute the cost (Mean Squared Error)
            matrix* err = matrix_sub(outputs[nn->n_layers - 1], input_gold);
            costs->data[i] += matrix_sum(matrix_square_inplace(err)) / err->cols;

            // Perform a backpropagation
            nn_backpropagate(nn, outputs, input_gold, learning_rate, dydx_act);

            // Cleanup the matrices that tracked the output of each layer and clean the help matrix
            for (size_t i = 0; i < nn->n_layers; i++) {
                destroy_matrix(outputs[i]);
            }
            destroy_matrix(err);
        }

        // Print the cost once every 100 iterations
        if (i % 100 == 0) {
            printf("    (Iter %lu) Cost: %.2f\n", i, costs->data[i]);
        }

        // Stop if the change in cost is small enough
        if (i > 0 && fabs(costs->data[i - 1] - costs->data[i]) < ITERATION_STOP_MARGIN) {
            // Copy the costs to a smaller area
            matrix* new_costs = create_empty_matrix(1, i + 1);
            for (size_t j = 0; j <= i; j++) {
                new_costs->data[j] = costs->data[j];
            }
            destroy_matrix(costs);
            costs = new_costs;
            break;
        }
    }

    // Clean the hacky matrix objects
    free(input);
    free(input_gold);

    // Return the costs list
    return costs;
}

void nn_train(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*)) {
    // Create some hacky matrix objects that will be used to quickly reference single rows
    matrix* input = malloc(sizeof(matrix));
    input->rows = 1;
    input->cols = inputs->cols;
    matrix* input_gold = malloc(sizeof(matrix));
    input_gold->rows = 1;
    input_gold->cols = expected->cols;
    
    // Perform the training
    matrix* outputs[nn->n_layers];
    for (size_t i = 0; i < n_iterations; i++) {
        for (size_t j = 0; j < inputs->rows; j++) {
            // Assign the current row to the hacky matrices
            input->data = inputs->data + j * inputs->cols;
            input_gold->data = expected->data + j * expected->cols;

            // First, perform a forward pass through the network
            nn_activate_all(nn, outputs, input, act);

            // Perform a backpropagation
            nn_backpropagate(nn, outputs, input_gold, learning_rate, dydx_act);

            // Cleanup the matrices that tracked the output of each layer
            for (size_t i = 0; i < nn->n_layers; i++) {
                destroy_matrix(outputs[i]);
            }
        }
    }

    // Clean the hacky matrix objects
    free(input);
    free(input_gold);
}



/***** USEFUL TOOLS *****/

matrix* nn_flatten_results(matrix* outputs) {
    for (size_t y = 0; y < outputs->rows; y++) {
        double highest_index = 0;
        double highest_value = -INFINITY;
        for (size_t x = 0; x < outputs->cols; x++) {
            double data = outputs->data[y * outputs->cols + x];
            if (data > highest_value) {
                highest_index = x;
                highest_value = data;
            }
        }
        for (size_t x = 0; x < outputs->cols; x++) {
            outputs->data[y * outputs->cols + x] = x == highest_index ? 1 : 0;
        }
    }

    // Return the matrix for chaining
    return outputs;
}
