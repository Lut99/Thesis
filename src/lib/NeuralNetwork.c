/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   28/04/2020, 01:17:36
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
    matrix* to_ret = create_empty_matrix(input_size, output_size);

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
    matrix* to_ret = create_empty_matrix(1, n_nodes);

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
    to_ret->n_weights = to_ret->n_layers - 1;

    // Store the number of nodes per layer
    to_ret->nodes_per_layer = malloc(sizeof(size_t) * to_ret->n_layers);
    for (size_t i = 0; i < n_hidden_layers; i++) {
        to_ret->nodes_per_layer[i + 1] = hidden_nodes[i];
    }
    // Also add the input and output layers
    to_ret->nodes_per_layer[0] = input_nodes;
    to_ret->nodes_per_layer[to_ret->n_layers - 1] = output_nodes;
    
    // Initialise the weights of the neural network randomly
    to_ret->biases = malloc(sizeof(matrix*) * to_ret->n_weights);
    to_ret->weights = malloc(sizeof(matrix*) * to_ret->n_weights);
    for (size_t i = 0; i < to_ret->n_weights; i++) {
        to_ret->biases[i] = initialise_biases(to_ret->nodes_per_layer[i + 1]);
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

#include "stdio.h"
void nn_activate_all(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* inputs, matrix* (*act)(matrix*)) {
    // Copy the input matrix to be sure we do not deallocate it
    matrix* inputs2 = copy_matrix_new(inputs);

    // Iterate over each layer to feedforward through the network
    for (size_t i = 0; i < nn->n_weights; i++) {
        // Store the output without bias in the list
        outputs[i] = inputs2;
        
        // Then, compute the input values for the nodes in this layer
        matrix* z = matrix_matmul(inputs2, nn->weights[i]);

        // Add the bias if we're in a hidden layer
        matrix_add_inplace(z, nn->biases[i]);

        // Apply the activation function
        act(z);

        // Set z as the new one
        inputs2 = z;
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

// void nn_backpropagate_old(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate) {
//     // Compute the error on the output layer
//     // Compute the activation of this layer
//     matrix* in = matrix_matmul(nn->weights[nn->n_weights - 1], outputs[nn->n_weights - 1]);
//     matrix* act = dydx_act(in);
//     // Compute the activation times the derivative of the cost function
//     matrix* error = matrix_mul_c_inplace(act, dydx_cost(outputs[nn->n_layers - 1], expected));
//     destroy_matrix(in);

//     // Loop backwards through all the layers
//     for (size_t i = nn->n_layers - 2; i <= nn->n_layers - 2; i--) {
//         // Compute the weights deltas with the old error
//         matrix* output_T = matrix_transpose(outputs[i]);
//         matrix* d_weights = matrix_matmul(error, output_T);

//         if (i > 0) {
//             // Compute the delta bias for this layer
//             double d_bias = matrix_sum(error);

//             // Get the input through the derivative of the activation function. Don't forget to add the bias.
//             in = matrix_matmul(nn->weights[i - 1], outputs[i - 1]);
//             matrix_add_c_inplace(act, nn->biases[i]);
//             act = dydx_act(in);

//             // Compute the new error based on the input
//             matrix* weights_T = matrix_transpose(nn->weights[i]);
//             matrix* weighted_err = matrix_matmul(weights_T, error);
//             destroy_matrix(error);
//             error = matrix_mul_inplace(weighted_err, act);

//             // Update the bias matrix
//             nn->biases[i] -= learning_rate * d_bias;

//             // Cleanup
//             destroy_matrix(in);
//             destroy_matrix(act);
//             destroy_matrix(weights_T);
//         }



//         // Update the weights of this layer
//         matrix_sub_inplace(nn->weights[i], matrix_mul_c_inplace(d_weights, learning_rate));

//         // Cleanup
//         destroy_matrix(output_T);
//         destroy_matrix(d_weights);
//     }

//     // Cleanup the error
//     destroy_matrix(error);
// }

void nn_backpropagate(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate, matrix* (*dydx_act)(const matrix*)) {
    // Compute the deltas at the output layer first
    matrix* error = matrix_sub(expected, outputs[nn->n_layers - 1]);
    matrix* deltas = matrix_mul_inplace(dydx_act(outputs[nn->n_layers - 1]), error);
    destroy_matrix(error);

    // For all other layers, update the weights and the biases. Only compute a new error for all hidden layers.
    for (size_t i = nn->n_layers - 1; i > 0; i--) {
        // Compute the d_weights and d_bias
        matrix* d_bias = matrix_mul_c(deltas, learning_rate);
        matrix* deltas_T = matrix_transpose(deltas);
        matrix* d_weights = matrix_mul_c_inplace(matrix_matmul(outputs[i - 1], deltas_T), learning_rate);

        // Compute a new deltas
        matrix* weight_T = matrix_transpose(nn->weights[i - 1]);
        error = matrix_matmul(deltas, weight_T);
        destroy_matrix(deltas);
        deltas = matrix_mul_inplace(dydx_act(outputs[i]), error);

        // Update the bias and the weights
        matrix_add_inplace(nn->biases[i - 1], d_bias);
        matrix_add_inplace(nn->weights[i - 1], d_weights);

        // Cleanup
        destroy_matrix(d_bias);
        destroy_matrix(deltas_T);
        destroy_matrix(d_weights);
        destroy_matrix(weight_T);
        destroy_matrix(error);
    }

    // Destroy the deltas
    destroy_matrix(deltas);
}



double* nn_train_costs(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*)) {
    // Allocate the list for the costs
    double* costs = malloc(sizeof(double) * n_iterations);

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

            // Compute the cost (Mean Squared Error)
            matrix* err = matrix_sub(outputs[nn->n_layers - 1], input_gold);
            costs[i] = matrix_sum(matrix_square_inplace(err)) / err->cols;

            // Print the cost once every 100 iterations
            if (i % 100 == 0) {
                printf("    (Iter %lu) Cost: %.2f\n", i, costs[i]);
            }
            
            // Perform a backpropagation
            nn_backpropagate(nn, outputs, input_gold, learning_rate, dydx_act);

            // Cleanup the matrices that tracked the output of each layer and clean the help matrix
            for (size_t i = 0; i < nn->n_layers; i++) {
                destroy_matrix(outputs[i]);
            }
            destroy_matrix(err);
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
