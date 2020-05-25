/* NEURAL NETWORK CUDA GPU1.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   5/25/2020, 9:33:39 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a neural network using a matrix-based
 *   implementation (using Matrix.c) rather than an object-oriented-based
 *   implementation. Any special functions used (such as activation or loss
 *   functions) are defined in Functions.c.
 * 
 *   This particular version implements a CUDA-accelerated version. It builds
 *   on the swapped loops as seen in the OpenMP-CPU variation 8, to exploit
 *   maximum parallelism. Note that the actual parts which use CUDA are
 *   offloaded to NeuralNetwork_CUDA_GPU1.cu, so that we can use C99+ features
 *   here.
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


/***** OPTIONAL PARAMETERS *****/
static unsigned int batch_size = 32;
static unsigned int n_batches = 256;



/***** HELPER FUNCTIONS *****/

/* Returns the maximum size_t in a list of size_ts. */
size_t max(size_t size, const size_t data[size]) {
    size_t m = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }
    return m;
}

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

    // Fill in the size values
    to_ret->n_layers = n_hidden_layers + 2;
    to_ret->n_weights = to_ret->n_layers - 1;

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
                    i + 1, to_ret->n_weights);
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

void nn_activate(neural_net* nn, array* outputs[nn->n_layers], const array* inputs, double (*act)(double)) {
    // Copy the inputs to the outputs array
    copy_array(outputs[0], inputs);

    // Iterate over each layer to feedforward through the network
    for (size_t l = 1; l < nn->n_layers; l++) {
        // Get some references to the bias list, weight matrix and outputs of the previous and this layer
        array* bias = nn->biases[l - 1];
        matrix* weight = nn->weights[l - 1];
        array* prev_output = outputs[l - 1];
        array* output = outputs[l];

        // Compute the activation for each node on this layer
        for (size_t n = 0; n < nn->nodes_per_layer[l]; n++) {
            // Sum the weighted inputs for this node
            double z = bias->d[n];
            for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l - 1]; prev_n++) {
                z += prev_output->d[prev_n] * INDEX(weight, prev_n, n);
            }

            // Run the activation function over this input and store it in the output
            output->d[n] = act(z);
        }
    }

    // Done forward pass.
}

void nn_forward(neural_net* nn, size_t n_samples, array* outputs[n_samples], array* inputs[n_samples], double (*act)(double)) {
    // Prepare a fully allocated list of arrays for the intermediate outputs for each sample
    array* layer_outputs[nn->n_layers];
    for (size_t l = 0; l < nn->n_layers; l++) {
        layer_outputs[l] = create_empty_array(nn->nodes_per_layer[l]);
    }

    // Loop through all samples
    for (size_t i = 0; i < n_samples; i++) {
        // Let the activation run through the matrix
        nn_activate(nn, layer_outputs, inputs[i], act);

        // Copy the elements of the last outputs to the general outputs
        copy_array(outputs[i], layer_outputs[nn->n_layers - 1]);
    }

    // Destroy the intermediate output arrays
    for (size_t l = 0; l < nn->n_layers; l++) {
        destroy_array(layer_outputs[l]);
    }
}

// Implementation: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
void nn_backpropagate(neural_net* nn, array* outputs[nn->n_layers], const array* expected, double learning_rate, double (*dydx_act)(double), array* deltas) {
    // Backpropagate the error from the last layer to the first. Note that the deltas are computed on non-updated matrices.
    for (size_t l = nn->n_layers - 1; l > 0; l--) {
        // Set shortcuts to some values used both in delta computing and weight / bias updating
        size_t this_nodes = nn->nodes_per_layer[l];
        array* output = outputs[l];

        // Compute the deltas of the correct layer
        if (l == nn->n_layers - 1) {
            // Deltas for output layer

            // Loop through all nodes in this layer to compute their deltas
            for (size_t n = 0; n < this_nodes; n++) {
                deltas->d[n] = (expected->d[n] - output->d[n]) * dydx_act(output->d[n]);
            }
        } else {
            // Deltas for any hidden layer
            
            // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
            size_t next_nodes = nn->nodes_per_layer[l + 1];
            matrix* weight_next = nn->weights[l];
            for (size_t n = 0; n < this_nodes; n++) {
                // Take the weighted sum of all connection of that node with this layer
                double error = 0;
                for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                    error += deltas->d[next_n] * INDEX(weight_next, n, next_n);
                }

                // Multiply the error with the derivative of the activation function to find the result
                deltas->d[n] = error * dydx_act(output->d[n]);
            }
        }

        // Set some shutcuts for weight updating alone so they don't have to be recomputed each iteration
        size_t prev_nodes = nn->nodes_per_layer[l - 1];
        array* bias = nn->biases[l - 1];
        matrix* weight = nn->weights[l - 1];
        array* prev_output = outputs[l - 1];

        // Updated all biases and weights for this layer
        for (size_t n = 0; n < this_nodes; n++) {
            bias->d[n] +=  deltas->d[n] * learning_rate;
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                INDEX(weight, prev_n, n) += prev_output->d[prev_n] * deltas->d[n] * learning_rate;
            }
        }
    }
}



array* nn_train_costs(neural_net* nn, size_t n_samples, array* inputs[n_samples], array* expected[n_samples], double learning_rate, size_t n_iterations, double (*act)(double), double (*dydx_act)(double)) {
    // Allocate a list for the costs and initialize the scratchpad memory to the correct size
    array* costs = create_empty_array(n_iterations);

    // Initialize the scratchpad memory to the correct size
    array* deltas = create_empty_array(max(nn->n_layers, nn->nodes_per_layer));
    
    // Create a list that is used to store intermediate outputs. Note that we create no create_empty_array
    //   for the first element, as this is simply a reference to the input.
    array* layer_outputs[n_samples][nn->n_layers];
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t l = 1; l < nn->n_layers; l++) {
            layer_outputs[s][l] = create_empty_array(nn->nodes_per_layer[l]);
        }
    }

    // Create the delta_biases and delta_weights arrays / matrices
    array* delta_biases[nn->n_layers - 1];
    matrix* delta_weights[nn->n_layers - 1];
    for (size_t l = 0; l < nn->n_layers - 1; l++) {
        delta_biases[l] = create_empty_array(nn->biases[l]->size);
        delta_weights[l] = create_empty_matrix(nn->weights[l]->rows, nn->weights[l]->cols);

        // Fill with zeros
        for (size_t n = 0; n < nn->nodes_per_layer[l + 1]; n++) {
            delta_biases[l]->d[n] = 0;
            for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l]; prev_n++) {
                INDEX(delta_weights[l], prev_n, n) = 0;
            }
        }
    }

    // Perform the training for n_iterations (always)
    for (size_t i = 0; i < n_iterations; i++) {
        // Set the cost for this iteration to 0
        costs->d[i] = 0;

        // Loop through all samples to compute the forward cost
        for (size_t s = 0; s < n_samples; s++) {

            // Perform a forward pass through the network to be able to say something about the performance
            array** sample_outputs = layer_outputs[s];
            
            // Copy the inputs to the outputs array
            sample_outputs[0] = inputs[s];

            // Iterate over each layer to feedforward through the network
            for (size_t l = 1; l < nn->n_layers; l++) {
                // Get some references to the bias list, weight matrix and outputs of the previous and this layer
                array* bias = nn->biases[l - 1];
                matrix* weight = nn->weights[l - 1];
                array* prev_output = sample_outputs[l - 1];
                array* output = sample_outputs[l];

                // Compute the activation for each node on this layer
                for (size_t n = 0; n < nn->nodes_per_layer[l]; n++) {
                    // Sum the weighted inputs for this node
                    double z = bias->d[n];
                    for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l - 1]; prev_n++) {
                        z += prev_output->d[prev_n] * INDEX(weight, prev_n, n);
                    }

                    // Run the activation function over this input and store it in the output
                    output->d[n] = act(z);
                }
            }

            // Compute the cost for this sample
            double cost = 0;
            for (size_t n = 0; n < nn->nodes_per_layer[nn->n_layers - 1]; n++)  {
                double err = (sample_outputs[nn->n_layers - 1]->d[n] - expected[s]->d[n]);
                cost += err * err;
            }
            costs->d[i] += cost / nn->nodes_per_layer[nn->n_layers - 1];
        }

        // Report it once every hundred
        if (i % 100 == 0) {
            printf("    (Iter %lu) Cost: %.4f\n", i, costs->d[i]);
        }

        // Loop through all samples to compute the backward cost
        for (size_t s = 0; s < n_samples; s++) {
            // Backpropagate the error from the last layer to the first.
            array** sample_outputs = layer_outputs[s];
            array* sample_expected = expected[s];
            for (size_t l = nn->n_layers - 1; l > 0; l--) {
                // Set shortcuts to some values used both in delta computing and weight / bias updating
                size_t this_nodes = nn->nodes_per_layer[l];
                array* output = sample_outputs[l];

                // Compute the deltas of the correct layer
                if (l == nn->n_layers - 1) {
                    // Deltas for output layer

                    // Loop through all nodes in this layer to compute their deltas
                    for (size_t n = 0; n < this_nodes; n++) {
                        deltas->d[n] = (sample_expected->d[n] - output->d[n]) * dydx_act(output->d[n]);
                    }
                } else {
                    // Deltas for any hidden layer
                    
                    // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
                    size_t next_nodes = nn->nodes_per_layer[l + 1];
                    matrix* weight_next = nn->weights[l];
                    for (size_t n = 0; n < this_nodes; n++) {
                        // Take the weighted sum of all connection of that node with this layer
                        double error = 0;
                        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                            error += deltas->d[next_n] * INDEX(weight_next, n, next_n);
                        }

                        // Multiply the error with the derivative of the activation function to find the result
                        deltas->d[n] = error * dydx_act(output->d[n]);
                    }
                }

                // Set some shutcuts for weight updating alone so they don't have to be recomputed each iteration
                size_t prev_nodes = nn->nodes_per_layer[l - 1];
                array* delta_bias = delta_biases[l - 1];
                matrix* delta_weight = delta_weights[l - 1];
                array* prev_output = sample_outputs[l - 1];

                // Updated all biases and weights for this layer
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_bias->d[n] +=  deltas->d[n];
                    for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                        INDEX(delta_weight, prev_n, n) += prev_output->d[prev_n] * deltas->d[n];
                    }
                }
            }
        }

        // Actually update the weights, and reset the delta updates to 0 for next iteration
        for (size_t l = 1; l < nn->n_layers; l++) {
            for (size_t n = 0; n < nn->nodes_per_layer[l]; n++) {
                nn->biases[l - 1]->d[n] += delta_biases[l - 1]->d[n] * learning_rate;
                delta_biases[l - 1]->d[n] = 0;
                for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l - 1]; prev_n++) {
                    INDEX(nn->weights[l - 1], prev_n, n) += INDEX(delta_weights[l - 1], prev_n, n) * learning_rate;
                    INDEX(delta_weights[l - 1], prev_n, n) = 0;
                }
            }   
        }
    }

    // Cleanup
    // Destroy the intermediate outputs - but not the first one, as this is simply copied by pointer from the input list
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t l = 1; l < nn->n_layers; l++) {
            destroy_array(layer_outputs[s][l]);
        }
    }
    // Destroy the delta_weights
    for (size_t l = 0; l < nn->n_layers - 1; l++) {
        destroy_array(delta_biases[l]);
        destroy_matrix(delta_weights[l]);
    }
    destroy_array(deltas);

    return costs;
}



/***** VALIDATION TOOLS *****/

void flatten_output(size_t n_samples, array* outputs[n_samples]) {
    for (size_t s = 0; s < n_samples; s++) {
        array* output = outputs[s];

        // First pass: collect the highest value of this sample
        double max_value = -INFINITY;
        double max_index = 0;
        for (size_t n = 0; n < output->size; n++) {
            if (output->d[n] > max_value) {
                max_value = output->d[n];
                max_index = n;
            }
        }

        // Second pass: set all to 0, save for the highest value, which will be set to 1
        for (size_t n = 0; n < output->size; n++) {
            output->d[n] = n == max_index ? 1.0 : 0.0;
        }
    }
}

void round_output(size_t n_samples, array* outputs[n_samples]) {
    for (size_t s = 0; s < n_samples; s++) {
        array* output = outputs[s];

        // Round each element
        for (size_t n = 0; n < output->size; n++) {
            output->d[n] = round(output->d[n]);
        }
    }
}

double compute_accuracy(size_t n_samples, array* outputs[n_samples], array* expected[n_samples]) {
    double correct = 0;
    for (size_t s = 0; s < n_samples; s++) {
        array* output = outputs[s];
        array* expect = expected[s];

        // Check if the lists are equally sized
        if (output->size != expect->size) {
            fprintf(stderr, "ERROR: compute_accuracy: the sizes of the output (%lu) and expected (%lu) are not equal for sample %lu.\n",
                    output->size, expect->size, s);
            return -1;
        }

        // Compare each element
        bool equal = true;
        for (size_t n = 0; n < output->size; n++) {
            equal = equal && fabs(output->d[n] - expect->d[n]) < 0.0001;
        }

        // Update correct based on if they were equal
        correct += equal ? 1.0 : 0.0;
    }
    return correct / n_samples;
}



/***** OTHER TOOLS *****/

void parse_opt_args(int argc, char** argv) {
    (void) argc;
    (void) argv;
}

void print_opt_args() {
    
}
