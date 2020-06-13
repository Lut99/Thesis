/* NEURAL NETWORK OMP CPU2.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   6/13/2020, 2:36:15 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The NeuralNetwork class implements a matrix-based Feedforward Neural
 *   Network which is hardcoded to use Mean Squared Error for cost function and
 *   sigmoid as activation function.
 * 
 *   This file implements the second of eight different OpenMP-optimised
 *   versions for the CPU. It optimises both the forward and the backward pass
 *   using threads, where any race conditions in the backward pass are fixed
 *   using critical regions.
**/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "NeuralNetwork.h"

#define WEIGHTS_MIN -3.0
#define WEIGHTS_MAX 3.0
#define BIAS_MIN -3.0
#define BIAS_MAX 3.0
#define NUM_THREADS 16


/***** OPTIONAL PARAMETERS *****/
static unsigned int n_threads = 16;


/***** OPENMP DECLARATIONS *****/
extern int omp_set_num_threads();
extern int omp_get_num_procs();
extern int omp_get_thread_num();


/***** HELPER FUNCTIONS *****/

extern size_t max(size_t length, const size_t* list);



/***** NEURAL NETWORK OPERATIONS *****/

void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Make some shortcuts for the number-of-nodes information
    size_t n_layers = nn->n_layers;
    size_t* nodes_per_layer = nn->nodes_per_layer;
    
    // Initialize the temporary delta memory (and previous list) to the correct size, one for each thread
    size_t deltas_size = max(n_layers, nodes_per_layer);
    double* deltas = malloc(sizeof(double) * n_threads * deltas_size);
    double* prev_deltas = malloc(sizeof(double) * n_threads * deltas_size);

    // Create a list that is used to store intermediate outputs. The first input layer (=first column)
    //   is linked and not copied to the input data
    double* layer_outputs[n_threads][n_layers];
    for (unsigned int t = 0; t < n_threads; t++) {
        // Allocate arrays for the other layers except 
        for (size_t l = 1; l < n_layers; l++) {
            layer_outputs[t][l] = malloc(sizeof(double) * nodes_per_layer[l]);
        }
    }

    // Create the delta_biases and delta_weights arrays / matrices
    double* delta_biases[nn->n_weights];
    double* delta_weights[nn->n_weights];
    for(size_t l = 0; l < nn->n_weights; l++) {
        delta_biases[l] = malloc(sizeof(double) * nodes_per_layer[l + 1]);
        delta_weights[l] = malloc(sizeof(double) * nodes_per_layer[l] * nodes_per_layer[l + 1]);

        // Fill with zeros
        for (size_t n = 0; n < nodes_per_layer[l + 1]; n++) {
            delta_biases[l][n] = 0;
            for (size_t prev_n = 0; prev_n < nodes_per_layer[l]; prev_n++) {
                delta_weights[l][prev_n * nodes_per_layer[l + 1] + n] = 0;
            }
        }
    }

    // Perform the training for n_iterations (always)
    size_t last_nodes = nodes_per_layer[n_layers - 1];
    size_t last_prev_nodes = nodes_per_layer[n_layers - 2];
    double* last_delta_bias = delta_biases[n_layers - 2];
    double* last_delta_weight = delta_weights[n_layers - 2];
    for (size_t i = 0; i < n_iterations; i++) {
        #pragma omp parallel
        {
            int TID = omp_get_thread_num();
            double* t_deltas = deltas + TID * deltas_size;
            double* t_prev_deltas = prev_deltas + TID * deltas_size;
            double** t_layer_outputs = layer_outputs[TID];

            #pragma omp for schedule(static)
            for (size_t s = 0; s < n_samples; s++) {
                /***** FORWARD PASS *****/

                // Set the inputs as the first layer
                t_layer_outputs[0] = inputs[s];

                // Iterate over each layer to feedforward through the network
                for (size_t l = 1; l < n_layers; l++) {
                    // Get some references to the bias list, weight matrix and outputs of the previous and this layer
                    double* bias = biases[l - 1];
                    double* weight = weights[l - 1];
                    double* prev_output = t_layer_outputs[l - 1];
                    double* output = t_layer_outputs[l];

                    // Compute the activation for each node on this layer
                    size_t this_nodes = nodes_per_layer[l];
                    size_t prev_nodes = nodes_per_layer[l - 1];
                    for (size_t n = 0; n < this_nodes; n++) {
                        // Sum the weighted inputs for this node
                        double z = bias[n];
                        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                            z += prev_output[prev_n] * weight[prev_n * this_nodes + n];
                        }

                        // Run the activation function over this input and store it in the output
                        output[n] = 1 / (1 + exp(-z));
                    }
                }

                /***** BACKWARD PASS *****/
                // Implementation: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547

                // Backpropagate the error from the last layer to the first.
                double* sample_expected = expected[s];

                // Do the output layer: compute the deltas
                double* output = t_layer_outputs[n_layers - 1];
                for (size_t n = 0; n < last_nodes; n++) {
                    double output_val = output[n];
                    t_prev_deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
                }

                // Do the output layer: compute the bias & weight updates
                #pragma omp critical
                {  
                    // Add all deltas as delta_biases for this layer
                    for (size_t n = 0; n < last_nodes; n++) {
                        last_delta_bias[n] += t_prev_deltas[n];
                    }
                    // Same for all the weights, except we compute the delta_weights first
                    double* last_prev_output = t_layer_outputs[n_layers - 2];
                    for (size_t prev_n = 0; prev_n < last_prev_nodes; prev_n++) {
                        for (size_t n = 0; n < last_nodes; n++) {
                            last_delta_weight[prev_n * last_nodes + n] += last_prev_output[prev_n] * t_prev_deltas[n];
                        }
                    }
                }
                

                // Then, the rest of the hidden layers
                for (size_t l = n_layers - 2; l > 0; l--) {
                    double* delta_bias = delta_biases[l - 1];
                    double* delta_weight = delta_weights[l - 1];
                    double* output = t_layer_outputs[l];
                    double* prev_output = t_layer_outputs[l - 1];
                    size_t next_nodes = nodes_per_layer[l + 1];
                    size_t this_nodes = nodes_per_layer[l];
                    size_t prev_nodes = nodes_per_layer[l - 1];
                    
                    // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
                    double* weight_next = weights[l];
                    for (size_t n = 0; n < this_nodes; n++) {
                        // Take the weighted sum of all connection of that node with this layer
                        double error = 0;
                        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                            error += t_prev_deltas[next_n] * weight_next[n * next_nodes + next_n];
                        }

                        // Multiply the error with the derivative of the activation function to find the result
                        double output_val = output[n];
                        t_deltas[n] = error * output_val * (1 - output_val);
                    }

                    // Add all deltas as delta_biases for this layer
                    #pragma omp critical
                    {
                        for (size_t n = 0; n < this_nodes; n++) {
                            delta_bias[n] += t_deltas[n];
                        }
                        // Same for all the weights, except we compute the delta_weights first
                        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                            for (size_t n = 0; n < this_nodes; n++) {
                                delta_weight[prev_n * this_nodes + n] += prev_output[prev_n] * t_deltas[n];
                            }
                        }
                    }

                    // Swap the two delta lists
                    double* temp = t_deltas;
                    t_deltas = t_prev_deltas;
                    t_prev_deltas = temp;
                }
            }

            // Actually update the weights, and reset the delta updates to 0 for next iteration
            #pragma omp for schedule(static)
            for (size_t l = 0; l < nn->n_weights; l++) {
                double* bias = biases[l];
                double* delta_bias = delta_biases[l];
                double* weight = weights[l];
                double* delta_weight = delta_weights[l];

                // Update the biases & reset delta_biases
                size_t this_nodes = nodes_per_layer[l + 1];
                for (size_t n = 0; n < this_nodes; n++) {
                    bias[n] += delta_bias[n] * learning_rate;
                    delta_bias[n] = 0;
                }

                // Update the weights & reset delta_weights
                size_t prev_nodes = nodes_per_layer[l];
                for (size_t i = 0; i < this_nodes * prev_nodes; i++) {
                    weight[i] += delta_weight[i] * learning_rate;
                    delta_weight[i] = 0;
                }
            }
        }
    }

    // Cleanup

    // Free the delta biases / weights
    for(size_t l = 0; l < n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (unsigned int t = 0; t < n_threads; t++) {
        for (size_t l = 1; l < n_layers; l++) {
            free(layer_outputs[t][l]);
        }
    }

    // Cleanup the deltas
    free(deltas);
    free(prev_deltas);
}



/***** OTHER TOOLS *****/

void parse_opt_args(int argc, char** argv) {
    // Parse and set number of threads as first argument
    if (argc >= 1) {
        // Set the number of threads
        n_threads = atoi(argv[0]);
    }
    omp_set_num_threads(n_threads);
}

void print_opt_args() {
    printf(" - Variation               : OpenMP CPU 2 (Forward & Backward, critical)\n");
    printf(" - Number of threads       : %u\n", n_threads);
}

