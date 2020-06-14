/* NEURAL NETWORK OMP CPU6.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   6/15/2020, 12:20:16 AM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The NeuralNetwork class implements a matrix-based Feedforward Neural
 *   Network which is hardcoded to use Mean Squared Error for cost function and
 *   sigmoid as activation function.
 * 
 *   This file implements the sixth of eight different OpenMP-optimised
 *   versions for the CPU. It optimises both the forward and the backward pass
 *   using threads, where any race conditions in the backward pass are fixed
 *   using critical regions. Additionally, inner loops are optimised with SIMD.
**/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

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

#define TIMEVAL_TO_MS(T_START, T_END) (((T_END.tv_sec - T_START.tv_sec) * 1000000 + (T_END.tv_usec - T_START.tv_usec)) / 1000000.0)

extern size_t max(size_t length, const size_t* list);



/***** NEURAL NETWORK OPERATIONS *****/

void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    #ifdef BENCHMARK
    // Declare all timers
    struct timeval s_total, e_total, s_iters, e_iters, s_fwd, e_fwd, s_bck_out, e_bck_out, s_bck_hid, e_bck_hid, s_upd, e_upd;

    // Set some shortcuts for the timers
    size_t half_iters = n_iterations / 2;
    size_t half_samples = n_samples / 2;

    // Start the total timer
    gettimeofday(&s_total, NULL);
    #endif

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
    for (size_t t = 0; t < n_threads; t++) {
        // Allocate arrays for the other layers
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

    #ifdef BENCHMARK
    // Start the iterations timer
    gettimeofday(&s_iters, NULL);
    #endif

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

            // Loop through all samples to compute the forward cost
            #pragma omp for schedule(static)
            for (size_t s = 0; s < n_samples; s++) {

                /***** FORWARD PASS *****/

                #ifdef BENCHMARK
                // Start the forward pass timer
                if (i == half_iters && s == half_samples) {
                    gettimeofday(&s_fwd, NULL);
                }
                #endif

                // Link first input to first layer output
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
                        #pragma omp simd
                        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                            z += prev_output[prev_n] * weight[prev_n * this_nodes + n];
                        }

                        // Run the activation function over this input and store it in the output
                        output[n] = 1 / (1 + exp(-z));
                    }
                }

                #ifdef BENCHMARK
                // End the forward timer, start the backward pass output timer
                if (i == half_iters && s == half_samples) {
                    gettimeofday(&e_fwd, NULL);
                    gettimeofday(&s_bck_out, NULL);
                }
                #endif

                /***** BACKWARD PASS *****/
                // Implementation: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547

                // Backpropagate the error from the last layer to the first.
                double* sample_expected = expected[s];

                // Do the output layer: compute the deltas
                double* output = t_layer_outputs[n_layers - 1];
                #pragma omp simd
                for (size_t n = 0; n < last_nodes; n++) {
                    double output_val = output[n];
                    t_prev_deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
                }

                // Do the output layer: compute the bias & weight updates
                #pragma omp critical
                {
                    // Add all deltas as delta_biases for this layer
                    #pragma omp simd
                    for (size_t n = 0; n < last_nodes; n++) {
                        last_delta_bias[n] += t_prev_deltas[n];
                    }
                    // Same for all the weights, except we compute the delta_weights first
                    double* last_prev_output = t_layer_outputs[n_layers - 2];
                    for (size_t prev_n = 0; prev_n < last_prev_nodes; prev_n++) {
                        #pragma omp simd
                        for (size_t n = 0; n < last_nodes; n++) {
                            last_delta_weight[prev_n * last_nodes + n] += last_prev_output[prev_n] * t_prev_deltas[n];
                        }
                    }
                }

                #ifdef BENCHMARK
                // End the backward pass output timer, start the backward pass hidden timer
                if (i == half_iters && s == half_samples) {
                    gettimeofday(&e_bck_out, NULL);
                    gettimeofday(&s_bck_hid, NULL);
                }
                #endif
                

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
                        #pragma omp simd
                        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                            error += t_prev_deltas[next_n] * weight_next[n * next_nodes + next_n];
                        }

                        // Multiply the error with the derivative of the activation function to find the result
                        double output_val = output[n];
                        t_deltas[n] = error * output_val * (1 - output_val);
                    }

                    #pragma omp critical
                    {  
                        // Add all deltas as delta_biases for this layer
                        #pragma omp simd
                        for (size_t n = 0; n < this_nodes; n++) {
                            delta_bias[n] += t_deltas[n];
                        }
                        // Same for all the weights, except we compute the delta_weights first
                        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                            #pragma omp simd
                            for (size_t n = 0; n < this_nodes; n++) {
                                delta_weight[prev_n * this_nodes + n] += prev_output[prev_n] * t_deltas[n];
                            }
                        }
                    }

                    // Swap the two deltas
                    double* temp = t_deltas;
                    t_deltas = t_prev_deltas;
                    t_prev_deltas = temp;
                }

                #ifdef BENCHMARK
                // End the backward pass hidden timer
                if (i == half_iters && s == half_samples) {
                    gettimeofday(&e_bck_hid, NULL);
                }
                #endif
            }

            #ifdef BENCHMARK
            // Start the updates timer
            if (i == half_iters) {
                gettimeofday(&s_upd, NULL);
            }
            #endif

            // Actually update the weights, and reset the delta updates to 0 for next iteration
            #pragma omp for schedule(static)
            for (size_t l = 0; l < nn->n_weights; l++) {
                double* bias = biases[l];
                double* delta_bias = delta_biases[l];
                double* weight = weights[l];
                double* delta_weight = delta_weights[l];

                // Update the biases & reset delta_biases
                size_t this_nodes = nodes_per_layer[l + 1];
                #pragma omp simd
                for (size_t n = 0; n < this_nodes; n++) {
                    bias[n] += delta_bias[n] * learning_rate;
                    delta_bias[n] = 0;
                }

                // Update the weights & reset delta_weights
                size_t prev_nodes = nodes_per_layer[l];
                #pragma omp simd
                for (size_t i = 0; i < this_nodes * prev_nodes; i++) {
                    weight[i] += delta_weight[i] * learning_rate;
                    delta_weight[i] = 0;
                }
            }

            #ifdef BENCHMARK
            // Stop the updates timer
            if (i == half_iters) {
                gettimeofday(&e_upd, NULL);
            }
            #endif
        }
    }

    #ifdef BENCHMARK
    // End the iterations timer
    gettimeofday(&e_iters, NULL);
    #endif

    // Cleanup

    // Free the delta biases / weights
    for(size_t l = 0; l < n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (size_t t = 0; t < n_threads; t++) {
        for (size_t l = 1; l < n_layers; l++) {
            free(layer_outputs[t][l]);
        }
    }

    // Cleanup the deltas
    free(deltas);
    free(prev_deltas);

    #ifdef BENCHMARK
    // End the total timer
    gettimeofday(&e_total, NULL);

    // Print the results
    printf("%f\n", TIMEVAL_TO_MS(s_total, e_total));
    printf("%f\n", TIMEVAL_TO_MS(s_iters, e_iters));
    printf("%f\n", TIMEVAL_TO_MS(s_fwd, e_fwd));
    printf("%f\n", TIMEVAL_TO_MS(s_bck_out, e_bck_out));
    printf("%f\n", TIMEVAL_TO_MS(s_bck_hid, e_bck_hid));
    printf("%f\n", TIMEVAL_TO_MS(s_upd, e_upd));
    #endif
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
    printf(" - Variation               : OpenMP CPU 6 (Forward & Backward, critical, with SIMD)\n");
    printf(" - Number of threads       : %u\n", n_threads);
}

