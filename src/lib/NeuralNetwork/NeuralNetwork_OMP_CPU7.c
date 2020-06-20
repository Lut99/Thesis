/* NEURAL NETWORK OMP CPU7.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   6/20/2020, 5:25:10 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The NeuralNetwork class implements a matrix-based Feedforward Neural
 *   Network which is hardcoded to use Mean Squared Error for cost function and
 *   sigmoid as activation function.
 * 
 *   This file implements the seventh of eight different OpenMP-optimised
 *   versions for the CPU. It applies algorithmic optimisations for for-loops
 *   to achieve better parallelism (moving to a pipelined structure) and then
 *   uses threads to achieve so. Note that race conditions for the backward
 *   pass are solved using reduction.
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


/***** OPTIONAL PARAMETERS *****/
static unsigned int n_threads = 16;


/***** OPENMP DECLARATIONS *****/
extern int omp_set_num_threads();
extern int omp_get_num_procs();
extern int omp_get_thread_num();


/***** HELPER FUNCTIONS *****/

#define TIMEVAL_TO_MS(T_START, T_END) (((T_END.tv_sec - T_START.tv_sec) * 1000000 + (T_END.tv_usec - T_START.tv_usec)) / 1000000.0)

extern size_t max(size_t length, const size_t* list);


double mean(size_t length, double* list) {
    double total = 0;
    for (size_t i = 0; i < length; i++) {
        total += list[i];
    }
    return total / length;
}


double variance(size_t length, double* list) {
    // COmpute the mean
    double m = mean(length, list);
    // Use that to compute the variance
    double total = 0;
    for (size_t i = 0; i < length; i++) {
        total += (list[i] - m) * (list[i] - m);
    }
    return total / length;
}



/***** NEURAL NETWORK OPERATIONS *****/

void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    #ifdef BENCHMARK
    // Declare all timers
    struct timeval s_total, e_total, s_iters, e_iters, s_fwd, e_fwd, s_bck_out, e_bck_out, s_bck_hid, e_bck_hid, s_upd, e_upd;

    // Set some shortcuts for the timers
    size_t half_iters = n_iterations / 2;

    // Start the total timer
    gettimeofday(&s_total, NULL);
    #else
    struct timeval s_fwd, e_fwd;
    double t_fwd = 0;
    #endif

    
    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Make some shortcuts for the number-of-nodes information
    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;
    
    // Initialize the temporary delta memory to the correct size, one for every sample
    size_t deltas_size = max(n_layers, nodes_per_layer);
    double* deltas = malloc(sizeof(double) * n_samples * deltas_size);
    double* prev_deltas = malloc(sizeof(double) * n_samples * deltas_size);

    // Create a list that is used to store intermediate outputs. Note that, unlike other variations,
    //   the layer_outputs is transposed (so layers on the rows rather than samples). Aside from cache
    //   friendliness, this also reduces memory accesses. Also note that this means the input is copied
    //   rather than linked.
    double* layer_outputs[n_layers];
    for (size_t l = 0; l < n_layers; l++) {
        // Create a memory allocation for this layer
        layer_outputs[l] = malloc(sizeof(double) * n_samples * nodes_per_layer[l]);
    }
    // Copy the input
    for (size_t s = 0; s < n_samples; s++) {
        memcpy(layer_outputs[0] + s * nodes_per_layer[0], inputs[s], sizeof(double) * nodes_per_layer[0]);
    }

    // Create the delta_biases and delta_weights arrays / matrices
    double* delta_biases[n_weights];
    double* delta_weights[n_weights];
    for(size_t l = 0; l < n_weights; l++) {
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

    // Perform the training for n_iterations (always) (20,000 iterations, non-parallelizable)
    for (size_t i = 0; i < n_iterations; i++) {
        /***** FORWARD PASS *****/

        #ifdef BENCHMARK
        // Start the forward pass timer
        if (i == half_iters) {
            gettimeofday(&s_fwd, NULL);
        }
        #else
        // Start the forward pass timer
        gettimeofday(&s_fwd, NULL);
        #endif

        // Loop through all layers forwardly so that we can compute errors later (2 iterations, non-parallelizable)
        for (size_t l = 1; l < nn->n_layers; l++) {
            // Create some shortcuts
            double* bias = biases[l - 1];
            double* weight = weights[l - 1];
            double* this_outputs = layer_outputs[l];
            double* prev_outputs = layer_outputs[l - 1];
            size_t this_nodes = nodes_per_layer[l];
            size_t prev_nodes = nodes_per_layer[l - 1];

            #ifndef BENCHMARK
            // Define a list to keep track of each sample's duration
            double benchmarks[n_samples * this_nodes];
            #endif
            

            // Iterate over all available samples (1797 x 20 first iteration of l, 1797 x 10 second iteration)
            #pragma omp parallel for schedule(static) collapse(2)
            for (size_t s = 0; s < n_samples; s++) {
                // Compute the activation for each node on this layer
                for (size_t n = 0; n < this_nodes; n++) {
                    #ifndef BENCHMARK
                    // Start the sample timer
                    struct timeval s_sample, e_sample;
                    gettimeofday(&s_sample, NULL);
                    #endif
                    // Sum the weighted inputs for this node (64 first iteration of l, 20 for second iteration)
                    double z = bias[n];
                    for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                        z += prev_outputs[s * prev_nodes + prev_n] * weight[prev_n * this_nodes + n];
                    }

                    // Run the activation function over this input and store it in the output
                    this_outputs[s * this_nodes + n] = 1 / (1 + exp(-z));

                    #ifndef BENCHMARK
                    // End the sample timer
                    gettimeofday(&e_sample, NULL);
                    benchmarks[s * this_nodes + n] = TIMEVAL_TO_MS(s_sample, e_sample);
                    #endif
                }
            }

            #ifndef BENCHMARK
            if (l == 1) {
                // Print the results
                printf("\nAverage thread duration : %fs\n", mean(n_samples * this_nodes, benchmarks));
                printf("               Variance : %f\n", variance(n_samples * this_nodes, benchmarks));
            }
            #endif
        }

        #ifdef BENCHMARK
        // End the forward timer, start the backward pass output timer
        if (i == half_iters) {
            gettimeofday(&e_fwd, NULL);
            gettimeofday(&s_bck_out, NULL);
        }
        #else
        // End forward timer, add to average
        gettimeofday(&e_fwd, NULL);
        t_fwd += TIMEVAL_TO_MS(s_fwd, e_fwd);
        #endif

        /***** BACKWARD PASS *****/

        // First, compute the error at the output layer
        size_t this_nodes = nodes_per_layer[n_layers - 1];
        size_t prev_nodes = nodes_per_layer[n_layers - 2];
        
        // Compute the deltas for all samples (1797 x 10 iterations)
        double* this_outputs = layer_outputs[n_layers - 1];
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t s = 0; s < n_samples; s++) {
            for (size_t n = 0; n < this_nodes; n++) {
                double output_val = this_outputs[s * this_nodes + n];
                prev_deltas[s * deltas_size + n] = (expected[s][n] - output_val) * output_val * (1 - output_val);
            }
        }

        // Use those deltas to update the change in biases and weights (1797 x 10 iterations, non-parallelizable)
        double* delta_bias = delta_biases[n_layers - 2];
        double* delta_weight = delta_weights[n_layers - 2];
        double* prev_outputs = layer_outputs[n_layers - 2];
        for (size_t s = 0; s < n_samples; s++) {
            double* sample_prev_deltas = prev_deltas + s * deltas_size;
            
            // Update the delta biases
            for (size_t n = 0; n < this_nodes; n++) {
                delta_bias[n] += sample_prev_deltas[n];
            }

            // Also do the weights but more cache-friendly
            double* sample_outputs = prev_outputs + s * prev_nodes;
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_weight[prev_n * this_nodes + n] += sample_outputs[prev_n] * sample_prev_deltas[n];
                }
            }
        }

        #ifdef BENCHMARK
        // End the backward pass output timer, start the backward pass hidden timer
        if (i == half_iters) {
            gettimeofday(&e_bck_out, NULL);
            gettimeofday(&s_bck_hid, NULL);
        }
        #endif
        
        // Do the other, hidden layers (1 iteration, non-parallelizable)
        for (size_t l = nn->n_layers - 2; l > 0; l--) {
            // Set some shortcuts
            double* weight_next = weights[l];
            delta_bias = delta_biases[l - 1];
            delta_weight = delta_weights[l - 1];
            this_outputs = layer_outputs[l];
            prev_outputs = layer_outputs[l - 1];
            size_t next_nodes = nodes_per_layer[l + 1];
            this_nodes = nodes_per_layer[l];
            prev_nodes = nodes_per_layer[l - 1];

            // Loop through all the samples available on this layer to compute the deltas (1797 x 20 iterations)
            #pragma omp parallel for schedule(static) collapse(2)
            for (size_t s = 0; s < n_samples; s++) {
                // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
                for (size_t n = 0; n < this_nodes; n++) {
                    double* sample_prev_deltas = prev_deltas + s * deltas_size;
                    double* sample_deltas = deltas + s * deltas_size;

                    // Take the weighted sum of all connection of that node with this layer (10 iterations)
                    double error = 0;
                    for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                        error += sample_prev_deltas[next_n] * weight_next[n * next_nodes + next_n];
                    }

                    // Multiply the error with the derivative of the activation function to find the result
                    double output_val = this_outputs[s * this_nodes + n];
                    sample_deltas[n] = error * output_val * (1 - output_val);
                }
            }

            // Use those to update the change in biases and weights (1797 x 20 iterations, non-paralellizable)
            for (size_t s = 0; s < n_samples; s++) {
                double* sample_deltas = deltas + s * deltas_size;
                
                // Update the delta biases
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_bias[n] += sample_deltas[n];
                }

                // Also do the weights but more cache-friendly
                double* sample_outputs = prev_outputs + s * prev_nodes;
                for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                    for (size_t n = 0; n < this_nodes; n++) {
                        delta_weight[prev_n * this_nodes + n] += sample_outputs[prev_n] * sample_deltas[n];
                    }
                }
            }

            // Swap the deltas and prev_deltas
            double* temp = deltas;
            deltas = prev_deltas;
            prev_deltas = temp;
        }

        #ifdef BENCHMARK
        // End the backward pass hidden timer
        if (i == half_iters) {
            gettimeofday(&e_bck_hid, NULL);
            gettimeofday(&s_upd, NULL);
        }
        #endif

        // Actually update the weights, and reset the delta updates to 0 for next iteration (2 iterations)
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

        #ifdef BENCHMARK
        // Stop the updates timer
        if (i == half_iters) {
            gettimeofday(&e_upd, NULL);
        }
        #endif
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
    for (size_t l = 0; l < n_layers; l++) {
        free(layer_outputs[l]);
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
    #else
    // Print dat average
    printf("Average time for the forward pass : %fs\n", t_fwd / n_iterations);
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
    printf(" - Variation               : OpenMP CPU 7 (Forward & Backward, algorithmic optimisation)\n");
    printf(" - Number of threads       : %u\n", n_threads);
}

