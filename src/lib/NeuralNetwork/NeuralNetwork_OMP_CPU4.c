/* NEURAL NETWORK OMP CPU4.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   6/1/2020, 12:11:21 AM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a neural network using a matrix-based
 *   implementation (using Matrix.c) rather than an object-oriented-based
 *   implementation. Any special functions used (such as activation or loss
 *   functions) are defined in Functions.c.
 * 
 *   This particular version implements an OpenMP-accelerated version. It
 *   optimises the inner loops that go over the layer, but not with threads
 *   (as this takes unreasonably long), but with SIMD.
**/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

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

/* Returns the maximum size_t in a list of size_ts. */
size_t max(size_t size, const size_t* data) {
    size_t m = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }
    return m;
}

/* Creates and initialises a bias matrix for the number of given nodes. */
double* initialize_biases(size_t n_nodes) {
    // Create a new matrix of the proper dimensions
    double* to_ret = malloc(sizeof(double) * n_nodes);

    // Set each value to a random one in the range BIAS_MIN (inclusive) and BIAS_MAX (exclusive)
    for (size_t i = 0; i < n_nodes; i++) {
        to_ret[i] = (double)rand()/RAND_MAX * (BIAS_MAX - BIAS_MIN) + BIAS_MIN;
    }

    // Return
    return to_ret;
}

/* Creates and initialises a weights matrix of given proportions. */
double* initialize_weights(size_t input_size, size_t output_size) {
    // Create a new matrix of the proper dimensions
    double* to_ret = malloc(sizeof(double) * input_size * output_size);

    // Set each value to a random one in the range WEIGHTS_MIN (inclusive) and WEIGHTS_MAX (exclusive)
    for (size_t i = 0; i < input_size * output_size; i++) {
        to_ret[i] = (double)rand()/RAND_MAX * (WEIGHTS_MAX - WEIGHTS_MIN) + WEIGHTS_MIN;
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
    to_ret->biases = malloc(sizeof(double*) * to_ret->n_weights);
    to_ret->weights = malloc(sizeof(double*) * to_ret->n_weights);
    if (to_ret->nodes_per_layer == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate nodes list (%lu bytes).\n",
                sizeof(size_t) * to_ret->n_layers);
        return NULL;
    } else if (to_ret->biases == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate biases list (%lu bytes).\n",
                sizeof(double*) * to_ret->n_weights);
        return NULL;
    } else if (to_ret->weights == NULL) {
        fprintf(stderr, "ERROR: create_nn: could not allocate weights list (%lu bytes).\n",
                sizeof(double*) * to_ret->n_weights);
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
        free(nn->biases[i]);
        free(nn->weights[i]);
    }
    free(nn->biases);
    free(nn->weights);
    free(nn);
}



/***** NEURAL NETWORK OPERATIONS *****/

void nn_forward(neural_net* nn, size_t n_samples, double* outputs, double** inputs) {
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Declare a temporary inputs and outputs list
    size_t max_nodes = max(nn->n_layers, nn->nodes_per_layer);
    double* intermediate_inputs = malloc(sizeof(double) * max_nodes);
    double* intermediate_outputs = malloc(sizeof(double) * max_nodes);

    // Loop through all samples to compute the forward cost
    size_t last_nodes = nn->nodes_per_layer[nn->n_layers - 1];
    for (size_t s = 0; s < n_samples; s++) {
        // Initialize the intermediate inputs to the real inputs for this sample
        memcpy(intermediate_inputs, inputs[s], sizeof(double) * nn->nodes_per_layer[0]);

        // Iterate over each layer to feedforward through the network
        for (size_t l = 1; l < nn->n_layers; l++) {
            // Get some references to the bias list and weight matrix
            double* bias = biases[l - 1];
            double* weight = weights[l - 1];

            // Compute the activation for each node on this layer
            size_t this_nodes = nn->nodes_per_layer[l];
            size_t prev_nodes = nn->nodes_per_layer[l - 1];
            for (size_t n = 0; n < this_nodes; n++) {
                // Sum the weighted inputs for this node
                double z = bias[n];
                for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                    z += intermediate_inputs[prev_n] * weight[prev_n * this_nodes + n];
                }

                // Run the activation function over this input and store it in the output
                intermediate_outputs[n] = 1 / (1 + exp(-z));
            }

            // Swap the pointers
            double* temp = intermediate_inputs;
            intermediate_inputs = intermediate_outputs;
            intermediate_outputs = temp;
        }

        // Copy the intermediate outputs (due to swapping, this is intermediate_inputs) for this sample to the eventual outputs of this sample
        memcpy(outputs + s * last_nodes, intermediate_inputs, sizeof(double) * last_nodes);
    }

    // Destroy the intermediate lists
    free(intermediate_inputs);
    free(intermediate_outputs);
}





array* nn_train_costs(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    // Allocate an array for the costs and initialize the scratchpad memory to the correct size
    array* costs = create_empty_array(n_iterations);

    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;
    
    // Initialize the temporary delta memory to the correct size
    double* deltas = malloc(sizeof(double) * max(nn->n_layers, nn->nodes_per_layer));

    // Create a list that is used to store intermediate outputs. The first input layer (=first column)
    //   is linked and not copied to the input data
    double* layer_outputs[n_samples][nn->n_layers];
    for (size_t s = 0; s < n_samples; s++) {
        // Link the input layer
        layer_outputs[s][0] = inputs[s];
        
        // Allocate arrays for the other layers
        for (size_t l = 1; l < nn->n_layers; l++) {
            layer_outputs[s][l] = malloc(sizeof(double) * nn->nodes_per_layer[l]);
        }
    }

    // Create the delta_biases and delta_weights arrays / matrices
    double* delta_biases[nn->n_weights];
    double* delta_weights[nn->n_weights];
    for(size_t l = 0; l < nn->n_weights; l++) {
        delta_biases[l] = malloc(sizeof(double) * nn->nodes_per_layer[l + 1]);
        delta_weights[l] = malloc(sizeof(double) * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1]);

        // Fill with zeros
        for (size_t n = 0; n < nn->nodes_per_layer[l + 1]; n++) {
            delta_biases[l][n] = 0;
            for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l]; prev_n++) {
                delta_weights[l][prev_n * nn->nodes_per_layer[l + 1] + n] = 0;
            }
        }
    }

    // Perform the training for n_iterations (always)
    for (size_t i = 0; i < n_iterations; i++) {
        /***** FORWARD PASS *****/

        // Loop through all samples to compute the forward cost
        for (size_t s = 0; s < n_samples; s++) {
            // Perform a forward pass through the network to be able to say something about the performance

            // sample_outputs is a 2D flattened array for this layer
            double** sample_outputs = layer_outputs[s];

            // Iterate over each layer to feedforward through the network
            for (size_t l = 1; l < nn->n_layers; l++) {
                // Get some references to the bias list, weight matrix and outputs of the previous and this layer
                double* bias = biases[l - 1];
                double* weight = weights[l - 1];
                double* prev_output = sample_outputs[l - 1];
                double* output = sample_outputs[l];

                // Compute the activation for each node on this layer
                size_t this_nodes = nn->nodes_per_layer[l];
                size_t prev_nodes = nn->nodes_per_layer[l - 1];
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

            // Compute the cost for this sample
            double cost = 0;
            for (size_t n = 0; n < nn->nodes_per_layer[nn->n_layers - 1]; n++)  {
                double err = (sample_outputs[nn->n_layers - 1][n] - expected[s][n]);
                cost += err * err;
            }
            costs->d[i] += cost / nn->nodes_per_layer[nn->n_layers - 1];
        }

        // Report it once every hundred
        if (i % 100 == 0) {
            printf("    (Iter %lu) Cost: %.4f\n", i, costs->d[i]);
        }

        /***** BACKWARD PASS *****/
        // Implementation: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547

        // Loop through all samples to compute the backward cost
        for (size_t s = 0; s < n_samples; s++) {
            // Backpropagate the error from the last layer to the first.
            double** sample_outputs = layer_outputs[s];
            double* sample_expected = expected[s];
            for (size_t l = nn->n_layers - 1; l > 0; l--) {
                // Set shortcuts to some values used both in delta computing and weight / bias updating
                size_t this_nodes = nn->nodes_per_layer[l];
                double* output = sample_outputs[l];

                // Compute the deltas of the correct layer
                if (l == nn->n_layers - 1) {
                    // Deltas for output layer

                    // Loop through all nodes in this layer to compute their deltas
                    for (size_t n = 0; n < this_nodes; n++) {
                        double output_val = output[n];
                        deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
                    }
                } else {
                    // Deltas for any hidden layer
                    
                    // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
                    size_t next_nodes = nn->nodes_per_layer[l + 1];
                    double* weight_next = weights[l];
                    for (size_t n = 0; n < this_nodes; n++) {
                        // Take the weighted sum of all connection of that node with this layer
                        double error = 0;
                        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                            error += deltas[next_n] * weight_next[n * next_nodes + next_n];
                        }

                        // Multiply the error with the derivative of the activation function to find the result
                        double output_val = output[n];
                        deltas[n] = error * output_val * (1 - output_val);
                    }
                }

                // Set some shutcuts for weight updating alone so they don't have to be recomputed each iteration
                size_t prev_nodes = nn->nodes_per_layer[l - 1];
                double* delta_bias = delta_biases[l - 1];
                double* delta_weight = delta_weights[l - 1];
                double* prev_output = sample_outputs[l - 1];

                // Add all deltas as delta_biases for this layer
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_bias[n] += deltas[n];
                }
                // Same for all the weights, except we compute the delta_weights first
                for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                    for (size_t n = 0; n < this_nodes; n++) {
                        delta_weight[prev_n * this_nodes + n] += prev_output[prev_n] * deltas[n];
                    }
                }
            }
        }

        // Actually update the weights, and reset the delta updates to 0 for next iteration
        for (size_t l = 0; l < nn->n_weights; l++) {
            double* bias = biases[l];
            double* delta_bias = delta_biases[l];
            double* weight = weights[l];
            double* delta_weight = delta_weights[l];

            // Update the biases & reset delta_biases
            size_t this_nodes = nn->nodes_per_layer[l + 1];
            for (size_t n = 0; n < this_nodes; n++) {
                bias[n] += delta_bias[n] * learning_rate;
                delta_bias[n] = 0;
            }

            // Update the weights & reset delta_weights
            size_t prev_nodes = nn->nodes_per_layer[l];
            for (size_t i = 0; i < this_nodes * prev_nodes; i++) {
                weight[i] += delta_weight[i] * learning_rate;
                delta_weight[i] = 0;
            }
        }
    }

    // Cleanup

    // Free the delta biases / weights
    for(size_t l = 0; l < nn->n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t l = 1; l < nn->n_layers; l++) {
            free(layer_outputs[s][l]);
        }
    }

    // Cleanup the deltas
    free(deltas);

    return costs;
}

void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Make some shortcuts for the number-of-nodes information
    size_t n_layers = nn->n_layers;
    size_t* nodes_per_layer = nn->nodes_per_layer;
    
    // Initialize the temporary delta memory to the correct size
    double* deltas = malloc(sizeof(double) * max(n_layers, nodes_per_layer));

    // Create a list that is used to store intermediate outputs. The first input layer (=first column)
    //   is linked and not copied to the input data
    double* layer_outputs[n_samples][n_layers];
    for (size_t s = 0; s < n_samples; s++) {
        // Link the input layer
        layer_outputs[s][0] = inputs[s];
        
        // Allocate arrays for the other layers
        for (size_t l = 1; l < n_layers; l++) {
            layer_outputs[s][l] = malloc(sizeof(double) * nodes_per_layer[l]);
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
    for (size_t i = 0; i < n_iterations; i++) {
        /***** FORWARD PASS *****/

        // Loop through all samples to compute the forward cost
        for (size_t s = 0; s < n_samples; s++) {
            // Perform a forward pass through the network to be able to say something about the performance

            // sample_outputs is a 2D flattened array for this layer
            double** sample_outputs = layer_outputs[s];

            // Iterate over each layer to feedforward through the network
            for (size_t l = 1; l < n_layers; l++) {
                // Get some references to the bias list, weight matrix and outputs of the previous and this layer
                double* bias = biases[l - 1];
                double* weight = weights[l - 1];
                double* prev_output = sample_outputs[l - 1];
                double* output = sample_outputs[l];

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
        }

        /***** BACKWARD PASS *****/
        // Implementation: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547

        // Loop through all samples to compute the backward cost
        for (size_t s = 0; s < n_samples; s++) {
            // Backpropagate the error from the last layer to the first.
            double** sample_outputs = layer_outputs[s];
            double* sample_expected = expected[s];
            for (size_t l = n_layers - 1; l > 0; l--) {
                // Set shortcuts to some values used both in delta computing and weight / bias updating
                size_t this_nodes = nodes_per_layer[l];
                double* output = sample_outputs[l];

                // Compute the deltas of the correct layer
                if (l == n_layers - 1) {
                    // Deltas for output layer

                    // Loop through all nodes in this layer to compute their deltas
                    #pragma omp simd
                    for (size_t n = 0; n < this_nodes; n++) {
                        double output_val = output[n];
                        deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
                    }
                } else {
                    // Deltas for any hidden layer
                    
                    // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
                    size_t next_nodes = nodes_per_layer[l + 1];
                    double* weight_next = weights[l];
                    for (size_t n = 0; n < this_nodes; n++) {
                        // Take the weighted sum of all connection of that node with this layer
                        double error = 0;
                        #pragma omp simd
                        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                            error += deltas[next_n] * weight_next[n * next_nodes + next_n];
                        }

                        // Multiply the error with the derivative of the activation function to find the result
                        double output_val = output[n];
                        deltas[n] = error * output_val * (1 - output_val);
                    }
                }

                // Set some shutcuts for weight updating alone so they don't have to be recomputed each iteration
                size_t prev_nodes = nodes_per_layer[l - 1];
                double* delta_bias = delta_biases[l - 1];
                double* delta_weight = delta_weights[l - 1];
                double* prev_output = sample_outputs[l - 1];

                // Add all deltas as delta_biases for this layer
                #pragma omp simd
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_bias[n] += deltas[n];
                }
                // Same for all the weights, except we compute the delta_weights first
                for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                    #pragma omp simd
                    for (size_t n = 0; n < this_nodes; n++) {
                        delta_weight[prev_n * this_nodes + n] += prev_output[prev_n] * deltas[n];
                    }
                }
            }
        }

        // Actually update the weights, and reset the delta updates to 0 for next iteration
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
    }

    // Cleanup

    // Free the delta biases / weights
    for(size_t l = 0; l < n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t l = 1; l < n_layers; l++) {
            free(layer_outputs[s][l]);
        }
    }

    // Cleanup the deltas
    free(deltas);
}



/***** VALIDATION TOOLS *****/

void flatten_output(size_t n_samples, size_t last_nodes, double* outputs) {
    for (size_t s = 0; s < n_samples; s++) {
        double* output = outputs + s * last_nodes;
        
        // First pass: collect the highest value of this sample
        double max_value = -INFINITY;
        double max_index = 0;
        for (size_t n = 0; n < last_nodes; n++) {
            if (output[n] > max_value) {
                max_value = output[n];
                max_index = n;
            }
        }

        // Second pass: set all to 0, save for the highest value, which will be set to 1
        for (size_t n = 0; n < last_nodes; n++) {
            output[n] = n == max_index ? 1.0 : 0.0;
        }
    }
}

void round_output(size_t n_samples, size_t last_nodes, double* outputs) {
    for (size_t s = 0; s < n_samples; s++) {
        double* output = outputs + s * last_nodes;
        
        // Round each element
        for (size_t n = 0; n < last_nodes; n++) {
            output[n] = round(output[n]);
        }
    }
}

double compute_accuracy(size_t n_samples, size_t last_nodes, double* outputs, double** expected) {
    double correct = 0;
    for (size_t s = 0; s < n_samples; s++) {
        double* output = outputs + s * last_nodes;
        double* expect = expected[s];
        
        // Compare each element
        bool equal = true;
        for (size_t n = 0; n < last_nodes; n++) {
            equal = equal && fabs(output[n] - expect[n]) < 0.0001;
        }

        // Update correct based on if they were equal
        correct += equal ? 1.0 : 0.0;
    }
    return correct / n_samples;
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
    printf("Configuration:\n");
    printf(" - Variation         : OpenMP CPU 4 (SIMD only)\n");
    printf(" - Number of threads : %u\n\n", n_threads);
}

