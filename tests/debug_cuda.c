/* DEBUG CUDA.c
 *   by Anonymous
 *
 * Created:
 *   6/11/2020, 9:31:55 PM
 * Last edited:
 *   6/13/2020, 1:19:40 AM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   File to debug the CUDA_GPU1 version
**/

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "NeuralNetwork.h"
extern void nn_forward_cuda(neural_net* nn, size_t n_samples, double* outputs, double** inputs);
extern void nn_backward_cuda(neural_net* nn, double** delta_biases_cpu, double** delta_weights_cpu, size_t n_samples, double** inputs, double** expected);
extern void nn_full_cuda(neural_net* nn, double** delta_biases_cpu, double** delta_weights_cpu, size_t n_samples, double** inputs, double** expected, double learning_rate);


extern size_t max(size_t size, const size_t* data);


/* Cleans given list of pointers, also free'ing the pointers (except when those pointers are NULL).
 *
 * Parameters:
 *   @param length length of the given list
 *   @param list the list itself
 */
 void clean(size_t length, double** list) {
    for (size_t l = 0; l < length; l++) {
        if (list[l] != NULL) {
            free(list[l]);
        }
    }
    free(list);
}


/* Generates a dataset with random doubles in the given range. Additionally, each element is given a random class, also in the specified range.
 *
 * Parameters:
 *   @param dataset the resulting pointer that is allocated by the function which will point to the 2D-array of the generated datapoints
 *   @param classes the resulting pointer that is allocated by the function which will point to the 2D-array of the randomly assigned classes
 *   @param n_samples desired number of samples in the dataset
 *   @param sample_size desired number of doubles for every sample
 *   @param data_min lower bound (inclusive) of the random range of values
 *   @param data_max upper bound (exclusive) of the random range of values
 *   @param n_classes number of classes for this dataset
 */
 void generate_random(double*** dataset, double*** classes, size_t n_samples, size_t sample_size, double data_min, double data_max, size_t n_classes) {
    // Seed the random
    srand(time(NULL));

    // First, malloc the datasets and classes main lists
    *dataset = malloc(sizeof(double*) * n_samples);
    *classes = malloc(sizeof(double*) * n_samples);

    // Next, fill 'em for every sample
    for (size_t s = 0; s < n_samples; s++) {
        (*dataset)[s] = malloc(sizeof(double) * sample_size);
        (*classes)[s] = malloc(sizeof(double) * n_classes);

        // First, fill the data
        for (size_t i = 0; i < sample_size; i++) {
            (*dataset)[s][i] = ((double) rand() / RAND_MAX) * (data_max - data_min) + data_min;
        }

        // Next, assign a random class
        size_t class = rand() % n_classes;
        for (size_t i = 0; i < n_classes; i++) {
            (*classes)[s][i] = i == class ? 1.0 : 0.0;
        }
    }
}


void nn_backward(neural_net* nn, double** delta_biases_cpu, double** delta_weights_cpu, size_t n_samples, double** inputs, double** expected) {
    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Make some shortcuts for the number-of-nodes information
    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;
    
    // Initialize the temporary delta memory to the correct size
    double* deltas = malloc(sizeof(double) * max(n_layers, nodes_per_layer));
    double* prev_deltas = malloc(sizeof(double) * max(n_layers, nodes_per_layer));

    // Create a list that is used to store intermediate outputs. The first input layer (=first column)
    //   is linked and not copied to the input data
    double* layer_outputs[n_layers];
    // Allocate arrays for all layers except the first layer, as that will
    //   be linked to the appropriate input layer
    for (size_t l = 1; l < n_layers; l++) {
        layer_outputs[l] = malloc(sizeof(double) * nodes_per_layer[l]);
    }

    // Create the delta_biases and delta_weights arrays / matrices
    double* delta_biases[nn->n_weights];
    double* delta_weights[nn->n_weights];
    for(size_t l = 0; l < nn->n_weights; l++) {
        delta_biases[l] = malloc(sizeof(double) * n_samples * nodes_per_layer[l + 1]);
        delta_weights[l] = malloc(sizeof(double) * n_samples * nodes_per_layer[l] * nodes_per_layer[l + 1]);
    }

    // Perform the training for n_iterations (always)
    size_t last_nodes = nodes_per_layer[n_layers - 1];
    size_t last_prev_nodes = nodes_per_layer[n_layers - 2];
    double* last_delta_bias = delta_biases[n_layers - 2];
    double* last_delta_weight = delta_weights[n_layers - 2];
    for (size_t s = 0; s < n_samples; s++) {
        /***** FORWARD PASS *****/

        // Link the first output to the input
        layer_outputs[0] = inputs[s];

        // Iterate over each layer to feedforward through the network
        for (size_t l = 1; l < n_layers; l++) {
            // Get some references to the bias list, weight matrix and outputs of the previous and this layer
            double* bias = biases[l - 1];
            double* weight = weights[l - 1];
            double* prev_output = layer_outputs[l - 1];
            double* output = layer_outputs[l];

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
        double* output = layer_outputs[n_layers - 1];
        for (size_t n = 0; n < last_nodes; n++) {
            double output_val = output[n];
            prev_deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
        }

        // Do the output layer: compute the bias & weight updates

        // Add all deltas as delta_biases for this layer
        for (size_t n = 0; n < last_nodes; n++) {
            last_delta_bias[s * last_nodes + n] = prev_deltas[n];
        }
        // Same for all the weights, except we compute the delta_weights first
        double* last_prev_output = layer_outputs[n_layers - 2];
        for (size_t prev_n = 0; prev_n < last_prev_nodes; prev_n++) {
            for (size_t n = 0; n < last_nodes; n++) {
                last_delta_weight[s * last_prev_nodes * last_nodes + prev_n * last_nodes + n] = last_prev_output[prev_n] * prev_deltas[n];
            }
        }

        // Then, the rest of the hidden layers
        for (size_t l = n_layers - 2; l > 0; l--) {
            size_t next_nodes = nodes_per_layer[l + 1];
            size_t this_nodes = nodes_per_layer[l];
            size_t prev_nodes = nodes_per_layer[l - 1];
            double* delta_bias = delta_biases[l - 1];
            double* delta_weight = delta_weights[l - 1];
            double* output = layer_outputs[l];
            double* prev_output = layer_outputs[l - 1];
            
            // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
            double* weight_next = weights[l];
            for (size_t n = 0; n < this_nodes; n++) {
                // Take the weighted sum of all connection of that node with this layer
                double error = 0;
                for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                    error += prev_deltas[next_n] * weight_next[n * next_nodes + next_n];
                }

                // Multiply the error with the derivative of the activation function to find the result
                double output_val = output[n];
                deltas[n] = error * output_val * (1 - output_val);
            }

            // Add all deltas as delta_biases for this layer
            for (size_t n = 0; n < this_nodes; n++) {
                delta_bias[s * this_nodes + n] = deltas[n];
            }
            // Same for all the weights, except we compute the delta_weights first
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_weight[s * prev_nodes * this_nodes + prev_n * this_nodes + n] = prev_output[prev_n] * deltas[n];
                }
            }

            // Swap prev_deltas and deltas
            double* temp = deltas;
            deltas = prev_deltas;
            prev_deltas = temp;
        }
    }

    // Copy the delta weights 'n' stuff
    for (size_t l = 0; l < n_weights; l++) {
        memcpy(delta_biases_cpu[l], delta_biases[l], sizeof(double) * n_samples * nodes_per_layer[l + 1]);
        memcpy(delta_weights_cpu[l], delta_weights[l], sizeof(double) * n_samples * nodes_per_layer[l] * nodes_per_layer[l + 1]);
    }

    // Free the delta biases / weights
    for(size_t l = 0; l < n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (size_t l = 1; l < n_layers; l++) {
        free(layer_outputs[l]);
    }

    // Cleanup the deltas
    free(deltas);
}

void nn_full(neural_net* nn, double** biases_cpu, double** weights_cpu, size_t n_samples, double** inputs, double** expected, double learning_rate) {
    // Also obtain links to all biases / matrices
    double** biases = nn->biases;
    double** weights = nn->weights;

    // Make some shortcuts for the number-of-nodes information
    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;
    
    // Initialize the temporary delta memory to the correct size
    double* deltas = malloc(sizeof(double) * max(n_layers, nodes_per_layer));
    double* prev_deltas = malloc(sizeof(double) * max(n_layers, nodes_per_layer));

    // Create a list that is used to store intermediate outputs. The first input layer (=first column)
    //   is linked and not copied to the input data
    double* layer_outputs[n_layers];
    // Allocate arrays for all layers except the first layer, as that will
    //   be linked to the appropriate input layer
    for (size_t l = 1; l < n_layers; l++) {
        layer_outputs[l] = malloc(sizeof(double) * nodes_per_layer[l]);
    }

    // Create the delta_biases and delta_weights arrays / matrices
    double* delta_biases[nn->n_weights];
    double* delta_weights[nn->n_weights];
    for(size_t l = 0; l < nn->n_weights; l++) {
        delta_biases[l] = malloc(sizeof(double) * n_samples * nodes_per_layer[l + 1]);
        delta_weights[l] = malloc(sizeof(double) * n_samples * nodes_per_layer[l] * nodes_per_layer[l + 1]);
    }

    // Perform the training for n_iterations (always)
    size_t last_nodes = nodes_per_layer[n_layers - 1];
    size_t last_prev_nodes = nodes_per_layer[n_layers - 2];
    double* last_delta_bias = delta_biases[n_layers - 2];
    double* last_delta_weight = delta_weights[n_layers - 2];
    for (size_t s = 0; s < n_samples; s++) {
        /***** FORWARD PASS *****/

        // Link the first output to the input
        layer_outputs[0] = inputs[s];

        // Iterate over each layer to feedforward through the network
        for (size_t l = 1; l < n_layers; l++) {
            // Get some references to the bias list, weight matrix and outputs of the previous and this layer
            double* bias = biases[l - 1];
            double* weight = weights[l - 1];
            double* prev_output = layer_outputs[l - 1];
            double* output = layer_outputs[l];

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
        double* output = layer_outputs[n_layers - 1];
        for (size_t n = 0; n < last_nodes; n++) {
            double output_val = output[n];
            prev_deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
        }

        // Do the output layer: compute the bias & weight updates

        // Add all deltas as delta_biases for this layer
        for (size_t n = 0; n < last_nodes; n++) {
            last_delta_bias[s * last_nodes + n] = prev_deltas[n];
        }
        // Same for all the weights, except we compute the delta_weights first
        double* last_prev_output = layer_outputs[n_layers - 2];
        for (size_t prev_n = 0; prev_n < last_prev_nodes; prev_n++) {
            for (size_t n = 0; n < last_nodes; n++) {
                last_delta_weight[s * last_prev_nodes * last_nodes + prev_n * last_nodes + n] = last_prev_output[prev_n] * prev_deltas[n];
            }
        }

        // Then, the rest of the hidden layers
        for (size_t l = n_layers - 2; l > 0; l--) {
            size_t next_nodes = nodes_per_layer[l + 1];
            size_t this_nodes = nodes_per_layer[l];
            size_t prev_nodes = nodes_per_layer[l - 1];
            double* delta_bias = delta_biases[l - 1];
            double* delta_weight = delta_weights[l - 1];
            double* output = layer_outputs[l];
            double* prev_output = layer_outputs[l - 1];
            
            // Loop through all nodes in this layer to compute their deltas by summing all deltas of the next layer in a weighted fashion
            double* weight_next = weights[l];
            for (size_t n = 0; n < this_nodes; n++) {
                // Take the weighted sum of all connection of that node with this layer
                double error = 0;
                for (size_t next_n = 0; next_n < next_nodes; next_n++) {
                    error += prev_deltas[next_n] * weight_next[n * next_nodes + next_n];
                }

                // Multiply the error with the derivative of the activation function to find the result
                double output_val = output[n];
                deltas[n] = error * output_val * (1 - output_val);
            }

            // Add all deltas as delta_biases for this layer
            for (size_t n = 0; n < this_nodes; n++) {
                delta_bias[s * this_nodes + n] = deltas[n];
            }
            // Same for all the weights, except we compute the delta_weights first
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                for (size_t n = 0; n < this_nodes; n++) {
                    delta_weight[s * prev_nodes * this_nodes + prev_n * this_nodes + n] = prev_output[prev_n] * deltas[n];
                }
            }

            // Swap prev_deltas and deltas
            double* temp = deltas;
            deltas = prev_deltas;
            prev_deltas = temp;
        }
    }

    // Sum all the delta stuff to their respective stuff
    for (size_t l = 0; l < nn->n_weights; l++) {
        double* bias = biases[l];
        double* delta_bias = delta_biases[l];
        double* weight = weights[l];
        double* delta_weight = delta_weights[l];

        size_t this_nodes = nodes_per_layer[l + 1];
        size_t prev_nodes = nodes_per_layer[l];

        // Update the biases & reset delta_biases
        for (size_t s = 0; s < n_samples; s++) {
            for (size_t n = 0; n < this_nodes; n++) {
                bias[n] += delta_bias[s * this_nodes + n] * learning_rate;
            }

            // Update the weights & reset delta_weights
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                for (size_t n = 0; n < this_nodes; n++) {
                    weight[prev_n * this_nodes + n] += delta_weight[s * prev_nodes * this_nodes + prev_n * this_nodes + n] * learning_rate;
                }
            }
        }
    }

    // Copy the delta weights 'n' stuff
    for (size_t l = 0; l < n_weights; l++) {
        memcpy(biases_cpu[l], delta_biases[l], sizeof(double) * nodes_per_layer[l + 1]);
        memcpy(weights_cpu[l], delta_weights[l], sizeof(double) * nodes_per_layer[l] * nodes_per_layer[l + 1]);
    }

    // Free the delta biases / weights
    for(size_t l = 0; l < n_layers - 1; l++) {
        free(delta_biases[l]);
        free(delta_weights[l]);
    }

    // Free the layer_outputs (skip the first, as these merely link the input rather than copy 'em)
    for (size_t l = 1; l < n_layers; l++) {
        free(layer_outputs[l]);
    }

    // Cleanup the deltas
    free(deltas);
}


int main() {
    // Generate a random testset
    size_t n_samples = 10000;
    double** dataset;
    double** classes;
    generate_random(&dataset, &classes, n_samples, 64, -3, 3, 10);

    // Create an NN
    size_t hidden_layers[1] = {20};
    neural_net* nn = create_nn(64, 1, hidden_layers, 10);

    // First, obtain the result for the normal forward pass
    size_t last_nodes = nn->nodes_per_layer[nn->n_layers - 1];
    double outputs_normal[n_samples * last_nodes];
    nn_forward(nn, n_samples, outputs_normal, dataset);

    // Then for the cuda pass
    double outputs_cuda[n_samples * last_nodes];
    nn_forward_cuda(nn, n_samples, outputs_cuda, dataset);

    // Compare the two
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t n = 0; n < last_nodes; n++) {
            if (fabs(outputs_normal[s * last_nodes + n] - outputs_cuda[s * last_nodes + n]) >= 0.00000000000001) {
                fprintf(stderr, "ERROR: Elements do not match @ (%lu, %lu): %f is not %f\n",
                        s, n, outputs_normal[s * last_nodes + n], outputs_cuda[s * last_nodes + n]);
                clean(n_samples, dataset);
                clean(n_samples, classes);
                destroy_nn(nn);
                return EXIT_FAILURE;
            }
        }
    }

    printf("Nice! Forward pass seems to work!\n");

    printf("Let's move to the backward pass...\n");


    // Run the normal kernel thing
    double* delta_bias_normal[nn->n_weights];
    double* delta_weight_normal[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        delta_bias_normal[l] = malloc(sizeof(double) * n_samples * nn->nodes_per_layer[l + 1]);
        delta_weight_normal[l] = malloc(sizeof(double) * n_samples * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1]);
    }

    nn_backward(nn, delta_bias_normal, delta_weight_normal, n_samples, dataset, classes);

    // Run the CUDA kernel thing
    double* delta_bias_cuda[nn->n_weights];
    double* delta_weight_cuda[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        delta_bias_cuda[l] = malloc(sizeof(double) * n_samples * nn->nodes_per_layer[l + 1]);
        delta_weight_cuda[l] = malloc(sizeof(double) * n_samples * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1]);
    }
    nn_backward_cuda(nn, delta_bias_cuda, delta_weight_cuda, n_samples, dataset, classes);

    // Compare them
    for (size_t l = 0; l < nn->n_weights; l++) {
        double* delta_bias_normal_l = delta_bias_normal[l];
        double* delta_bias_cuda_l = delta_bias_cuda[l];
        double* delta_weight_normal_l = delta_weight_normal[l];
        double* delta_weight_cuda_l = delta_weight_cuda[l];
        for (size_t s = 0; s < n_samples; s++) {
            for (size_t n = 0; n < nn->nodes_per_layer[l + 1]; n++) {
                size_t index_b = s * nn->nodes_per_layer[l + 1] + n;
                if (fabs(delta_bias_normal_l[index_b] - delta_bias_cuda_l[index_b]) >= 0.00000001) {
                    fprintf(stderr, "ERROR: Delta biases: Elements do not match @ (l=%lu, %lu, %lu): %f is not %f\n",
                            l, s, n, delta_bias_normal_l[index_b], delta_bias_cuda_l[index_b]);
                    for (size_t l2 = 0; l2 < nn->n_weights; l2++) {
                        free(delta_bias_normal[l2]);
                        free(delta_bias_cuda[l2]);
                        free(delta_weight_normal[l2]);
                        free(delta_weight_cuda[l2]);
                    }
                    clean(n_samples, dataset);
                    clean(n_samples, classes);
                    destroy_nn(nn);
                    return EXIT_FAILURE;
                }
                for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l]; prev_n++) {
                    size_t index = s * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1] + prev_n * nn->nodes_per_layer[l + 1] + n;
                    if (fabs(delta_weight_normal_l[index] - delta_weight_cuda_l[index]) >= 0.00000001) {
                        fprintf(stderr, "ERROR: Delta weights: Elements do not match @ (l=%lu, %lu, %lu, %lu): %f is not %f\n",
                                l, s, prev_n, n, delta_weight_normal_l[index], delta_weight_cuda_l[index]);
                        for (size_t l2 = 0; l2 < nn->n_weights; l2++) {
                            free(delta_bias_normal[l2]);
                            free(delta_bias_cuda[l2]);
                            free(delta_weight_normal[l2]);
                            free(delta_weight_cuda[l2]);
                        }
                        clean(n_samples, dataset);
                        clean(n_samples, classes);
                        destroy_nn(nn);
                        return EXIT_FAILURE;
                    }
                }
            }
        }
    }

    printf("Sweet! Backward pass work as well!\n");

    for (size_t l = 0; l < nn->n_weights; l++) {
        free(delta_bias_normal[l]);
        free(delta_bias_cuda[l]);
        free(delta_weight_normal[l]);
        free(delta_weight_cuda[l]);
    }

    printf("Onwards we go, onto the final update step...\n");

    // Run the normal kernel thing
    double* biases_normal[nn->n_weights];
    double* weights_normal[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        biases_normal[l] = malloc(sizeof(double) * nn->nodes_per_layer[l + 1]);
        weights_normal[l] = malloc(sizeof(double) * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1]);
    }

    nn_full(nn, biases_normal, weights_normal, n_samples, dataset, classes, 0.5);

    // Run the CUDA kernel thing
    double* biases_cuda[nn->n_weights];
    double* weights_cuda[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        biases_cuda[l] = malloc(sizeof(double) * nn->nodes_per_layer[l + 1]);
        weights_cuda[l] = malloc(sizeof(double) * nn->nodes_per_layer[l] * nn->nodes_per_layer[l + 1]);
    }

    nn_full_cuda(nn, biases_normal, weights_normal, n_samples, dataset, classes, 0.5);

    printf("Check 1\n");

    // Compare them
    for (size_t l = 0; l < nn->n_weights; l++) {
        double* delta_bias_normal_l = delta_bias_normal[l];
        double* delta_bias_cuda_l = delta_bias_cuda[l];
        double* delta_weight_normal_l = delta_weight_normal[l];
        double* delta_weight_cuda_l = delta_weight_cuda[l];
        for (size_t n = 0; n < nn->nodes_per_layer[l + 1]; n++) {
            size_t index_b = n;
            if (fabs(delta_bias_normal_l[index_b] - delta_bias_cuda_l[index_b]) >= 0.00000001) {
                fprintf(stderr, "ERROR: Biases: Elements do not match @ (l=%lu, %lu): %f is not %f\n",
                        l, n, delta_bias_normal_l[index_b], delta_bias_cuda_l[index_b]);
                for (size_t l2 = 0; l2 < nn->n_weights; l2++) {
                    free(biases_normal[l2]);
                    free(biases_cuda[l2]);
                    free(weights_normal[l2]);
                    free(weights_cuda[l2]); 
                }
                clean(n_samples, dataset);
                clean(n_samples, classes);
                destroy_nn(nn);
                return EXIT_FAILURE;
            }
            for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l]; prev_n++) {
                size_t index = prev_n * nn->nodes_per_layer[l + 1] + n;
                if (fabs(delta_weight_normal_l[index] - delta_weight_cuda_l[index]) >= 0.00000001) {
                    fprintf(stderr, "ERROR: Weights: Elements do not match @ (l=%lu, %lu, %lu): %f is not %f\n",
                            l,  prev_n, n, delta_weight_normal_l[index], delta_weight_cuda_l[index]);
                    for (size_t l2 = 0; l2 < nn->n_weights; l2++) {
                        free(biases_normal[l2]);
                        free(biases_cuda[l2]);
                        free(weights_normal[l2]);
                        free(weights_cuda[l2]);  
                    }
                    clean(n_samples, dataset);
                    clean(n_samples, classes);
                    destroy_nn(nn);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    for (size_t l = 0; l < nn->n_weights; l++) {
        free(biases_normal[l]);
        free(biases_cuda[l]);
        free(weights_normal[l]);
        free(weights_cuda[l]);
    }
    clean(n_samples, dataset);
    clean(n_samples, classes);
    destroy_nn(nn);
    return EXIT_SUCCESS;
}