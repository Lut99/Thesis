/* DEBUG CUDA.c
 *   by Anonymous
 *
 * Created:
 *   6/11/2020, 9:31:55 PM
 * Last edited:
 *   6/12/2020, 4:50:46 PM
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
extern void nn_backward_output_cuda(neural_net* nn, double* delta_biases_cpu, double* delta_weights_cpu, size_t n_samples, double** inputs, double** expected);


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


void nn_backward_output(neural_net* nn, double* delta_biases_cpu, double* delta_weights_cpu, size_t n_samples, double** inputs, double** expected) {
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
            deltas[n] = (sample_expected[n] - output_val) * output_val * (1 - output_val);
        }

        // Do the output layer: compute the bias & weight updates

        // Add all deltas as delta_biases for this layer
        for (size_t n = 0; n < last_nodes; n++) {
            last_delta_bias[s * last_nodes + n] = deltas[n];
        }
        // Same for all the weights, except we compute the delta_weights first
        double* last_prev_output = layer_outputs[n_layers - 2];
        for (size_t prev_n = 0; prev_n < last_prev_nodes; prev_n++) {
            for (size_t n = 0; n < last_nodes; n++) {
                last_delta_weight[s * last_prev_nodes * last_nodes + prev_n * last_nodes + n] = last_prev_output[prev_n] * deltas[n];
            }
        }
    }

    // Copy the delta weights 'n' stuff
    memcpy(delta_biases_cpu, last_delta_bias, sizeof(double) * n_samples * nodes_per_layer[n_layers - 1]);
    memcpy(delta_weights_cpu, last_delta_weight, sizeof(double) * n_samples * nodes_per_layer[n_layers - 2] * nodes_per_layer[n_layers - 1]);

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

    printf("Let's move to the backward pass - outer layer...\n");


    // Run the normal kernel thing
    size_t prev_nodes = nn->nodes_per_layer[nn->n_layers - 2];
    double delta_bias_normal[n_samples * last_nodes];
    double* delta_weight_normal = malloc(sizeof(double) * n_samples * prev_nodes * last_nodes);

    nn_backward_output(nn, delta_bias_normal, delta_weight_normal, n_samples, dataset, classes);

    // Run the CUDA kernel thing
    double delta_bias_cuda[n_samples * last_nodes];
    double* delta_weight_cuda = malloc(sizeof(double) * n_samples * prev_nodes * last_nodes);
    nn_backward_output_cuda(nn, delta_bias_cuda, delta_weight_cuda, n_samples, dataset, classes);

    // Compare them
    for (size_t s = 0; s < n_samples; s++) {
        for (size_t n = 0; n < last_nodes; n++) {
            if (fabs(delta_bias_normal[s * last_nodes + n] - delta_bias_cuda[s * last_nodes + n]) >= 0.00000000000001) {
                fprintf(stderr, "ERROR: Delta biases: Elements do not match @ (%lu, %lu): %f is not %f\n",
                        s, n, delta_bias_normal[s * last_nodes + n], delta_bias_cuda[s * last_nodes + n]);
                free(delta_weight_normal);
                free(delta_weight_cuda);
                clean(n_samples, dataset);
                clean(n_samples, classes);
                destroy_nn(nn);
                return EXIT_FAILURE;
            }
            for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
                double normal = delta_weight_normal[s * prev_nodes * last_nodes + prev_n * last_nodes + n];
                double cuda = delta_weight_cuda[s * prev_nodes * last_nodes + prev_n * last_nodes + n];
                if (fabs(normal - cuda) >= 0.00000000000001) {
                    fprintf(stderr, "ERROR: Delta weights: Elements do not match @ (%lu, %lu, %lu): %f is not %f\n",
                            s, prev_n, n, normal, cuda);
                    free(delta_weight_normal);
                    free(delta_weight_cuda);
                    clean(n_samples, dataset);
                    clean(n_samples, classes);
                    destroy_nn(nn);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    printf("Sweet! Backward pass - outputs work as well!\n");

    free(delta_weight_normal);
    free(delta_weight_cuda);
    clean(n_samples, dataset);
    clean(n_samples, classes);
    destroy_nn(nn);
    return EXIT_SUCCESS;
}

