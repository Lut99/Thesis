/* NEURAL NETWORK CUDA GPU1.cu
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   5/24/2020, 11:07:35 PM
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
 *   maximum parallelism.
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



/***** CUDA KERNELS *****/

/* Kernel that computes the forward pass for a single layer. This version implements a sigmoid activation function.
 * Parameters:
 *   @param outputs: a 2D, pitched array which will store the output of this layer (columns) for every sample (rows)
 *   @param outputs_pitch: the pitch of the outputs array
 *   @param biases: a list of biases for this layer
 *   @param weights: a pitched matrix of weights for this layer to the next
 *   @param weights_pitch: the pitch for this weights matrix
 *   @param inputs: a 2D, pitched array with inputs from the previous layer (columns) for every sample (rows)
 *   @param inputs_pitch: the pitch of the inputs array
 *   @param prev_nodes: number of nodes in the layer before this one
 *   @param this_nodes: number of nodes in this layer
 *   @param n_samples: total number of samples to process
 */
__global__ void FwdPassKernel(double* outputs, size_t outputs_pitch,
                              double* biases,
                              double* weights, size_t weights_pitch,
                              double* inputs, size_t inputs_pitch,
                              size_t prev_nodes, size_t, this_nodes, size_t n_samples) {
    // Get the index of this particular thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int s = i / this_nodes;
    int n = i % this_nodes;

    // Only do work if still within range
    if (s < n_samples && n < this_nodes) {
        // Sum the weighted inputs for this node (64 first iteration of l, 20 for second iteration)
        double z = biases[n];
        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
            double input_val = *((double*) ((char*) inputs + s * inputs_pitch) + prev_n);
            double weight_val = *((double*) ((char*) weights + prev_n * weights_pitch) + n);
            z += input_val * weight_val;
        }

        // Run the activation function over this input and store it in the output (using sigmoid)
        double* output_ptr = (double*) ((char*) outputs + s * outputs_pitch) + n;
        *output_ptr = 1 / (1 + exp(-z));
    }
}

/* Kernel that computes the output layer-backward pass. This version implements Mean Square Error, and assumes
 *   a sigmoid activation function. Also note that the reduction of delta_weights should be done using a later
 *   kernel. Finally, note that the deltas are not explicitly returned, as they are equal to the delta_bias for this node.
 * Parameters:
 *   @param delta_biases: a 2D, pitched matrix containing the delta_biases computed for each node in the output layer and each sample.
 *   @param delta_biases_pitch: the pitch of the delta_biases matrix.
 *   @param delta_weights: a 3D, pitched tensor containing the weight updates for the last layer across all samples.
 *   @param delta_weights_pitch: the pitch for the delta_weights 3D-array.
 *   @param layer_inputs: a 2D, pitched matrix containing the inputs for the last layer.
 *   @param layer_inputs_pitch: pitch for the layer_inputs.
 *   @param layer_outputs: a 2D, pitched matrix containing the outputs for the last layer.
 *   @param layer_outputs_pitch: pitch for the layer_inputs.
 *   @param expected: a 2D, pitched matrix containing the expected outputs for the output layer.
 *   @param expected_pitch: the pitch of the expected matrix.
 *   @param prev_nodes: number of nodes in the layer before this one
 *   @param this_nodes: number of nodes in this layer
 *   @param n_samples: total number of samples to process
 */
 __global__ void BckPassOutputKernel(double* delta_biases, size_t delta_biases_pitch,
                                     double* delta_weights, size_t delta_weights_pitch,
                                     double* layer_inputs, size_t layer_inputs_pitch,
                                     double* layer_outputs, size_t layer_outputs_pitch,
                                     double* expected, size_t expected_pitch,
                                     size_t prev_nodes, size_t this_nodes, size_t n_samples) {
     // Get the index of this particular thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int s = i / this_nodes;
    int n = i % this_nodes;

    // Only do work if still within range
    if (s < n_samples && n < this_nodes) {
        // First, compute the delta for this specific node and sample pair
        double output_val = *((double*) ((char*) layer_outputs + s * layer_outputs_pitch) + n);
        double expected_val = *((double*) ((char*) expected + s * expected_pitch) + n);
        double delta = (expected_val - output_val) * output_val * (1 - output_val);

        // Compute the change in biases (aka, store the deltas for the next node)
        double* delta_biases_ptr = (double*) ((char*) delta_biases + s * delta_biases_pitch) + n;
        *delta_biases_ptr = delta;

        // Compute the weight updates
        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
            double* delta_weight_ptr = (double*) ((char*) delta_weights + s * prev_n * delta_weights_pitch) + n;
            double input_val = *((double*) ((char*) layer_inputs + s * layer_inputs_pitch) + n);
            *delta_weight_ptr = input_val * delta;
        }
    }
 }

 /* Kernel that computes a hidden layer-backward pass. This version implements Mean Square Error, and assumes
 *   a sigmoid activation function. Also note that the reduction of delta_weights should be done using a later
 *   kernel. Finally, note that the deltas are not explicitly returned, as they are equal to the delta_bias for this node.
 * Parameters:
 *   @param delta_biases: a 2D, pitched matrix containing the delta_biases computed for each node in the output layer and each sample.
 *   @param delta_biases_pitch: the pitch of the delta_biases matrix.
 *   @param delta_weights: a 3D, pitched tensor containing the weight updates for the last layer across all samples.
 *   @param delta_weights_pitch: the pitch for the delta_weights 3D-array.
 *   @param deltas: a 2D, pitched matrix containing the deltas computed for each node / sample pair in the previous layer.
 *   @param deltas_pitch: the pitch of the deltas matrix.
 *   @param weights: a 2D, pitched matrix containing the weights from this layer to the next one.
 *   @param weights_pitch: pitch for the weights array.
 *   @param layer_inputs: a 2D, pitched matrix containing the inputs for the last layer.
 *   @param layer_inputs_pitch: pitch for the layer_inputs.
 *   @param layer_outputs: a 2D, pitched matrix containing the outputs for the last layer.
 *   @param layer_outputs_pitch: pitch for the layer_inputs.
 *   @param prev_nodes: number of nodes in the layer before this one
 *   @param this_nodes: number of nodes in this layer
 *   @param next_nodes: number of nodes in the layer after this one
 *   @param n_samples: total number of samples to process
 */
 __global__ void BckPassHiddenKernel(double* delta_biases, size_t delta_biases_pitch,
                                     double* delta_weights, size_t delta_weights_pitch,
                                     double* deltas, size_t deltas_pitch,
                                     double* weights, size_t weights_pitch,
                                     double* layer_inputs, size_t layer_inputs_pitch,
                                     double* layer_outputs, size_t layer_outputs_pitch,
                                     size_t prev_nodes, size_t this_nodes, size_t next_nodes,
                                     size_t n_samples) {
    // Get the index of this particular thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int s = i / this_nodes;
    int n = i % this_nodes;

    // Only do work if still within range
    if (s < n_samples && n < this_nodes) {
        // Take the weighted sum of all connection of that node with this layer (10 iterations)
        double error = 0;
        for (size_t next_n = 0; next_n < next_nodes; next_n++) {
            double deltas_val = *((double*) ((char*) deltas + s * deltas_pitch) + n);
            double weight_val = *((double*) ((char*) weights + n * weights_pitch) + next_n);
            error += deltas_val * weight_val;
        }

        // Multiply the error with the derivative of the activation function to find the result
        double output_val = *((double*) ((char*) layer_outputs + s * layer_outputs_pitch) + n);
        double delta = error * output_val * (1 - output_val);

        // Compute the change in biases (aka, store the deltas for the next node)
        double* delta_biases_ptr = (double*) ((char*) delta_biases + s * delta_biases_pitch) + n;
        *delta_biases_ptr = delta;

        // Compute the weight updates
        for (size_t prev_n = 0; prev_n < prev_nodes; prev_n++) {
            double* delta_weight_ptr = (double*) ((char*) delta_weights + s * prev_n * delta_weights_pitch) + n;
            double input_val = *((double*) ((char*) layer_inputs + s * layer_inputs_pitch) + n);
            *delta_weight_ptr = input_val * delta;
        }
    }
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

// Cuda memory help from https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
void nn_train(neural_net* nn, size_t n_samples, array* inputs[n_samples], array* expected[n_samples], double learning_rate, size_t n_iterations, double (*act)(double), double (*dydx_act)(double)) {
    // Create a list of deltas, one for each thread
    array* deltas[n_samples];
    for (size_t s = 0; s < n_samples; s++) {
        deltas[s] = create_empty_array(max(nn->n_layers, nn->nodes_per_layer));
    }

    /***** GPU MEMORY INITIALISATION *****/

    
    /* BIASES */
    
    // Copy the biases of the network to the GPU. Since the lists have different lengths, it is important to not make it a 2D-array.
    double* biases[nn->n_weights];
    for (size_t l = 0; l < nn->n_layers - 1; l++) {
        cudaMalloc((void**) (biases + l), sizeof(double) * nn->nodes_per_layer[l + 1]);
        cudaMemcpy((void*) biases[l], nn->biases[l]->d, sizeof(double) * nn->nodes_per_layer[l + 1], cudaMemcpyHostToDevice);
    }


    /* WEIGHTS */

    // Also copy the weights in practically the same way, except that now the inner values are pitched arrays.
    //   We store the pitches in a similarly lengthy weights_pitches list.
    double* weights[nn->n_weights];
    size_t weights_pitches[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        size_t w = sizeof(double) * nn->nodes_per_layer[l];
        size_t h = nn->nodes_per_layer[l + 1];
        cudaMallocPitch((void**) (weights + l), weights_pitches + l, w, h);
        cudaMemcpy2D((void*) weights[l], weights_pitches[l], (void*) nn->weights[l]->data, w, w, h, cudaMemcpyHostToDevice);
    }

    
    /* LAYER OUTPUTS */
    
    // The layer outputs is for every layer a matrix of samples by nodes_for_that_layer elements.
    //   Just as with the weights do we use pitches, and create a list to store those. Note that
    //   only the inputs need to be copied, as the rest serves as a buffer.
    double* layer_outputs[nn->n_layers];
    size_t layer_outputs_pitches[nn->n_layers];
    for (size_t l = 0; l < nn->n_layers; l++) {
        size_t w = sizeof(double) * nn->nodes_per_layer[l];
        size_t h = n_samples;
        cudaMallocPitch((void**) layer_outputs + l, layer_outputs_pitches + l, w, h);
    }
    // Copy all sample inputs. Because of the unhappy alginment of inputs, we have to do this manually row-by-row.
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) layer_outputs[0] + s * layer_outputs_pitches[0]);
        cudaMemcpy((void*) ptr, (void*) inputs[s]->d, sizeof(double) * inputs[s]->size, cudaMemcpyHostToDevice);
    }


    // TODO: (V) Restructure the delta biases into a list of matrices (samples x nodes)
    // TODO: (V) Restructure the delta weights into a list of tensors (samples x prev_nodes x nodes)
    // TODO: (V) Update the delta & weight resetting
    // TODO: (V) Restructure the backward pass to leave summation for later so one kernel can do an entire layer
    // TODO: (V) Make a kernel for the output layer and then each hidden layer
    // TODO: Reduce all delta biases and weights using the recursion / sum-two-or-more-at-a-time kernel trick
    //       (https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

    
    /* DELTA BIASES */

    // We also have to declare the delta biases. Simultaneously, we allocate a host-side, zero-filled counterpart,
    //   so that resetting the deltas is nothing more than copying 0 over the old values.
    double* delta_biases[nn->n_weights];
    size_t delta_biases_pitches[nn->n_weights];
    double* delta_biases_zero[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        // First, we allocate all memory
        size_t this_nodes = nn->nodes_per_layer[l + 1];
        cudaMallocPitch((void**) (delta_biases + l), delta_biases_pitches + l, sizeof(double) * this_nodes, s_samples);
        delta_biases_zero[l] = (double*) malloc(sizeof(double) * this_nodes * s_samples);

        // Set the host-side array to 0
        for (size_t s = 0; s < this_nodes; s++) {
            for (size_t n = 0; n < this_nodes; n++) {
                delta_biases_zero[l][s][n] = 0;
            }
        }
    }


    /* DELTA WEIGHTS */
    
    // Declare the delta weights. Note that we pitch a 3D-array here. Not pretty, but better than
    //   the endless structs 3D require from us. Just as with the delta biases, create a host-side
    //   zero-filled one that is used for resetting.
    double* delta_weights[nn->n_weights];
    size_t delta_weights_pitches[nn->n_weights];
    double* delta_weights_zero[nn->n_weights];
    for (size_t l = 0; l < nn->n_weights; l++) {
        // Prepare CUDA structs for allocation
        size_t w = nn->nodes_per_layer[l + 1];
        size_t h = nn->nodes_per_layer[l];
        size_t d = n_samples;

        // First, we allocate all memory
        cudaMallocPitch((void**) (delta_weights + l), delta_weights_pitches + l, sizeof(double) * w, h * d);
        delta_biases_zero[l] = (double*) malloc(sizeof(double) * w * h * d);

        // Set the host-side array to 0
        for (size_t z = 0; z < d; z++) {
            for (size_t y = 0; y < h; y++) {
                for (size_t x = 0; x < w; x++) {
                    delta_biases_zero[l][z * (w * h) + y * w + x] = 0;
                }
            }
        }
    }


    /* EXPECTED */
    
    // The expected values are formatted as a 2D, n_samples x nodes_in_output_layer pitched matrix.
    double* expected_gpu;
    size_t expected_gpu_pitch;
    cudaMallocPitch((void**) &expecteds, &expecteds_pitch, sizeof(double) * nn->nodes_per_layer[nn->n_layers - 1], n_samples);
    // Copy all expected values for each sample, which we have to do row-by-row due to unfortunate formatting of expected
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) expected_gpu + s * expected_gpu_pitch);
        cudaMemcpy((void*) ptr, (void*) expected[s]->d, sizeof(double) * expected[s]->size, cudaMemcpyHostToDevice);
    }


    /***** ITERATIONS *****/

    // Choose block size
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Perform the training for n_iterations (always) (20,000 iterations, non-parallelizable)
    for (size_t i = 0; i < n_iterations; i++) {
        /***** FORWARD PASS *****/

        // Loop through all layers forwardly so that we can compute errors later (2 iterations, non-parallelizable)
        for (size_t l = 1; l < nn->n_layers; l++) {
            // Call upon the actiation kernel (should do 1797 x 20 elements for first iteration, 1797 x 10 elements for second)
            FwdPassKernel<<<blocksPerGrid, threadsPerBlock>>>(
                layer_outputs[l], layer_outputs_pitches[l],
                biases[l - 1],
                weights[l - 1], weights_pitches[l - 1],
                layer_outputs[l - 1], layer_outputs_pitches[l - 1],
                nn->nodes_per_layer[l - 1], nn->nodes_per_layer[l], n_samples
            );
        }


        /***** BACKWARD PASS *****/
        
        // Reset the delta biases and delta weights by copying the host-side, 0-filled ones over them
        for (size_t l = 0; l < nn->n_weights; l++) {
            size_t w = sizeof(double) * nn->nodes_per_layer[l + 1];
            cudaMemcpy2D((void*) delta_biases[l], delta_biases_pitches[l], (void*) delta_biases_zero[l], w, w, n_samples, cudaMemcpyHostToDevice);
            cudaMemcpy2D((void*) delta_weights[l], delta_weights_pitches[l], (void*) delta_weights_zero[l], w, w, nn->nodes_per_layer[l] * n_samples, cudaMemcpyHostToDevice);
        }

        // Then, compute the error at the output laye (1797 x 10 iterations)
        size_t l = nn->n_layers - 1;
        BckPassOutputKernel<<<blockPerGrid, threadsPerBlock>>>(
            delta_biases[l - 1], delta_biases_pitches[l - 1],
            delta_weights[l - 1], delta_weights_pitches[l - 1],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            layer_outputs[l], layer_outputs_pitches[l],
            expected_gpu, expected_gpu_pitch,
            nn->nodes_per_layer[nn->n_layers - 2], nn->nodes_per_layer[nn->n_layers - 1], n_samples
        );

        // Loop through all hidden layers in the other direction so that we can compute their weight updates (1 iteration, non-parallelizable)
        for (l = nn->n_layers - 2; l > 0; l--) {
            BckPassHiddenKernel<<<blockPerGrid, threadsPerBlock>>>(
                delta_biases[l - 1], delta_biases_pitches[l - 1],
                delta_weights[l - 1], delta_weights_pitches[l - 1],
                delta_biases[l], delta_biases_pitches[l],
                weights[l - 1], weights_pitches[l - 1],
                layer_outputs[l - 1], layer_outputs_pitches[l - 1],
                layer_outputs[l], layer_outputs_pitches[l],
                nn->nodes_per_layer[l - 1], nn->nodes_per_layer[l], nn->nodes_per_layer[l + 1],
                n_samples
            );
        }


        /***** WEIGHT UPDATES *****/

        // Actually update the weights, and reset the delta updates to 0 for next iteration (2 iterations)
        for (size_t l = 1; l < nn->n_layers; l++) {
            // 20 for first iteration of l, 10 for second iteration of l
            for (size_t n = 0; n < nn->nodes_per_layer[l]; n++) {
                nn->biases[l - 1]->d[n] += delta_biases[l - 1]->d[n] * learning_rate;
                // 64 for first iteration of l, 20 for second iteration of l
                for (size_t prev_n = 0; prev_n < nn->nodes_per_layer[l - 1]; prev_n++) {
                    INDEX(nn->weights[l - 1], prev_n, n) += INDEX(delta_weights[l - 1], prev_n, n) * learning_rate;
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
    // Destroy the temporary deltas
    for (size_t s = 0; s < n_samples; s++) {
        destroy_array(deltas[s]);
    }
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
