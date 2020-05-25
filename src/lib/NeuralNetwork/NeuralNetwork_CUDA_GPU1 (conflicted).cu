/* NEURAL NETWORK CUDA GPU 1.cu
 *   by Lut99
 *
 * Created:
 *   5/25/2020, 9:30:27 PM
 * Last edited:
 *   5/25/2020, 9:42:54 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This version implements the CUDA parts of the
 *   NeuralNetwork_CUDA_GPU1.c variation on the NeuralNetwork
 *   implementation. Specifically, it provides code for the NeuralNetwork's
 *   train function, and the kernels used there.
**/

#include "NeuralNetwork.h"


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



/***** NEURAL NETWORK OPERATIONS *****/

// Cuda memory help from https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
void nn_train(neural_net* nn, size_t n_samples, array* inputs[n_samples], array* expected[n_samples], double learning_rate, size_t n_iterations, double (*act)(double), double (*dydx_act)(double)) {
    
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


    /***** CLEANUP *****/

    /* BIASES & WEIGHTS */

    // Simply loop through all layers (except the last one), and clean everything weight & bias related.
    for (size_t l = 0; l < nn->n_layers - 1; l++) {
        // Free the device-side stuff
        cudaFree(biases[l]);
        cudaFree(delta_biases[l]);
        cudaFree(weights[l]);
        cudaFree(delta_weights[l]);

        // But don't forget the two host-side arrays
        free(delta_weights_zero[l]);
        free(delta_biases_zero[l]);
    }

    /* WEIGHTS */

    
    /* LAYER OUTPUTS */

    
    /* DELTA BIASES */


    /* DELTA WEIGHTS */


    /* EXPECTED */
}
