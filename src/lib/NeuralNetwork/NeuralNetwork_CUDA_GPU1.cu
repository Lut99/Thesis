/* NEURAL NETWORK CUDA GPU 1.cu
 *   by Lut99
 *
 * Created:
 *   5/25/2020, 9:30:27 PM
 * Last edited:
 *   6/24/2020, 8:35:22 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This version implements the CUDA parts of the
 *   NeuralNetwork_CUDA_GPU1.c variation on the NeuralNetwork
 *   implementation. Specifically, it provides code for the NeuralNetwork's
 *   train function, and the kernels used there.
**/

extern "C" {
#include "NeuralNetwork.h"
}
#include <sys/time.h>


/***** GLOBALS *****/

static unsigned int threads_per_block = 64;


/***** HELPER FUNCTIONS *****/

#define TIMEVAL_TO_MS(T_START, T_END) (((T_END.tv_sec - T_START.tv_sec) * 1000000 + (T_END.tv_usec - T_START.tv_usec)) / 1000000.0)


/***** IDEAS *****/

/* Could be worth to split single phases into multiple, so that the for-loop
 * areas could be updated even further.
 * (Expected time expense: medium, as it would mostly be splitting kernels)
 */

/* Also, I can try to copy relavant loop data to shared cache mem or something
 *   first, if feasable (look into this) to speed that up, hopefully by quite
 *   some. EDIT: Maybe not that useful as we re-use NO data (within one kernel)
 * (Expected time expense: larger)
 */


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
                               size_t prev_nodes, size_t this_nodes, size_t n_samples) {
    // Get the index of this particular thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int n = i % this_nodes;
    int s = i / this_nodes;

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

    int n = i % this_nodes;
    int s = i / this_nodes;

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
            double* delta_weight_ptr = (double*) ((char*) delta_weights + s * delta_weights_pitch * prev_nodes + prev_n * delta_weights_pitch) + n;
            double input_val = *((double*) ((char*) layer_inputs + s * layer_inputs_pitch) + prev_n);
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
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    size_t n = i % this_nodes;
    size_t s = i / this_nodes;

    // Only do work if still within range
    if (s < n_samples && n < this_nodes) {
        // Take the weighted sum of all connection of that node with this layer (10 iterations)
        double error = 0;
        for (size_t next_n = 0; next_n < next_nodes; next_n++) {

            double deltas_val = *((double*) ((char*) deltas + s * deltas_pitch) + next_n);
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
            double* delta_weight_ptr = (double*) ((char*) delta_weights + s * delta_weights_pitch * prev_nodes + prev_n * delta_weights_pitch) + n;
            double input_val = *((double*) ((char*) layer_inputs + s * layer_inputs_pitch) + prev_n);
            *delta_weight_ptr = input_val * delta;
        }
    }
}

/* This kernel will be used to accelerate bias updating. In particular, it will reduce the
 *   update biases for a particular layer for every sample, i.e., the first row will contain
 *   the summed updates for the node denoted by the column.
 * Parameters:
 *   @param delta_biases: a 2D, pitched list of whom we want to reduce one dimension
 *   @param delta_biases_pitch: pitch of the 2D array delta_biases
 *   @param this_nodes: the number of nodes on the current layer
 *   @param n_samples: the number of samples
 */
__global__ void BiasReduceKernel(double* delta_biases, size_t delta_biases_pitch,
                                 size_t this_nodes, size_t n_samples) {
    // Get the index of this particular thread
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // Decode the i
    size_t n = i % this_nodes;
    size_t s = i / this_nodes;

    // Only do work if still within range
    if (n < this_nodes && s < n_samples / 2) {
        // Simply sum this and the next element. Make sure to stay in bounds
        size_t half_H = ceil(n_samples / 2.0);
        double* delta_biases_ptr = (double*) ((char*) delta_biases + s * delta_biases_pitch) + n;
        double delta_biases_val = *((double*) ((char*) delta_biases + (s + half_H) * delta_biases_pitch) + n);
        *delta_biases_ptr += delta_biases_val;
    }
}

/* This kernel will be used to accelerate weight updating. In particular, it will reduce the
 *   update weights for a particular layer for every sample, i.e., the first matrix will contain
 *   the summed updates for the node denoted by the depth.
 * Parameters:
 *   @param delta_weights: a 3D, pitched list of whom we want to reduce one dimension
 *   @param delta_weights_pitch: the pitch of the delta tensor
 *   @param this_nodes: the number of nodes on this layer
 *   @param next_nodes: the number of nodes on the next layer
 *   @param n_samples: the number of samples
 */
 __global__ void WeightReduceKernel(double* delta_weights, size_t delta_weights_pitch,
                                    size_t this_nodes, size_t next_nodes,
                                    size_t n_samples) {
    // Get the index of this particular thread
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // Decode the i
    size_t x = i % next_nodes;
    size_t yz = i / next_nodes;
    size_t y = yz % this_nodes;
    size_t z = yz / this_nodes;

    // Only do work if still within range
    if (x < next_nodes && y < this_nodes && z < n_samples / 2) {
        // Simply sum this and the next element. Make sure to stay in bounds
        size_t half_H = ceil(n_samples / 2.0);
        double* delta_weights_ptr = (double*) ((char*) delta_weights + z * delta_weights_pitch * this_nodes + y * delta_weights_pitch) + x;
        double delta_weights_val = *((double*) ((char*) delta_weights + (z + half_H) * delta_weights_pitch * this_nodes + y * delta_weights_pitch) + x);
        *delta_weights_ptr += delta_weights_val;
    }
}

/* This kernel finally applies the bias updates to the biases. Normally this
 *   would be a waste, but since the data is already on the GPU and we would
 *   otherwise have to copy it, this kernel is here.
 *
 * Parameters:
 *   @param biases: the destination vector.
 *   @param delta_biases: the bias updates to add to the first vector.
 *   @param this_nodes: the number of nodes for this layer, aka, length (in number of elements) for both vectors.
 *   @param learning_rate: the learning rate by which to reduce the effect of the updates.
 */
 __global__ void BiasUpdateKernel(double* biases, double* delta_biases,
                                  size_t this_nodes, double learning_rate) {
     // Get the index of this particular thread
     size_t n = blockDim.x * blockIdx.x + threadIdx.x;

     // Only do work if still within bounds
     if (n < this_nodes) {
         // Just add lol
         biases[n] += learning_rate * delta_biases[n];
     }
 }

 /* This kernel finally applies the weight updates. Normally this would be a
 *   waste, but since the data is already on the GPU and we would otherwise
 *   have to copy it, this kernel is here.
 *
 * Parameters:
 *   @param weights: the destination matrix containing the existing weights
 *   @oaram weights_pitch: the pitch of the weight matrix
 *   @param delta_weights: the source matrix containing the updates
 *   @oaram delta_weights_pitch: the pitch of the delta_weights matrix
 *   @param prev_nodes: the number of nodes in the previous layer, aka, the number of rows for both matrices.
 *   @param this_nodes: the number of nodes in this layer, aka, the number of columns for both matrices.
 *   @param learning_rate: the learning rate by which to reduce the effect of the updates.
 */
 __global__ void WeightUpdateKernel(double* weights, size_t weights_pitch,
                                    double* delta_weights, size_t delta_weights_pitch,
                                    size_t prev_nodes, size_t this_nodes, double learning_rate) {
    // Get the index of this particular thread
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // Decode the i
    size_t n = i % this_nodes;
    size_t prev_n = i / this_nodes;

    // Only do work if still within bounds
    if (n < this_nodes && prev_n < prev_nodes) {
        // Just add lol
        double* weights_ptr = (double*) ((char*) weights + prev_n * weights_pitch) + n;
        double delta_weights_val = *((double*) ((char*) delta_weights + prev_n * delta_weights_pitch) + n);
        *weights_ptr += learning_rate * delta_weights_val;
    }
}



/***** NEURAL NETWORK OPERATIONS *****/

extern "C" void nn_forward_cuda(neural_net* nn, size_t n_samples, double* outputs, double** inputs) {
    // Shortcut to stuff
    size_t* nodes_per_layer = nn->nodes_per_layer;
    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    int threads_per_block = 32;
    int blocks_per_grid;

    // Create a CUDA-side spot to put the layer outputs in
    double* layer_outputs[n_layers];
    size_t layer_outputs_pitches[n_layers];
    for (size_t l = 0; l < n_layers; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l];
        size_t h = n_samples;
        cudaMallocPitch((void**) layer_outputs + l, layer_outputs_pitches + l, w, h);
    }
    // Copy all sample inputs. Because of the unhappy alginment of inputs, we have to do this manually row-by-row.
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) layer_outputs[0] + s * layer_outputs_pitches[0]);
        cudaMemcpy((void*) ptr, (void*) inputs[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }

    // Copy the list of biases
    double* biases[n_weights];
    for (size_t l = 0; l < n_layers - 1; l++) {
        cudaMalloc((void**) (biases + l), sizeof(double) * nodes_per_layer[l + 1]);
        cudaMemcpy((void*) biases[l], nn->biases[l], sizeof(double) * nodes_per_layer[l + 1], cudaMemcpyHostToDevice);
    }

    // Copy the list of weights
    double* weights[n_weights];
    size_t weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMallocPitch((void**) (weights + l), weights_pitches + l, w, h);
        cudaMemcpy2D((void*) weights[l], weights_pitches[l], (void*) nn->weights[l], w, w, h, cudaMemcpyHostToDevice);
    }

    for (size_t l = 1; l < n_layers; l++) {
        // Call upon the actiation kernel (should do 1797 x 20 elements for first iteration, 1797 x 10 elements for second)
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        FwdPassKernel<<<blocks_per_grid, threads_per_block>>>(
            layer_outputs[l], layer_outputs_pitches[l],
            biases[l - 1],
            weights[l - 1], weights_pitches[l - 1],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            nodes_per_layer[l - 1], nodes_per_layer[l], n_samples
        );
    }

    // Copy the last layer back
    cudaMemcpy2D((void*) outputs, sizeof(double) * nodes_per_layer[n_layers - 1], (void*) layer_outputs[n_layers - 1], layer_outputs_pitches[n_layers - 1], sizeof(double) * nodes_per_layer[n_layers - 1], n_samples, cudaMemcpyDeviceToHost);

    // Simply loop through all layers (except the last one), and clean everything weight & bias related.
    for (size_t l = 0; l < n_weights; l++) {
        // Free the device-side stuff
        cudaFree(biases[l]);
        cudaFree(weights[l]);
    }

    // Loop through all layers, including the input as this is copied to the GPU
    for (size_t l = 0; l < n_layers; l++) {
        // Free the device-side stuff
        cudaFree(layer_outputs[l]);
    }
}

extern "C" void nn_backward_cuda(neural_net* nn, double** delta_biases_cpu, double** delta_weights_cpu, size_t n_samples, double** inputs, double** expected) {
    /***** NN SHORTCUTS *****/

    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;

    
    /***** GPU MEMORY INITIALISATION *****/

    /* BIASES */
    
    // Copy the biases of the network to the GPU. Since the lists have different lengths, it is important to not make it a 2D-array.
    double* biases[n_weights];
    for (size_t l = 0; l < n_layers - 1; l++) {
        cudaMalloc((void**) (biases + l), sizeof(double) * nodes_per_layer[l + 1]);
        cudaMemcpy((void*) biases[l], nn->biases[l], sizeof(double) * nodes_per_layer[l + 1], cudaMemcpyHostToDevice);
    }


    /* WEIGHTS */

    // Also copy the weights in practically the same way, except that now the inner values are pitched arrays.
    //   We store the pitches in a similarly lengthy weights_pitches list.
    double* weights[n_weights];
    size_t weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMallocPitch((void**) (weights + l), weights_pitches + l, w, h);
        cudaMemcpy2D((void*) weights[l], weights_pitches[l], (void*) nn->weights[l], w, w, h, cudaMemcpyHostToDevice);
    }


    /* LAYER OUTPUTS */

    // The layer outputs is for every layer a matrix of samples by nodes_for_that_layer elements.
    //   Just as with the weights do we use pitches, and create a list to store those. Note that
    //   only the inputs need to be copied, as the rest serves as a buffer.
    double* layer_outputs[n_layers];
    size_t layer_outputs_pitches[n_layers];
    for (size_t l = 0; l < n_layers; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l];
        size_t h = n_samples;
        cudaMallocPitch((void**) layer_outputs + l, layer_outputs_pitches + l, w, h);
    }
    // Copy all sample inputs. Because of the unhappy alginment of inputs, we have to do this manually row-by-row.
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) layer_outputs[0] + s * layer_outputs_pitches[0]);
        cudaMemcpy((void*) ptr, (void*) inputs[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }


    /* DELTA BIASES */

    // We also have to declare the delta biases. Note that resetting is not needed, as every cell is not updated
    //   anymore but instead set to.
    double* delta_biases[n_weights];
    size_t delta_biases_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t this_nodes = nodes_per_layer[l + 1];
        cudaMallocPitch((void**) (delta_biases + l), delta_biases_pitches + l, sizeof(double) * this_nodes, n_samples);
    }


    /* DELTA WEIGHTS */

    // Declare the delta weights. Note that we pitch a 3D-array here. Not pretty, but better than
    //   the endless structs 3D require from us. Note that, just as with the biases, resetting is
    //   not needed, as every cell is not updated anymore but instead set to.
    double* delta_weights[n_weights];
    size_t delta_weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        // Prepare CUDA structs for allocation
        size_t w = nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        size_t d = n_samples;
        cudaMallocPitch((void**) (delta_weights + l), delta_weights_pitches + l, sizeof(double) * w, h * d);
    }


    /* EXPECTED */

    // The expected values are formatted as a 2D, n_samples x nodes_in_output_layer pitched matrix.
    double* expected_gpu;
    size_t expected_gpu_pitch;
    cudaMallocPitch((void**) &expected_gpu, &expected_gpu_pitch, sizeof(double) * nodes_per_layer[n_layers - 1], n_samples);
    // Copy all expected values for each sample, which we have to do row-by-row due to unfortunate formatting of expected
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) expected_gpu + s * expected_gpu_pitch);
        cudaMemcpy((void*) ptr, (void*) expected[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }


    /***** ITERATIONS *****/

    // Choose block size, but leave computation of the number of blocks to later as these are
    //   layer-dependent
    int threads_per_block = 32;
    int blocks_per_grid;

    /***** FORWARD PASS *****/

    // Loop through all layers forwardly so that we can compute errors later (2 iterations, non-parallelizable)
    for (size_t l = 1; l < n_layers; l++) {
        // Call upon the actiation kernel (should do 1797 x 20 elements for first iteration, 1797 x 10 elements for second)
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        FwdPassKernel<<<blocks_per_grid, threads_per_block>>>(
            layer_outputs[l], layer_outputs_pitches[l],
            biases[l - 1],
            weights[l - 1], weights_pitches[l - 1],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            nodes_per_layer[l - 1], nodes_per_layer[l], n_samples
        );
    }


    /***** BACKWARD PASS *****/

    // Then, compute the error at the output laye (1797 x 10 iterations)
    size_t l = n_layers - 1;
    blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
    BckPassOutputKernel<<<blocks_per_grid, threads_per_block>>>(
        delta_biases[l - 1], delta_biases_pitches[l - 1],
        delta_weights[l - 1], delta_weights_pitches[l - 1],
        layer_outputs[l - 1], layer_outputs_pitches[l - 1],
        layer_outputs[l], layer_outputs_pitches[l],
        expected_gpu, expected_gpu_pitch,
        nodes_per_layer[n_layers - 2], nodes_per_layer[n_layers - 1], n_samples
    );

    // Loop through all hidden layers in the other direction so that we can compute their weight updates (1 iteration, non-parallelizable)
    for (l = n_layers - 2; l > 0; l--) {
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        BckPassHiddenKernel<<<blocks_per_grid, threads_per_block>>>(
            delta_biases[l - 1], delta_biases_pitches[l - 1],
            delta_weights[l - 1], delta_weights_pitches[l - 1],
            delta_biases[l], delta_biases_pitches[l],
            weights[l], weights_pitches[l],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            layer_outputs[l], layer_outputs_pitches[l],
            nodes_per_layer[l - 1], nodes_per_layer[l], nodes_per_layer[l + 1],
            n_samples
        );
    }


    /***** CLEANUP *****/

    cudaDeviceSynchronize();

    /* COPY */

    // Copy the delta biases 'n' stuff back to the CPU
    for (size_t l = 0; l < n_weights; l++) {
        cudaMemcpy2D((void*) delta_biases_cpu[l], sizeof(double) * nodes_per_layer[l + 1], (void*) delta_biases[l], delta_biases_pitches[l], sizeof(double) * nodes_per_layer[l + 1], n_samples, cudaMemcpyDeviceToHost);
        cudaMemcpy2D((void*) delta_weights_cpu[l], sizeof(double) * nodes_per_layer[l + 1], (void*) delta_weights[l], delta_weights_pitches[l], sizeof(double) * nodes_per_layer[l + 1], nodes_per_layer[l] * n_samples, cudaMemcpyDeviceToHost);
    }

    /* BIASES & WEIGHTS */

    // Simply loop through all layers (except the last one), and clean everything weight & bias related.
    for (size_t l = 0; l < n_weights; l++) {
        // Free the device-side stuff
        cudaFree(biases[l]);
        cudaFree(delta_biases[l]);
        cudaFree(weights[l]);
        cudaFree(delta_weights[l]);
    }

    
    /* LAYER OUTPUTS */

    // Loop through all layers, including the input as this is copied to the GPU
    for (size_t l = 0; l < n_layers; l++) {
        // Free the device-side stuff
        cudaFree(layer_outputs[l]);
    }


    /* EXPECTED */

    // Finally, clear the expected list
    cudaFree(expected_gpu);
}

extern "C" void nn_full_cuda(neural_net* nn, double** biases_cpu, double** weights_cpu, size_t n_samples, double** inputs, double** expected, double learning_rate) {
    /***** NN SHORTCUTS *****/

    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;

    
    /***** GPU MEMORY INITIALISATION *****/

    /* BIASES */
    
    // Copy the biases of the network to the GPU. Since the lists have different lengths, it is important to not make it a 2D-array.
    double* biases[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        cudaMalloc((void**) (biases + l), sizeof(double) * nodes_per_layer[l + 1]);
        cudaMemcpy((void*) biases[l], nn->biases[l], sizeof(double) * nodes_per_layer[l + 1], cudaMemcpyHostToDevice);
    }


    /* WEIGHTS */

    // Also copy the weights in practically the same way, except that now the inner values are pitched arrays.
    //   We store the pitches in a similarly lengthy weights_pitches list.
    double* weights[n_weights];
    size_t weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMallocPitch((void**) (weights + l), weights_pitches + l, w, h);
        cudaMemcpy2D((void*) weights[l], weights_pitches[l], (void*) nn->weights[l], w, w, h, cudaMemcpyHostToDevice);
    }


    /* LAYER OUTPUTS */

    // The layer outputs is for every layer a matrix of samples by nodes_for_that_layer elements.
    //   Just as with the weights do we use pitches, and create a list to store those. Note that
    //   only the inputs need to be copied, as the rest serves as a buffer.
    double* layer_outputs[n_layers];
    size_t layer_outputs_pitches[n_layers];
    for (size_t l = 0; l < n_layers; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l];
        size_t h = n_samples;
        cudaMallocPitch((void**) layer_outputs + l, layer_outputs_pitches + l, w, h);
    }
    // Copy all sample inputs. Because of the unhappy alginment of inputs, we have to do this manually row-by-row.
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) layer_outputs[0] + s * layer_outputs_pitches[0]);
        cudaMemcpy((void*) ptr, (void*) inputs[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }


    /* DELTA BIASES */

    // We also have to declare the delta biases. Note that resetting is not needed, as every cell is not updated
    //   anymore but instead set to.
    double* delta_biases[n_weights];
    size_t delta_biases_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t this_nodes = nodes_per_layer[l + 1];
        cudaMallocPitch((void**) (delta_biases + l), delta_biases_pitches + l, sizeof(double) * this_nodes, n_samples);
    }


    /* DELTA WEIGHTS */

    // Declare the delta weights. Note that we pitch a 3D-array here. Not pretty, but better than
    //   the endless structs 3D require from us. Note that, just as with the biases, resetting is
    //   not needed, as every cell is not updated anymore but instead set to.
    double* delta_weights[n_weights];
    size_t delta_weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        // Prepare CUDA structs for allocation
        size_t w = nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        size_t d = n_samples;
        cudaMallocPitch((void**) (delta_weights + l), delta_weights_pitches + l, sizeof(double) * w, h * d);
    }


    /* EXPECTED */

    // The expected values are formatted as a 2D, n_samples x nodes_in_output_layer pitched matrix.
    double* expected_gpu;
    size_t expected_gpu_pitch;
    cudaMallocPitch((void**) &expected_gpu, &expected_gpu_pitch, sizeof(double) * nodes_per_layer[n_layers - 1], n_samples);
    // Copy all expected values for each sample, which we have to do row-by-row due to unfortunate formatting of expected
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) expected_gpu + s * expected_gpu_pitch);
        cudaMemcpy((void*) ptr, (void*) expected[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }


    /***** ITERATIONS *****/

    // Choose block size, but leave computation of the number of blocks to later as these are
    //   layer-dependent
    int threads_per_block = 32;
    int blocks_per_grid;

    /***** FORWARD PASS *****/

    // Loop through all layers forwardly so that we can compute errors later (2 iterations, non-parallelizable)
    for (size_t l = 1; l < n_layers; l++) {
        // Call upon the actiation kernel (should do 1797 x 20 elements for first iteration, 1797 x 10 elements for second)
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        FwdPassKernel<<<blocks_per_grid, threads_per_block>>>(
            layer_outputs[l], layer_outputs_pitches[l],
            biases[l - 1],
            weights[l - 1], weights_pitches[l - 1],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            nodes_per_layer[l - 1], nodes_per_layer[l], n_samples
        );
    }


    /***** BACKWARD PASS *****/

    // Then, compute the error at the output laye (1797 x 10 iterations)
    size_t l = n_layers - 1;
    blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
    BckPassOutputKernel<<<blocks_per_grid, threads_per_block>>>(
        delta_biases[l - 1], delta_biases_pitches[l - 1],
        delta_weights[l - 1], delta_weights_pitches[l - 1],
        layer_outputs[l - 1], layer_outputs_pitches[l - 1],
        layer_outputs[l], layer_outputs_pitches[l],
        expected_gpu, expected_gpu_pitch,
        nodes_per_layer[n_layers - 2], nodes_per_layer[n_layers - 1], n_samples
    );

    // Loop through all hidden layers in the other direction so that we can compute their weight updates (1 iteration, non-parallelizable)
    for (l = n_layers - 2; l > 0; l--) {
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        BckPassHiddenKernel<<<blocks_per_grid, threads_per_block>>>(
            delta_biases[l - 1], delta_biases_pitches[l - 1],
            delta_weights[l - 1], delta_weights_pitches[l - 1],
            delta_biases[l], delta_biases_pitches[l],
            weights[l], weights_pitches[l],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            layer_outputs[l], layer_outputs_pitches[l],
            nodes_per_layer[l - 1], nodes_per_layer[l], nodes_per_layer[l + 1],
            n_samples
        );
    }
    
    /***** BIAS & WEIGHT UPDATES *****/
        
    // Start with the biases
    for (size_t l = 0; l < n_weights; l++) {
        // Get some shortcuts
        double* delta_biases_l = delta_biases[l];
        size_t delta_biases_pitch = delta_biases_pitches[l];
        double* delta_weights_l = delta_weights[l];
        size_t delta_weights_pitch = delta_weights_pitches[l];
        size_t prev_nodes = nodes_per_layer[l];
        size_t this_nodes = nodes_per_layer[l + 1];

        // Use the recursive kernel call to sum all columns in the delta_biases array
        size_t to_do = n_samples;
        while (to_do > 1) {
            // Next, launch the kernel
            blocks_per_grid = ceil((to_do * this_nodes) / (double) threads_per_block);
            BiasReduceKernel<<<blocks_per_grid, threads_per_block>>>(
                delta_biases_l, delta_biases_pitch,
                this_nodes, to_do
            );

            // Don't forget to decrease to_do
            to_do = ceil(to_do / 2.0);
        }

        // Add the summed list of bias updates to the actual list of biases
        blocks_per_grid = ceil(this_nodes / (double) threads_per_block);
        BiasUpdateKernel<<<blocks_per_grid, threads_per_block>>>(
            biases[l], delta_biases_l,
            this_nodes, learning_rate
        );

        // Use the recursive kernel call to sum all columns in the delta_biases array
        to_do = n_samples;
        while (to_do > 1) {
            // Next, launch the kernel
            blocks_per_grid = ceil((to_do * this_nodes * prev_nodes) / (double) threads_per_block);
            WeightReduceKernel<<<blocks_per_grid, threads_per_block>>>(
                delta_weights_l, delta_weights_pitch,
                prev_nodes, this_nodes, to_do
            );

            // Don't forget to decrease to_do
            to_do = ceil(to_do / 2.0);
        }
        
        // Finally, add the summed matrix of weight updates to the actual matrix of weights
        blocks_per_grid = ceil((prev_nodes * this_nodes) / (double) threads_per_block);
        WeightUpdateKernel<<<blocks_per_grid, threads_per_block>>>(
            weights[l], weights_pitches[l],
            delta_weights_l, delta_weights_pitch,
            prev_nodes, this_nodes, learning_rate
        );
    }


    /***** CLEANUP *****/

    cudaDeviceSynchronize();

    /* COPY */

    // Copy the resulting weights and biases back to the CPU-side
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMemcpy((void*) biases_cpu[l], (void*) biases[l], w, cudaMemcpyDeviceToHost);
        cudaMemcpy2D((void*) weights_cpu[l], w, (void*) weights[l], weights_pitches[l], w, h, cudaMemcpyDeviceToHost);
    }

    /* BIASES & WEIGHTS */

    // Simply loop through all layers (except the last one), and clean everything weight & bias related.
    for (size_t l = 0; l < n_weights; l++) {
        // Free the device-side stuff
        cudaFree(biases[l]);
        cudaFree(delta_biases[l]);
        cudaFree(weights[l]);
        cudaFree(delta_weights[l]);
    }

    
    /* LAYER OUTPUTS */

    // Loop through all layers, including the input as this is copied to the GPU
    for (size_t l = 0; l < n_layers; l++) {
        // Free the device-side stuff
        cudaFree(layer_outputs[l]);
    }


    /* EXPECTED */

    // Finally, clear the expected list
    cudaFree(expected_gpu);
}



// Cuda memory help from https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
extern "C" void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations) {
    #ifdef BENCHMARK
    // Declare all timers
    struct timeval s_total, e_total, s_iters, e_iters, s_fwd, e_fwd, s_bck_out, e_bck_out, s_bck_hid, e_bck_hid, s_upd, e_upd;

    // Set some shortcuts for the timers
    size_t half_iters = n_iterations / 2;

    // Start the total timer
    gettimeofday(&s_total, NULL);
    #endif

    /***** NN SHORTCUTS *****/

    size_t n_layers = nn->n_layers;
    size_t n_weights = nn->n_weights;
    size_t* nodes_per_layer = nn->nodes_per_layer;

    
    /***** GPU MEMORY INITIALISATION *****/

    /* BIASES */
    
    // Copy the biases of the network to the GPU. Since the lists have different lengths, it is important to not make it a 2D-array.
    double* biases[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        cudaMalloc((void**) (biases + l), sizeof(double) * nodes_per_layer[l + 1]);
        cudaMemcpy((void*) biases[l], nn->biases[l], sizeof(double) * nodes_per_layer[l + 1], cudaMemcpyHostToDevice);
    }


    /* WEIGHTS */

    // Also copy the weights in practically the same way, except that now the inner values are pitched arrays.
    //   We store the pitches in a similarly lengthy weights_pitches list.
    double* weights[n_weights];
    size_t weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMallocPitch((void**) (weights + l), weights_pitches + l, w, h);
        cudaMemcpy2D((void*) weights[l], weights_pitches[l], (void*) nn->weights[l], w, w, h, cudaMemcpyHostToDevice);
    }


    /* LAYER OUTPUTS */

    // The layer outputs is for every layer a matrix of samples by nodes_for_that_layer elements.
    //   Just as with the weights do we use pitches, and create a list to store those. Note that
    //   only the inputs need to be copied, as the rest serves as a buffer.
    double* layer_outputs[n_layers];
    size_t layer_outputs_pitches[n_layers];
    for (size_t l = 0; l < n_layers; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l];
        size_t h = n_samples;
        cudaMallocPitch((void**) layer_outputs + l, layer_outputs_pitches + l, w, h);
    }
    // Copy all sample inputs. Because of the unhappy alginment of inputs, we have to do this manually row-by-row.
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) layer_outputs[0] + s * layer_outputs_pitches[0]);
        cudaMemcpy((void*) ptr, (void*) inputs[s], sizeof(double) * nodes_per_layer[0], cudaMemcpyHostToDevice);
    }


    /* DELTA BIASES */

    // We also have to declare the delta biases. Note that resetting is not needed, as every cell is not updated
    //   anymore but instead set to.
    double* delta_biases[n_weights];
    size_t delta_biases_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        size_t this_nodes = nodes_per_layer[l + 1];
        cudaMallocPitch((void**) (delta_biases + l), delta_biases_pitches + l, sizeof(double) * this_nodes, n_samples);
    }


    /* DELTA WEIGHTS */

    // Declare the delta weights. Note that we pitch a 3D-array here. Not pretty, but better than
    //   the endless structs 3D require from us. Note that, just as with the biases, resetting is
    //   not needed, as every cell is not updated anymore but instead set to.
    double* delta_weights[n_weights];
    size_t delta_weights_pitches[n_weights];
    for (size_t l = 0; l < n_weights; l++) {
        // Prepare CUDA structs for allocation
        size_t w = nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        size_t d = n_samples;
        cudaMallocPitch((void**) (delta_weights + l), delta_weights_pitches + l, sizeof(double) * w, h * d);
    }


    /* EXPECTED */

    // The expected values are formatted as a 2D, n_samples x nodes_in_output_layer pitched matrix.
    double* expected_gpu;
    size_t expected_gpu_pitch;
    cudaMallocPitch((void**) &expected_gpu, &expected_gpu_pitch, sizeof(double) * nodes_per_layer[n_layers - 1], n_samples);
    // Copy all expected values for each sample, which we have to do row-by-row due to unfortunate formatting of expected
    for (size_t s = 0; s < n_samples; s++) {
        double* ptr = (double*) ((char*) expected_gpu + s * expected_gpu_pitch);
        cudaMemcpy((void*) ptr, (void*) expected[s], sizeof(double) * nodes_per_layer[n_layers - 1], cudaMemcpyHostToDevice);
    }


    /***** ITERATIONS *****/

    // Choose block size, but leave computation of the number of blocks to later as these are
    //   layer-dependent
    unsigned int blocks_per_grid;

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
            cudaDeviceSynchronize();

            gettimeofday(&s_fwd, NULL);
        }
        #endif

        // Loop through all layers forwardly so that we can compute errors later (2 iterations, non-parallelizable)
        for (size_t l = 1; l < n_layers; l++) {
            // Call upon the actiation kernel (should do 1797 x 20 elements for first iteration, 1797 x 10 elements for second)
            blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
            FwdPassKernel<<<blocks_per_grid, threads_per_block>>>(
                layer_outputs[l], layer_outputs_pitches[l],
                biases[l - 1],
                weights[l - 1], weights_pitches[l - 1],
                layer_outputs[l - 1], layer_outputs_pitches[l - 1],
                nodes_per_layer[l - 1], nodes_per_layer[l], n_samples
            );
        }

        #ifdef BENCHMARK
        // End the forward timer, start the backward pass output timer
        if (i == half_iters) {
            cudaDeviceSynchronize();

            gettimeofday(&e_fwd, NULL);
            gettimeofday(&s_bck_out, NULL);
        }
        #endif


        /***** BACKWARD PASS *****/

        // Then, compute the error at the output laye (1797 x 10 iterations)
        size_t l = n_layers - 1;
        blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
        BckPassOutputKernel<<<blocks_per_grid, threads_per_block>>>(
            delta_biases[l - 1], delta_biases_pitches[l - 1],
            delta_weights[l - 1], delta_weights_pitches[l - 1],
            layer_outputs[l - 1], layer_outputs_pitches[l - 1],
            layer_outputs[l], layer_outputs_pitches[l],
            expected_gpu, expected_gpu_pitch,
            nodes_per_layer[n_layers - 2], nodes_per_layer[n_layers - 1], n_samples
        );

        #ifdef BENCHMARK
        // End the backward pass output timer, start the backward pass hidden timer
        if (i == half_iters) {
            cudaDeviceSynchronize();

            gettimeofday(&e_bck_out, NULL);
            gettimeofday(&s_bck_hid, NULL);
        }
        #endif

        // Loop through all hidden layers in the other direction so that we can compute their weight updates (1 iteration, non-parallelizable)
        for (l = n_layers - 2; l > 0; l--) {
            blocks_per_grid = (n_samples * nodes_per_layer[l] + threads_per_block - 1) / threads_per_block;
            BckPassHiddenKernel<<<blocks_per_grid, threads_per_block>>>(
                delta_biases[l - 1], delta_biases_pitches[l - 1],
                delta_weights[l - 1], delta_weights_pitches[l - 1],
                delta_biases[l], delta_biases_pitches[l],
                weights[l], weights_pitches[l],
                layer_outputs[l - 1], layer_outputs_pitches[l - 1],
                layer_outputs[l], layer_outputs_pitches[l],
                nodes_per_layer[l - 1], nodes_per_layer[l], nodes_per_layer[l + 1],
                n_samples
            );
        }

        #ifdef BENCHMARK
        // End the backward pass hidden timer
        if (i == half_iters) {
            cudaDeviceSynchronize();

            gettimeofday(&e_bck_hid, NULL);
            gettimeofday(&s_upd, NULL);
        }
        #endif

        /***** BIAS & WEIGHT UPDATES *****/
        
        // Start with the biases
        for (size_t l = 0; l < n_weights; l++) {
            // Get some shortcuts
            double* delta_biases_l = delta_biases[l];
            size_t delta_biases_pitch = delta_biases_pitches[l];
            double* delta_weights_l = delta_weights[l];
            size_t delta_weights_pitch = delta_weights_pitches[l];
            size_t prev_nodes = nodes_per_layer[l];
            size_t this_nodes = nodes_per_layer[l + 1];

            // Use the recursive kernel call to sum all columns in the delta_biases array
            size_t to_do = n_samples;
            while (to_do > 1) {
                // Next, launch the kernel
                blocks_per_grid = ceil((to_do * this_nodes) / (double) threads_per_block);
                BiasReduceKernel<<<blocks_per_grid, threads_per_block>>>(
                    delta_biases_l, delta_biases_pitch,
                    this_nodes, to_do
                );

                // Don't forget to decrease to_do
                to_do = ceil(to_do / 2.0);
            }

            // Add the summed list of bias updates to the actual list of biases
            blocks_per_grid = ceil(this_nodes / (double) threads_per_block);
            BiasUpdateKernel<<<blocks_per_grid, threads_per_block>>>(
                biases[l], delta_biases_l,
                this_nodes, learning_rate
            );

            // Use the recursive kernel call to sum all columns in the delta_biases array
            to_do = n_samples;
            while (to_do > 1) {
                // Next, launch the kernel
                blocks_per_grid = ceil((to_do * this_nodes * prev_nodes) / (double) threads_per_block);
                WeightReduceKernel<<<blocks_per_grid, threads_per_block>>>(
                    delta_weights_l, delta_weights_pitch,
                    prev_nodes, this_nodes, to_do
                );

                // Don't forget to decrease to_do
                to_do = ceil(to_do / 2.0);
            }
            
            // Finally, add the summed matrix of weight updates to the actual matrix of weights
            blocks_per_grid = ceil((prev_nodes * this_nodes) / (double) threads_per_block);
            WeightUpdateKernel<<<blocks_per_grid, threads_per_block>>>(
                weights[l], weights_pitches[l],
                delta_weights_l, delta_weights_pitch,
                prev_nodes, this_nodes, learning_rate
            );
        }

        #ifdef BENCHMARK
        // Stop the updates timer
        if (i == half_iters) {
            cudaDeviceSynchronize();
            
            gettimeofday(&e_upd, NULL);
        }
        #endif
    }

    /***** CLEANUP *****/

    cudaDeviceSynchronize();

    #ifdef BENCHMARK
    // End the iterations timer
    gettimeofday(&e_iters, NULL);
    #endif

    /* COPY */

    // Copy the resulting weights and biases back to the CPU-side
    for (size_t l = 0; l < n_weights; l++) {
        size_t w = sizeof(double) * nodes_per_layer[l + 1];
        size_t h = nodes_per_layer[l];
        cudaMemcpy((void*) nn->biases[l], (void*) biases[l], w, cudaMemcpyDeviceToHost);
        cudaMemcpy2D((void*) nn->weights[l], w, (void*) weights[l], weights_pitches[l], w, h, cudaMemcpyDeviceToHost);
    }

    /* BIASES & WEIGHTS */

    // Simply loop through all layers (except the last one), and clean everything weight & bias related.
    for (size_t l = 0; l < n_weights; l++) {
        // Free the device-side stuff
        cudaFree(biases[l]);
        cudaFree(delta_biases[l]);
        cudaFree(weights[l]);
        cudaFree(delta_weights[l]);
    }

    
    /* LAYER OUTPUTS */

    // Loop through all layers, including the input as this is copied to the GPU
    for (size_t l = 0; l < n_layers; l++) {
        // Free the device-side stuff
        cudaFree(layer_outputs[l]);
    }


    /* EXPECTED */

    // Finally, clear the expected list
    cudaFree(expected_gpu);

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

extern "C" void parse_opt_args(int argc, char** argv) {
    // // Parse threads per block as first argument
    // if (argc >= 1) {
    //     threads_per_block = atoi(argv[0]);
    // }
    // Do nothing instead
    (void) argc;
    (void) argv;
}

extern "C" void print_opt_args() {
    printf(" - Variation               : CUDA GPU 1\n");
    printf(" - Threads per block       : %u\n", threads_per_block);
}
