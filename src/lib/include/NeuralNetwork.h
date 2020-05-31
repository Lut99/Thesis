/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   5/31/2020, 8:39:57 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file implements a neural network using a matrix-based
 *   implementation (using Matrix.c) rather than an object-oriented-based
 *   implementation. Any special functions used (such as activation or loss
 *   functions) are defined in Functions.c. This is the header file.
**/

#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H

#include "Array.h"


/* Defines the NeuralNetwork. */
typedef struct NEURALNET {
    /* The number of layers in the neural network. Note that this includes input and output layers. */
    size_t n_layers;
    /* The number of nodes per layer in the network. Note that this includes input and output layers. */
    size_t* nodes_per_layer;

    /* The number of weights in the neural network. Equal to the number of layers minus 1. */
    size_t n_weights;
    /* List of biases for each layer, save the output layer (each one has size this_layer_nodes). */
    double** biases;
    /* List of weights for each layer (each one has size this_layer_nodes x next_layer_nodes) */
    double** weights;
} neural_net;



/***** MEMORY MANAGEMENT *****/

/* Initializes a neural network. The first argument is the number of nodes in the input layer, the second argument the number of hidden layers, the third argument is a list of the number of nodes for each of those hidden layers and the last argument is the number of nodes in the output layer. */
neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t* hidden_nodes, size_t output_nodes);

/* Destroys given neural network. */
void destroy_nn(neural_net* nn);



/***** NEURAL NETWORK OPERATIONS *****/

/* Computes a forward pass through the network for n_samples using the given activation function. The outputs of the final layer for each sample is returned in the given output list. */
void nn_forward(neural_net* nn, size_t n_samples, double* outputs, double** inputs);

/* Performs training for n_iterations and returns the costs. Like nn_forward, it is designed to take in all the samples in a training set at once and parse them. The learning rate, also called eta, determines how fast the network learns which can be tweaked to avoid overfitting. The activiation function is given in act, and its derivative in dydx_act. */
array* nn_train_costs(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t max_iterations);

/* Performs training for n_iterations. Like nn_forward, it is designed to take in all the samples in a training set at once and parse them. The learning rate, also called eta, determines how fast the network learns which can be tweaked to avoid overfitting. The activiation function is given in act, and its derivative in dydx_act. */
void nn_train(neural_net* nn, size_t n_samples, double** inputs, double** expected, double learning_rate, size_t n_iterations);



/***** VALIDATION TOOLS *****/

/* For a given 2D array outputs (n_samples x last_nodes), flatting it so that the highest element each row is set to 1 and the others to 0. */
void flatten_output(size_t n_samples, size_t last_nodes, double* outputs) ;

/* For a given 2D array outputs (n_samples x last_nodes), rounds the value of each element */
void round_output(size_t n_samples, size_t last_nodes, double* outputs);

/* For a given 2D array outputs (n_samples x last_nodes) and a list-of-lists expected (n_samples x last_nodes), compares given output of each sample with given expected output of that sample and returns the accuracy measure. */
double compute_accuracy(size_t n_samples, size_t last_nodes, double* outputs, double** expected);


/***** OTHER TOOLS *****/

/* Parses given optional arguments according to the specific NeuralNetwork implementation. Note that these need to be offsetted to exclude the bin and dataset location. */
void parse_opt_args(int argc, char** argv);

/* Prints the optional arguments for to the specific NeuralNetwork implementation. */
void print_opt_args();

#endif
