/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   01/05/2020, 13:42:45
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
#include "Matrix.h"


/* Defines the NeuralNetwork. */
typedef struct NEURALNET {
    /* The number of layers in the neural network. Note that this includes input and output layers. */
    size_t n_layers;
    /* The number of nodes per layer in the network. Note that this includes input and output layers. */
    size_t* nodes_per_layer;

    /* The number of weights in the neural network. Equal to the number of layers minus 1. */
    size_t n_weights;
    /* List of biases for each layer, save the output layer. */
    array** biases;
    /* List of weights for each layer. */
    matrix** weights;
} neural_net;



/***** MEMORY MANAGEMENT *****/

/* Initializes a neural network. The first argument is the number of nodes in the input layer, the second argument the number of hidden layers, the third argument is a list of the number of nodes for each of those hidden layers and the last argument is the number of nodes in the output layer. */
neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t hidden_nodes[n_hidden_layers], size_t output_nodes);

/* Destroys given neural network. */
void destroy_nn(neural_net* nn);



/***** NEURAL NETWORK OPERATIONS *****/

/* Computes a forward pass through the network for the inputs of a single sample using the given activation function. The outputs for each layer are returned in the given output list. */
void nn_activate(neural_net* nn, array* outputs[nn->n_layers], const array* inputs, double (*activation_function)(double));

/* Computes a forward pass through the network for n_samples using the given activation function. The outputs of the final layer for each sample is returned in the given output list. */
void nn_forward(neural_net* nn, size_t n_samples, array* outputs[n_samples], array* inputs[n_samples], double (*activation_function)(double));

/* Backpropagates through the network to update the weights. The learning rate is equal to eta, as sometimes seen in tutorials, and determines the speed of the gradient descent. While the cost function is fixed (Mean Square Error), the derivative of the activation function is provided via a function pointer. Finally, the scratchpad argument is a list of at least the maximum number of nodes on the layers of the neural network to use as re-usable temporary array that needn't be re-allocated all the time. */
void nn_backpropagate(neural_net* nn, array* outputs[nn->n_layers], const array* expected, double learning_rate, double (*dydx_act)(double), array* scratchpad);

/* Performs training for n_iterations and returns the costs. Like nn_forward, it is designed to take in all the samples in a training set at once and parse them. The learning rate, also called eta, determines how fast the network learns which can be tweaked to avoid overfitting. The activiation function is given in act, and its derivative in dydx_act. */
array* nn_train_costs(neural_net* nn, size_t n_samples, array* inputs[n_samples], array* expected[n_samples], double learning_rate, size_t max_iterations, double (*act)(double), double (*dydx_act)(double));

/* Performs training for n_iterations. Like nn_forward, it is designed to take in all the samples in a training set at once and parse them. The learning rate, also called eta, determines how fast the network learns which can be tweaked to avoid overfitting. The activiation function is given in act, and its derivative in dydx_act. */
void nn_train(neural_net* nn, size_t n_samples, array* inputs[n_samples], array* expected[n_samples], double learning_rate, size_t n_iterations, double (*act)(double), double (*dydx_act)(double));



/***** VALIDATION TOOLS *****/

/* Flattens given list of outputs so that only the highest value of each output is set to 1, the rest to 0. */
void flatten_output(size_t n_samples, array* outputs[n_samples]);

/* Round the output to the nearest integer. */
void round_output(size_t n_samples, array* outputs[n_samples]);

/* Compares given output with given expected output. Returns an accuracy measure of how many samples were (almost) equal divided by the total amount of samples. */
double compute_accuracy(size_t n_samples, array* outputs[n_samples], array* expected[n_samples]);

#endif
