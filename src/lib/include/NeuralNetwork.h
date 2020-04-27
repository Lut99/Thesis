/* NEURAL NETWORK.c
 *   by Lut99
 *
 * Created:
 *   4/18/2020, 11:25:46 PM
 * Last edited:
 *   28/04/2020, 00:47:29
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
    matrix** biases;
    /* List of weights for each layer. */
    matrix** weights;
} neural_net;



/***** MEMORY MANAGEMENT *****/

/* Initializes a neural network. The first argument is the number of nodes in the input layer, the second argument the number of hidden layers, the third argument is a list of the number of nodes for each of those hidden layers and the last argument is the number of nodes in the output layer. */
neural_net* create_nn(size_t input_nodes, size_t n_hidden_layers, size_t hidden_nodes[n_hidden_layers], size_t output_nodes);

/* Destroys given neural network. */
void destroy_nn(neural_net* nn);



/***** NEURAL NETWORK OPERATIONS *****/

/* Activates the neural network using the given activation function (which should operate on matrices) and using given inputs. The results of each layer are returned in the given output list of newly allocated matrices. */
void nn_activate_all(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* inputs, matrix* (*act)(matrix*));

/* Activates the neural network using the given activation function (which should operate on matrices) and using given inputs, except that this only returns the outputs of the last layer. */
matrix* nn_activate(neural_net* nn, const matrix* inputs, matrix* (*act)(matrix*));

/* Backpropagates through the network. The learning rate is equal to eta, as sometimes seen in tutorials, and determines the speed of the gradient descent. The given partial derivative of the cost function is used to translate from deltas to the change in weights. */
void nn_backpropagate(neural_net* nn, matrix* outputs[nn->n_layers], const matrix* expected, double learning_rate, matrix* (*dydx_act)(const matrix*));

/* Trains the network for n_iterations iterations. Note that this version also returns a newly allocated list of costs for each iteration so that plots can be made based on the given cost function. */
double* nn_train_costs(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*));

/* Trains the network for n_iterations iterations. Note that this version also returns a newly allocated list of costs for each iteration so that plots can be made based on the given cost function. */
void nn_train(neural_net* nn, const matrix* inputs, const matrix* expected, double learning_rate, size_t n_iterations, matrix* (*act)(matrix*), matrix* (*dydx_act)(const matrix*));


/***** USEFUL TOOLS *****/

/* For all columns in the matrix, sets each value to zero except the largest value. For two equal values, the first one is chosen so that there is only one. The result is returned in the given matrix (inplace). */
matrix* nn_flatten_results(matrix* outputs);

#endif
