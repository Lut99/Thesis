/* TEST NN.c
 *   by Lut99
 *
 * Created:
 *   4/19/2020, 11:19:47 PM
 * Last edited:
 *   4/20/2020, 12:11:07 AM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file performs test on the basic operations of the NeuralNetwork
 *   library. Can be run using 'make tests'.
**/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "Functions.h"
#include "NeuralNetwork.h"


/***** TEST FUNCTIONS *****/

/* Tests the feedforward capibility. */
bool test_activation() {
    // Define the input and output values. Note that we want to test an AND-function here.
    double start[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected[4] = {0, 0, 0, 1};

    // Define the custom weights
    double weights[1][3] = {{-30, -10, -10}};

    // Create a neural network with no hidden layers but remove the random weights that are initialised
    neural_net* nn = create_nn(2, 0, NULL, 1);
    for (size_t i = 0; i < nn->n_weights; i++) {
        destroy_matrix(nn->weights[i]);
    }
    free(nn->weights);

    // Set the weights custom weights
    matrix* custom_weights = create_matrix(1, 3, weights);
    nn->weights = malloc(sizeof(matrix*));
    nn->weights[0] = custom_weights;
    
    // Loop through the test cases to test the pre-trained network
    bool succes = true;
    for (int i = 0; succes && i < 4; i++) {
        // Create the input matrix and placeholder for the output matrix
        matrix* input = create_vector(2, start[i]);
        matrix* output = create_empty_matrix(1, 1);

        // Active the network
        nn_activate(nn, output, input, sigmoid);

        // Check if the output is expected
        if (round(output->data[0]) != expected[i]) {
            succes = false;
            printf(" [FAIL]\n");
            fprintf(stderr, "\nNeural network returned %f, but expected %f for testcase [%f, %f]\n\n",
                    output->data[0], expected[i], start[i][0], start[i][1]);
            fprintf(stderr, "Testing activation failed.\n\n");
        }

        // Free the two matrices
        destroy_matrix(input);
        destroy_matrix(output);
    }
    
    // Free the neural network
    destroy_nn(nn);

    return succes;
}



/***** MAIN *****/

int main() {
    printf("  Testing activation...");
    if (!test_activation()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("NeuralNetwork tests succes.\n\n");
}
