/* TEST NN.c
 *   by Lut99
 *
 * Created:
 *   4/19/2020, 11:19:47 PM
 * Last edited:
 *   30/04/2020, 22:00:01
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

#include "Array.h"
#include "Matrix.h"
#include "Functions.h"
#include "NeuralNetwork.h"


/***** TOOLS *****/

/* Writes the data to a .dat file so GNUplot can plot it later. */
void write_costs(const array* costs) {
    FILE* dat = fopen("./nn_costs.dat", "w");
    fprintf(dat, "# Iteration / Costs\n");
    for (size_t i = 0; i < costs->size; i++) {
        fprintf(dat, "%ld %f\n", i, costs->d[i]);
    }
    fclose(dat);
}


/***** TEST FUNCTIONS *****/

/* Tests the entire network on a simple toy problem (AND-function) */
bool test_training_simple() {
    // Define the input. Each row is sample, where each column is one of the input nodes
    double inputs_vanilla[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    // Define the output
    double expected_vanilla[4][1] = {
        {0},
        {0},
        {0},
        {1}
    };

    // Create a new neural network
    size_t nodes_per_layer[] = {2};
    neural_net* nn = create_nn(2, 1, nodes_per_layer, 1);

    // Prepare the input, expected and output lists
    array* inputs[4] = {};
    array* expected[4] = {};
    array* outputs[4] = {};
    for (int i = 0; i < 4; i++) {
        inputs[i] = create_array(2, inputs_vanilla[i]);
        expected[i] = create_array(1, expected_vanilla[i]);
        outputs[i] = create_empty_array(1);
    }

    // Train it with 500 iterations, noting the costs
    array* avg_costs = nn_train_costs(nn, 4, inputs, expected, 0.9, 5000, sigmoid, dydx_sigmoid);

    // Write the data for the graph showing the average cost
    write_costs(avg_costs);

    // Now, run a test
    nn_forward(nn, 4, outputs, inputs, sigmoid);
    round_output(4, outputs);

    // Check if it is expected (either correct or not, there is nothing in between >:( )
    bool succes = true;
    double accuracy = compute_accuracy(4, outputs, expected);
    if (accuracy < 1.0) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "\nAccuracy is less than perfect (%.2f)\n\nGot output:\n", accuracy);
        for (size_t i = 0; i < 4; i++) {
            array_print(stderr, outputs[i]);
        }
        fprintf(stderr, "\nExpected output:\n");
        for (size_t i = 0; i < 4; i++) {
            array_print(stderr, expected[i]);
        }
        fprintf(stderr, "\nTesting neural network failed.\n");
    }

    // Cleanup
    for (int i = 0; i < 4; i++) {
        destroy_array(inputs[i]);
        destroy_array(expected[i]);
        destroy_array(outputs[i]);
    }
    destroy_array(avg_costs);

    // Return the succes status
    return succes;
}



/***** MAIN *****/

int main() {
    printf("  Testing training (AND-function)...       ");
    if (!test_training_simple()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("NeuralNetwork tests succes.\n\n");
}
