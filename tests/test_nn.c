/* TEST NN.c
 *   by Lut99
 *
 * Created:
 *   4/19/2020, 11:19:47 PM
 * Last edited:
 *   20/04/2020, 23:25:13
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

/* Tests the feedforward capibility by entering all elements vector-by-vector. */
bool test_activation_vec() {
    // Define the input and output values. Note that we want to test an AND-function here.
    double start[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected[4] = {0, 0, 0, 1};

    // Define the custom weights
    double weights[1][3] = {{-30, 20, 20}};

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

        // Active the network
        matrix* output = nn_activate(nn, input, sigmoid);

        // Check if the output is expected
        if (round(output->data[0]) != expected[i]) {
            succes = false;
            printf(" [FAIL]\n");
            fprintf(stderr, "\nNeural network returned %f, but expected %f for testcase [%f, %f]\n\n",
                    output->data[0], expected[i], start[i][0], start[i][1]);
            fprintf(stderr, "Testing activation (vectors) failed.\n\n");
        }

        // Free the two matrices
        destroy_matrix(input);
        destroy_matrix(output);
    }
    
    // Free the neural network
    destroy_nn(nn);

    return succes;
}

/* Tests the feedforward capibility by entering all samples at once in a matrix */
bool test_activation_mat() {
    // Define the input. Each test case is a row to not upset the underlying math
    double start[2][4] = {{0, 0, 1, 1},
                          {0, 1, 0, 1}};
    // Define the output
    double expec[1][4] = {{0, 1, 1, 1}};

    // Define the custom weights
    double weights[1][3] = {{-10, 20, 20}};

    // Create a neural network with no hidden layers but remove the random weights that are initialised
    neural_net* nn = create_nn(2, 0, NULL, 1);
    for (size_t i = 0; i < nn->n_weights; i++) {
        destroy_matrix(nn->weights[i]);
    }
    free(nn->weights);

    // Set the custom weights
    matrix* custom_weights = create_matrix(1, 3, weights);
    nn->weights = malloc(sizeof(matrix*));
    nn->weights[0] = custom_weights;

    // Prepare the input, output and expected matrices
    matrix* m_in = create_matrix(2, 4, start);
    matrix* m_exp = create_matrix(1, 4, expec);
    
    // Activate the network
    matrix* m_out = nn_activate(nn, m_in, sigmoid);

    // Check if it is what we expect
    bool succes = true;
    if (m_out->rows != m_exp->rows || m_out->cols != m_exp->cols) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices do not have the same shape: got %ldx%ld, expected %ldx%ld\n",
                m_out->rows,
                m_out->cols,
                m_exp->rows,
                m_exp->cols);
    } else {
        // Loop through the elements
        for (size_t i = 0; succes && i < m_out->rows * m_out->cols; i++) {
            if (round(m_out->data[i]) != m_exp->data[i]) {
                succes = false;
                printf(" [FAIL]\n");
                fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
                matrix_print(m_out);
                fprintf(stderr, "\nExpected:\n");
                matrix_print(m_exp);
                fprintf(stderr, "\nTesting activation (matrices) failed.\n\n");
            }
        }
    }
    
    // Cleanup
    destroy_nn(nn);
    destroy_matrix(m_in);
    destroy_matrix(m_exp);
    destroy_matrix(m_out);

    return succes;
}

/* Tests the entire network on a simple toy problem (AND-function) */
bool test_training_simple() {
    // Define the input. Each test case is a row to not upset the underlying math
    double start[2][4] = {{0, 0, 1, 1},
                          {0, 1, 0, 1}};
    // Define the output
    double expec[1][4] = {{0, 1, 1, 1}};

    // Create a new neural network
    size_t nodes_per_layer[] = {2};
    neural_net* nn = create_nn(2, 1, nodes_per_layer, 1);

    // Prepare the inputs & expected matrices
    matrix* m_in = create_matrix(2, 4, start);
    matrix* m_exp = create_matrix(1, 4, expec);

    // Train it with 500 iterations, noting the costs
    double* costs = nn_train_costs(nn, m_in, m_exp, 0.9, 5000, sigmoid, other_cost_func, dydx_other_cost_func);

    // Write the graph showing the cost
    // TBD

    // Now, test it
    matrix* m_out = nn_activate(nn, m_in, sigmoid);

    // Check if it is expected
    bool succes = true;
    if (m_out->rows != m_exp->rows || m_out->cols != m_exp->cols) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices do not have the same shape: got %ldx%ld, expected %ldx%ld\n",
                m_out->rows,
                m_out->cols,
                m_exp->rows,
                m_exp->cols);
    } else {
        // Loop through the elements
        for (size_t i = 0; succes && i < m_out->rows * m_out->cols; i++) {
            if (round(m_out->data[i]) != m_exp->data[i]) {
                succes = false;
                printf(" [FAIL]\n");
                fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
                matrix_print(m_out);
                fprintf(stderr, "\nExpected:\n");
                matrix_print(m_exp);
                fprintf(stderr, "\nTesting training (AND-function) failed.\n\n");
            }
        }
    }

    // Cleanup
    destroy_nn(nn);
    destroy_matrix(m_in);
    destroy_matrix(m_exp);
    destroy_matrix(m_out);
    free(costs);

    // Return the succes status
    return succes;
}



/***** MAIN *****/

int main() {
    printf("  Testing activation (vectors)...          ");
    if (!test_activation_vec()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing activation (matrices)...         ");
    if (!test_activation_mat()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing training (AND-function)...       ");
    if (!test_training_simple()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("NeuralNetwork tests succes.\n\n");
}
