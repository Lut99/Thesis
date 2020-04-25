/* TEST MEMORY.c
 *   by Lut99
 *
 * Created:
 *   4/25/2020, 8:25:15 PM
 * Last edited:
 *   4/25/2020, 8:31:06 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file is used for basic memory debugging
**/

#include <stdio.h>
#include <stdbool.h>

#include "Functions.h"
#include "NeuralNetwork.h"


/* Test activate_all memory */
int main() {
    double inputs[3][3] = {
        {1, 2, 3},
        {4, 5, 6}.
        {7, 8, 9}
    };

    // This just runs a very simplified test on nn->activate_all to test for memory leaks
    matrix* m_in = create_matrix(3, 3, inputs);
    neural_net* nn = create_nn(3, 0, NULL, 3);
    matrix* m_out[2];
    nn_activate_all(nn, m_out, m_in, sigmoid);

    // Done, cleanup
    destroy_matrix(m_in);
    destroy_nn(nn);
}

