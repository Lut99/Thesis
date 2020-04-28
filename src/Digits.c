/* DIGITS.c
 *   by Tim Müller
 *
 * Created:
 *   21/04/2020, 11:46:37
 * Last edited:
 *   28/04/2020, 19:28:19
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file uses the NeuralNetwork class to try and predict the correct
 *   digit from a training set of hand-drawn digits.
**/

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <math.h>

#include "Functions.h"
#include "NeuralNetwork.h"

/* Percentage of data that will be used for training, the rest is for testing. */
#define TRAIN_RATIO 0.8
/* Number of iterations that the neural network will be trained on. */
#define TRAIN_ITERATIONS 100000
/* Learning rate of the Neural Network. */
#define TRAIN_ETA 0.0005


static unsigned int row = 1;
static unsigned int col = 1;


/***** TOOLS *****/

/* Fetches a number from the file. If something goes wrong, prints an error and returns -1. If EOF is reached, it return 0. Otherwise, 1 is returned, indicating that more is to be read. */
int get_num(FILE* f, int* num) {
    // Get character-by-character until comma, newline or EOF
    int value = 0;
    bool comment = false;
    bool seen_digit = false;
    while (true) {
        char c = fgetc(f);

        // Fetch an element into the correct buffer position
        if (c == EOF) {
            if (!feof(f)) {
                fprintf(stderr, "ERROR: get_num: (%u:%u): could not read from file: %s\n",
                        row, col, strerror(errno));
                return -1;
            }

            // Check if we have got a number
            if (!seen_digit) {
                return 0;
            }

            // Otherwise, return the value
            (*num) = value;
            return 1;
        }
        
        if (c == '#') {
            // Set comment mode to skip all until newline
            comment = true;
        } else if ((!comment && c == ',') || c == '\n') {
            if (c == '\n') {
                // Reset the comment line & increment row
                comment = false;
                row++;
                col = 0;

                // If nothing's found, just continue
                if (!seen_digit) { continue; }
            } else {
                // Check if we have got a number
                if (!seen_digit) {
                    fprintf(stderr, "ERROR: get_num: (%u:%u): no number given\n",
                            row, col);
                    return -1;
                }
            }

            // Otherwise, update col and return the value
            col++;
            (*num) = value;
            return 1;
        } else if (!comment && c >= '0' && c <= '9') {
            seen_digit = true;
            // Make sure value will not overflow
            int ic = (int)(c - '0');
            if (value > INT_MAX / 10 || value * 10 > INT_MAX - ic) {
                fprintf(stderr, "ERROR: get_num: (%u:%u): integer overflow occured\n",
                        row, col);
                return -1;
            }

            // Update the value
            value = value * 10 + ic;
        } else if (!comment && c != ' ' && c != '\t') {
            fprintf(stderr, "ERROR: get_num: (%u:%u): illegal character '%c'\n",
                    row, col, c);
            return -1;
        }

        // Don't forget to update col
        col++;
    }
}

/* Cleans a given string by removing all spaces and # and everything after (as comment). This is done in-place, so after this operation the string be shorter or the same size as it was before. */
void clean_input(char* input) {
    int write_i = 0;
    for (int i = 0; input[i] != '\0'; i++) {
        char c = input[i];
        if (c == '#') {
            // Immediately done
            break;
        } else if (c != ' ') {
            // Write this char to the write_i and update it
            input[write_i] = input[i];
            write_i++;
        }
    }

    // Set zero-termination
    input[write_i] = '\0';
}

/* Writes the data to a .dat file so GNUplot can plot it later. */
void write_costs(size_t n_iterations, double* costs) {
    FILE* dat = fopen("./nn_costs.dat", "w");
    fprintf(dat, "# Iteration / Costs\n");
    for (size_t i = 0; i < n_iterations; i++) {
        fprintf(dat, "%ld %f\n", i, costs[i]);
    }
    fclose(dat);
}



/***** MAIN *****/

int main(int argc, char** argv) {
    // Check argument validity
    if (argc != 2) {
        printf("Usage: %s <path_to_digits_datafile>\n", argv[0]);
    }

    printf("\n*** NEURAL NETWORK training DIGITS ***\n\n");

    printf("Loading digit dataset \"%s\"...\n", argv[1]);
    
    // Try to open the file
    FILE* data = fopen(argv[1], "r");
    if (data == NULL) {
        fprintf(stderr, "ERROR: could not open file: %s\n", strerror(errno));
        exit(errno);
    }

    // The resulting linkedlist of matrices
    int n_samples = -1;
    int n_classes = -1;
    int matrix_i = -1;
    matrix* digits = NULL;
    matrix* classes = NULL;

    // Loop through all elements (numbers enclosed in ',')
    int num;
    int status = get_num(data, &num);
    int i;
    for (i = 0; status > 0; i++) {
        int i_row = (i - 2) % 65;
        if (i == 0) {
            // Fill the number of sampels in first so we can allocate the array
            n_samples = num;
            digits = create_empty_matrix(n_samples, 64);
        } else if (i == 1) {
            // Fill in the number of classes for the output layer
            n_classes = num;
            classes = create_empty_matrix(n_samples, n_classes);
            // Fill it with zeros
            for (size_t i = 0; i < classes->rows * classes->cols; i++) {
                classes->data[i] = 0;
            }
        } else if (i_row == 0) {
            // First every 65: note the class
            matrix_i++;
            if (num >= n_classes) {
                destroy_matrix(digits);
                destroy_matrix(classes);
                fclose(data);
                fprintf(stderr, "ERROR: (%u:%u): given class is too high (%d > %d)\n",
                        row, col, num, n_classes - 1);
                return -1;
            }
            classes->data[matrix_i * classes->cols + num] = 1;
        } else {
            // One of the 64 datapoints: fill in the matrix
            if (num > 16) {
                destroy_matrix(digits);
                destroy_matrix(classes);
                fclose(data);
                fprintf(stderr, "ERROR: (%u:%u): given datapoint is too high for pixel value (%d > 16)\n",
                        row, col, num);
                return -1;
            }
            digits->data[matrix_i * digits->cols + (i_row - 1)] = num;
        }
        
        status = get_num(data, &num);
    }
    // Stop if an error occured
    if (status < 0) {
        destroy_matrix(digits);
        destroy_matrix(classes);
        fclose(data);
        return -1;
    }
    
    // Make sure n_samples and n_classes are set
    if (n_samples == -1) {
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): number of samples not specified\n",
                row, col);
        return -1;
    }
    if (n_classes == -1) {
        destroy_matrix(digits);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): number of classes not specified\n",
                row, col);
        return -1;
    }

    // Throw a complaint about too few data if that's the case
    if (matrix_i != n_samples - 1) {
        destroy_matrix(digits);
        destroy_matrix(classes);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): not enough samples provided (got %d, expected %d)\n",
                row, col, matrix_i + 1, n_samples);
        return -1;
    } else if ((i - 2) % 65 != 0) {
        destroy_matrix(digits);
        destroy_matrix(classes);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): not enough datapoints provided for sample (got %d, expected 64)\n",
                row, col, (i - 2) % 65);
        return -1;
    }

    printf("Done loading (loaded %d samples)\n\n", n_samples);

    printf("Training network...\n");

    // Create training and testing subsets of the digits and classes matrix
    printf("  Splitting test and train sets...\n");
    int training_size = n_samples * TRAIN_RATIO;
    matrix* digits_train = subset_matrix(digits, 0, training_size, 0, 64);
    matrix* digits_test = subset_matrix(digits, training_size + 1, n_samples, 0, 64);
    matrix* classes_train = subset_matrix(classes, 0, training_size, 0, n_classes);
    matrix* classes_test = subset_matrix(classes, training_size + 1, n_samples, 0, n_classes);

    // Create a new neural network
    printf("  Initializing Neural Network...\n");
    size_t hidden_layer_nodes[] = {20};
    neural_net* nn = create_nn(64, 1, hidden_layer_nodes, n_classes);

    // Train the neural network for ITERATIONS iterations
    printf("  Training...\n");
    matrix* costs = nn_train_costs(nn, digits_train, classes_train, TRAIN_ETA, TRAIN_ITERATIONS, sigmoid, dydx_sigmoid);
    printf("  Writing costs...\n\n");
    // Write the costs for plotting
    write_costs(costs->cols, costs->data);

    printf("Validating network...\n");

    // Test the network and report the accuracy
    matrix* outputs = nn_activate(nn, digits_test, sigmoid);
    nn_flatten_results(outputs);
    int correct = 0;
    for (size_t y = 0; y < outputs->rows; y++) {
        bool error = false;
        for (size_t x = 0; x < outputs->cols; x++) {
            error = error || ((int) outputs->data[y * outputs->cols + x] != (int) classes_test->data[y * classes_test->cols + x]);
        }
        correct += !error ? 1 : 0;
    }

    printf("Network accuracy: %f\n\n", (correct / (double)outputs->rows));

    // Cleanup
    printf("Cleaning up...\n");
    destroy_matrix(digits);
    destroy_matrix(classes);
    destroy_matrix(digits_train);
    destroy_matrix(digits_test);
    destroy_matrix(classes_train);
    destroy_matrix(classes_test);
    destroy_matrix(outputs);
    destroy_matrix(costs);
    destroy_nn(nn);
    fclose(data);

    printf("Done.\n\n");
}
