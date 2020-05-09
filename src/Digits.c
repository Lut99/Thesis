/* DIGITS.c
 *   by Tim Müller
 *
 * Created:
 *   21/04/2020, 11:46:37
 * Last edited:
 *   09/05/2020, 14:03:49
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
#include <time.h>

#include "Functions.h"
#include "NeuralNetwork.h"

/* If left uncommented, reports and writes the costs. */
// #define PLOT 1
/* Percentage of data that will be used for training, the rest is for testing. */
#define TRAIN_RATIO 0.8
/* Number of iterations that the neural network will be trained on. */
#define TRAIN_ITERATIONS 30000
/* Learning rate of the Neural Network. */
#define TRAIN_ETA 0.001

static unsigned int row = 1;
static unsigned int col = 1;


/***** TOOLS *****/

/* Deallocates a list of arrays, up to the number of samples given. */
void destroy_array_list(size_t last_sample, array** list) {
    for (size_t i = 0; i <= last_sample; i++) {
        destroy_array(list[i]);
    }
    free(list);
}

/* Fetches a number from the file and returns it in the given num argument. If something goes wrong, prints an error and returns -1. If EOF is reached, it return 0. Otherwise, 1 is returned, indicating that more is to be read. */
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

/* Cleans a given string by removing all spaces and # and everything after (as comment). This is done in-place, so after this operation the string will be shorter or the same size as it was before. */
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
void write_costs(array* costs) {
    FILE* dat = fopen("./nn_costs.dat", "w");
    fprintf(dat, "# Iteration / Costs\n");
    for (size_t i = 0; i < costs->size; i++) {
        fprintf(dat, "%ld %f\n", i, costs->d[i]);
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
    int sample = -1;
    array** digits = NULL;
    array** classes = NULL;

    // Loop through all elements (numbers enclosed in ',')
    int num;
    int status = get_num(data, &num);
    int i;
    for (i = 0; status > 0; i++) {
        int i_row = (i - 2) % 65;
        if (i == 0) {
            // Fill the number of samples first so we can allocate the array
            n_samples = num;
            digits = malloc(sizeof(array*) * n_samples);
            classes = malloc(sizeof(array*) * n_samples);
        } else if (i == 1) {
            // Fill in the number of classes for the output layer
            n_classes = num;
        } else if (i_row == 0) {
            // First every 65: new sample
            sample++;
            // Make sure the num is within range
            if (num >= n_classes) {
                destroy_array_list(sample, digits);
                destroy_array_list(sample, classes);
                fclose(data);
                fprintf(stderr, "ERROR: (%u:%u): given class is too high (%d > %d)\n",
                        row, col, num, n_classes - 1);
                return -1;
            }
            // Create an empty array in the digits class
            digits[sample] = create_empty_array(64);

            // Allocate an output array and set the correct index to 1
            classes[sample] = create_empty_array(n_classes);
            for (int i = 0; i < n_classes; i++) {
                classes[sample]->d[i] = i == num ? 1.0 : 0.0;
            }
        } else {
            // One of the 64 datapoints: fill in the matrix
            if (num > 16) {
                destroy_array_list(sample, digits);
                destroy_array_list(sample, classes);
                fclose(data);
                fprintf(stderr, "ERROR: (%u:%u): given datapoint is too high for pixel value (%d > 16)\n",
                        row, col, num);
                return -1;
            }
            digits[sample]->d[i_row - 1] = num;
        }
        
        status = get_num(data, &num);
    }
    // Stop if an error occured
    if (status < 0) {
        destroy_array_list(sample, digits);
        destroy_array_list(sample, classes);
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
        destroy_array_list(-1, digits);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): number of classes not specified\n",
                row, col);
        return -1;
    }

    // Throw a complaint about too few data if that's the case
    if (sample != n_samples - 1) {
        destroy_array_list(sample, digits);
        destroy_array_list(sample, classes);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): not enough samples provided (got %d, expected %d)\n",
                row, col, sample + 1, n_samples);
        return -1;
    } else if ((i - 2) % 65 != 0) {
        destroy_array_list(sample, digits);
        destroy_array_list(sample, classes);
        fclose(data);
        fprintf(stderr, "ERROR: (%u:%u): not enough datapoints provided for last sample (got %d, expected 64)\n",
                row, col, (i - 2) % 65);
        return -1;
    }

    printf("Done loading (loaded %d samples)\n\n", n_samples);

    printf("Training network...\n");

    // Create training and testing subsets of the digits and classes matrix
    printf("  Splitting test and train sets...\n");
    size_t training_size = n_samples * TRAIN_RATIO;
    size_t testing_size = n_samples - training_size;
    array** digits_train = digits;
    array** digits_test = digits + training_size;
    array** classes_train = classes;
    array** classes_test = classes + training_size;

    // Create a new neural network
    printf("  Initializing Neural Network...\n");
    size_t hidden_layer_nodes[] = {20};
    neural_net* nn = create_nn(64, 1, hidden_layer_nodes, n_classes);

    // Train the neural network for ITERATIONS iterations
    #ifdef PLOT
    printf("  Training...\n");
    array* costs = nn_train_costs(nn, training_size, digits_train, classes_train, TRAIN_ETA, TRAIN_ITERATIONS, sigmoid, dydx_sigmoid);
    printf("  Writing costs...\n\n");
    // Write the costs for plotting
    write_costs(costs);
    #else
    printf("  Training...\n");
    time_t start_ms = time(NULL);
    clock_t start = clock();
    nn_train(nn, training_size, digits_train, classes_train, TRAIN_ETA, TRAIN_ITERATIONS, sigmoid, dydx_sigmoid);
    clock_t end = clock();
    time_t end_ms = time(NULL);
    printf("  Done (time taken: %ld seconds / CPU time taken: %f seconds)\n", end_ms - start_ms, (end - start) / (double) CLOCKS_PER_SEC);
    #endif

    printf("Validating network...\n");

    // Test the network
    array** outputs = malloc(sizeof(array*) * testing_size);
    for (size_t i = 0; i < testing_size; i++) {
        outputs[i] = create_empty_array(n_classes);
    }
    nn_forward(nn, testing_size, outputs, digits_test, sigmoid);

    // Flatten the results
    flatten_output(testing_size, outputs);

    // Compute the accuracy
    double accuracy = compute_accuracy(testing_size, outputs, classes_test);
    printf("Network accuracy: %f\n\n", accuracy);
    
    // Cleanup
    printf("Cleaning up...\n");
    destroy_array_list(sample, digits);
    destroy_array_list(sample, classes);
    destroy_array_list(testing_size - 1, outputs);
    #ifdef PLOT
    destroy_array(costs);
    #endif
    destroy_nn(nn);
    fclose(data);

    printf("Done.\n\n");
}
