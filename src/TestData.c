/* TEST DATA.c
 *   by Anonymous
 *
 * Created:
 *   6/2/2020, 3:40:16 PM
 * Last edited:
 *   6/2/2020, 5:09:39 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file generates test sets of desired size and content.
 *   Additionally, the testset also provides functionality to easily tweak
 *   other NeuralNetwork parameters, such as the number of epochs, the
 *   learning rate and the network size.
**/

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include "NeuralNetwork.h"
#include "Array.h"


/***** HELPER FUNCTIONS *****/

/* Cleans given list of pointers, also free'ing the pointers (except when those pointers are NULL).
 *   @param length length of the given list
 *   @param list the list itself
 */
void clean(size_t length, double** list) {
    for (size_t l = 0; l < length; l++) {
        if (list[l] != NULL) {
            free(list[l]);
        }
    }
    free(list);
}

/* Prints given 2D array of pointers to the giben file
 *   @param file file to print to
 *   @param rows number of rows
 *   @param cols number of columns
 *   @param data the list itself
 */
void fprint_ptrlist(FILE* file, size_t rows, size_t cols, double** arr) {
    char buffer[128];
    for (size_t y = 0; y < rows; y++) {
        double* row = arr[y];
        fprintf(file, "[");
        for (size_t x = 0; x < cols; x++) {
            if (x > 0) { fprintf(file, ", "); }

            // Create a string from the value
            sprintf(buffer, "%.3f", row[x]);

            // Check the length
            size_t len = strlen(buffer);
            // Pad some spaces
            for (size_t i = len; i < 8; i++) {
                fprintf(file, " ");
            }

            // Print the string
            fprintf(file, "%s", buffer);
        }
        fprintf(file, "]\n");
    }
}

/* Prints given list of classes by simply printing the indices of the highest value as a 1D-list.
 *   @param file file to print to
 *   @param n_samples number of samples
 *   @param n_classes number of classes
 *   @param classes the list of classes itself
 */
void fprint_ptrlist(FILE* file, size_t rows, size_t cols, double** arr) {
    char buffer[128];
    for (size_t y = 0; y < rows; y++) {
        double* row = arr[y];
        fprintf(file, "[");
        for (size_t x = 0; x < cols; x++) {
            if (x > 0) { fprintf(file, ", "); }

            // Create a string from the value
            sprintf(buffer, "%.3f", row[x]);

            // Check the length
            size_t len = strlen(buffer);
            // Pad some spaces
            for (size_t i = len; i < 8; i++) {
                fprintf(file, " ");
            }

            // Print the string
            fprintf(file, "%s", buffer);
        }
        fprintf(file, "]\n");
    }
}



/***** DATASET GENERATORS *****/

/* Generates a dataset with random doubles in the given range. Additionally, each element is given a random class, also in the specified range.
 *   @param dataset the resulting pointer that is allocated by the function which will point to the 2D-array of the generated datapoints
 *   @param classes the resulting pointer that is allocated by the function which will point to the 2D-array of the randomly assigned classes
 *   @param n_samples desired number of samples in the dataset
 *   @param sample_size desired number of doubles for every sample
 *   @param data_min lower bound (inclusive) of the random range of values
 *   @param data_max upper bound (exclusive) of the random range of values
 *   @param n_classes number of classes for this dataset
 */
void generate_random(double*** dataset, double*** classes, size_t n_samples, size_t sample_size, double data_min, double data_max, size_t n_classes) {
    // Seed the random
    srand(time(NULL));

    // First, malloc the datasets and classes main lists
    *dataset = malloc(sizeof(double*) * n_samples);
    *classes = malloc(sizeof(double*) * n_samples);

    // Next, fill 'em for every sample
    for (size_t s = 0; s < n_samples; s++) {
        (*dataset)[s] = malloc(sizeof(double) * sample_size);
        (*classes)[s] = malloc(sizeof(double) * n_classes);

        // First, fill the data
        for (size_t i = 0; i < sample_size; i++) {
            (*dataset)[s][i] = ((double) rand() / RAND_MAX) * (data_max - data_min) + data_min;
        }

        // Next, assign a random class
        size_t class = rand() % n_classes;
        for (size_t i = 0; i < n_classes; i++) {
            (*classes)[s][i] = i == class ? 1.0 : 0.0;
        }
    }
}


/***** MAIN *****/
int main(int argc, char** argv) {
    // Generate a random data
    double** dataset;
    double** classes;
    generate_random(&dataset, &classes, 10, 10, -3, 3, 5);
    
    printf("\nDataset:\n");
    fprint_ptrlist(stdout, 10, 10, dataset);
    printf("\nClasses:\n");
    fprint_ptrlist(stdout, 10, 5, classes);
    printf("\nClasses (numerical): ");
    fprint_classes(stdout, 10, 5, classes);
    printf("\n\n");

    // Clean 'em
    clean(10, dataset);
    clean(10, classes);

    return 0;
}
