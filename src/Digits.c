/* DIGITS.c
 *   by Tim Müller
 *
 * Created:
 *   21/04/2020, 11:46:37
 * Last edited:
 *   21/04/2020, 12:59:49
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file uses the NeuralNetwork class to try and predict the correct
 *   digit from a training set of hand-drawn digits.
**/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>

#include "NeuralNetwork.h"

#define BUFFER_SIZE 512


int main(int argc, char** argv) {
    // Check argument validity
    if (argc != 2) {
        printf("Usage: %s <path_to_digits_datafile>\n", argv[0]);
    }

    printf("\n*** NEURAL NETWORK training DIGITS ***\n\n");

    printf("Loading digit dataset...\n");
    
    // Try to open the file
    FILE* data = fopen(argv[1], "r");
    if (data == NULL) {
        fprintf(stderr, "Could not open file \"%s\": %s\n", argv[1], strerror(errno));
        exit(errno);
    }

    // Loop through all characters to parse the file
    matrix** digits;
    char buffer[BUFFER_SIZE];
    while (fgets(buffer, BUFFER_SIZE, data)) {
        // Loop through all characters
        for (int i = 0; i < BUFFER_SIZE; i++) {
            char c = buffer[i];
            
        }
    }

    if (!feof(data)) {
        fprintf(stderr, "Error reading file \"%s\": %s\n", argv[1], strerror(errno));
        exit(errno);
    }

    printf("Parsed %ld characters.\n", chars);
}
