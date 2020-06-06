/* TEST DATA.c
 *   by Anonymous
 *
 * Created:
 *   6/2/2020, 3:40:16 PM
 * Last edited:
 *   6/6/2020, 3:20:53 PM
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
#include <sys/time.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <limits.h>
#include <math.h>

#include "NeuralNetwork.h"
#include "Array.h"


/***** DEFAULT VALUES *****/

/* Converts the expanded macros to string. */
#define STR_IMPL_(x) #x
/* Is able to convert macros to their string counterpart. */
#define STR(x) STR_IMPL_(x)

#define DEFAULT_EPOCHS 20000
#define DEFAULT_LEARNING_RATE 0.005
#define DEFAULT_N_SAMPLES 10000
#define DEFAULT_SAMPLE_SIZE 64
#define DEFAULT_N_CLASSES 10
#define DEFAULT_N_HIDDEN_LAYERS 1
static size_t DEFAULT_NODES_PER_HIDDEN_LAYER[DEFAULT_N_HIDDEN_LAYERS] = {20};
#define DEFAULT_DATA_MIN -3.0
#define DEFAULT_DATA_MAX 3.0


/***** STRUCT DEFINITIONS *****/

/* The Options struct stores all options. */
typedef struct OPTIONS {
    unsigned int epochs;
    double learning_rate;

    size_t n_samples;
    size_t sample_size;
    size_t n_classes;

    size_t n_hidden_layers;
    size_t* nodes_per_hidden_layer;

    double data_min;
    double data_max;
} options;


/***** HELPER FUNCTIONS *****/

/* Cleans given list of pointers, also free'ing the pointers (except when those pointers are NULL).
 *
 * Parameters:
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

/* Prints given 1D array to the given file
 *
 * Parameters:
 *   @param file file to print to
 *   @param size number of elements in the list
 *   @param data the list itself
 */
void fprint_array(FILE* file, size_t size, size_t* data) {
    fprintf(file, "[");
    for (size_t i = 0; i < size; i++) {
        if (i > 0) { fprintf(file, ","); }
        fprintf(file, "%lu", data[i]);
    }
    fprintf(file, "]");
}

/* Prints given 2D array of pointers to the giben file
 *
 * Parameters:
 *   @param file file to print to
 *   @param rows number of rows
 *   @param cols number of columns
 *   @param data the list itself
 */
void fprint_ptrlist(FILE* file, size_t rows, size_t cols, double** data) {
    char buffer[128];
    for (size_t y = 0; y < rows; y++) {
        double* row = data[y];
        fprintf(file, "[");
        for (size_t x = 0; x < cols; x++) {
            if (x > 0) { fprintf(file, ","); }

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
 *
 * Parameters:
 *   @param file file to print to
 *   @param n_samples number of samples
 *   @param n_classes number of classes
 *   @param classes the list of classes itself
 */
void fprint_classes(FILE* file, size_t n_samples, size_t n_classes, double** classes) {
    fprintf(file, "[");
    for (size_t s = 0; s < n_samples; s++) {
        double* sample_output = classes[s];
        size_t highest_value = sample_output[0];
        size_t highest_index = 0;
        for (size_t n = 1; n < n_classes; n++) {
            if (sample_output[n] > highest_value) {
                highest_index = n;
                highest_value = sample_output[n];
            }
        }
        if (s > 0) { fprintf(file, ","); }
        fprintf(file, "%lu", highest_index);
    }
    fprintf(file, "]");
}

/* Converts given text to an unsigned long. Note that it prints an error message and exists the program whenever an error occurs.
 * 
 * Parameters:
 *   @param opt the option we are currently parsing. Used for errors.
 *   @param str the zero-terminated string to convert to an unsigned long
 */
unsigned long str_to_ulong(char opt, char* str) {
    // Because '-' is not picked up as illegal, scout for those first
    for (int i = 0; ; i++) {
        char c = str[i];
        if (c == '-') {
            fprintf(stderr, "ERROR: Option '%c': cannot convert \"%s\" to an unsigned long.\n", opt, str);
            exit(EXIT_FAILURE);
        } else if (c == '\0') {
            break;
        }
    }

    char* end;
    unsigned long to_ret = strtoul(str, &end, 10);
    
    // Error if we encountered an illegal character
    if (*end != '\0') {
        fprintf(stderr, "ERROR: Option '%c': cannot convert \"%s\" to an unsigned long.\n", opt, str);
        exit(EXIT_FAILURE);
    }

    // Also stop if we are out of range
    if (to_ret == ULONG_MAX && errno == ERANGE) {
        fprintf(stderr, "ERROR: Option '%c': value \"%s\" is out of range for unsigned long.\n", opt, str);
        exit(EXIT_FAILURE);
    }

    // Otherwise, return the value
    return to_ret;
}

/* Converts given text to a double. Note that it prints an error message and exists the program whenever an error occurs.
 * 
 * Parameters:
 *   @param opt the option we are currently parsing. Used for errors.
 *   @param str the zero-terminated string to convert to a double
 */
double str_to_double(char opt, char* str) {
    char* end;
    double to_ret = strtod(str, &end);
    
    // Error if we encountered an illegal character
    if (*end != '\0') {
        fprintf(stderr, "ERROR: Option '%c': cannot convert \"%s\" to an double.\n", opt, str);
        exit(EXIT_FAILURE);
    }

    // Also stop if we are out of range
    if ((to_ret == HUGE_VAL || to_ret == -HUGE_VAL) && errno == ERANGE) {
        fprintf(stderr, "ERROR: Option '%c': value \"%s\" is out of range for double.\n", opt, str);
        exit(EXIT_FAILURE);
    }

    // Otherwise, return the value
    return to_ret;
}

/* Converts given text to a list of unsigned longs. Note that it prints an error message and exists the program whenever an error occurs.
 * 
 * Parameters:
 *   @param n_hidden_layers the number of hidden layers, aka, the length of the list
 *   @param opt the option we are currently parsing. Used for errors.
 *   @param str the zero-terminated string to convert to a list
 */
size_t* str_to_list(size_t n_hidden_layers, char opt, char* str) {
    // If the number of hidden layers is 0, simply return NULL
    if (n_hidden_layers == 0) {
        return NULL;
    }

    // Declare the return list
    size_t* to_ret = malloc(sizeof(size_t) * n_hidden_layers);
    size_t to_ret_i = 0;

    // Create a buffer to store the individual numbers
    char buffer[512];
    int buffer_i = 0;

    // Loop through the string to parse
    char c = str[0];
    for (int i = 0; c != '\0'; i++) {
        // Make sure to break on commas
        if (c == ',') {
            // If we already hit our target, throw an error
            if (to_ret_i >= n_hidden_layers - 1) {
                free(to_ret);
                fprintf(stderr, "ERROR: Option '%c': Given list is too long (larger than n_hidden_layers, or %lu).\n", opt, n_hidden_layers);
                exit(EXIT_FAILURE);
            }

            // Zero-terminate the buffer and convert to size_t
            buffer[buffer_i] = '\0';
            to_ret[to_ret_i] = str_to_ulong(opt, buffer);

            // Reset the buffer_i, increment to_ret_i
            buffer_i = 0;
            to_ret_i++;
        } else {
            // Add the value to the buffer & increment
            buffer[buffer_i] = c;
            buffer_i++;
            if (buffer_i >= 512) {
                free(to_ret);
                fprintf(stderr, "ERROR: Option '%c': Buffer overflow for element %lu\n", opt, to_ret_i);
                exit(EXIT_FAILURE);
            }
        }

        c = str[i + 1];
    }

    // Make sure we seen enough elements
    if (to_ret_i < n_hidden_layers - 1) {
        free(to_ret);
        fprintf(stderr, "ERROR: Option '%c': Given list is too short (shorter than n_hidden_layers, or %lu).\n", opt, n_hidden_layers);
        exit(EXIT_FAILURE);
    }

    // Add the remaining buffer
    buffer[buffer_i] = '\0';
    to_ret[to_ret_i] = str_to_ulong(opt, buffer);

    // Otherwise, return the value
    return to_ret;
}

/* Prints a usage string to the given file.
 *
 * Parameters:
 *   @param file the file to write to
 *   @param name the name of the program that we are running
 */
void print_usage(FILE* file, char* name) {
    fprintf(file, "Usage: %s [-h] [-SsceH <ulong>] [-Ddl <float>] [-N <list>] [<var_args>]\n", name);
}

/* Prints a help string to the given file.
 *
 * Parameters:
 *   @param file the file to write to
 *   @param name the name of the program that we are running
 */
void print_help(FILE* file, char* name) {
    fprintf(file, "\n");
    fprintf(file, "Usage:\n    %s [-h] [-SsceH <ulong>] [-Ddl <float>] [-N <list>] [<var_args>]\n", name);

    fprintf(file, "\nDataset options:\n");
    fprintf(file, "  -S <ulong>\tThe number of samples generated in the dataset (default: " STR(DEFAULT_N_SAMPLES) ")\n");
    fprintf(file, "  -s <ulong>\tThe number of elements for each sample in the dataset (default: " STR(DEFAULT_SAMPLE_SIZE) ")\n");
    fprintf(file,          "\t\tNote that the first layer of the neural network has this many nodes.\n");
    fprintf(file, "  -c <ulong>\tThe number of classes that the generated dataset can take (default: " STR(DEFAULT_N_CLASSES) ")\n");
    fprintf(file,          "\t\tNote that the last layer of the neural network has this many nodes.\n");
    fprintf(file, "  -D <float>\tThe upperbound value (exclusive) for the random values in the dataset (default: " STR(DEFAULT_DATA_MAX) ")\n");
    fprintf(file, "  -d <float>\tThe lowerbound value (inclusive) for the random values in the dataset (default: " STR(DEFAULT_DATA_MIN) ")\n");
    
    fprintf(file, "\nNeural network options:\n");
    fprintf(file, "  -e <ulong>\tThe number of epochs for the neural network (default: " STR(DEFAULT_EPOCHS) ")\n");
    fprintf(file, "  -l <float>\tThe learning rate for the neural network (default: " STR(DEFAULT_LEARNING_RATE) ")\n");
    fprintf(file, "  -H <ulong>\tThe number of hidden layers in the neural network (default: " STR(DEFAULT_N_HIDDEN_LAYERS) ")\n");
    fprintf(file,           "\t\tNote: -H must be specified before -N to make it use the correct number of hidden layers.\n");
    fprintf(file, "  -N <list>\tThe number of nodes per hidden layer. Should be a comma-separated list (without whitespaces).\n");
    fprintf(file,          "\t\tNote that the length has to be equal to the number of hidden layers, and that for any number\n");
    fprintf(file,          "\t\tof hidden layers other than 1 this argument is not optional (default: ");
    for (size_t i = 0; i < DEFAULT_N_HIDDEN_LAYERS; i++) { if (i > 0) { fprintf(file, ","); } fprintf(file, "%lu", DEFAULT_NODES_PER_HIDDEN_LAYER[i]); }
    fprintf(file, ")\n");

    fprintf(file, "\nMiscellaneous options:\n");
    fprintf(file, "  -h\t\tPrints this help message.\n");
    fprintf(file, "  <var_args>\tAny number of optional positional arguments that are specific to the variations.\n");
    fprintf(file, "\n");
}

/* Parses the command line arguments and stores the result into the given options struct.
 * Note: it exits the program when the user entered something invalid, printing an
 *   appropriate error message.
 * 
 * Parameters:
 *   @param opts the options struct to store all results in
 *   @param argc the number of arguments passed to the program
 *   @param argv the arguments, as list of pointers, passed to the program
 */
void parse_arguments(options* opts, int argc, char** argv) {
    // Set as default first
    opts->epochs = DEFAULT_EPOCHS;
    opts->learning_rate = DEFAULT_LEARNING_RATE;
    opts->n_samples = DEFAULT_N_SAMPLES;
    opts->sample_size = DEFAULT_SAMPLE_SIZE;
    opts->n_classes = DEFAULT_N_CLASSES;
    opts->n_hidden_layers = DEFAULT_N_HIDDEN_LAYERS;
    opts->nodes_per_hidden_layer = DEFAULT_NODES_PER_HIDDEN_LAYER;
    opts->data_min = DEFAULT_DATA_MIN;
    opts->data_max = DEFAULT_DATA_MAX;

    // Then, go through all inputs
    int opt;
    while ((opt = getopt(argc, argv, ":S:s:c:D:d:e:l:H:N:h")) != -1) {
        switch(opt) {
            case '?':
                print_usage(stderr, argv[0]);
                fprintf(stderr, "\nERROR: Unknown option '%c'. Run '%s -h' to see a list of possible options.\n", optopt, argv[0]);
                exit(EXIT_FAILURE);
            case ':':
                print_usage(stderr, argv[0]);
                fprintf(stderr, "\nERROR: Missing value for option '%c'. Run '%s -h' to see a description.\n", optopt, argv[0]);
                exit(EXIT_FAILURE);
            case 'S':
                // Number of samples
                opts->n_samples = str_to_ulong(opt, optarg);
                break;
            case 's':
                // Sample size
                opts->sample_size = str_to_ulong(opt, optarg);
                break;
            case 'c':
                // Number of classes
                opts->n_classes = str_to_ulong(opt, optarg);
                break;
            case 'D':
                // Data max
                opts->data_max = str_to_double(opt, optarg);
                break;
            case 'd':
                // Number of samples
                opts->data_min = str_to_double(opt, optarg);
                break;
            case 'e':
                // Number of epochs
                opts->epochs = str_to_ulong(opt, optarg);
                break;
            case 'l':
                // Learning rate
                opts->learning_rate = str_to_double(opt, optarg);
                break;
            case 'H':
                // Number of hidden layers
                opts->n_hidden_layers = str_to_ulong(opt, optarg);
                break;
            case 'N':
                // Number of nodes per hidden layer
                opts->nodes_per_hidden_layer = str_to_list(opts->n_hidden_layers, opt, optarg);
                break;
            case 'h':
                print_help(stdout, argv[0]);
                exit(EXIT_SUCCESS);
        }
    }

    // Make sure the nodes_per_hidden_layer has changed if needed
    if (opts->n_hidden_layers != DEFAULT_N_HIDDEN_LAYERS &&
        opts->n_hidden_layers != 0 &&
        opts->nodes_per_hidden_layer == DEFAULT_NODES_PER_HIDDEN_LAYER)
    {
        fprintf(stderr, "ERROR: Changed number of hidden layers to %lu, but did not specify new list of nodes per layer.\n",
                opts->n_hidden_layers);
        exit(EXIT_FAILURE);
    }

    // For any other, positional arguments, pass those to the variations
    parse_opt_args(argc - optind, argv + optind);
}



/***** DATASET GENERATORS *****/

/* Generates a dataset with random doubles in the given range. Additionally, each element is given a random class, also in the specified range.
 *
 * Parameters:
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
    // The options
    options opts;

    // Parse the command line arguments
    parse_arguments(&opts, argc, argv);

    // Do the intro print
    #ifndef BENCHMARK
    printf("\n*** NEURAL NETWORK training RANDOM DATASET ***\n\n");

    printf("Dataset configuration:\n");
    printf(" - Number of samples       : %lu\n", opts.n_samples);
    printf(" - Size of each sample     : %lu\n", opts.sample_size);
    printf(" - Number of classes       : %lu\n", opts.n_classes);
    printf(" - Data range upperbound   : %.2f\n", opts.data_max);
    printf(" - Data range lowerbound   : %.2f\n", opts.data_min);
    printf("\n");

    printf("Neural network configuration:\n");
    print_opt_args();
    printf(" - Number of epochs        : %u\n", opts.epochs);
    printf(" - Learning rate (eta)     : %f\n", opts.learning_rate);
    printf(" - Number of hidden layers : %lu\n", opts.n_hidden_layers);
    printf(" - Nodes per hidden layer  : "); fprint_array(stdout, opts.n_hidden_layers, opts.nodes_per_hidden_layer); printf("\n");
    printf("\n");

    printf("Generating random data...");
    #endif

    // Generate a random data
    double** dataset;
    double** classes;
    generate_random(&dataset, &classes, opts.n_samples, opts.sample_size, opts.data_min, opts.data_max, opts.n_classes);

    #ifndef BENCHMARK
    printf(" Done\n");
    #endif

    #ifndef BENCHMARK
    printf("Creating neural network...");
    #endif
    neural_net* nn = create_nn(opts.sample_size, opts.n_hidden_layers, opts.nodes_per_hidden_layer, opts.n_classes);
    #ifndef BENCHMARK
    printf(" Done\n");
    #endif

    #ifndef BENCHMARK
    printf("Training neural network...");
    fflush(stdout);
    #endif

    // Declare some time structs
    struct timeval start_ms, end_ms;
    clock_t start, end;

    // Start recording the time
    start = clock();
    gettimeofday(&start_ms, NULL);

    // Run the training
    nn_train(nn, opts.n_samples, dataset, classes, opts.learning_rate, opts.epochs);

    // Stop recording the time
    gettimeofday(&end_ms, NULL);
    end = clock();

    // Print the results
    #ifdef BENCHMARK
    printf("%f,%f",
           ((end_ms.tv_sec - start_ms.tv_sec) * 1000000 + (end_ms.tv_usec - start_ms.tv_usec)) / 1000000.0,
           (end - start) / (double) CLOCKS_PER_SEC);
    #else
    printf(" Done (%f seconds, %f seconds CPU time)\n",
           ((end_ms.tv_sec - start_ms.tv_sec) * 1000000 + (end_ms.tv_usec - start_ms.tv_usec)) / 1000000.0,
           (end - start) / (double) CLOCKS_PER_SEC);
    #endif

    // Clean 'em
    #ifndef BENCHMARK
    printf("Cleaning up...");
    #endif

    clean(opts.n_samples, dataset);
    clean(opts.n_samples, classes);
    destroy_nn(nn);
    // Don't forget to clean the hidden list
    if (opts.nodes_per_hidden_layer != DEFAULT_NODES_PER_HIDDEN_LAYER && opts.nodes_per_hidden_layer != NULL) {
        free(opts.nodes_per_hidden_layer);
    }

    #ifndef BENCHMARK
    printf(" Done\n");

    printf("\nDone.\n\n");
    #endif

    return 0;
}
