/* ARRAY.c
 *   by Tim Müller
 *
 * Created:
 *   28/04/2020, 19:36:38
 * Last edited:
 *   5/24/2020, 3:17:26 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Defines a very simple array class that enhances usage of basic C
 *   arrays.
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Array.h"


/***** MEMORY MANAGEMENT *****/



array* create_empty_array(size_t size) {
    // Allocate the struct
    array* to_ret = (array*) malloc(sizeof(array) + size * sizeof(double));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_empty_array: could not allocate memory (%lu bytes).\n",
                sizeof(array) + size * sizeof(double));
        return NULL;
    }

    // Set the size and the data pointer
    to_ret->size = size;
    to_ret->d = (double*) (((char*) to_ret) + sizeof(array));

    // Return
    return to_ret;
}

array* create_array(size_t size, double* data) {
    // Create an empty array
    array* to_ret = create_empty_array(size);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_array: empty array creation failed.\n");
        return NULL;
    }

    // Fill it with the data
    for (size_t i = 0; i < to_ret->size; i++) {
        to_ret->d[i] = data[i];
    }

    // Return
    return to_ret;
}

array* create_linked_array(size_t size, double* data) {
    // Create an array without place for extra data
    array* to_ret = (array*) malloc(sizeof(array));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_linked_array: could not allocate array struct (%lu bytes).\n",
                sizeof(array));
        return NULL;
    }

    // Set the size and the data pointer
    to_ret->size = size;
    to_ret->d = data;

    // Return
    return to_ret;
}



array* initialize_array(size_t size, array* a) {
    // Set the size and linked status
    a->size = size;

    // Set the data pointer to the continued value
    a->d = (double*) (((char*) a) + a->size);

    return a;
}



array* fill_array(array* a, const double* data) {
    // Copy all the data and return
    for (size_t i = 0; i < a->size; i++) {
        a->d[i] = data[i];
    }
    return a;
}



array* copy_array(array* target, const array* source) {
    // Throw an error if not equally sized
    if (target->size != source->size) {
        fprintf(stderr, "ERROR: copy_array: target and source array have different lengths (%lu vs %lu).\n",
                target->size,
                source->size);
        return NULL;
    }

    // Copy all elements from one to the other
    fill_array(target, source->d);

    // Done, return
    return target;
}

array* copy_create_array(const array* a) {
    // Create a new array with the same size
    array* to_ret = create_empty_array(a->size);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: copy_create_array: empty array creation failed.\n");
        return NULL;
    }
    
    // Use copy_array to copy the data
    if (copy_array(to_ret, a) == NULL) {
        fprintf(stderr, "ERROR: copy_create_array: copying data failed.\n");
        return NULL;
    }

    // Return
    return to_ret;
}



void destroy_array(array* a) {
    // Due to the genius of making the array in one memory space, we can simply call free
    free(a);
}



/***** USEFUL FUNCTIONS *****/

double array_sum(const array* a) {
    double sum = 0;
    for (size_t i = 0; i < a->size; i++) {
        sum += a->d[i];
    }
    return sum;
}

double array_max(const array* a) {
    double max = -INFINITY;
    for (size_t i = 0; i < a->size; i++) {
        if (a->d[i] > max) {
            max = a->d[i];
        }
    }
    return max;
}



/***** DEBUG FUNCTIONS *****/

void array_write(FILE* handle, const array* a) {
    fprintf(handle, "[");
    for (size_t i = 0; i < a->size; i++) {
        if (i > 0) {
            fprintf(handle, ", ");
        }
        fprintf(handle, "%.2f", a->d[i]);
    }
    fprintf(handle, "]");
}

void array_print(FILE* handle, const array* a) {
    fprintf(handle, "[");
    for (size_t i = 0; i < a->size; i++) {
        if (i > 0) {
            fprintf(handle, ", ");
        }
        fprintf(handle, "%.2f", a->d[i]);
    }
    fprintf(handle, "]\n");
}

bool array_equals(const array* a1, const array* a2) {
    // Check if the sizes are equal
    if (a1->size != a2->size) {
        return false;
    }

    // Check each element
    for (size_t i = 0; i < a1->size; i++) {
        if (a1->d[i] != a2->d[i]) {
            return false;
        }
    }

    // Succes
    return true;
}

bool array_equals2(const array* a, const double* data) {
    // Check each element
    for (size_t i = 0; i < a->size; i++) {
        if (a->d[i] != data[i]) {
            return false;
        }
    }

    // Succes
    return true;
}
