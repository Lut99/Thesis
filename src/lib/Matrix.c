/* MATRIX.c
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:37
 * Last edited:
 *   5/25/2020, 10:08:02 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains the necessary matrix operations for a Neural
 *   Network.
**/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Matrix.h"


/***** MEMORY MANAGEMENT *****/

matrix* create_empty_matrix(size_t rows, size_t cols) {
    // Compute the total size of the matrix
    size_t size = rows * cols;

    // Allocate enough data for the struct to include the data part
    matrix* to_ret = malloc(sizeof(matrix) + size * sizeof(double));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_empty_matrix: could not allocate memory (%lu bytes).\n",
                sizeof(sizeof(matrix) + size * sizeof(double)));
        return NULL;
    }

    // Set the parameters
    to_ret->size = size;
    to_ret->rows = rows;
    to_ret->cols = cols;

    // Set the data pointer to the second bit of the allocated memory
    to_ret->data = (double*) (((char*) to_ret) + sizeof(matrix));

    // Return it
    return to_ret;
}

matrix* create_empty_vector(size_t size) {
    // Allocate enough data for the struct to include the data part
    matrix* to_ret = malloc(sizeof(matrix) + size * sizeof(double));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_empty_vector: could not allocate memory (%lu bytes).\n",
                sizeof(sizeof(matrix) + size * sizeof(double)));
        return NULL;
    }

    // Set the parameters
    to_ret->size = size;
    to_ret->rows = size;
    to_ret->cols = 1;

    // Set the data pointer to the second bit of the allocated memory
    to_ret->data = (double*) (((char*) to_ret) + sizeof(matrix));

    // Return it
    return to_ret;
}

matrix* create_matrix(size_t rows, size_t cols, const double* data) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(rows, cols);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_matrix: empty matrix creation failed.\n");
        return NULL;
    }

    // Copy the data from the given pointer
    for (size_t i = 0; i < rows * cols; i++) {
        to_ret->data[i] = data[i];
    }

    // Return
    return to_ret;
}

matrix* create_vector(size_t size, const double* data) {
    // Create an empty matrix with the correct dimensions
    matrix* to_ret = create_empty_vector(size);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_vector: empty vector creation failed.\n");
        return NULL;
    }

    // Copy the data from the given pointer
    for (size_t i = 0; i < size; i++) {
        to_ret->data[i] = data[i];
    }

    // Return
    return to_ret;
}

matrix* initialize_empty_matrix(matrix* block, size_t rows, size_t cols) {
    // Set the parameters
    block->size = rows * cols;
    block->rows = rows;
    block->cols = cols;

    // Set the data pointer
    block->data = (double*)(((char*) block) + sizeof(matrix));

    // Return
    return block;
}

matrix* link_matrix(size_t rows, size_t cols, double* data) {
    // Allocate space for just the struct
    matrix* to_ret = malloc(sizeof(matrix));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: link_matrix: could not allocate memory (%lu bytes).\n",
                sizeof(sizeof(matrix)));
        return NULL;
    }

    // Set the parameters
    to_ret->size = rows * cols;
    to_ret->rows = rows;
    to_ret->cols = cols;

    // Link the data
    to_ret->data = data;

    // Return
    return to_ret;
}

matrix* link_vector(size_t size, double* data) {
    // Allocate space for just the struct
    matrix* to_ret = malloc(sizeof(matrix));
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: link_vector: could not allocate memory (%lu bytes).\n",
                sizeof(sizeof(matrix)));
        return NULL;
    }

    // Set the parameters
    to_ret->size = size;
    to_ret->rows = size;
    to_ret->cols = 1;

    // Link the data
    to_ret->data = data;

    // Return
    return to_ret;
}



matrix* copy_matrix(matrix* target, const matrix* source) {
    // Sanity check that the matrices are correctly sized
    if (target->rows != source->rows || target->cols != source->cols) {
        fprintf(stderr, "ERROR: copy_matrix: matrix target (%ldx%ld) and source (%ldx%ld) do not have the same sizes.\n",
                target->rows,
                target->cols,
                source->rows,
                source->cols);
        return NULL;
    }

    // Copy the data
    for (size_t i = 0; i < target->size; i++) {
        target->data[i] = source->data[i];
    }

    // Return the target for chaining
    return target;
}

matrix* copy_matrix_new(const matrix* m) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(m->rows, m->cols);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: copy_matrix_new: empty matrix creation failed.\n");
        return NULL;
    }

    // Copy the data
    if (copy_matrix(to_ret, m) == NULL) {
        fprintf(stderr, "ERROR: copy_matrix_new: data copying failed.\n");
        return NULL;
    }

    // Return
    return to_ret;
}



matrix* subset_matrix(const matrix* m, size_t row_min, size_t row_max, size_t col_min, size_t col_max) {
    // Make sure that everything is within bounds
    if (row_min > m->rows) { row_min = m->rows; }
    if (row_max > m->rows) { row_max = m->rows; }
    if (col_min > m->cols) { col_min = m->cols; }
    if (col_max > m->cols) { col_max = m->cols; }

    // Make sure that they aren't incorrectly ordered
    if (row_min > row_max) {
        fprintf(stderr, "ERROR: subset_matrix: row_min (%lu) cannot be larger than row_max (%lu)\n",
                row_min, row_max);
        return NULL;
    }
    if (col_min > col_max) {
        fprintf(stderr, "ERROR: subset_matrix: col_min (%lu) cannot be larger than col_max (%lu)\n",
                col_min, col_max);
        return NULL;
    }

    // Create a new matrix and copy the correct subset
    matrix* to_ret = create_empty_matrix(row_max - row_min, col_max - col_min);
    for (size_t y = row_min; y < row_max; y++) {
        for (size_t x = col_min; x < col_max; x++) {
            INDEX(to_ret, y - row_min, x - col_min) = INDEX(m, y, x);
        }
    }

    // Return
    return to_ret;
}



void destroy_matrix(matrix* m) {
    // Due to the genius of making the array in one memory space, we can simply call free
    free(m);
}





/***** DEBUG TOOLS *****/

void matrix_print(matrix* m) {
    // Early quit if there is nothing to print
    if (m->rows == 0 || m->cols == 0) {
        fprintf(stdout, "(empty)\n");
        return;
    }
    for (size_t y = 0; y < m->rows; y++) {
        fprintf(stdout, "[%7.2f", m->data[y * m->cols]);
        for (size_t x = 1; x < m->cols; x++) {
            fprintf(stdout, " %7.2f", m->data[y * m->cols + x]);
        }
        fprintf(stdout, "]\n");
    }
}

bool matrix_equals(matrix* m1, matrix* m2) {
    // Check if they are the same size
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        return false;
    }

    // Check each element
    for (size_t y = 0; y < m1->rows; y++) {
        for (size_t x = 0; x < m1->cols; x++) {
            if (m1->data[y * m1->cols + x] != m2->data[y * m2->cols + x]) {
                return false;
            }
        }
    }

    // Match!
    return true;
}
