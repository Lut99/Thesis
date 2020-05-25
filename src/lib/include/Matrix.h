/* MATRIX.h
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:54
 * Last edited:
 *   5/25/2020, 10:07:06 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains the necessary matrix operations for a Neural
 *   Network. This is the header file.
**/

#ifndef _MATRIX_H
#define _MATRIX_H

#include <stddef.h>
#include <stdbool.h>

#include "Scratchpad.h"

/* The struct that contains all data for a matrix, i.e., the size and a matrix. */
typedef struct MATRIX {
    size_t rows;
    size_t cols;
    size_t size;
    double* data;
} matrix;



/* Expands to an expression that indices the data in a multi-dimensional matrix. */
#define INDEX(MATRIX, ROWS, COLS) (MATRIX->data[(ROWS) * MATRIX->cols + (COLS)])



/***** NORMAL MEMORY MANAGEMENT *****/

/* Creates a new matrix object with the given size. Values are left unitialised, and note that this object will have to be destroyed. If it fails to allocate, prints to stderr and returns NULL. */
matrix* create_empty_matrix(size_t rows, size_t cols);
/* Creates a new matrix object with the given size as number of rows and one column. Values are left unitialised, and note that this object will have to be destroyed. If it fails to allocate, prints to stderr and returns NULL. */
matrix* create_empty_vector(size_t size);

/* Creates a new matrix object with the given size. Values are copied from given data array, and note that this object will have to be destroyed. If it fails to allocate, prints to stderr and returns NULL. */
matrix* create_matrix(size_t rows, size_t cols, const double* data);
/* Creates a new matrix object with the given size as number of rows and one column. Values are copied from given data array, and note that this object will have to be destroyed. If it fails to allocate, prints to stderr and returns NULL. */
matrix* create_vector(size_t size, const double* data);

/* Creates a new matrix object with the given size. The matrix will simply reference to the given data, i.e., does not copy it. Note that the matrix object itself will be destroyed (which will not destroy the referenced data object). If it fails to allocate, prints to stderr and returns NULL. */
matrix* link_matrix(size_t rows, size_t cols, double* data);
/* Creates a new matrix object with the given size as number of rows and one column. The matrix will simply reference to the given data, i.e., does not copy it. Note that the matrix object itself will be destroyed (which will not destroy the referenced data object). If it fails to allocate, prints to stderr and returns NULL. */
matrix* link_vector(size_t size, double* data);

/* Copies the data from the source matrix into the target matrix, and returns the target matrix. If the sizes of the matrices are not equal, prints to stderr and returns NULL. */
matrix* copy_matrix(matrix* target, const matrix* source);
/* Copies the data from the given matrix into a new matrix, which has to be deallocated later on. If it fails to allocate, prints to stderr and returns NULL. */
matrix* copy_matrix_new(const matrix* m);

/* Returns a new matrix which references the data range in given matrix. Note that this matrix has to be deallocated. Prints to stderr and return NULL is the given indices are invalid. */
matrix* subset_matrix(const matrix* m, size_t row_min, size_t row_max, size_t col_min, size_t col_max);

/* Destroys given matrix. Note that its equal to calling free(m) on it. */
void destroy_matrix(matrix* m);


/***** DEBUG TOOLS *****/

/* Prints a matrix to stdout. */
void matrix_print(matrix* m);

/* Checks if two matrices are the same and, if not, prints them both out with an error. */
bool matrix_equals(matrix* m1, matrix* m2);

#endif