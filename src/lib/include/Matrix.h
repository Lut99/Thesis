/* MATRIX.h
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:54
 * Last edited:
 *   4/19/2020, 12:01:31 AM
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

/* The struct that contains all data for a matrix, i.e., the size and a matrix. */
typedef struct MATRIX {
    size_t rows;
    size_t cols;
    double* data;
} matrix;



/***** MEMORY MANAGEMENT *****/

/* Creates a new matrix object with the given size. Values are uninitialised, and note that this object has to be deallocated. */
matrix* create_empty_matrix(size_t rows, size_t cols);

/* Creates a new matrix object from given multi-dimensional array. */
matrix* create_matrix(size_t rows, size_t cols, const double data[rows][cols]);

/* Copies given matrix into a new matrix. */
matrix* copy_matrix(const matrix* m);

/* Destroys a given matrix object. */
void destroy_matrix(matrix* m);



/***** MATH OPERATIONS *****/

/* Transposes given matrix and returns the result as a new matrix. */
matrix* matrix_transpose(const matrix *m);

/* Adds two matrices and returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_add(const matrix *m1, const matrix *m2);
/* Adds two matrices and returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_add_inplace(matrix* m1, const matrix *m2);

/* Multiplies given matrix with a scaler and returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_s(const matrix *m1, double s);
/* Multiplies given matrix with a scaler and returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_s_inplace(matrix* m1, double s);

/* Performs a matrix multiplication on given matrices. Returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_matmul(const matrix *m1, const matrix *m2);

/* Performs an element-wise multiplication on given matrices. Returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul(const matrix *m1, const matrix *m2);
/* Performs an element-wise multiplication on given matrices. Returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_inplace(matrix* m1, const matrix *m2);

/* Returns a new matrix containing the tensor product of the given two matrices. */
matrix* matrix_tensor(const matrix *m1, const matrix *m2);

/* Concatenates two matrices horizontally. The result is returned in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_concat_h(const matrix *m1, const matrix *m2);

#endif