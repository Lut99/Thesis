/* MATRIX.h
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:54
 * Last edited:
 *   20/04/2020, 16:46:28
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
/* Creates a new matrix object from given single-dimensional array. Note that the resulting matrix will always have one column, i.e., it's vertical. */
matrix* create_vector(size_t rows, const double data[rows]);

/* Copies given source matrix into the target matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* copy_matrix(matrix* target, const matrix* source);
/* Copies given matrix into a new matrix. */
matrix* copy_matrix_new(const matrix* m);

/* Destroys a given matrix object. */
void destroy_matrix(matrix* m);



/***** MATH OPERATIONS *****/

/* Transposes given matrix and returns the result as a new matrix. */
matrix* matrix_transpose(const matrix *m);

/* Adds a constant to given matrix and returns the result in a new matrix. */
matrix* matrix_add_c(const matrix *m1, double c);
/* Adds a constant to given matrix and returns the result in the given matrix. */
matrix* matrix_add_c_inplace(matrix* m1, double c);

/* Adds two matrices and returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_add(const matrix *m1, const matrix *m2);
/* Adds two matrices and returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_add_inplace(matrix* m1, const matrix *m2);

/* Subtracts given constant from given matrix (m1 - c). Returns the result in a new matrix. */
matrix* matrix_sub1_c(const matrix *m1, double c);
/* Subtracts given constant from given matrix (m1 - c). Returns the result in a new matrix. */
matrix* matrix_sub1_c_inplace(matrix *m1, double c);
/* Subtracts given matrix from given constant (c - m1). Returns the result in the given matrix. */
matrix* matrix_sub2_c(double c, const matrix *m1);
/* Subtracts given matrix from given constant (c - m1). Returns the result in the given matrix. */
matrix* matrix_sub2_c_inplace(double c, matrix *m1);

/* Subtracts the second matrix from the first and returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_sub(const matrix *m1, const matrix *m2);
/* Subtracts the second matrix from the first and returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_sub_inplace(matrix* m1, const matrix *m2);

/* Multiplies given matrix with a scaler and returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_c(const matrix *m1, double c);
/* Multiplies given matrix with a scaler and returns the result in the given matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_c_inplace(matrix* m1, double c);

/* Performs an element-wise multiplication on given matrices. Returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul(const matrix *m1, const matrix *m2);
/* Performs an element-wise multiplication on given matrices. Returns the result in the first matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_mul_inplace(matrix* m1, const matrix *m2);

/* Performs a matrix multiplication on given matrices. Returns the result in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_matmul(const matrix *m1, const matrix *m2);

/* Returns a new matrix containing the tensor product of the given two matrices. */
matrix* matrix_tensor(const matrix *m1, const matrix *m2);

/* Inverts all elements in given matrix, and returns the result in a new matrix. */
matrix* matrix_inv(const matrix* m1);
/* Inverts all elements in given matrix, and returns the result in the given matrix. */
matrix* matrix_inv_inplace(matrix* m1);

/* Takes the exponent of each element, e.g., x = e^x. Returns the result in a new matrix. */
matrix* matrix_exp(const matrix* m1);
/* Takes the exponent of each element, e.g., x = e^x. Returns the result in the given matrix. */
matrix* matrix_exp_inplace(matrix* m1);

/* Takes the natural logarithm of each element. Returns the result in a new matrix. */
matrix* matrix_ln(const matrix* m1);
/* Takes the natural logarithm of each element. Returns the result in the given matrix. */
matrix* matrix_ln_inplace(matrix* m1);

/* Squares each element in the matrix. Returns the result in a new matrix. */
matrix* matrix_square(const matrix* m1);
/* Squares each element in the matrix. Returns the result in the given matrix. */
matrix* matrix_square_inplace(matrix* m1);

/* Sums all elements in the matrix and returns the result. */
double matrix_sum(const matrix* m1);

/* Concatenates two matrices horizontally. The result is returned in a new matrix. Returns NULL and prints to stderr if the sizes are not correct. */
matrix* matrix_concat_h(const matrix *m1, const matrix *m2);


/***** DEBUG TOOLS *****/

/* Prints a matrix to stderr. */
void matrix_print(matrix* m);

/* Checks if two matrices are the same and, if not, prints them both out with an error. */
bool matrix_equals(matrix* m1, matrix* m2);

#endif