/* MATRIX.c
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:37
 * Last edited:
 *   16/04/2020, 23:16:07
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains the necessary matrix operations for a Neural
 *   Network.
**/

#include <stdio.h>

#include "Matrix.h"


matrix* create_empty_matrix(size_t rows, size_t cols) {
    // Create a new struct and allocate the pointer to the data
    matrix* to_ret = malloc(sizeof(matrix));
    to_ret->rows = rows;
    to_ret->cols = cols;
    to_ret->data = malloc(rows * cols * sizeof(double));

    // Return it
    return to_ret;
}

matrix* create_matrix(double* data, size_t rows, size_t cols) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(rows, cols);

    // Copy the data from the given pointer
    for (size_t i = 0; i < rows * cols; i++) {
        to_ret->data[i] = data[i];
    }

    // Return
    return to_ret;
}

void destroy_matrix(matrix* m) {
    // Free the internal data, then the struct
    free(m->data);
    free(m);
}



matrix* matrix_transpose(matrix* m) {
    // Create a new matrix with inverted size
    matrix* to_ret = create_empty_matrix(m->cols, m->rows);

    // Loop and put 'em there
    for (size_t y = 0; y < m->rows; y++) {
        for (size_t x = 0; x < m->cols; x++) {
            to_ret->data[(x * m->cols) + y] = m->data[(y * m->rows) + x];
        }
    }

    // Return
    return to_ret;
}

matrix* matrix_add(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf("ERROR: matrix_add: matrix m1 (%dx%d) and m2 (%dx%d) do not have the same sizes",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Create a new matrix, copy the the addition for each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] + m2->data[i];
    }
    return to_ret;
}
matrix* matrix_add_inplace(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf("ERROR: matrix_add_inplace: matrix m1 (%dx%d) and m2 (%dx%d) do not have the same sizes",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Add each element of m1 to m2, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] += m2->data[i];
    }
    return m1;
}

matrix* matrix_mul_s(matrix* m1, double s) {
    // Create a new matrix, copy the the multiplication of each element of m1 and s and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = s * m1->data[i];
    }
    return to_ret;
}
matrix* matrix_mul_s_inplace(matrix* m1, double s) {
    // Multiply each element of m1 with s, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] *= s;
    }
    return m1;
}

matrix* matrix_matmul(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->cols != m2->rows) {
        fprintf("ERROR: matrix_matmul: matrix m1 (%dx%d) and m2 (%dx%d) have illegal sizes for matrix multiplication",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Perform the matrix multiplication and store the result in to_ret, then return
    matrix* to_ret = create_empty_matrix(m1->rows, m2->cols);
    for (size_t y = 0; y < m1->rows; y++) {
        for (size_t x = 0; x < m2->cols; x++) {
            double sum = 0;
            for (size_t i = 0; i < m1->cols; i++) {
                sum += m1->data[y * m1->rows + i] * m2->data[i * m1->rows + x];
            }
            to_ret->data[y * m1->rows + x] = sum;
        }
    }
    return to_ret;
}

matrix* matrix_mul(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf("ERROR: matrix_mul: matrix m1 (%dx%d) and m2 (%dx%d) do not have the same sizes",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Create a new matrix, copy the the addition for each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] * m2->data[i];
    }
    return to_ret;
}
matrix* matrix_mul_inplace(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf("ERROR: matrix_mul_inplace: matrix m1 (%dx%d) and m2 (%dx%d) do not have the same sizes",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Multiply each element of m1 with m2, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] *= m2->data[i];
    }
    return m1;
}

matrix* matrix_tensor(matrix* m1, matrix* m2) {
    // Create a matrix to return
    size_t w = m1->cols * m2->cols;
    size_t h = m1->rows * m2->rows;
    matrix* to_ret = create_empty_matrix;

    // Loop through the m1 matrix first
    for (size_t y1 = 0; y1 < m1->rows; y1++) {
        for (size_t x1 = 0; x1 < m1->cols; x1++) {
            size_t x = x1 * m2->cols;
            size_t y = y1 * m2->rows;

            // Copy every element of m2 with this element of m1 and store it in the proper location
            double value = m1->data[y1 * m1->rows + x1];
            for (size_t y2 = 0; y2 < m2->rows; y2++) {
                for (size_t x2 = 0; x2 < m2->cols; x2++) {
                    to_ret->data[(y + y2) * h + (x + x2)] = value * m2->data[y2 * m2->rows + x2];
                }
            }
        }
    }
    return to_ret;
}

matrix* matrix_concat_h(matrix* m1, matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf("ERROR: matrix_concat: matrix m1 (%dx%d) and m2 (%dx%d) do not have the same number of rows",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols + m2->cols);
    for (size_t y = 0; y < m1->rows; y++) {
        for (size_t x = 0; x < m1->cols; x++) {
            to_ret->data[y * m1->rows + x] = m1->data[y * m1->rows + x];
        }
        for (size_t x = 0; x < m2->cols; x++) {
            to_ret->data[y * m1->rows + m1->cols + x] = m2->data[y * m1->rows + x];
        }
    }
    return to_ret;
}