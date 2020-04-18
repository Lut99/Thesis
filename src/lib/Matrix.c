/* MATRIX.c
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:37
 * Last edited:
 *   4/19/2020, 12:01:30 AM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains the necessary matrix operations for a Neural
 *   Network.
**/

#include <stdlib.h>
#include <stdio.h>

#include "Matrix.h"


/***** MEMORY MANAGEMENT *****/

matrix* create_empty_matrix(size_t rows, size_t cols) {
    // Create a new struct and allocate the pointer to the data
    matrix* to_ret = malloc(sizeof(matrix));
    to_ret->rows = rows;
    to_ret->cols = cols;
    to_ret->data = malloc(rows * cols * sizeof(double));

    // Return it
    return to_ret;
}

matrix* create_matrix(size_t rows, size_t cols, const double data[rows][cols]) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(rows, cols);

    // Copy the data from the given pointer
    for (size_t y = 0; y < rows; y++) {
        for (size_t x = 0; x < cols; x++) {
            to_ret->data[y * cols + x] = data[y][x];
        }
    }

    // Return
    return to_ret;
}

matrix* copy_matrix(const matrix* m) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(m->rows, m->cols);

    // Copy the data
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        to_ret->data[i] = m->data[i];
    }

    // Return
    return to_ret;
}

void destroy_matrix(matrix* m) {
    // Free the internal data, then the struct
    free(m->data);
    free(m);
}



/***** MATH *****/

matrix* matrix_transpose(const matrix* m) {
    // Create a new matrix with inverted size
    matrix* to_ret = create_empty_matrix(m->cols, m->rows);

    // Loop and put 'em there
    for (size_t y = 0; y < m->rows; y++) {
        for (size_t x = 0; x < m->cols; x++) {
            to_ret->data[x * m->rows + y] = m->data[y * m->cols + x];
        }
    }

    // Return
    return to_ret;
}

matrix* matrix_add(const matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_add: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
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
matrix* matrix_add_inplace(matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_add_inplace: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
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

matrix* matrix_mul_s(const matrix* m1, double s) {
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

matrix* matrix_matmul(const matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->cols != m2->rows) {
        fprintf(stderr, "ERROR: matrix_matmul: matrix m1 (%ldx%ld) and m2 (%ldx%ld) have illegal sizes for matrix multiplication\n",
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
                sum += m1->data[y * m1->cols + i] * m2->data[i * m2->cols + x];
            }
            to_ret->data[y * m1->cols + x] = sum;
        }
    }
    return to_ret;
}

matrix* matrix_mul(const matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_mul: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
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
matrix* matrix_mul_inplace(matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_mul_inplace: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
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

matrix* matrix_tensor(const matrix* m1, const matrix* m2) {
    // Create a matrix to return
    size_t w = m1->cols * m2->cols;
    size_t h = m1->rows * m2->rows;
    matrix* to_ret = create_empty_matrix(h, w);

    // Loop through the m1 matrix first
    for (size_t y1 = 0; y1 < m1->rows; y1++) {
        for (size_t x1 = 0; x1 < m1->cols; x1++) {
            size_t x = x1 * m2->cols;
            size_t y = y1 * m2->rows;

            // Copy every element of m2 with this element of m1 and store it in the proper location
            double value = m1->data[y1 * m1->cols + x1];
            for (size_t y2 = 0; y2 < m2->rows; y2++) {
                for (size_t x2 = 0; x2 < m2->cols; x2++) {
                    to_ret->data[(y + y2) * w + (x + x2)] = value * m2->data[y2 * m2->cols + x2];
                }
            }
        }
    }
    return to_ret;
}

matrix* matrix_concat_h(const matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows) {
        fprintf(stderr, "ERROR: matrix_concat: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same number of rows\n",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    size_t w = m1->cols + m2->cols;
    matrix* to_ret = create_empty_matrix(m1->rows, w);
    for (size_t y = 0; y < m1->rows; y++) {
        // First, handle everything in m1 on this row
        for (size_t x = 0; x < m1->cols; x++) {
            to_ret->data[y * w + x] = m1->data[y * m1->cols + x];
        }
        // Next, handle everything in m2 on this row
        for (size_t x = 0; x < m2->cols; x++) {
            to_ret->data[y * w + m1->cols + x] = m2->data[y * m2->cols + x];
        }
    }
    return to_ret;
}