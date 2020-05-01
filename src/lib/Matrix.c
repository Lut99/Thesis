/* MATRIX.c
 *   by Tim Müller
 *
 * Created:
 *   16/04/2020, 22:19:37
 * Last edited:
 *   01/05/2020, 13:50:10
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

matrix* create_matrix(size_t rows, size_t cols, const double data[rows][cols]) {
    // Create an empty matrix with the same dimensions
    matrix* to_ret = create_empty_matrix(rows, cols);
    if (to_ret == NULL) {
        fprintf(stderr, "ERROR: create_matrix: empty matrix creation failed.\n");
        return NULL;
    }

    // Copy the data from the given pointer
    for (size_t y = 0; y < rows; y++) {
        for (size_t x = 0; x < cols; x++) {
            to_ret->data[y * cols + x] = data[y][x];
        }
    }

    // Return
    return to_ret;
}

matrix* create_vector(size_t size, const double data[size]) {
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



/***** MATH *****/

matrix* matrix_transpose(const matrix* m1) {
    // Loop and put 'em there
    matrix* to_ret = create_empty_matrix(m1->cols, m1->rows);
    for (size_t y = 0; y < m1->rows; y++) {
        for (size_t x = 0; x < m1->cols; x++) {
            INDEX(to_ret, x, y) = INDEX(m1, y, x);
        }
    }
    // Return
    return to_ret;
}

matrix* matrix_add_c(const matrix* m1, double c) {
    // Create a new matrix, copy the the addition for each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] + c;
    }
    return to_ret;
}
matrix* matrix_add_c_inplace(matrix* m1, double c) {
    // Add c to each element of m1, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] += c;
    }
    return m1;
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

    // Add each element of m2 to m1, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] += m2->data[i];
    }
    return m1;
}

matrix* matrix_add_vec(const matrix* m1, const matrix* v1) {
    // Sanity check that the matrix and the vector have correct sizes
    if (v1->cols != 1) {
        fprintf(stderr, "ERROR: matrix_add_vec: vector v1 (%ldx%ld) is not a vector\n",
                v1->rows,
                v1->cols);
        return NULL;
    } else if (m1->rows != v1->rows) {
        fprintf(stderr, "ERROR: matrix_add_vec: matrix m1 (%ldx%ld) and vector v1 (%ld) do not have the same number of rows\n",
                m1->rows,
                m1->cols,
                v1->rows);
        return NULL;
    }

    // Add each element of the vector to each column of a new vector
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t y = 0; y < m1->rows; y++) {
        double elem = v1->data[y];
        for (size_t x = 0; x < m1->cols; x++) {
            to_ret->data[y * m1->cols + x] = m1->data[y * m1->cols + x] + elem;
        }
    }
    return to_ret;
}
matrix* matrix_add_vec_inplace(matrix* m1, const matrix* v1) {
    // Sanity check that the matrix and the vector have correct sizes
    if (v1->cols != 1) {
        fprintf(stderr, "ERROR: matrix_add_vec_inplace: vector v1 (%ldx%ld) is not a vector\n",
                v1->rows,
                v1->cols);
        return NULL;
    } else if (m1->rows != v1->rows) {
        fprintf(stderr, "ERROR: matrix_add_vec_inplace: matrix m1 (%ldx%ld) and vector v1 (%ld) do not have the same number of rows\n",
                m1->rows,
                m1->cols,
                v1->rows);
        return NULL;
    }

    // Add each element of the vector to each column of a new vector
    for (size_t y = 0; y < m1->rows; y++) {
        double elem = v1->data[y];
        for (size_t x = 0; x < m1->cols; x++) {
            m1->data[y * m1->cols + x] += elem;
        }
    }
    return m1;
}

matrix* matrix_sub1_c(const matrix *m1, double c) {
    // Create a new matrix, copy m1i - c of each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] - c;
    }
    return to_ret;
}
matrix* matrix_sub1_c_inplace(matrix *m1, double c) {
    // Replace every element of m1 with m1i - c, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] -= c;
    }
    return m1;
}
matrix* matrix_sub2_c(double c, const matrix *m1) {
    // Create a new matrix, copy c - m1i of each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = c - m1->data[i];
    }
    return to_ret;
}
matrix* matrix_sub2_c_inplace(double c, matrix *m1) {
    // Replace every element of m1 with c - m1i, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] = c - m1->data[i];
    }
    return m1;
}

matrix* matrix_sub(const matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_sub: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Create a new matrix, copy the the subtraction for each element and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] - m2->data[i];
    }
    return to_ret;
}
matrix* matrix_sub_inplace(matrix* m1, const matrix* m2) {
    // Sanity check that the matrices are correctly sized
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "ERROR: matrix_sub_inplace: matrix m1 (%ldx%ld) and m2 (%ldx%ld) do not have the same sizes\n",
                m1->rows,
                m1->cols,
                m2->rows,
                m2->cols);
        return NULL;
    }

    // Subtract each element of m2 from m1, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] -= m2->data[i];
    }
    return m1;
}


matrix* matrix_mul_c(const matrix* m1, double c) {
    // Create a new matrix, copy the the multiplication of each element of m1 and c and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] * c;
    }
    return to_ret;
}
matrix* matrix_mul_c_inplace(matrix* m1, double c) {
    // Multiply each element of m1 with c, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] *= c;
    }
    return m1;
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
            to_ret->data[y * to_ret->cols + x] = sum;
        }
    }
    return to_ret;
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

matrix* matrix_inv(const matrix *m1) {
    // Create a new matrix, copy the the inversion of each element of m1 and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = 1 / m1->data[i];
    }
    return to_ret;
}
matrix* matrix_inv_inplace(matrix *m1) {
    // Inverse each element of m1, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] = 1 / m1->data[i];
    }
    return m1;
}

matrix* matrix_exp(const matrix *m1) {
    // Create a new matrix, copy the the exp of each element of m1 and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = exp(m1->data[i]);
    }
    return to_ret;
}
matrix* matrix_exp_inplace(matrix *m1) {
    // Take exp of each element, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] = exp(m1->data[i]);
    }
    return m1;
}

matrix* matrix_tanh(const matrix* m1) {
    // Create a new matrix, copy the the tanh of each element of m1 and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = tanh(m1->data[i]);
    }
    return to_ret;
}
matrix* matrix_tanh_inplace(matrix* m1) {
    // Take tanh of each element, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] = tanh(m1->data[i]);
    }
    return m1;
}

matrix* matrix_ln(const matrix *m1) {
    // Create a new matrix, copy the the ln of each element of m1 and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        if (fabs(m1->data[i]) > 0.0001) {
            to_ret->data[i] = log(m1->data[i]);
        } else {
            to_ret->data[i] = 0.0;
        }
    }
    return to_ret;
}
matrix* matrix_ln_inplace(matrix *m1) {
    // Take ln of each element, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        if (fabs(m1->data[i]) > 0.0001) {
            m1->data[i] = log(m1->data[i]);
        } else {
            m1->data[i] = 0.0;
        }
    }
    return m1;
}

matrix* matrix_square(const matrix* m1) {
    // Create a new matrix, copy the the square of each element of m1 and return
    matrix* to_ret = create_empty_matrix(m1->rows, m1->cols);
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        to_ret->data[i] = m1->data[i] * m1->data[i];
    }
    return to_ret;
}
matrix* matrix_square_inplace(matrix* m1) {
    // Multiply each element with itself, then return m1 to allow chaining
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        m1->data[i] *= m1->data[i];
    }
    return m1;
}

double matrix_sum(const matrix* m1) {
    // Loop through all elements to sum them
    double total = 0;
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        total += m1->data[i];
    }
    return total;
}
matrix* matrix_sum_h(const matrix* m1) {
    // Create a vector of the correct size
    matrix* to_ret = create_empty_matrix(m1->rows, 1);
    
    // Loop through all rows to sum them
    for (size_t y = 0; y < m1->rows; y++) {
        double sum = 0;
        for (size_t x = 0; x < m1->cols; x++) {
            sum += m1->data[y * m1->cols + x];
        }
        to_ret->data[y] = sum;
    }

    // Return
    return to_ret;
}
double matrix_max(const matrix* m1) {
    double max = -INFINITY;
    for (size_t i = 0; i < m1->rows * m1->cols; i++) {
        if (m1->data[i] > max) {
            max = m1->data[i];
        }
    }
    return max;
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
