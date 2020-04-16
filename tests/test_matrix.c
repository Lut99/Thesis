/* TEST MATRIX.c
 *   by Lut99
 *
 * Created:
 *   16/04/2020, 23:18:21
 * Last edited:
 *   17/04/2020, 00:40:23
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file performs tests for Matrix.c. Can be run using 'make tests'.
**/

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "Matrix.h"


/***** TOOLS *****/

/* Prints a matrix to stderr. */
void matrix_print(matrix* m) {
    // Early quit if there is nothing to print
    if (m->rows == 0 || m->cols == 0) {
        fprintf(stderr, "(empty)\n");
        return;
    }
    for (size_t y = 0; y < m->rows; y++) {
        fprintf(stderr, "[%7.2f", m->data[y * m->cols]);
        for (size_t x = 1; x < m->cols; x++) {
            fprintf(stderr, " %7.2f", m->data[y * m->cols + x]);
        }
        fprintf(stderr, "]\n");
    }
}

/* Checks if two matrices are the same and, if not, prints them both out with an error. */
bool matrix_equals(matrix* m1, matrix* m2) {
    // Check if they are the same size
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not the same (incorrect size):\n\nMatrix 1:\n");
        matrix_print(m1);
        fprintf(stderr, "\nMatrix 2:\n");
        matrix_print(m2);
        return false;
    }

    // Check each element
    for (size_t y = 0; y < m1->rows; y++) {
        for (size_t x = 0; x < m1->cols; x++) {
            if (m1->data[y * m1->cols + x] != m2->data[y * m2->cols + x]) {
                printf(" [FAIL]\n");
                fprintf(stderr, "Matrices are not the same (mismatching element at (%ld,%ld)):\n\nMatrix 1:\n",
                        y, x);
                matrix_print(m1);
                fprintf(stderr, "\nMatrix 2:\n");
                matrix_print(m2);
                return false;
            }
        }
    }

    // Match!
    return true;
}


/***** TEST FUNCTIONS *****/

/* Tests matrix transposes. */
bool test_transpose() {
    // Set the begin array and expected array
    double start[5][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    double expec[3][5] = {{1, 4, 7, 10, 13}, {2, 5, 8, 11, 14}, {3, 6, 9, 12, 15}};

    // Create a matrix
    matrix* m_res = create_matrix(5, 3, start);

    // Transpose it
    matrix* m_t = matrix_transpose(m_res);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 5, expec);

    bool succes = true;
    if (!matrix_equals(m_t, m_exp)) {
        succes = false;
        fprintf(stderr, "\nTesting transpose failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_res),
    destroy_matrix(m_t),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix addition. */
bool test_addition() {
    // Set the begin array and expected array
    double start1[5][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    double start2[5][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    double expect[5][3] = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}, {20, 22, 24}, {26, 28, 30}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = create_matrix(5, 3, start2);

    // Do the addition (also in-place)
    matrix* m_res = matrix_add(m_1, m_2);
    matrix_add_inplace(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        fprintf(stderr, "\nAddition transpose failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1),
    destroy_matrix(m_2),
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix scalar multiplication. */
bool test_scalar_mult() {
    // Set the begin array and expected array
    double start1[5][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    double start2 = 2;
    double expect[5][3] = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}, {20, 22, 24}, {26, 28, 30}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);

    // Do the addition (also in-place)
    matrix* m_res = matrix_mul_s(m_1, start2);
    matrix_mul_s_inplace(m_1, start2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        fprintf(stderr, "\nAddition transpose failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1),
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix multiplication. */
bool test_matmult() {
    return false;
}

/* Tests matrix element-wise multiplication. */
bool test_elem_mult() {
    return false;
}

/* Tests matrix tensor product. */
bool test_tensor() {
    return false;
}

/* Tests matrix horizontal concatenation. */
bool test_concat() {
    return false;
}


int main() {
    printf("  Testing transposing...                 ");
    if (!test_transpose()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing addition...                    ");
    if (!test_addition()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix-scalar multiplication...");
    if (!test_scalar_mult()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix multiplication...       ");
    if (!test_matmult()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing element-wise multiplication... ");
    if (!test_elem_mult()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing tensor product...              ");
    if (!test_tensor()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing concatenation product...       ");
    if (!test_concat()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("Tranpose tests succes.\n\n");
}