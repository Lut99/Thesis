/* TEST MATRIX.c
 *   by Lut99
 *
 * Created:
 *   16/04/2020, 23:18:21
 * Last edited:
 *   27/04/2020, 22:28:58
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


/***** TEST FUNCTIONS *****/

/* Tests matrix subset. */
bool test_subset() {
    // Set the begin arrays and expected arrays
    double start[5][3] = {{ 1,  2,  3},
                          { 4,  5,  6},
                          { 7,  8,  9},
                          {10, 11, 12},
                          {13, 14, 15}};
    double expec1[4][2] = {{ 1,  2},
                           { 4,  5},
                           { 7,  8},
                           {10, 11}};
    double expec2[4][2] = {{ 5,  6},
                           { 8,  9},
                           {11, 12},
                           {14, 15}};
    
    // Create the start matrix
    matrix* m_1 = create_matrix(5, 3, start);

    // Fetch the subsets
    matrix* m_res1 = subset_matrix(m_1, 0, 4, 0, 2);
    matrix* m_res2 = subset_matrix(m_1, 1, 5, 1, 3);

    // Compare if they are equal
    matrix* m_exp1 = create_matrix(4, 2, expec1);
    matrix* m_exp2 = create_matrix(4, 2, expec2);

    bool succes = true;
    if (!matrix_equals(m_res1, m_exp1)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res1);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp1);
        fprintf(stderr, "\nTesting subset failed.\n\n");
    } else if (!matrix_equals(m_res2, m_exp2)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res2);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp2);
        fprintf(stderr, "\nTesting subset failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res1);
    destroy_matrix(m_res2);
    destroy_matrix(m_exp1);
    destroy_matrix(m_exp2);

    return succes;
}

/* Tests matrix transposes. */
bool test_transpose() {
    // Set the begin array and expected array
    double start[5][3] = {{ 1,  2,  3},
                          { 4,  5,  6},
                          { 7,  8,  9},
                          {10, 11, 12},
                          {13, 14, 15}};
    double expec[3][5] = {{1, 4, 7, 10, 13},
                          {2, 5, 8, 11, 14},
                          {3, 6, 9, 12, 15}};

    // Create a matrix
    matrix* m_1 = create_matrix(5, 3, start);

    // Transpose it
    matrix* m_res = matrix_transpose(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 5, expec);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting transpose failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res);
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix constant addition. */
bool test_constant_add() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double start2 = 2;
    double expect[5][3] = {{ 3,  4,  5},
                           { 6,  7,  8},
                           { 9, 10, 11},
                           {12, 13, 14},
                           {15, 16, 17}};

    // Create the matrix
    matrix* m_1 = create_matrix(5, 3, start1);

    // Do the addition (also in-place)
    matrix* m_res = matrix_add_c(m_1, start2);
    matrix_add_c_inplace(m_1, start2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting constant addition failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix addition. */
bool test_addition() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double expect[5][3] = {{ 2,  4,  6},
                           { 8, 10, 12},
                           {14, 16, 18},
                           {20, 22, 24},
                           {26, 28, 30}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = copy_matrix_new(m_1);

    // Do the addition (also in-place)
    matrix* m_res = matrix_add(m_1, m_2);
    matrix_add_inplace(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting addition failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res);
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix addition. */
bool test_vec_add() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double start2[5] = {1,
                        2,
                        3,
                        4,
                        5};
    double expect[5][3] = {{ 2,  3,  4},
                           { 6,  7,  8},
                           {10, 11, 12},
                           {14, 15, 16},
                           {18, 19, 20}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* v_1 = create_vector(5, start2);

    // Do the addition (also in-place)
    matrix* m_res = matrix_add_vec(m_1, v_1);
    matrix_add_vec_inplace(m_1, v_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting vector addition failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(v_1);
    destroy_matrix(m_res);
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix constant subtraction. */
bool test_constant_sub() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double start2 = 2;
    double expect1[5][3] = {{-1,  0,  1},
                            { 2,  3,  4},
                            { 5,  6,  7},
                            { 8,  9, 10},
                            {11, 12, 13}};
    double expect2[5][3] = {{  1,   0,  -1},
                            { -2,  -3,  -4},
                            { -5,  -6,  -7},
                            { -8,  -9, -10},
                            {-11, -12, -13}};

    // Create the matrix, but create a copy to test the other way around
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = copy_matrix_new(m_1);

    // Do the subtraction both ways (also in-place)
    matrix* m_res1 = matrix_sub1_c(m_1, start2);
    matrix_sub1_c_inplace(m_1, start2);
    
    matrix* m_res2 = matrix_sub2_c(start2, m_2);
    matrix_sub2_c_inplace(start2, m_2);

    // Compare if they are equal
    matrix* m_exp1 = create_matrix(5, 3, expect1);
    matrix* m_exp2 = create_matrix(5, 3, expect2);

    bool succes = true;
    if (!matrix_equals(m_res1, m_exp1) || !matrix_equals(m_1, m_exp1) ||
        !matrix_equals(m_res2, m_exp2) || !matrix_equals(m_2, m_exp2)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res1, m_exp1)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res1);
        } else if (!matrix_equals(m_1, m_exp1)) {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        } else if (!matrix_equals(m_res2, m_exp2)) {
            fprintf(stderr, "Got (reversed):\n");
            matrix_print(m_res2);
        } else {
            fprintf(stderr, "Got (reversed, inplace):\n");
            matrix_print(m_2);
        }
        fprintf(stderr, "\nExpected:\n");
        if (!matrix_equals(m_res1, m_exp1) || !matrix_equals(m_1, m_exp1)) {
            matrix_print(m_exp1);
        } else {
            matrix_print(m_exp2);
        }
        fprintf(stderr, "\nTesting constant subtraction failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res1),
    destroy_matrix(m_res2),
    destroy_matrix(m_exp1);
    destroy_matrix(m_exp2);

    return succes;
}

/* Tests matrix subtraction. */
bool test_subtraction() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double expect[5][3] = {{0, 0, 0},
                           {0, 0, 0},
                           {0, 0, 0},
                           {0, 0, 0},
                           {0, 0, 0}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = copy_matrix_new(m_1);

    // Do the subtraction (also in-place)
    matrix* m_res = matrix_sub(m_1, m_2);
    matrix_sub_inplace(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting subtraction failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res);
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix constant multiplication. */
bool test_constant_mul() {
    // Set the begin array and expected array
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double start2 = 2;
    double expect[5][3] = {{ 2,  4,  6},
                           { 8, 10, 12},
                           {14, 16, 18},
                           {20, 22, 24},
                           {26, 28, 30}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);

    // Do the addition (also in-place)
    matrix* m_res = matrix_mul_c(m_1, start2);
    matrix_mul_c_inplace(m_1, start2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting constant multiplication failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix element-wise multiplication. */
bool test_elem_mult() {
    // Set the begin array and expected array          
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double expect[5][3] = {{  1,   4,   9},
                           { 16,  25,  36},
                           { 49,  64,  81},
                           {100, 121, 144},
                           {169, 196, 225}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = copy_matrix_new(m_1);

    // Do the matrix multiplication, also inplace
    matrix *m_res = matrix_mul(m_1, m_2);
    matrix_mul_inplace(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting element-wise multiplication failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix multiplication. */
bool test_matmult() {
    // Set the begin array and expected array          
    double start1[5][3] = {{ 1,  2,  3},
                           { 4,  5,  6},
                           { 7,  8,  9},
                           {10, 11, 12},
                           {13, 14, 15}};
    double start2[3][3] = {{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};
    double expect[5][3] = {{ 30,  36,  42},
                           { 66,  81,  96},
                           {102, 126, 150},
                           {138, 171, 204},
                           {174, 216, 258}};

    // Create the two matrices
    matrix* m_1 = create_matrix(5, 3, start1);
    matrix* m_2 = create_matrix(3, 3, start2);

    // Do the matrix multiplication
    matrix *m_res = matrix_matmul(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(5, 3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting matrix multiplication failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix tensor product. */
bool test_tensor() {
    // Set the begin array and expected array          
    double start1[3][2] = {{1, 2},
                           {3, 4},
                           {5, 6}};
    double start2[2][3] = {{1, 2, 3},
                           {4, 5, 6}};
    double expect[6][6] = {{ 1,  2,  3,  2,  4,  6},
                           { 4,  5,  6,  8, 10, 12},
                           { 3,  6,  9,  4,  8, 12},
                           {12, 15, 18, 16, 20, 24},
                           { 5, 10, 15,  6, 12, 18},
                           {20, 25, 30, 24, 30, 36}};

    // Create the two matrices
    matrix* m_1 = create_matrix(3, 2, start1);
    matrix* m_2 = create_matrix(2, 3, start2);

    // Do the matrix multiplication, also inplace
    matrix *m_res = matrix_tensor(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(6, 6, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting tensor product failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix inverse. */
bool test_inverse() {
    // Set the begin array and expected array          
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3][4] = {{1.0 / 1, 1.0 /  2, 1.0 /  3, 1.0 /  4},
                           {1.0 / 5, 1.0 /  6, 1.0 /  7, 1.0 /  8},
                           {1.0 / 9, 1.0 / 10, 1.0 / 11, 1.0 / 12}};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Do the inverse, also in-place
    matrix *m_res = matrix_inv(m_1);
    matrix_inv_inplace(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 4, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting inverse failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix exp. */
bool test_exponent() {
    // Set the begin array and expected array          
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3][4] = {{exp(1), exp( 2), exp( 3), exp( 4)},
                           {exp(5), exp( 6), exp( 7), exp( 8)},
                           {exp(9), exp(10), exp(11), exp(12)}};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Do the inverse, also in-place
    matrix *m_res = matrix_exp(m_1);
    matrix_exp_inplace(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 4, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting exponent failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix tanh. */
bool test_tanh() {
    // Set the begin array and expected array          
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3][4] = {{tanh(1), tanh( 2), tanh( 3), tanh( 4)},
                           {tanh(5), tanh( 6), tanh( 7), tanh( 8)},
                           {tanh(9), tanh(10), tanh(11), tanh(12)}};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Do the inverse, also in-place
    matrix *m_res = matrix_tanh(m_1);
    matrix_tanh_inplace(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 4, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting tanh failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix log. */
bool test_nat_log() {
    // Set the begin array and expected array          
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3][4] = {{log(1), log( 2), log( 3), log( 4)},
                           {log(5), log( 6), log( 7), log( 8)},
                           {log(9), log(10), log(11), log(12)}};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Do the inverse, also in-place
    matrix *m_res = matrix_ln(m_1);
    matrix_ln_inplace(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 4, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting natural logarithm failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix square. */
bool test_square() {
    // Set the begin array and expected array          
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3][4] = {{ 1,   4,   9,  16},
                           {25,  36,  49,  64},
                           {81, 100, 121, 144}};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Do the inverse, also in-place
    matrix *m_res = matrix_square(m_1);
    matrix_square_inplace(m_1);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 4, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp) || !matrix_equals(m_1, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\n");
        if (!matrix_equals(m_res, m_exp)) {
            fprintf(stderr, "Got:\n");
            matrix_print(m_res);
        } else {
            fprintf(stderr, "Got (inplace):\n");
            matrix_print(m_1);
        }
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting square failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}

/* Tests matrix sum. */
bool test_sum() {
    // Set the begin array and expected return value       
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect = 78;

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Take the sum
    double m_res = matrix_sum(m_1);

    // Compare if they are equal
    bool succes = true;
    if (m_res != expect) {
        printf(" [FAIL]\n");
        fprintf(stderr, "Summed value is incorrect: expected %f, got %f\n\n",
                expect, m_res);
        fprintf(stderr, "Testing sum failed.\n\n");

        succes = false;
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    return succes;
}

/* Tests horizontal matrix sum. */
bool test_hsum() {
    // Set the begin array and expected return value       
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect[3] = {10, 26, 42};

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Take the sum
    matrix* m_res = matrix_sum_h(m_1);

    // Compare if they are equal
    matrix* m_exp = create_vector(3, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp)) {
        printf(" [FAIL]\n");
        fprintf(stderr, "Summed vector is incorrect:\n\nGot:\n");
        matrix_print(m_res);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting horizontal sum failed.\n\n");

        succes = false;
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_res);
    destroy_matrix(m_exp);
    return succes;
}

/* Tests matrix max. */
bool test_max() {
    // Set the begin array and expected return value       
    double start1[3][4] = {{1,  2,  3,  4},
                           {5,  6,  7,  8},
                           {9, 10, 11, 12}};
    double expect = 12;

    // Create the matrix
    matrix* m_1 = create_matrix(3, 4, start1);

    // Take the sum
    double m_res = matrix_max(m_1);

    // Compare if they are equal
    bool succes = true;
    if (m_res != expect) {
        printf(" [FAIL]\n");
        fprintf(stderr, "Max value is incorrect: expected %f, got %f\n\n",
                expect, m_res);
        fprintf(stderr, "Testing max failed.\n\n");

        succes = false;
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    return succes;
}

/* Tests matrix horizontal concatenation. */
bool test_concat() {
    // Set the begin array and expected array          
    double start1[3][2] = {{1, 2},
                           {3, 4},
                           {5, 6}};
    double start2[3][3] = {{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};
    double expect[3][5] = {{1, 2, 1, 2, 3},
                           {3, 4, 4, 5, 6},
                           {5, 6, 7, 8, 9}};

    // Create the two matrices
    matrix* m_1 = create_matrix(3, 2, start1);
    matrix* m_2 = create_matrix(3, 3, start2);

    // Do the matrix multiplication, also inplace
    matrix *m_res = matrix_concat_h(m_1, m_2);

    // Compare if they are equal
    matrix* m_exp = create_matrix(3, 5, expect);

    bool succes = true;
    if (!matrix_equals(m_res, m_exp)) {
        succes = false;
        printf(" [FAIL]\n");
        fprintf(stderr, "Matrices are not equal:\n\nGot:\n");
        matrix_print(m_res);
        fprintf(stderr, "\nExpected:\n");
        matrix_print(m_exp);
        fprintf(stderr, "\nTesting horizontal concatenation failed.\n\n");
    }

    // Clean up and return the succes status
    destroy_matrix(m_1);
    destroy_matrix(m_2);
    destroy_matrix(m_res),
    destroy_matrix(m_exp);

    return succes;
}



/***** MAIN *****/

int main() {
    printf("  Testing subset...                        ");
    if (!test_subset()) {
        return -1;
    }
    printf(" [ OK ]\n");
    
    printf("  Testing transposing...                   ");
    if (!test_transpose()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix-constant addition...      ");
    if (!test_constant_add()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing addition...                      ");
    if (!test_addition()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing vector addition...               ");
    if (!test_vec_add()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix-constant subtraction...   ");
    if (!test_constant_sub()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing subtraction...                   ");
    if (!test_subtraction()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix-constant multiplication...");
    if (!test_constant_mul()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing element-wise multiplication...   ");
    if (!test_elem_mult()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing matrix multiplication...         ");
    if (!test_matmult()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing tensor product...                ");
    if (!test_tensor()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing inverse...                       ");
    if (!test_inverse()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing exponent...                      ");
    if (!test_exponent()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing tanh...                          ");
    if (!test_tanh()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing natural logarithm...             ");
    if (!test_nat_log()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing square...                        ");
    if (!test_square()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing sum...                           ");
    if (!test_sum()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing horizontal sum...                ");
    if (!test_hsum()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing max...                           ");
    if (!test_max()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("  Testing concatenation...                 ");
    if (!test_concat()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("Matrix tests succes.\n\n");
}