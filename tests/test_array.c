/* TEST ARRAY.c
 *   by Lut99
 *
 * Created:
 *   28/04/2020, 20:55:21
 * Last edited:
 *   28/04/2020, 21:24:59
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Tests for the array class.
**/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "Array.h"


/* Some tests for the new memory system. */
bool test_mem() {
    double test_data[] = {1, 2, 3, 4, 5, 42, 42, 55, 12, 21};

    // Try a dynamically allocated pointer
    array* a_1 = create_array(10, test_data);

    // Also try a dynamic linked array
    array* a_2 = create_linked_array(10, test_data);

    // Also try a stack-allocated array
    CREATE_STACK_ARRAY(a_3, 10);
    fill_array(a_3, test_data);

    // And a linked stack-allocated array
    CREATE_LINKED_STACK_ARRAY(a_4, 10, test_data);

    // Check the validity of all arrays
    bool succes = true;
    if (!array_equals2(a_1, test_data)) {
        fprintf(stderr, " [FAIL]\n");
        fprintf(stderr, "\nArrays are not equal:\n\na_1:\n");
        array_print(stdout, a_1);
        fprintf(stderr, "\nExpected:\n[");
        for (size_t i = 0; i < 10; i++) {
            if (i > 0) {
                fprintf(stderr, " ");
            }
            fprintf(stderr, "%f", test_data[i]);
        }
        fprintf(stderr, "]\n");

        succes = false;
    } else if (!array_equals2(a_2, test_data)) {
        fprintf(stderr, " [FAIL]\n");
        fprintf(stderr, "\nArrays are not equal:\n\na_2:\n");
        array_print(stdout, a_2);
        fprintf(stderr, "\nExpected:\n[");
        for (size_t i = 0; i < 10; i++) {
            if (i > 0) {
                fprintf(stderr, " ");
            }
            fprintf(stderr, "%f", test_data[i]);
        }
        fprintf(stderr, "]\n");

        succes = false;
    } else if (!array_equals2(a_3, test_data)) {
        fprintf(stderr, " [FAIL]\n");
        fprintf(stderr, "\nArrays are not equal:\n\na_3:\n");
        array_print(stdout, a_3);
        fprintf(stderr, "\nExpected:\n[");
        for (size_t i = 0; i < 10; i++) {
            if (i > 0) {
                fprintf(stderr, " ");
            }
            fprintf(stderr, "%f", test_data[i]);
        }
        fprintf(stderr, "]\n");

        succes = false;
    } else if (!array_equals2(a_4, test_data)) {
        fprintf(stderr, " [FAIL]\n");
        fprintf(stderr, "\nArrays are not equal:\n\na_4:\n");
        array_print(stdout, a_4);
        fprintf(stderr, "\nExpected:\n[");
        for (size_t i = 0; i < 10; i++) {
            if (i > 0) {
                fprintf(stderr, " ");
            }
            fprintf(stderr, "%f", test_data[i]);
        }
        fprintf(stderr, "]\n");

        succes = false;
    }

    // Deallocate the dynamic arrays
    free(a_1);
    free(a_2);

    return succes;
}


int main() {
    printf("  Testing memory...                        ");
    if (!test_mem()) {
        return -1;
    }
    printf(" [ OK ]\n");

    printf("Matrix tests succes.\n\n");
}
