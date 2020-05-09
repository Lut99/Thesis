/* PLAYGROUND.c
 *   by Lut99
 *
 * Created:
 *   07/05/2020, 22:11:32
 * Last edited:
 *   07/05/2020, 22:42:40
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A file just to test some things in.
**/

#include <stdio.h>
#include <stddef.h>


int main() {
    double zs[100];
    #pragma omp parallel for reduction(+:zs[:100]) collapse(2)
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            printf("(NESTED - i = %lu) Thread id: %d\n", i, __builtin_omp_get_thread_num());
            zs[i]++;
        }
    }

    for (size_t i = 0; i < 100; i++) {
        zs[i] = (zs[i] + i) / 2;
    }

    printf("All z's : [");
    for (size_t i = 0; i < 100; i++) {
        if (i > 0) {
            printf(", ");
        }
        printf("%f", zs[i]);
    }
    printf("]\n");
}