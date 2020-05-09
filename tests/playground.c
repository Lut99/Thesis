/* PLAYGROUND.c
 *   by Lut99
 *
 * Created:
 *   07/05/2020, 22:11:32
 * Last edited:
 *   09/05/2020, 17:37:40
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A file just to test some things in.
**/

#include <stdio.h>
#include <stddef.h>
#include <sys/time.h>


extern int omp_get_thread_num();


int main() {
    struct timeval start, stop;

    gettimeofday(&start, NULL);

    // #pragma omp parallel
    {
        #pragma omp parallel
        {
        int num = omp_get_thread_num();

        printf("(Thread %d) Hello there!\n", num);
        }

        #pragma omp parallel for
        for (int i = 0; i < 64; i++) {
            int num = omp_get_thread_num();
            printf("(Thread %d) General Kenobi!\n", num);
        }

        #pragma omp parallel for
        for (int i = 0; i < 64; i++) {
            int num = omp_get_thread_num();
            printf("(Thread %d) You are a bold one!\n", num);
        }
    }

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;

    printf("Time taken: %lu ms\n", time_taken);
}