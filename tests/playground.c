/* PLAYGROUND.c
 *   by Lut99
 *
 * Created:
 *   07/05/2020, 22:11:32
 * Last edited:
 *   5/23/2020, 10:22:35 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A file just to test some things in.
**/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


extern int omp_get_thread_num();


int main() {
    struct timeval start, stop;

    gettimeofday(&start, NULL);

    // Try some wacky kernel
    int m = 50000;
    int n = 50000;
    float A[n][m];
    float Anew[n][m];

    // Fill A with some nice values
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[j][i] = (float)rand()/(float)(RAND_MAX);
        }
    }

    // Do the parallel stuff!
    float error = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:error) collapse(2)
    for(int j = 1; j < n-1; j++) {
        for(int i= 1; i< m-1; i++ ) {
            Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]+ A[j-1][i] + A[j+1][i]);
            error = fmax(error, fabs(Anew[j][i] -A[j][i]));
        }
    }

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;

    printf("Time taken: %lu ms\n", time_taken);
}