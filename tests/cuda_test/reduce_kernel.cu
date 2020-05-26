/* REDUCE KERNEL.cu
 *   by Anonymous
 *
 * Created:
 *   5/24/2020, 9:25:20 PM
 * Last edited:
 *   5/26/2020, 6:04:01 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file tests and implements te reduction kernel as seen by the CUDA
 *   slides: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
**/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


__global__ void reduceKernelA(unsigned long* result, unsigned long* to_reduce, size_t N) {
    // Make sure we are allowed to do work
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    if (i < N / 2) {
        // Simply sum this and the next element
        result[i] = to_reduce[i] + to_reduce[i + N / 2];
    }
}


int main() {
    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Create a list of elements to reduce (make it large, for fun)
    size_t N = 1797 * 20;
    int max = 50;
    unsigned long to_reduce[N];
    for (size_t i = 0; i < N; i++) {
        to_reduce[i] = rand() % max;
    }




    
    // Acquire a correct value, and benchmark the sequential version while we're at it
    gettimeofday(&start, NULL);
    
    unsigned long correct = 0;
    for (size_t i = 0; i < N; i++) {
        correct += to_reduce[i];
    }

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("Sequential time taken: %lu ms\n", time_taken);
    printf("Sequential result: %lu\n", correct);




    
    // Now, enter CUDA!

    gettimeofday(&start, NULL);

    // Allocate space for N (with a possible pad), then copy all the data
    unsigned long* to_reduce_gpu;
    cudaMalloc(&to_reduce_gpu, sizeof(unsigned long) * (N + (N % 2 == 0 ? 0 : 1)));
    cudaMemcpy(to_reduce_gpu, to_reduce, sizeof(unsigned long) * N, cudaMemcpyHostToDevice);

    // Allocate space for the result
    unsigned long* result_gpu;
    cudaMalloc(&result_gpu, sizeof(unsigned long) * N);

    // Next, invoke the kernel as many times as needed. Let's say that we do the rest manually from 32 and down
    int threads_per_block = 32;
    int to_do = N;
    while (to_do > 32) {
        // First, if to_do is uneven, set the value after that to 0
        if (to_do % 2 != 0) {
            cudaMemsetAsync(to_reduce_gpu + to_do, 0, sizeof(unsigned long));
            to_do++;
        }

        // Next, launch the kernel
        int blocks_per_grid = to_do / threads_per_block + (to_do % threads_per_block == 0 ? 0 : 1);
        reduceKernelA<<<blocks_per_grid, threads_per_block>>>(result_gpu, to_reduce_gpu, to_do);

        // We can already decrease to_do and swap the pointers
        to_do >>= 1;
        unsigned long* temp = to_reduce_gpu;
        to_reduce_gpu = result_gpu;
        result_gpu = temp;
    }

    // Copy the memory back
    cudaMemcpy(to_reduce, to_reduce_gpu, sizeof(unsigned long) * to_do, cudaMemcpyDeviceToHost);

    // Manually combine all intermediate results
    unsigned long result = 0;
    for (size_t i = 0; i < to_do; i++) {
        result += to_reduce[i];
    }

    cudaFree(to_reduce_gpu);
    cudaFree(result_gpu);

    gettimeofday(&stop, NULL);

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA time taken: %lu ms\n", time_taken);
    printf("CUDA result: %lu\n", result);

    return 0;
}

