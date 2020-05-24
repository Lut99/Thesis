/* REDUCE KERNEL.cu
 *   by Anonymous
 *
 * Created:
 *   5/24/2020, 9:25:20 PM
 * Last edited:
 *   5/24/2020, 9:46:11 PM
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


__global__ void reduceKernel(unsigned long* to_reduce) {
    // Step 1: fetch global memory to a local shared cache for this block alone

    // First, we allocate shared dynamic memory in the block memory
    extern __shared__ unsigned long cache[];

    // Then, we load our part of the job in it
    int tid = threadIdx.x;
    cache[tid] = to_reduce[blockIdx.x * blockDim.x + tid];
}


int main() {
    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Create a list of elements to reduce (make it large, for fun)
    size_t N = 5000000;
    int min = -50;
    int max = 50;
    unsigned long to_reduce[N];
    for (size_t i = 0; i < N; i++) {
        to_reduce[i] = (rand() % abs(max - min)) - min;
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
    
    // Start by copying the data
    unsigned long* to_reduce_gpu;
    cudaMalloc(&to_reduce_gpu, sizeof(unsigned long) * N);
    cudaMemcpy(to_reduce_gpu, to_reduce, sizeof(unsigned long) * N, cudaMemcpyHostToDevice);

    // Next, invoke the kernel as many times as needed
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    while (true) {
        reduceKernel<<<blocksPerGrid, threadsPerBlock, sizeof(unsigned long) * threadsPerBlock>>>(to_reduce_gpu);
    }

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA time taken: %lu ms\n", time_taken);
    printf("CUDA result: %lu\n", result);

    return 0;
}

