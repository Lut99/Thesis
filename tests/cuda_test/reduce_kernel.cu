/* REDUCE KERNEL.cu
 *   by Anonymous
 *
 * Created:
 *   5/24/2020, 9:25:20 PM
 * Last edited:
 *   5/24/2020, 11:07:09 PM
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


__global__ void reduceKernel(unsigned long* to_reduce, size_t N) {
    // Step 1: fetch global memory to a local shared cache for this block alone

    // First, we allocate shared dynamic memory in the block memory
    extern __shared__ unsigned long cache[];

    // Then, we load our part of the job in it
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    cache[tid] = to_reduce[i];

    // Make sure also threads outside of our warp did their thing
    __syncthreads();

    // Step 2: do the reduction

    // In our local, superfast cache, let's reduce it
    int upper = blockDim.x/2;
    if (upper > N / 2) { upper = N / 2; }
    for (int s = upper; s > 0; s>>=1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    // Step 3: Write the cache memory back
    if (tid == 0) { to_reduce[blockIdx.x] = cache[0]; }
}


int main() {
    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Create a list of elements to reduce (make it large, for fun)
    size_t N = 50000;
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
    
    // Start by copying the data
    unsigned long* to_reduce_gpu;
    cudaMalloc(&to_reduce_gpu, sizeof(unsigned long) * N);
    cudaMemcpy(to_reduce_gpu, to_reduce, sizeof(unsigned long) * N, cudaMemcpyHostToDevice);

    // Next, invoke the kernel as many times as needed. Let's say that we do the rest manually from 32 and down
    int threadsPerBlock = 32;
    int to_go = N;
    while (to_go > 32) {
        int blocksPerGrid = to_go / threadsPerBlock + 1;
        reduceKernel<<<blocksPerGrid, threadsPerBlock, sizeof(unsigned long) * threadsPerBlock>>>(to_reduce_gpu, to_go);

        to_go = blocksPerGrid;

        printf("Next round:\n");
        printf("to_go=%d\n", to_go);
        cudaMemcpy(to_reduce, to_reduce_gpu, sizeof(unsigned long) * N, cudaMemcpyDeviceToHost);
        printf("First 100 elements of list: [");
        for (size_t i = 0; i < 100; i++) {
            if (i > 0) { printf(", "); }
            printf("%lu", to_reduce[i]);
        }
        printf("]\n");
    }

    // Copy the memory back
    cudaMemcpy(to_reduce, to_reduce_gpu, sizeof(unsigned long) * N, cudaMemcpyDeviceToHost);

    // Manually combine all intermediate results
    unsigned long result = 0;
    for (size_t i = 0; i < to_go; i++) {
        result += to_reduce[i];
    }

    cudaFree(to_reduce_gpu);

    gettimeofday(&stop, NULL);

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA time taken: %lu ms\n", time_taken);
    printf("CUDA result: %lu\n", result);

    return 0;
}

