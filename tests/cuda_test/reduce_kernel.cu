/* REDUCE KERNEL.cu
 *   by Anonymous
 *
 * Created:
 *   5/24/2020, 9:25:20 PM
 * Last edited:
 *   6/12/2020, 4:08:19 PM
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


#define cudaSafe() \
    if(cudaPeekAtLastError() != cudaSuccess) {\
        printf("ERROR: CUDA: %s\n", cudaGetErrorString(cudaGetLastError())); \
        exit(EXIT_FAILURE); \
    }


__global__ void reduceKernelA(unsigned long* to_reduce, size_t N) {
    // Make sure we are allowed to do work
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    int half_N = N / 2;
    if (i < half_N) {
        // Simply sum this and the next element
        to_reduce[i] += to_reduce[i + half_N];
    }
}


__global__ void reduceKernelB(unsigned long* to_reduce, size_t N) {
    // Make sure we are allowed to do work
    int i = (blockDim.x * blockIdx.x + threadIdx.x);
    int half_N = N / 2;
    if (i < half_N) {
        // Simply sum this and the next element
        to_reduce[i] += to_reduce[i + half_N];
        // If we are the last node, don't forget to sum any unevens
        if (i == half_N - 1 && (half_N + half_N) != N) {
            printf("There was an odd case!\n");
            to_reduce[i] += to_reduce[i + half_N + 1];
        }
    }
}


__global__ void reduceKernel2D(unsigned long* list, size_t list_pitch,
                               size_t width, size_t height) {
    // Get the index of this particular thread
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // Decode the i
    size_t x = i % width;
    size_t y = i / width;

    // Only do work if still within range
    if (x < width && y < height / 2) {
        // Simply sum this and the next element. Make sure to stay in bounds
        size_t half_N = ceil(height / 2.0);
        unsigned long* list_ptr = (unsigned long*) ((char*) list + y * list_pitch) + x;
        unsigned long list_val = *((unsigned long*) ((char*) list + (y + half_N) * list_pitch) + x);
        *list_ptr += list_val;
    }
}


__global__ void reduceKernel3D(unsigned long* list, size_t list_pitch,
                               size_t width, size_t height, size_t depth) {
    // Get the index of this particular thread
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // Decode the i
    size_t x = i % width;
    size_t yz = i / width;
    size_t y = yz % height;
    size_t z = yz / height;

    // Only do work if still within range
    if (x < width && y < height && z < depth / 2) {
        // Simply sum this and the next element. Make sure to stay in bounds
        size_t half_N = ceil(depth / 2.0);
        unsigned long* list_ptr = (unsigned long*) ((char*) list + z * list_pitch * height + y * list_pitch) + x;
        unsigned long list_val = *((unsigned long*) ((char*) list + (z + half_N) * list_pitch * height + y * list_pitch) + x);
        // unsigned long old = *list_ptr;
        *list_ptr += list_val;

        // printf("(%lu,%lu,%lu): %lu = %lu + %lu\n",
        //        x, y, z, *list_ptr, old, list_val);
    }
}


void reduction_1D() {
    printf("\n\n\n***** ONE-DIMENSIONAL *****\n\n");

    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Create a list of elements to reduce (make it large, for fun)
    size_t N = 5000000;
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
    printf("Sequential result: %lu\n\n", correct);




    
    // Now, enter CUDA!

    // Allocate space for N (with a possible pad), then copy all the data
    unsigned long* to_reduce_gpu;
    cudaMalloc(&to_reduce_gpu, sizeof(unsigned long) * (N + (N % 2 == 0 ? 0 : 1)));
    cudaMemcpy(to_reduce_gpu, to_reduce, sizeof(unsigned long) * N, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

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
        reduceKernelA<<<blocks_per_grid, threads_per_block>>>(to_reduce_gpu, to_do);

        // We can already decrease to_do
        to_do >>= 1;
    }

    // Copy the memory back
    unsigned long result_list[N];
    cudaMemcpy(result_list, to_reduce_gpu, sizeof(unsigned long) * to_do, cudaMemcpyDeviceToHost);

    // Manually combine all intermediate results
    unsigned long result = 0;
    for (size_t i = 0; i < to_do; i++) {
        result += result_list[i];
    }

    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);

    cudaFree(to_reduce_gpu);

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA A time taken: %lu ms\n", time_taken);
    printf("CUDA A result: %lu\n\n", result);






    // Now, enter CUDA (version B!

    // Allocate space for N (with a possible pad), then copy all the data
    cudaMalloc(&to_reduce_gpu, sizeof(unsigned long) * N);
    cudaMemcpy(to_reduce_gpu, to_reduce, sizeof(unsigned long) * N, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

    // Next, invoke the kernel as many times as needed. Let's say that we do the rest manually from 32 and down
    threads_per_block = 32;
    to_do = N;
    while (to_do > 32) {
        // Next, launch the kernel
        int blocks_per_grid = to_do / threads_per_block;
        reduceKernelA<<<blocks_per_grid, threads_per_block>>>(to_reduce_gpu, to_do);

        // We can already decrease to_do
        to_do >>= 1;
    }

    // Copy the memory back
    cudaMemcpy(result_list, to_reduce_gpu, sizeof(unsigned long) * to_do, cudaMemcpyDeviceToHost);

    // Manually combine all intermediate results
    result = 0;
    for (size_t i = 0; i < to_do; i++) {
        result += result_list[i];
    }

    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);

    cudaFree(to_reduce_gpu);

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA B time taken: %lu ms\n", time_taken);
    printf("CUDA B result: %lu\n", result);
}


void reduction_2D() {
    printf("\n\n\n***** TWO-DIMENSIONAL *****\n\n");

    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Initialize a 2D matrix
    size_t W = 5000;
    size_t H = 50000;
    int max = 50;
    unsigned long to_reduce[W * H];
    for (size_t i = 0; i < W * H; i++) {
        to_reduce[i] = rand() % max;
    }

    // Test the sequential version
    // Acquire a correct value, and benchmark the sequential version while we're at it
    gettimeofday(&start, NULL);
    
    unsigned long correct[W];
    for (size_t x = 0; x < W; x++) {
        correct[x] = 0;
        for (size_t y = 0; y < H; y++) {
            correct[x] += to_reduce[y * W + x];
        }
    }

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("Sequential time taken: %lu ms\n\n", time_taken);




    // Now, for CUDA...
    
    // Create the array on the GPU-side
    unsigned long* to_reduce_gpu;
    size_t to_reduce_gpu_pitch;
    cudaMallocPitch((void**) &to_reduce_gpu, &to_reduce_gpu_pitch, sizeof(unsigned long) * W, H);
    cudaSafe();
    cudaMemcpy2D(to_reduce_gpu, to_reduce_gpu_pitch, to_reduce, sizeof(unsigned long) * W, sizeof(unsigned long) * W, H, cudaMemcpyHostToDevice);
    cudaSafe();

    gettimeofday(&start, NULL);

    // unsigned long debug[W * (H + (H % 2 == 0 ? 0 : 1))];
    // cudaMemcpy2D(debug, sizeof(unsigned long) * W, to_reduce_gpu, to_reduce_gpu_pitch, sizeof(unsigned long) * W, H + (H % 2 == 0 ? 0 : 1), cudaMemcpyDeviceToHost);
    // for (size_t y = 0; y < H + (H % 2 == 0 ? 0 : 1); y++) {
    //     printf("[");
    //     for (size_t x = 0; x < W; x++) {
    //         if (x > 0) { printf(", "); }
    //         printf("%04lu", debug[y * W + x]);
    //     }
    //     printf("]\n");
    // }
    // printf("\n");

    // Time to invoke the kernel as many times as needed
    int threads_per_block = 32;
    size_t to_do = H;
    while (to_do > 1) {
        // Launch the kernel
        size_t blocks_per_grid = ceil((to_do * W) / (double) threads_per_block);
        reduceKernel2D<<<blocks_per_grid, threads_per_block>>>(
            to_reduce_gpu, to_reduce_gpu_pitch,
            W, to_do
        );
        cudaSafe();

        // cudaMemcpy2D(debug, sizeof(unsigned long) * W, to_reduce_gpu, to_reduce_gpu_pitch, sizeof(unsigned long) * W, H + (H % 2 == 0 ? 0 : 1), cudaMemcpyDeviceToHost);
        // for (size_t y = 0; y < H + (H % 2 == 0 ? 0 : 1); y++) {
        //     printf("[");
        //     for (size_t x = 0; x < W; x++) {
        //         if (x > 0) { printf(", "); }
        //         printf("%04lu", debug[y * W + x]);
        //     }
        //     printf("]\n");
        // }
        // printf("\n");

        // Don't forget to decrease to_do
        to_do = ceil(to_do / 2.0);
    }

    // Copy the resulting list of values back
    unsigned long result[W];
    cudaMemcpy(result, to_reduce_gpu, sizeof(unsigned long) * W, cudaMemcpyDeviceToHost);
    cudaSafe();

    cudaDeviceSynchronize();
    cudaSafe();

    gettimeofday(&stop, NULL);

    cudaFree(to_reduce_gpu);
    cudaSafe();

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("CUDA time taken: %lu ms\n", time_taken);
    printf("CUDA same as correct? ");
    for (size_t x = 0; x < W; x++) {
        if (correct[x] != result[x]) {
            printf("false\n");
            printf("CUDA vs Correct: %lu vs %lu\n", result[0], correct[0]);
            return;
        }
    }
    printf("true\n");
}


void reduction_3D(size_t W, size_t H, size_t D) {
    printf("\n\n\n***** THREE-DIMENSIONAL *****\n\n");

    struct timeval start, stop;

    // Get us a random seed
    srand(time(NULL));

    // Initialize a 2D matrix
    int max = 50;
    unsigned long* to_reduce = (unsigned long*) malloc(sizeof(unsigned long) * W * H * D);
    for (size_t i = 0; i < W * H * D; i++) {
        to_reduce[i] = rand() % max;
    }

    // Test the sequential version
    // Acquire a correct value, and benchmark the sequential version while we're at it
    gettimeofday(&start, NULL);
    
    unsigned long correct[W * H];
    for (size_t x = 0; x < W; x++) {
        for (size_t y = 0; y < H; y++) {
            correct[y * W + x] = 0;
            for (size_t z = 0; z < D; z++) {
                correct[y * W + x] += to_reduce[z * W * H + y * W + x];
            }
        }
    }

    // for (size_t y = 0; y < H; y++) {
    //     printf("[");
    //     for (size_t x = 0; x < W; x++) {
    //         if (x > 0) { printf(", "); }
    //         printf("%04lu", correct[y * W + x]);
    //     }
    //     printf("]\n");
    // }
    // printf("\n");

    gettimeofday(&stop, NULL);

    unsigned long time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;
    printf("Sequential time taken: %lu ms\n\n", time_taken);




    // Now, for CUDA...
    
    // Create the array on the GPU-side
    unsigned long* to_reduce_gpu;
    size_t to_reduce_gpu_pitch;
    cudaMallocPitch((void**) &to_reduce_gpu, &to_reduce_gpu_pitch, sizeof(unsigned long) * W, H * D);
    cudaSafe();
    cudaMemcpy2D(to_reduce_gpu, to_reduce_gpu_pitch, to_reduce, sizeof(unsigned long) * W, sizeof(unsigned long) * W, H * D, cudaMemcpyHostToDevice);
    cudaSafe();

    gettimeofday(&start, NULL);

    // unsigned long debug[W * H * D];
    // cudaMemcpy2D(debug, sizeof(unsigned long) * W, to_reduce_gpu, to_reduce_gpu_pitch, sizeof(unsigned long) * W, H * D, cudaMemcpyDeviceToHost);
    // for (size_t z = 0; z < D; z++) {
    //     printf("(%lu/%lu)\n", z + 1, D);
    //     for (size_t y = 0; y < H; y++) {
    //         printf("[");
    //         for (size_t x = 0; x < W; x++) {
    //             if (x > 0) { printf(", "); }
    //             printf("%04lu", debug[z * W * H + y * W + x]);
    //         }
    //         printf("]\n");
    //     }
    // }
    // printf("\n");

    // Time to invoke the kernel as many times as needed
    int threads_per_block = 32;
    size_t to_do = D;
    while (to_do > 1) {
        // Launch the kernel
        size_t blocks_per_grid = ceil((to_do * W * H) / (double) threads_per_block);
        reduceKernel3D<<<blocks_per_grid, threads_per_block>>>(
            to_reduce_gpu, to_reduce_gpu_pitch,
            W, H, to_do
        );
        cudaSafe();

        // cudaMemcpy2D(debug, sizeof(unsigned long) * W, to_reduce_gpu, to_reduce_gpu_pitch, sizeof(unsigned long) * W, H * D, cudaMemcpyDeviceToHost);
        // for (size_t z = 0; z < D; z++) {
        //     printf("(%lu/%lu)\n", z + 1, D);
        //     for (size_t y = 0; y < H; y++) {
        //         printf("[");
        //         for (size_t x = 0; x < W; x++) {
        //             if (x > 0) { printf(", "); }
        //             printf("%04lu", debug[z * W * H + y * W + x]);
        //         }
        //         printf("]\n");
        //     }
        // }
        // printf("\n");

        // Don't forget to decrease to_do
        to_do = ceil(to_do / 2.0);
    }

    // Copy the resulting list of values back
    unsigned long result[W * H];
    cudaMemcpy2D(result, sizeof(unsigned long) * W, to_reduce_gpu, to_reduce_gpu_pitch, sizeof(unsigned long) * W, H, cudaMemcpyDeviceToHost);
    cudaSafe();

    cudaDeviceSynchronize();
    cudaSafe();

    gettimeofday(&stop, NULL);

    cudaFree(to_reduce_gpu);
    cudaSafe();

    time_taken = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)) / 1000;

    free(to_reduce);

    printf("CUDA time taken: %lu ms\n", time_taken);
    printf("CUDA same as correct? ");
    for (size_t y = 0; y < H; y++) {
        for (size_t x = 0; x < W; x++) {
            if (correct[y * W + x] != result[y * W + x]) {
                printf("false\n");
                printf("CUDA vs Correct @ (%lu,%lu): %lu vs %lu\n", x, y, result[y * W + x], correct[y * W + x]);
                return;
            }
        }
    }
    printf("true\n");
}


int main() {
    // reduction_1D();

    // reduction_2D();

    reduction_3D(250, 500, 100);

    printf("\n\n\n");
    return 0;
}