/* CUDA KERNEL.cu
 *   by Anonymous
 *
 * Created:
 *   5/24/2020, 1:06:04 PM
 * Last edited:
 *   5/24/2020, 3:15:42 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Test file for CUDA-related operations
**/

#include <stdio.h>

#include "Array.h"

__global__ void testKernel(double* test, int N, int M) {
    // Simply add one to our element in the array
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        test[i] += 5;
    }
}


int main() {
    int N = 50000;

    // Create an array with all-zeroes here
    array* all_zeroes = create_empty_array(N);
    for (int i = 0; i < N; i++) {
        all_zeroes->d[i] = i;
    }

    // Copy that data to the GPU
    double* gpu_arr;
    cudaMalloc((void**) &gpu_arr, all_zeroes->size * sizeof(double));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        destroy_array(all_zeroes);
        printf("ERROR: Could not allocate memory on GPU: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the array
    cudaMemcpy(gpu_arr, all_zeroes->d, all_zeroes->size * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        destroy_array(all_zeroes);
        cudaFree(gpu_arr);
        printf("ERROR: Could not copy memory to GPU: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Now, run the kernel for that pointer
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    testKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_arr, all_zeroes->size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        destroy_array(all_zeroes);
        cudaFree(gpu_arr);
        printf("ERROR: Could not launch kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the memory back
    cudaMemcpy(all_zeroes->d, gpu_arr, all_zeroes->size * sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        destroy_array(all_zeroes);
        cudaFree(gpu_arr);
        printf("ERROR: Could not copy memory back to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free GPU memory
    cudaFree(gpu_arr);

    // Make sure everything is 1
    for (int i = 0; i < all_zeroes->size; i++) {
        if (all_zeroes->d[i] != i + 5) {
            fprintf(stderr, "ERROR: Element %d is not %d, but %f\n", i, i + 5, all_zeroes->d[i]);
            break;
        }
    }

    // // Print as well
    // array_print(stdout, all_zeroes);

    // Cleanup and done
    destroy_array(all_zeroes);

    return 0;
}
