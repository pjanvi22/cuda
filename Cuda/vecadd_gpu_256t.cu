#include <iostream>

__global__ void vectorAdditionGPU(int size, float* a, float* b, float* c) {
    int tid = threadIdx.x; // Thread index within the block
    int stride = blockDim.x; // Number of threads in a block

    // Compute the index and stride for each thread
    int index = tid + blockIdx.x * stride;

    // Perform vector addition
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int size = 1<<20; // Size of the input vectors

    // Allocate managed memory for the input and output vectors
    float* a, * b, * c;
    cudaMallocManaged(&a, size * sizeof(float));
    cudaMallocManaged(&b, size * sizeof(float));
    cudaMallocManaged(&c, size * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Perform vector addition on the GPU
    int blockSize = 256; // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize; // Number of blocks needed
    vectorAdditionGPU<<<numBlocks, blockSize>>>(size, a, b, c);
    cudaDeviceSynchronize();


    // Free the managed memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}