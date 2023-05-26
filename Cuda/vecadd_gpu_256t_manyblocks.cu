#include <iostream>

__global__ void vectorAdditionGPU(int size, float* a, float* b, float* c) {
    int tid = threadIdx.x; // Thread index within the block
    int stride = blockDim.x * gridDim.x; // Total number of threads

    // Compute the index for each thread
    int index = tid + blockIdx.x * blockDim.x;

    // Perform vector addition
    for (int i = index; i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int size = 1000000; // Size of the input vectors

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

    // Set the number of threads per block and number of blocks
    int blockSize = 256; // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize; // Number of blocks needed

    // Perform vector addition on the GPU
    vectorAdditionGPU<<<numBlocks, blockSize>>>(size, a, b, c);
    cudaDeviceSynchronize();



    // Print the number of thread blocks used
    std::cout << "Number of Thread Blocks: " << numBlocks << std::endl;

    // Free the managed memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}