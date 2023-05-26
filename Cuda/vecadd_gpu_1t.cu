#include <iostream>

__global__ void vectorAdditionGPU(int size, float* a, float* b, float* c) {
    for (int i = 0; i < size; i++){
          c[i] = a[i] + b[i];
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
    vectorAdditionGPU<<<1, 1>>>(size, a, b, c);
    cudaDeviceSynchronize();

    // Free the managed memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}