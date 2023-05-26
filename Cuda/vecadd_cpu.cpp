#include <iostream>
#include <chrono>

void vectorAdditionCPU(float* a, float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int size = 1<<26; // Size of the input vectors

    // Allocate memory for the input and output vectors
    float* a = new float[size];
    float* b = new float[size];
    float* c = new float[size];

    // Initialize input vectors
    for (int i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Perform vector addition on the CPU
    auto start = std::chrono::high_resolution_clock::now();
    vectorAdditionCPU(a, b, c, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Calculate the MFLOP/s and memory bandwidth utilized
    float mflops = (2 * size) / (duration.count() / 1000.0f) / 1e6;
    float bandwidth = (3 * size * sizeof(float)) / (duration.count() / 1000.0f) / 1e9;

    std::cout << "Elapsed Time: " << duration.count() << " ms" << std::endl;
    std::cout << "MFLOP/s: " << mflops << std::endl;
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Clean up
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}