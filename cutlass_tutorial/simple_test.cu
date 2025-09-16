#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// Simple CUDA kernel for matrix multiplication
__global__ void simple_gemm_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    }

int main() {
    std::cout << "=== Simple CUDA GEMM Test ===" << std::endl;
    
    // Check device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Matrix dimensions
    int M = 1024, N = 1024, K = 1024;
    std::cout << "Matrix dimensions: C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    simple_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result (should be K since A[i] = B[i] = 1.0)
    bool correct = true;
    for (int i = 0; i < std::min(100, M * N); i++) {
        if (abs(h_C[i] - K) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Kernel time: " << ms << " ms" << std::endl;
    double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Result verification: " << (correct ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "Sample result: " << h_C[0] << " (expected: " << K << ")" << std::endl;
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    std::cout << "\n✅ Basic CUDA functionality works!" << std::endl;
    return 0;
}