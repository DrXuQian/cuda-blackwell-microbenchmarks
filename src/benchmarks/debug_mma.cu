#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

constexpr int M = 16, N = 16, K = 16;

__global__ void debug_mma_kernel(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    wmma::load_matrix_sync(frag_A, A, K);
    wmma::load_matrix_sync(frag_B, B, N);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::store_matrix_sync(C, frag_C, N, wmma::mem_row_major);
}

void print_matrix(const char* name, const float* matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_half(const char* name, const half* matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", __half2float(matrix[i * cols + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("Debug MMA kernel with %dx%dx%d matrices\n", M, N, K);
    
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);
    
    half* h_A = (half*)malloc(size_A);
    half* h_B = (half*)malloc(size_B);
    float* h_C_mma = (float*)malloc(size_C);
    float* h_C_cublas = (float*)malloc(size_C);
    
    // Initialize with simple patterns
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            h_A[i * K + k] = __float2half(i == k ? 1.0f : 0.0f);  // Identity-like
        }
    }
    
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            h_B[k * N + j] = __float2half(k == j ? 2.0f : 0.0f);  // Scaled identity-like
        }
    }
    
    print_matrix_half("Matrix A", h_A, M, K);
    print_matrix_half("Matrix B", h_B, K, N);
    
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Test MMA kernel
    dim3 grid(1, 1);
    dim3 block(32);
    
    debug_mma_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaMemcpy(h_C_mma, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Compare with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost);
    
    print_matrix("MMA Result", h_C_mma, M, N);
    print_matrix("cuBLAS Result", h_C_cublas, M, N);
    
    // Check differences
    bool match = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_C_mma[i] - h_C_cublas[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > 1e-5) {
            match = false;
        }
    }
    
    printf("Match: %s, Max difference: %f\n", match ? "YES" : "NO", max_diff);
    
    cublasDestroy(handle);
    
    free(h_A);
    free(h_B);
    free(h_C_mma);
    free(h_C_cublas);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}