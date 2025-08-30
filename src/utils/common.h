#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M = 16, N = 16, K = 16;
constexpr int BLOCK_SIZE = 128;

// Cosine distance accuracy verification
struct AccuracyResult {
    bool passed;
    double cosine_similarity;
    double cosine_distance;
    float max_abs_error;
    float mean_abs_error;
};

__host__ AccuracyResult verify_with_cosine_distance(
    const float* gpu_result, const float* cpu_result, int size, 
    double similarity_threshold = 0.9999) {
    
    AccuracyResult result = {};
    
    // Calculate dot product and norms
    double dot_product = 0.0;
    double norm_gpu = 0.0;
    double norm_cpu = 0.0;
    float max_error = 0.0f;
    double sum_abs_error = 0.0;
    
    for (int i = 0; i < size; i++) {
        double gpu_val = static_cast<double>(gpu_result[i]);
        double cpu_val = static_cast<double>(cpu_result[i]);
        
        dot_product += gpu_val * cpu_val;
        norm_gpu += gpu_val * gpu_val;
        norm_cpu += cpu_val * cpu_val;
        
        float abs_error = fabsf(gpu_result[i] - cpu_result[i]);
        max_error = fmaxf(max_error, abs_error);
        sum_abs_error += abs_error;
    }
    
    norm_gpu = sqrt(norm_gpu);
    norm_cpu = sqrt(norm_cpu);
    
    // Compute cosine similarity and distance
    result.cosine_similarity = dot_product / (norm_gpu * norm_cpu);
    result.cosine_distance = 1.0 - result.cosine_similarity;
    result.max_abs_error = max_error;
    result.mean_abs_error = static_cast<float>(sum_abs_error / size);
    result.passed = result.cosine_similarity >= similarity_threshold;
    
    return result;
}

__host__ void print_accuracy_result(const AccuracyResult& result, const char* kernel_name) {
    printf("%s Accuracy Results:\n", kernel_name);
    printf("  Cosine Similarity: %.8f\n", result.cosine_similarity);
    printf("  Cosine Distance:   %.8f\n", result.cosine_distance);
    printf("  Max Abs Error:     %.8f\n", result.max_abs_error);
    printf("  Mean Abs Error:    %.8f\n", result.mean_abs_error);
    printf("  Status: %s\n", result.passed ? "✅ PASSED" : "❌ FAILED");
}

__host__ bool verify_with_cublas(const half* h_A, const half* h_B, const float* h_C_gpu, 
                                 int M_dim, int N_dim, int K_dim, AccuracyResult* acc_result) {
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C_cublas;
    
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_cublas, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N_dim, M_dim, K_dim,
                 &alpha,
                 d_B, CUDA_R_16F, N_dim,
                 d_A, CUDA_R_16F, K_dim,
                 &beta,
                 d_C_cublas, CUDA_R_32F, N_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    float* h_C_cublas = (float*)malloc(size_C);
    cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);
    
    // Compute cosine distance
    *acc_result = verify_with_cosine_distance(h_C_gpu, h_C_cublas, M_dim * N_dim);
    
    // Cleanup
    cublasDestroy(handle);
    free(h_C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_cublas);
    
    return acc_result->passed;
}

__host__ void benchmark_kernel(void (*kernel)(const half*, const half*, float*, int, int, int),
                              const char* kernel_name,
                              const half* d_A, const half* d_B, float* d_C,
                              int M_dim, int N_dim, int K_dim,
                              size_t shmem_size,
                              int num_iterations = 100) {
    
    dim3 grid((N_dim + N - 1) / N, (M_dim + M - 1) / M);
    dim3 block(BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(error));
        return;
    }
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double flops = 2.0 * M_dim * N_dim * K_dim * num_iterations;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;
    
    printf("%s: %.3f ms (avg), %.1f GFLOPS\n", kernel_name, elapsed_time / num_iterations, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__host__ void benchmark_cublas(const half* d_A, const half* d_B, float* d_C,
                              int M_dim, int N_dim, int K_dim,
                              int num_iterations = 100) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_dim, M_dim, K_dim,
                     &alpha, d_B, CUDA_R_16F, N_dim,
                     d_A, CUDA_R_16F, K_dim, &beta,
                     d_C, CUDA_R_32F, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_dim, M_dim, K_dim,
                     &alpha, d_B, CUDA_R_16F, N_dim,
                     d_A, CUDA_R_16F, K_dim, &beta,
                     d_C, CUDA_R_32F, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double flops = 2.0 * M_dim * N_dim * K_dim * num_iterations;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;
    
    printf("cuBLAS: %.3f ms (avg), %.1f GFLOPS\n", elapsed_time / num_iterations, gflops);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}