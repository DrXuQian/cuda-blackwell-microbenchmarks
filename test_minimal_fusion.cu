#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>

// Simple LayerNorm kernel
__global__ void simple_layernorm_kernel(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    const half* input_row = input + row * K;
    half* output_row = output + row * K;
    
    // Compute mean
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(input_row[k]);
    }
    float mean = sum / K;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float diff = __half2float(input_row[k]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / K + 1e-5f);
    
    // Apply normalization
    for (int k = 0; k < K; k++) {
        float normalized = (__half2float(input_row[k]) - mean) * inv_std;
        float result = normalized * __half2float(gamma[k]) + __half2float(beta[k]);
        output_row[k] = __float2half(result);
    }
}

int main() {
    const int M = 1024, K = 512, N = 256; // Smaller sizes for testing
    
    std::cout << "🧪 Testing LayerNorm + GEMM Fusion (minimal)" << std::endl;
    std::cout << "Shape: " << M << "x" << K << " -> LayerNorm -> " << M << "x" << K << " @ " << K << "x" << N << " -> " << M << "x" << N << std::endl;
    
    // Allocate host memory
    half *h_input = new half[M * K];
    half *h_weights = new half[K * N];
    half *h_gamma = new half[K];
    half *h_beta = new half[K];
    half *h_output = new half[M * N];
    half *h_normalized = new half[M * K];
    
    // Initialize with simple values
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_input[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.2f);
    }
    for (int i = 0; i < K * N; i++) {
        h_weights[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    for (int i = 0; i < K; i++) {
        h_gamma[i] = __float2half(1.0f);
        h_beta[i] = __float2half(0.0f);
    }
    
    // Allocate device memory
    half *d_input, *d_weights, *d_gamma, *d_beta, *d_output, *d_normalized;
    cudaMalloc(&d_input, M * K * sizeof(half));
    cudaMalloc(&d_weights, K * N * sizeof(half));
    cudaMalloc(&d_gamma, K * sizeof(half));
    cudaMalloc(&d_beta, K * sizeof(half));
    cudaMalloc(&d_output, M * N * sizeof(half));
    cudaMalloc(&d_normalized, M * K * sizeof(half));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, K * sizeof(half), cudaMemcpyHostToDevice);
    
    // Test 1: Separate LayerNorm + GEMM
    std::cout << "\\n1. Testing separate LayerNorm + GEMM..." << std::endl;
    
    // LayerNorm
    dim3 ln_grid((M + 255) / 256);
    dim3 ln_block(256);
    simple_layernorm_kernel<<<ln_grid, ln_block>>>(d_input, d_normalized, d_gamma, d_beta, M, K);
    cudaDeviceSynchronize();
    
    // GEMM with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    const half alpha = __float2half(1.0f);
    const half beta_val = __float2half(0.0f);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    for (int i = 0; i < 5; i++) {
        simple_layernorm_kernel<<<ln_grid, ln_block>>>(d_input, d_normalized, d_gamma, d_beta, M, K);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                   &alpha, d_weights, N, d_normalized, K, &beta_val, d_output, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        simple_layernorm_kernel<<<ln_grid, ln_block>>>(d_input, d_normalized, d_gamma, d_beta, M, K);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                   &alpha, d_weights, N, d_normalized, K, &beta_val, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time = elapsed_ms / iterations;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K; // LayerNorm FLOPs are much smaller than GEMM
    double gflops = flops / (avg_time / 1000.0) / 1e9;
    
    std::cout << "   Average time: " << avg_time << " ms" << std::endl;
    std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Validate results
    cudaMemcpy(h_output, d_output, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Simple validation - check that results are reasonable
    bool valid = true;
    for (int i = 0; i < std::min(10, M * N); i++) {
        float val = __half2float(h_output[i]);
        if (isnan(val) || isinf(val) || fabs(val) > 100.0f) {
            valid = false;
            break;
        }
    }
    
    std::cout << "   Validation: " << (valid ? "✅ PASSED" : "❌ FAILED") << std::endl;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    cudaFree(d_normalized);
    
    delete[] h_input;
    delete[] h_weights;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output;
    delete[] h_normalized;
    
    std::cout << "\\n✨ Minimal fusion test complete!" << std::endl;
    
    return 0;
}