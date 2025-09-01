#include "../utils/common.h"
#include "../kernels/layernorm_gemm_fusion.cu"
#include "../kernels/naive_gemm.cu"
#include <cublas_v2.h>
#include <chrono>
#include <algorithm>

// Simple timing function
double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

// Simple reference LayerNorm
__global__ void simple_layernorm_kernel(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    const int lane_id = threadIdx.x % 32;
    const half* input_row = input + row * K;
    half* output_row = output + row * K;
    
    // Compute mean
    float sum = 0.0f;
    for (int k = lane_id; k < K; k += 32) {
        sum += __half2float(input_row[k]);
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / K;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int k = lane_id; k < K; k += 32) {
        float diff = __half2float(input_row[k]) - mean;
        var_sum += diff * diff;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    float inv_std = rsqrtf(__shfl_sync(0xFFFFFFFF, var_sum, 0) / K + 1e-5f);
    
    // Apply normalization
    for (int k = lane_id; k < K; k += 32) {
        float normalized = (__half2float(input_row[k]) - mean) * inv_std;
        float result = normalized * __half2float(gamma[k]) + __half2float(beta[k]);
        output_row[k] = __float2half(result);
    }
}

int main() {
    printf("🧪 Simple Async Pipeline Performance Test\n");
    printf("=========================================\n\n");
    
    const int M = 9600, K = 2730, N = 1024;
    const int warmup_iters = 5;
    const int test_iters = 20;
    
    // Allocate device memory
    half *d_input, *d_weights, *d_gamma, *d_beta, *d_output, *d_normalized;
    cudaMalloc(&d_input, M * K * sizeof(half));
    cudaMalloc(&d_weights, K * N * sizeof(half));
    cudaMalloc(&d_gamma, K * sizeof(half));
    cudaMalloc(&d_beta, K * sizeof(half));
    cudaMalloc(&d_output, M * N * sizeof(half));
    cudaMalloc(&d_normalized, M * K * sizeof(half));
    
    // Simple manual initialization to avoid cuRAND issues
    half *h_temp = new half[std::max(M * K, K * N)];
    srand(42);
    
    // Initialize input
    for (int i = 0; i < M * K; i++) {
        h_temp[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.2f);
    }
    cudaMemcpy(d_input, h_temp, M * K * sizeof(half), cudaMemcpyHostToDevice);
    
    // Initialize weights
    for (int i = 0; i < K * N; i++) {
        h_temp[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    cudaMemcpy(d_weights, h_temp, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Initialize gamma
    for (int i = 0; i < K; i++) {
        h_temp[i] = __float2half(1.0f + ((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    cudaMemcpy(d_gamma, h_temp, K * sizeof(half), cudaMemcpyHostToDevice);
    
    // Initialize beta
    for (int i = 0; i < K; i++) {
        h_temp[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }
    cudaMemcpy(d_beta, h_temp, K * sizeof(half), cudaMemcpyHostToDevice);
    
    delete[] h_temp;
    
    cudaDeviceSynchronize();
    
    printf("📊 Benchmarking Results:\n");
    printf("=======================\n");
    
    // Test 1: Naive GEMM only (baseline)
    printf("[1/4] Naive GEMM only (baseline)...\n");
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        launch_cublas_gemm(d_input, d_weights, d_output, M, K, N);
    }
    cudaDeviceSynchronize();
    
    double start_time = get_time_ms();
    for (int i = 0; i < test_iters; i++) {
        launch_cublas_gemm(d_input, d_weights, d_output, M, K, N);
    }
    cudaDeviceSynchronize();
    double naive_gemm_time = (get_time_ms() - start_time) / test_iters;
    
    double flops = 2.0 * M * N * K;
    double naive_gflops = flops / (naive_gemm_time / 1000.0) / 1e9;
    printf("  Time: %.3f ms, GFLOPS: %.1f\n", naive_gemm_time, naive_gflops);
    
    // Test 2: Separate LayerNorm + GEMM
    printf("[2/4] Separate LayerNorm + cuBLAS GEMM...\n");
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const half alpha = __float2half(1.0f);
    const half beta_val = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        simple_layernorm_kernel<<<ceildiv_naive(M, 256), 256>>>(d_input, d_normalized, d_gamma, d_beta, M, K);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     d_weights, CUDA_R_16F, N, d_normalized, CUDA_R_16F, K, &beta_val,
                     d_output, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    start_time = get_time_ms();
    for (int i = 0; i < test_iters; i++) {
        simple_layernorm_kernel<<<ceildiv_naive(M, 256), 256>>>(d_input, d_normalized, d_gamma, d_beta, M, K);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     d_weights, CUDA_R_16F, N, d_normalized, CUDA_R_16F, K, &beta_val,
                     d_output, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    double separate_time = (get_time_ms() - start_time) / test_iters;
    
    double separate_gflops = flops / (separate_time / 1000.0) / 1e9;
    double layernorm_overhead = ((separate_time - naive_gemm_time) / naive_gemm_time) * 100;
    printf("  Time: %.3f ms, GFLOPS: %.1f, LayerNorm overhead: %.1f%%\n", 
           separate_time, separate_gflops, layernorm_overhead);
    
    // Test 3: Fused Kernel
    printf("[3/4] Fused LayerNorm + GEMM kernel...\n");
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        layernorm_gemm_fused(d_input, d_weights, d_output, d_gamma, d_beta, M, K, N);
    }
    cudaDeviceSynchronize();
    
    start_time = get_time_ms();
    for (int i = 0; i < test_iters; i++) {
        layernorm_gemm_fused(d_input, d_weights, d_output, d_gamma, d_beta, M, K, N);
    }
    cudaDeviceSynchronize();
    double fused_time = (get_time_ms() - start_time) / test_iters;
    
    double fused_gflops = flops / (fused_time / 1000.0) / 1e9;
    printf("  Time: %.3f ms, GFLOPS: %.1f\n", fused_time, fused_gflops);
    
    // Test 4: Custom tensor core GEMM
    printf("[4/4] Custom tensor core GEMM...\n");
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        launch_naive_tensor_core_gemm(d_input, d_weights, d_output, M, K, N);
    }
    cudaDeviceSynchronize();
    
    start_time = get_time_ms();
    for (int i = 0; i < test_iters; i++) {
        launch_naive_tensor_core_gemm(d_input, d_weights, d_output, M, K, N);
    }
    cudaDeviceSynchronize();
    double custom_time = (get_time_ms() - start_time) / test_iters;
    
    double custom_gflops = flops / (custom_time / 1000.0) / 1e9;
    printf("  Time: %.3f ms, GFLOPS: %.1f\n", custom_time, custom_gflops);
    
    cublasDestroy(handle);
    
    printf("\n🏆 Performance Summary:\n");
    printf("======================\n");
    printf("%-35s: %8.3f ms (%6.1f GFLOPS) - Baseline\n", 
           "Naive GEMM only", naive_gemm_time, naive_gflops);
    printf("%-35s: %8.3f ms (%6.1f GFLOPS) - %.1f%% overhead\n", 
           "Separate LayerNorm + cuBLAS", separate_time, separate_gflops, layernorm_overhead);
    printf("%-35s: %8.3f ms (%6.1f GFLOPS) - %.2fx vs separate\n", 
           "Fused Kernel", fused_time, fused_gflops, separate_time / fused_time);
    printf("%-35s: %8.3f ms (%6.1f GFLOPS) - Custom reference\n", 
           "Custom Tensor Core GEMM", custom_time, custom_gflops);
    
    printf("\n💡 Key Insights:\n");
    printf("===============\n");
    printf("• LayerNorm adds %.1f%% computational overhead to GEMM\n", layernorm_overhead);
    printf("• Fused kernel achieves %.2fx %s vs separate approach\n", 
           separate_time / fused_time, 
           (separate_time > fused_time) ? "speedup" : "slowdown");
    printf("• Custom GEMM efficiency: %.1f%% of cuBLAS performance\n", 
           (custom_gflops / naive_gflops) * 100);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    cudaFree(d_normalized);
    
    printf("\n✨ Simple benchmark complete!\n");
    return 0;
}