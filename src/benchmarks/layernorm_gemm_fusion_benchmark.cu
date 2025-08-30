#include "../utils/common.h"
#include "../kernels/layernorm_gemm_fusion.cu"
#include <cublas_v2.h>
#include <curand.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

// Reference LayerNorm implementation for comparison
__global__ void reference_layernorm_kernel(
    const half* input,      // [M, K]
    half* output,          // [M, K] 
    const half* gamma,      // [K]
    const half* beta,       // [K]
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
    
    // Warp reduction for mean
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

void launch_reference_layernorm(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K, cudaStream_t stream = 0
) {
    dim3 grid(ceildiv(M, 256));
    dim3 block(256);
    reference_layernorm_kernel<<<grid, block, 0, stream>>>(input, output, gamma, beta, M, K);
}

// Use cuBLAS for reference GEMM comparison

// Benchmark result structure
struct FusionBenchmarkResult {
    std::string method_name;
    double total_time_ms;
    double layernorm_time_ms;  // For separate implementations
    double gemm_time_ms;       // For separate implementations
    double gflops;
    double bandwidth_gb_s;
    bool accuracy_passed;
    double cosine_similarity;
    float max_abs_error;
};

// Performance measurement utility
class PerformanceTimer {
private:
    cudaEvent_t start_, stop_;
    
public:
    // Prevent copying to avoid double-destroy of CUDA events
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    
    PerformanceTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~PerformanceTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() { cudaEventRecord(start_); }
    
    void stop() { cudaEventRecord(stop_); }
    
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time;
    }
};

// Data generator and validator
class TestDataManager {
private:
    int M_, K_, N_;
    half *h_input_, *h_weights_, *h_gamma_, *h_beta_;
    half *d_input_, *d_weights_, *d_gamma_, *d_beta_;
    half *d_output_ref_, *d_output_test_, *d_normalized_;
    
public:
    // Prevent copying to avoid double-free issues
    TestDataManager(const TestDataManager&) = delete;
    TestDataManager& operator=(const TestDataManager&) = delete;
    
    TestDataManager(int M, int K, int N) : M_(M), K_(K), N_(N),
        h_input_(nullptr), h_weights_(nullptr), h_gamma_(nullptr), h_beta_(nullptr),
        d_input_(nullptr), d_weights_(nullptr), d_gamma_(nullptr), d_beta_(nullptr),
        d_output_ref_(nullptr), d_output_test_(nullptr), d_normalized_(nullptr) {
        
        // Allocate host memory
        h_input_ = new half[M * K];
        h_weights_ = new half[K * N];
        h_gamma_ = new half[K];
        h_beta_ = new half[K];
        
        // Allocate device memory
        cudaMalloc(&d_input_, M * K * sizeof(half));
        cudaMalloc(&d_weights_, K * N * sizeof(half));
        cudaMalloc(&d_gamma_, K * sizeof(half));
        cudaMalloc(&d_beta_, K * sizeof(half));
        cudaMalloc(&d_output_ref_, M * N * sizeof(half));
        cudaMalloc(&d_output_test_, M * N * sizeof(half));
        cudaMalloc(&d_normalized_, M * K * sizeof(half));
        
        generate_test_data();
    }
    
    ~TestDataManager() {
        if (h_input_) { delete[] h_input_; h_input_ = nullptr; }
        if (h_weights_) { delete[] h_weights_; h_weights_ = nullptr; }
        if (h_gamma_) { delete[] h_gamma_; h_gamma_ = nullptr; }
        if (h_beta_) { delete[] h_beta_; h_beta_ = nullptr; }
        
        if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
        if (d_weights_) { cudaFree(d_weights_); d_weights_ = nullptr; }
        if (d_gamma_) { cudaFree(d_gamma_); d_gamma_ = nullptr; }
        if (d_beta_) { cudaFree(d_beta_); d_beta_ = nullptr; }
        if (d_output_ref_) { cudaFree(d_output_ref_); d_output_ref_ = nullptr; }
        if (d_output_test_) { cudaFree(d_output_test_); d_output_test_ = nullptr; }
        if (d_normalized_) { cudaFree(d_normalized_); d_normalized_ = nullptr; }
    }
    
private:
    void generate_test_data() {
        // Use simple deterministic initialization to avoid std library issues
        srand(42);
        
        // Generate input data with realistic distribution
        for (int i = 0; i < M_ * K_; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // -0.1 to 0.1
            h_input_[i] = __float2half(val);
        }
        
        // Generate weights  
        for (int i = 0; i < K_ * N_; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // -0.05 to 0.05
            h_weights_[i] = __float2half(val);
        }
        
        // Generate LayerNorm parameters
        for (int i = 0; i < K_; i++) {
            float gamma_val = 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // Around 1.0
            float beta_val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;          // Around 0.0
            h_gamma_[i] = __float2half(gamma_val);
            h_beta_[i] = __float2half(beta_val);
        }
        
        // Copy to device
        cudaMemcpy(d_input_, h_input_, M_ * K_ * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights_, h_weights_, K_ * N_ * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma_, h_gamma_, K_ * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_, h_beta_, K_ * sizeof(half), cudaMemcpyHostToDevice);
    }
    
public:
    // Getters for device pointers
    const half* input() const { return d_input_; }
    const half* weights() const { return d_weights_; }
    const half* gamma() const { return d_gamma_; }
    const half* beta() const { return d_beta_; }
    half* output_ref() const { return d_output_ref_; }
    half* output_test() const { return d_output_test_; }
    half* normalized() const { return d_normalized_; }
    
    // Generate reference result using separate LayerNorm + cuBLAS
    void generate_reference_result() {
        // Step 1: LayerNorm
        launch_reference_layernorm(d_input_, d_normalized_, d_gamma_, d_beta_, M_, K_);
        cudaDeviceSynchronize();
        
        // Step 2: cuBLAS GEMM
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_, M_, K_,
                     &alpha,
                     d_weights_, CUDA_R_16F, N_,
                     d_normalized_, CUDA_R_16F, K_,
                     &beta,
                     d_output_ref_, CUDA_R_16F, N_,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        cublasDestroy(handle);
        cudaDeviceSynchronize();
    }
    
    // Validate results against reference
    bool validate_result(const half* test_output, double& cosine_sim, float& max_error) {
        half* h_ref = new half[M_ * N_];
        half* h_test = new half[M_ * N_];
        
        cudaMemcpy(h_ref, d_output_ref_, M_ * N_ * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_test, test_output, M_ * N_ * sizeof(half), cudaMemcpyDeviceToHost);
        
        // Convert to float for accurate comparison
        std::vector<float> ref_float(M_ * N_), test_float(M_ * N_);
        for (int i = 0; i < M_ * N_; i++) {
            ref_float[i] = __half2float(h_ref[i]);
            test_float[i] = __half2float(h_test[i]);
        }
        
        AccuracyResult result = verify_with_cosine_distance(
            test_float.data(), ref_float.data(), M_ * N_, 0.99
        );
        
        cosine_sim = result.cosine_similarity;
        max_error = result.max_abs_error;
        
        delete[] h_ref;
        delete[] h_test;
        
        return result.passed;
    }
};

// Individual benchmark functions
FusionBenchmarkResult benchmark_separate_layernorm_cublas(TestDataManager& data, int iterations = 100) {
    PerformanceTimer timer_total, timer_ln, timer_gemm;
    const int M = 9600, K = 2730, N = 1024;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_reference_layernorm(data.input(), data.normalized(), data.gamma(), data.beta(), M, K);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     data.weights(), CUDA_R_16F, N, data.normalized(), CUDA_R_16F, K, &beta,
                     data.output_test(), CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer_total.start();
    
    double ln_time = 0.0, gemm_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        timer_ln.start();
        launch_reference_layernorm(data.input(), data.normalized(), data.gamma(), data.beta(), M, K);
        timer_ln.stop();
        ln_time += timer_ln.elapsed_ms();
        
        timer_gemm.start();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     data.weights(), CUDA_R_16F, N, data.normalized(), CUDA_R_16F, K, &beta,
                     data.output_test(), CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        timer_gemm.stop();
        gemm_time += timer_gemm.elapsed_ms();
    }
    
    timer_total.stop();
    cublasDestroy(handle);
    
    double total_time = timer_total.elapsed_ms() / iterations;
    double avg_ln_time = ln_time / iterations;
    double avg_gemm_time = gemm_time / iterations;
    
    // Calculate metrics
    double flops = 2.0 * M * N * K; // GEMM FLOPs
    double gflops = flops / (total_time / 1000.0) / 1e9;
    
    double bytes = M * K * 2 + K * N * 2 + M * N * 2; // Input + weights + output (fp16)
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    // Validate accuracy
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Separate LayerNorm + cuBLAS",
        total_time,
        avg_ln_time,
        avg_gemm_time,
        gflops,
        bandwidth,
        passed,
        cosine_sim,
        max_error
    };
}

FusionBenchmarkResult benchmark_fused_kernel(TestDataManager& data, int iterations = 100) {
    PerformanceTimer timer;
    const int M = 9600, K = 2730, N = 1024;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        layernorm_gemm_fused(data.input(), data.weights(), data.output_test(), 
                           data.gamma(), data.beta(), M, K, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++) {
        layernorm_gemm_fused(data.input(), data.weights(), data.output_test(), 
                           data.gamma(), data.beta(), M, K, N);
    }
    timer.stop();
    
    double total_time = timer.elapsed_ms() / iterations;
    
    // Calculate metrics
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    // Validate accuracy
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Fused LayerNorm + GEMM",
        total_time,
        0.0, // No separate LayerNorm timing
        0.0, // No separate GEMM timing
        gflops,
        bandwidth,
        passed,
        cosine_sim,
        max_error
    };
}

FusionBenchmarkResult benchmark_streaming_pipeline(TestDataManager& data, int iterations = 100) {
    PerformanceTimer timer;
    const int M = 9600, K = 2730, N = 1024;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_streaming_layernorm_gemm(data.input(), data.weights(), data.output_test(), 
                                      data.gamma(), data.beta(), M, K, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++) {
        launch_streaming_layernorm_gemm(data.input(), data.weights(), data.output_test(), 
                                      data.gamma(), data.beta(), M, K, N);
    }
    timer.stop();
    
    double total_time = timer.elapsed_ms() / iterations;
    
    // Calculate metrics
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    // Validate accuracy
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Streaming Pipeline",
        total_time,
        0.0,
        0.0,
        gflops,
        bandwidth,
        passed,
        cosine_sim,
        max_error
    };
}

void print_results(const std::vector<FusionBenchmarkResult>& results) {
    printf("\n🚀 LayerNorm + GEMM Fusion Benchmark Results\n");
    printf("===========================================\n");
    printf("Target Shape: 9600×2730 → LayerNorm → 9600×2730 @ 2730×1024 → 9600×1024\n");
    printf("Data Type: fp16\n\n");
    
    printf("📊 Performance Comparison:\n");
    printf("=========================\n");
    printf("%-30s %10s %10s %10s %12s %10s %8s\n",
           "Method", "Total(ms)", "LN(ms)", "GEMM(ms)", "GFLOPS", "BW(GB/s)", "Accuracy");
    printf("%-30s %10s %10s %10s %12s %10s %8s\n",
           "------", "--------", "------", "--------", "------", "--------", "--------");
    
    for (const auto& result : results) {
        printf("%-30s %10.3f %10.3f %10.3f %12.1f %10.1f %8s\n",
               result.method_name.c_str(),
               result.total_time_ms,
               result.layernorm_time_ms,
               result.gemm_time_ms,
               result.gflops,
               result.bandwidth_gb_s,
               result.accuracy_passed ? "✅" : "❌");
    }
    
    // Calculate speedups
    if (results.size() > 1) {
        printf("\n🏆 Speedup Analysis:\n");
        printf("==================\n");
        double baseline_time = results[0].total_time_ms;
        
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = baseline_time / results[i].total_time_ms;
            printf("%-30s: %.2fx speedup (%.1f%% %s)\n",
                   results[i].method_name.c_str(),
                   speedup,
                   std::abs(speedup - 1.0) * 100,
                   speedup > 1.0 ? "faster" : "slower");
        }
    }
    
    printf("\n🔬 Latency Breakdown:\n");
    printf("====================\n");
    for (const auto& result : results) {
        if (result.layernorm_time_ms > 0 && result.gemm_time_ms > 0) {
            double ln_percent = (result.layernorm_time_ms / result.total_time_ms) * 100;
            double gemm_percent = (result.gemm_time_ms / result.total_time_ms) * 100;
            printf("%-30s: LayerNorm %.1f%% (%.3fms), GEMM %.1f%% (%.3fms)\n",
                   result.method_name.c_str(),
                   ln_percent, result.layernorm_time_ms,
                   gemm_percent, result.gemm_time_ms);
        }
    }
}

int main() {
    printf("🧪 LayerNorm + GEMM Fusion Performance Analysis\n");
    printf("===============================================\n\n");
    
    // Initialize test data
    const int M = 9600, K = 2730, N = 1024;
    TestDataManager data(M, K, N);
    
    printf("⚙️  Generating reference results...\n");
    data.generate_reference_result();
    
    std::vector<FusionBenchmarkResult> results;
    
    printf("🔬 Running benchmarks...\n\n");
    
    // Benchmark 1: Separate LayerNorm + cuBLAS GEMM (baseline)
    printf("[1/3] Separate LayerNorm + cuBLAS GEMM...\n");
    results.push_back(benchmark_separate_layernorm_cublas(data));
    
    // Benchmark 2: Fused kernel
    printf("[2/3] Fused LayerNorm + GEMM kernel...\n");
    results.push_back(benchmark_fused_kernel(data));
    
    // Benchmark 3: Streaming pipeline
    printf("[3/3] Streaming pipeline approach...\n");
    results.push_back(benchmark_streaming_pipeline(data));
    
    // Print comprehensive results
    print_results(results);
    
    printf("\n💡 Optimization Insights:\n");
    printf("========================\n");
    printf("• LayerNorm latency hiding effectiveness depends on GEMM compute intensity\n");
    printf("• Fused kernels reduce memory traffic but may have lower compute utilization\n");
    printf("• Streaming approaches work best when LayerNorm and GEMM have similar runtimes\n");
    printf("• Memory bandwidth is often the limiting factor for this fusion\n");
    
    printf("\n🔍 For detailed profiling, run:\n");
    printf("ncu --set full ./build/bin/layernorm_gemm_fusion_benchmark\n");
    
    printf("\n✨ Benchmark complete!\n");
    return 0;
}