#include "../utils/common.h"
#include "../kernels/layernorm_gemm_fusion.cu"
#include "../kernels/layernorm_gemm_async_pipeline.cu"
#include "../kernels/naive_gemm.cu"
#include <cublas_v2.h>
#include <curand.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

// Forward declarations
__global__ void reference_layernorm_kernel_enhanced(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K
);

void launch_reference_layernorm_enhanced(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K, cudaStream_t stream
);

// Enhanced benchmark result structure
struct EnhancedBenchmarkResult {
    std::string method_name;
    double total_time_ms;
    double layernorm_time_ms;
    double gemm_time_ms;
    double gflops;
    double bandwidth_gb_s;
    double compute_utilization;
    double memory_utilization;
    bool accuracy_passed;
    double cosine_similarity;
    float max_abs_error;
    size_t shared_memory_usage;
};

// Performance measurement utility with more detailed metrics
class EnhancedPerformanceTimer {
private:
    cudaEvent_t start_, stop_, ln_start_, ln_stop_, gemm_start_, gemm_stop_;
    
public:
    EnhancedPerformanceTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventCreate(&ln_start_);
        cudaEventCreate(&ln_stop_);
        cudaEventCreate(&gemm_start_);
        cudaEventCreate(&gemm_stop_);
    }
    
    ~EnhancedPerformanceTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaEventDestroy(ln_start_);
        cudaEventDestroy(ln_stop_);
        cudaEventDestroy(gemm_start_);
        cudaEventDestroy(gemm_stop_);
    }
    
    void start() { cudaEventRecord(start_); }
    void stop() { cudaEventRecord(stop_); }
    void start_layernorm() { cudaEventRecord(ln_start_); }
    void stop_layernorm() { cudaEventRecord(ln_stop_); }
    void start_gemm() { cudaEventRecord(gemm_start_); }
    void stop_gemm() { cudaEventRecord(gemm_stop_); }
    
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time;
    }
    
    float layernorm_elapsed_ms() {
        cudaEventSynchronize(ln_stop_);
        float time;
        cudaEventElapsedTime(&time, ln_start_, ln_stop_);
        return time;
    }
    
    float gemm_elapsed_ms() {
        cudaEventSynchronize(gemm_stop_);
        float time;
        cudaEventElapsedTime(&time, gemm_start_, gemm_stop_);
        return time;
    }
};

// Enhanced test data manager
class EnhancedTestDataManager {
private:
    int M_, K_, N_;
    half *h_input_, *h_weights_, *h_gamma_, *h_beta_;
    half *d_input_, *d_weights_, *d_gamma_, *d_beta_;
    half *d_output_ref_, *d_output_test_, *d_normalized_;
    half *d_output_naive_gemm_;  // For naive GEMM baseline
    
public:
    EnhancedTestDataManager(int M, int K, int N) : M_(M), K_(K), N_(N),
        h_input_(nullptr), h_weights_(nullptr), h_gamma_(nullptr), h_beta_(nullptr),
        d_input_(nullptr), d_weights_(nullptr), d_gamma_(nullptr), d_beta_(nullptr),
        d_output_ref_(nullptr), d_output_test_(nullptr), d_normalized_(nullptr),
        d_output_naive_gemm_(nullptr) {
        
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
        cudaMalloc(&d_output_naive_gemm_, M * N * sizeof(half));
        
        generate_test_data();
    }
    
    ~EnhancedTestDataManager() {
        if (h_input_) { delete[] h_input_; }
        if (h_weights_) { delete[] h_weights_; }
        if (h_gamma_) { delete[] h_gamma_; }
        if (h_beta_) { delete[] h_beta_; }
        
        if (d_input_) { cudaFree(d_input_); }
        if (d_weights_) { cudaFree(d_weights_); }
        if (d_gamma_) { cudaFree(d_gamma_); }
        if (d_beta_) { cudaFree(d_beta_); }
        if (d_output_ref_) { cudaFree(d_output_ref_); }
        if (d_output_test_) { cudaFree(d_output_test_); }
        if (d_normalized_) { cudaFree(d_normalized_); }
        if (d_output_naive_gemm_) { cudaFree(d_output_naive_gemm_); }
    }
    
private:
    void generate_test_data() {
        srand(42);
        
        // Generate input data with realistic distribution
        for (int i = 0; i < M_ * K_; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            h_input_[i] = __float2half(val);
        }
        
        // Generate weights  
        for (int i = 0; i < K_ * N_; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            h_weights_[i] = __float2half(val);
        }
        
        // Generate LayerNorm parameters
        for (int i = 0; i < K_; i++) {
            float gamma_val = 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            float beta_val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            h_gamma_[i] = __float2half(gamma_val);
            h_beta_[i] = __float2half(beta_val);
        }
        
        // Copy to device
        cudaMemcpy(d_input_, h_input_, M_ * K_ * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights_, h_weights_, K_ * N_ * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma_, h_gamma_, K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_, h_beta_, K * sizeof(half), cudaMemcpyHostToDevice);
    }
    
public:
    // Getters
    const half* input() const { return d_input_; }
    const half* weights() const { return d_weights_; }
    const half* gamma() const { return d_gamma_; }
    const half* beta() const { return d_beta_; }
    half* output_ref() const { return d_output_ref_; }
    half* output_test() const { return d_output_test_; }
    half* normalized() const { return d_normalized_; }
    half* output_naive_gemm() const { return d_output_naive_gemm_; }
    
    void generate_reference_result() {
        // Step 1: LayerNorm - use the reference kernel defined below
        launch_reference_layernorm_enhanced(d_input_, d_normalized_, d_gamma_, d_beta_, M_, K_, 0);
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
    
    void generate_naive_gemm_baseline() {
        // Just naive GEMM without LayerNorm
        launch_cublas_gemm(d_input_, d_weights_, d_output_naive_gemm_, M_, K_, N_);
        cudaDeviceSynchronize();
    }
    
    bool validate_result(const half* test_output, double& cosine_sim, float& max_error) {
        half* h_ref = new half[M_ * N_];
        half* h_test = new half[M_ * N_];
        
        cudaMemcpy(h_ref, d_output_ref_, M_ * N_ * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_test, test_output, M_ * N_ * sizeof(half), cudaMemcpyDeviceToHost);
        
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

// Reference LayerNorm implementation (same as before)
__global__ void reference_layernorm_kernel_enhanced(
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

void launch_reference_layernorm_enhanced(
    const half* input, half* output, const half* gamma, const half* beta,
    int M, int K, cudaStream_t stream = 0
) {
    dim3 grid(ceildiv(M, 256));
    dim3 block(256);
    reference_layernorm_kernel_enhanced<<<grid, block, 0, stream>>>(input, output, gamma, beta, M, K);
}

// Benchmark functions
EnhancedBenchmarkResult benchmark_naive_gemm_only(EnhancedTestDataManager& data, int iterations = 100) {
    EnhancedPerformanceTimer timer;
    const int M = 9600, K = 2730, N = 1024;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_cublas_gemm(data.input(), data.weights(), data.output_test(), M, K, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++) {
        launch_cublas_gemm(data.input(), data.weights(), data.output_test(), M, K, N);
    }
    timer.stop();
    
    double total_time = timer.elapsed_ms() / iterations;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    return {
        "Naive GEMM (cuBLAS, no LayerNorm)",
        total_time, 0.0, total_time, gflops, bandwidth, 0.0, 0.0,
        true, 1.0, 0.0, 0
    };
}

EnhancedBenchmarkResult benchmark_separate_layernorm_cublas(EnhancedTestDataManager& data, int iterations = 100) {
    EnhancedPerformanceTimer timer;
    const int M = 9600, K = 2730, N = 1024;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_reference_layernorm_enhanced(data.input(), data.normalized(), data.gamma(), data.beta(), M, K, 0);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     data.weights(), CUDA_R_16F, N, data.normalized(), CUDA_R_16F, K, &beta,
                     data.output_test(), CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    double ln_time = 0.0, gemm_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        timer.start_layernorm();
        launch_reference_layernorm_enhanced(data.input(), data.normalized(), data.gamma(), data.beta(), M, K, 0);
        timer.stop_layernorm();
        ln_time += timer.layernorm_elapsed_ms();
        
        timer.start_gemm();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     data.weights(), CUDA_R_16F, N, data.normalized(), CUDA_R_16F, K, &beta,
                     data.output_test(), CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        timer.stop_gemm();
        gemm_time += timer.gemm_elapsed_ms();
    }
    timer.stop();
    cublasDestroy(handle);
    
    double total_time = timer.elapsed_ms() / iterations;
    double avg_ln_time = ln_time / iterations;
    double avg_gemm_time = gemm_time / iterations;
    
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Separate LayerNorm + cuBLAS",
        total_time, avg_ln_time, avg_gemm_time, gflops, bandwidth, 0.0, 0.0,
        passed, cosine_sim, max_error, 0
    };
}

EnhancedBenchmarkResult benchmark_async_pipeline(EnhancedTestDataManager& data, int iterations = 100) {
    EnhancedPerformanceTimer timer;
    const int M = 9600, K = 2730, N = 1024;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        layernorm_gemm_async_pipeline(data.input(), data.weights(), data.output_test(), 
                                    data.gamma(), data.beta(), M, K, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++) {
        layernorm_gemm_async_pipeline(data.input(), data.weights(), data.output_test(), 
                                    data.gamma(), data.beta(), M, K, N);
    }
    timer.stop();
    
    double total_time = timer.elapsed_ms() / iterations;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Async Pipeline (NEW)",
        total_time, 0.0, 0.0, gflops, bandwidth, 0.0, 0.0,
        passed, cosine_sim, max_error, 0
    };
}

EnhancedBenchmarkResult benchmark_fused_kernel(EnhancedTestDataManager& data, int iterations = 100) {
    EnhancedPerformanceTimer timer;
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
    double flops = 2.0 * M * N * K;
    double gflops = flops / (total_time / 1000.0) / 1e9;
    double bytes = M * K * 2 + K * N * 2 + M * N * 2;
    double bandwidth = bytes / (total_time / 1000.0) / 1e9;
    
    double cosine_sim;
    float max_error;
    bool passed = data.validate_result(data.output_test(), cosine_sim, max_error);
    
    return {
        "Fused LayerNorm + GEMM (Original)",
        total_time, 0.0, 0.0, gflops, bandwidth, 0.0, 0.0,
        passed, cosine_sim, max_error, 0
    };
}

void print_enhanced_results(const std::vector<EnhancedBenchmarkResult>& results) {
    printf("\n🚀 Enhanced LayerNorm + GEMM Benchmark Results\n");
    printf("==============================================\n");
    printf("Target Shape: 9600×2730 → LayerNorm → 9600×2730 @ 2730×1024 → 9600×1024\n");
    printf("Data Type: fp16, Using Tensor Cores\n\n");
    
    printf("📊 Performance Comparison:\n");
    printf("=========================\n");
    printf("%-35s %10s %12s %10s %8s\n",
           "Method", "Time(ms)", "GFLOPS", "BW(GB/s)", "Accuracy");
    printf("%-35s %10s %12s %10s %8s\n",
           "------", "--------", "------", "--------", "--------");
    
    for (const auto& result : results) {
        printf("%-35s %10.3f %12.1f %10.1f %8s\n",
               result.method_name.c_str(),
               result.total_time_ms,
               result.gflops,
               result.bandwidth_gb_s,
               result.accuracy_passed ? "✅" : "❌");
    }
    
    // Calculate speedups vs naive GEMM and vs separate implementation
    if (results.size() > 2) {
        printf("\n🏆 Speedup Analysis:\n");
        printf("==================\n");
        
        // Find naive GEMM baseline and separate implementation
        double naive_gemm_time = results[0].total_time_ms;  // Assuming first is naive GEMM
        double separate_time = results[1].total_time_ms;    // Assuming second is separate
        
        printf("Baseline: Naive GEMM (%.3f ms)\n", naive_gemm_time);
        printf("Reference: Separate LayerNorm + cuBLAS (%.3f ms)\n", separate_time);
        printf("\n");
        
        for (size_t i = 0; i < results.size(); i++) {
            if (i == 0) continue;  // Skip naive GEMM baseline
            
            double speedup_vs_naive = naive_gemm_time / results[i].total_time_ms;
            double speedup_vs_separate = separate_time / results[i].total_time_ms;
            double overhead_vs_naive = ((results[i].total_time_ms - naive_gemm_time) / naive_gemm_time) * 100;
            
            printf("%-35s:\n", results[i].method_name.c_str());
            printf("  vs Naive GEMM:     %.2fx speedup (%.1f%% overhead)\n", 
                   speedup_vs_naive, overhead_vs_naive);
            printf("  vs Separate:       %.2fx %s\n", 
                   speedup_vs_separate, speedup_vs_separate > 1.0 ? "speedup" : "slowdown");
            printf("\n");
        }
    }
    
    printf("💡 Key Insights:\n");
    printf("===============\n");
    printf("• LayerNorm overhead: %.1f%% of GEMM computation time\n", 
           results.size() > 1 ? ((results[1].total_time_ms - results[0].total_time_ms) / results[0].total_time_ms) * 100 : 0.0);
    printf("• Software pipelining effectiveness depends on compute-memory balance\n");
    printf("• Async operations can hide LayerNorm latency during GEMM compute phases\n");
    printf("• Memory bandwidth utilization is critical for fusion benefits\n");
}

int main() {
    printf("🧪 Enhanced LayerNorm + GEMM Fusion Performance Analysis\n");
    printf("========================================================\n\n");
    
    const int M = 9600, K = 2730, N = 1024;
    EnhancedTestDataManager data(M, K, N);
    
    printf("⚙️  Generating reference results...\n");
    data.generate_reference_result();
    data.generate_naive_gemm_baseline();
    
    std::vector<EnhancedBenchmarkResult> results;
    
    printf("🔬 Running enhanced benchmarks...\n\n");
    
    // Benchmark 1: Naive GEMM only (baseline)
    printf("[1/4] Naive GEMM without LayerNorm (baseline)...\n");
    results.push_back(benchmark_naive_gemm_only(data));
    
    // Benchmark 2: Separate LayerNorm + cuBLAS GEMM
    printf("[2/4] Separate LayerNorm + cuBLAS GEMM...\n");
    results.push_back(benchmark_separate_layernorm_cublas(data));
    
    // Benchmark 3: Original fused kernel
    printf("[3/4] Original fused LayerNorm + GEMM kernel...\n");
    results.push_back(benchmark_fused_kernel(data));
    
    // Benchmark 4: New async pipeline
    printf("[4/4] New async pipelined implementation...\n");
    results.push_back(benchmark_async_pipeline(data));
    
    // Print comprehensive results
    print_enhanced_results(results);
    
    printf("\n🔍 For detailed profiling, run:\n");
    printf("ncu --set full ./build/bin/enhanced_layernorm_gemm_benchmark\n");
    
    printf("\n✨ Enhanced benchmark complete!\n");
    return 0;
}