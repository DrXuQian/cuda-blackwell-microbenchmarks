#include "common.h"
#include "marlin/marlin_cuda_kernel.cu"
#include "kernels/marlin_gemv_optimized.cu"
#include <random>
#include <vector>
#include <iomanip>

// Enhanced benchmark utilities for GEMV
struct GEMVBenchmarkResult {
    std::string kernel_name;
    double avg_time_ms;
    double gflops;
    double bandwidth_gb_s;
    bool correctness_passed;
    double cosine_similarity;
    float max_abs_error;
};

// Quantization utilities for 4-bit weights
class WeightQuantizer {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;
    
public:
    WeightQuantizer(int seed = 42) : gen(seed), dis(-0.5f, 0.5f) {}
    
    void quantize_weights(const half* fp16_weights, int4* quantized_weights, 
                         half* scales, int N, int M, int group_size = 128) {
        // Simple quantization scheme - in practice would use more sophisticated methods
        for (int col = 0; col < M; col++) {
            for (int group_start = 0; group_start < N; group_start += group_size) {
                int group_end = std::min(group_start + group_size, N);
                
                // Find min/max for this group
                float min_val = 1e6f, max_val = -1e6f;
                for (int row = group_start; row < group_end; row++) {
                    float val = __half2float(fp16_weights[row * M + col]);
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
                
                // Calculate scale and zero point
                float scale = (max_val - min_val) / 15.0f; // 4-bit range: 0-15
                scales[col] = __float2half(scale);
                
                // Quantize weights in this group
                for (int row = group_start; row < group_end; row += 8) {
                    int packed_weights = 0;
                    for (int i = 0; i < 8 && row + i < group_end; i++) {
                        float val = __half2float(fp16_weights[(row + i) * M + col]);
                        int quantized = static_cast<int>((val - min_val) / scale + 0.5f);
                        quantized = std::max(0, std::min(15, quantized)); // Clamp to 4-bit
                        packed_weights |= (quantized << (i * 4));
                    }
                    
                    int weight_idx = ((row * M + col) / 8);
                    reinterpret_cast<int*>(quantized_weights)[weight_idx] = packed_weights;
                }
            }
        }
    }
    
    void generate_test_data(half* A, half* B_fp16, int4* B_quantized, half* scales,
                           half* C_ref, int N, int M) {
        // Generate input vector A
        for (int i = 0; i < N; i++) {
            A[i] = __float2half(dis(gen));
        }
        
        // Generate weight matrix B (fp16 for reference)
        for (int i = 0; i < N * M; i++) {
            B_fp16[i] = __float2half(dis(gen) * 0.1f);
        }
        
        // Quantize weights
        quantize_weights(B_fp16, B_quantized, scales, N, M);
        
        // Compute reference result using cuBLAS
        half *d_A, *d_B_fp16, *d_C_ref;
        cudaMalloc(&d_A, N * sizeof(half));
        cudaMalloc(&d_B_fp16, N * M * sizeof(half));
        cudaMalloc(&d_C_ref, M * sizeof(half));
        
        cudaMemcpy(d_A, A, N * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_fp16, B_fp16, N * M * sizeof(half), cudaMemcpyHostToDevice);
        
        cublasHandle_t handle;
        cublasCreate(&handle);
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        
        // GEMV: C = A * B^T (1xN * NxM = 1xM)
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     M, 1, N,
                     &alpha,
                     d_B_fp16, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, N,
                     &beta,
                     d_C_ref, CUDA_R_16F, M,
                     CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        cudaMemcpy(C_ref, d_C_ref, M * sizeof(half), cudaMemcpyDeviceToHost);
        
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B_fp16);
        cudaFree(d_C_ref);
    }
};

// Enhanced benchmarking function with bandwidth calculation
GEMVBenchmarkResult benchmark_gemv_kernel(
    std::function<void(const half*, const int4*, const half*, half*, cudaStream_t)> kernel_func,
    const std::string& kernel_name,
    const half* h_A, const int4* h_B, const half* h_scales, const half* h_C_ref,
    int N, int M, int num_iterations = 1000
) {
    // Allocate device memory
    half *d_A, *d_C, *d_scales;
    int4 *d_B;
    
    cudaMalloc(&d_A, N * sizeof(half));
    cudaMalloc(&d_B, (N * M / 8) * sizeof(int4));
    cudaMalloc(&d_scales, M * sizeof(half));
    cudaMalloc(&d_C, M * sizeof(half));
    
    cudaMemcpy(d_A, h_A, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (N * M / 8) * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, M * sizeof(half), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup runs
    for (int i = 0; i < 10; i++) {
        kernel_func(d_A, d_B, d_scales, d_C, 0);
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel_func(d_A, d_B, d_scales, d_C, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double avg_time_ms = elapsed_time_ms / num_iterations;
    
    // Calculate performance metrics
    double flops = 2.0 * N * M; // GEMV: N*M multiply-adds
    double gflops = flops / (avg_time_ms / 1000.0) / 1e9;
    
    // Memory bandwidth calculation
    double bytes_read = N * sizeof(half) + (N * M / 2) + M * sizeof(half); // A + B(4-bit) + scales
    double bytes_write = M * sizeof(half); // C
    double total_bytes = bytes_read + bytes_write;
    double bandwidth_gb_s = total_bytes / (avg_time_ms / 1000.0) / 1e9;
    
    // Copy result back and verify correctness
    half* h_C_gpu = new half[M];
    cudaMemcpy(h_C_gpu, d_C, M * sizeof(half), cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result = verify_with_cosine_distance(
        reinterpret_cast<const float*>(h_C_gpu), 
        reinterpret_cast<const float*>(h_C_ref), 
        M, 0.95 // Lower threshold for quantized results
    );
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scales);
    cudaFree(d_C);
    delete[] h_C_gpu;
    
    return {
        kernel_name,
        avg_time_ms,
        gflops,
        bandwidth_gb_s,
        acc_result.passed,
        acc_result.cosine_similarity,
        acc_result.max_abs_error
    };
}

// Wrapper for our optimized kernel
void launch_optimized_gemv(const half* A, const int4* B, const half* scales, 
                          half* C, cudaStream_t stream) {
    launch_marlin_gemv_w4a16f(A, B, scales, C, 3584, 18944, stream);
}

// Wrapper for Marlin's original kernel (adapted for GEMV)
void launch_marlin_original(const half* A, const int4* B, const half* scales, 
                           half* C, cudaStream_t stream) {
    // Note: Marlin's original kernel is designed for GEMM, so we'd need to adapt it
    // For now, we'll use a simplified version or skip this comparison
    printf("Marlin original kernel adaptation needed for GEMV\n");
}

// NCU profiling helper
void run_ncu_profiling(const std::string& kernel_name, std::function<void()> kernel_launch) {
    printf("\n=== NCU Profiling for %s ===\n", kernel_name.c_str());
    printf("Run the following command to profile this kernel:\n");
    printf("ncu --set full --target-processes all ./marlin_gemv_test\n");
    printf("Or for specific metrics:\n");
    printf("ncu --metrics sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum ./marlin_gemv_test\n");
    
    // Launch kernel for profiling
    kernel_launch();
    cudaDeviceSynchronize();
}

int main() {
    printf("🚀 Marlin-Optimized GEMV w4a16f Benchmark\n");
    printf("==========================================\n\n");
    
    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Max Shared Memory per Block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);
    
    // Test configuration
    const int N = 3584;  // Input dimension
    const int M = 18944; // Output dimension
    printf("Target Shape: 1×%d @ %d×%d (%.2f M parameters)\n\n", N, N, M, (N * M) / 1e6);
    
    // Allocate host memory
    half* h_A = new half[N];
    half* h_B_fp16 = new half[N * M];
    int4* h_B_quantized = new int4[N * M / 8];
    half* h_scales = new half[M];
    half* h_C_ref = new half[M];
    
    // Generate test data
    WeightQuantizer quantizer;
    printf("⚙️  Generating and quantizing test data...\n");
    quantizer.generate_test_data(h_A, h_B_fp16, h_B_quantized, h_scales, h_C_ref, N, M);
    printf("✅ Test data ready\n\n");
    
    // Benchmark results
    std::vector<GEMVBenchmarkResult> results;
    
    // Benchmark our optimized kernel
    printf("🧪 Benchmarking Optimized GEMV Kernel...\n");
    auto result_optimized = benchmark_gemv_kernel(
        [](const half* A, const int4* B, const half* scales, half* C, cudaStream_t stream) {
            launch_marlin_gemv_w4a16f(A, B, scales, C, 3584, 18944, stream);
        },
        "Optimized Marlin GEMV",
        h_A, h_B_quantized, h_scales, h_C_ref, N, M
    );
    results.push_back(result_optimized);
    
    // Benchmark cuBLAS fp16 (reference)
    printf("🔬 Benchmarking cuBLAS fp16 Reference...\n");
    
    half *d_A, *d_B_fp16, *d_C_cublas;
    cudaMalloc(&d_A, N * sizeof(half));
    cudaMalloc(&d_B_fp16, N * M * sizeof(half));
    cudaMalloc(&d_C_cublas, M * sizeof(half));
    
    cudaMemcpy(d_A, h_A, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, N * M * sizeof(half), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N, &alpha,
                     d_B_fp16, CUDA_R_16F, N, d_A, CUDA_R_16F, N, &beta,
                     d_C_cublas, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N, &alpha,
                     d_B_fp16, CUDA_R_16F, N, d_A, CUDA_R_16F, N, &beta,
                     d_C_cublas, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_cublas;
    cudaEventElapsedTime(&elapsed_cublas, start, stop);
    double cublas_gflops = (2.0 * N * M * 1000) / (elapsed_cublas / 1000.0) / 1e9;
    
    results.push_back({
        "cuBLAS fp16 (Reference)",
        elapsed_cublas / 1000.0,
        cublas_gflops,
        0.0, // Bandwidth not calculated for cuBLAS
        true,
        1.0,
        0.0f
    });
    
    // Print results table
    printf("\n📊 Performance Results\n");
    printf("======================\n");
    printf("%-25s %12s %12s %15s %10s %8s\n", 
           "Kernel", "Time (ms)", "GFLOPS", "Bandwidth (GB/s)", "Correct", "Sim");
    printf("%-25s %12s %12s %15s %10s %8s\n", 
           "-----", "--------", "------", "-------------", "-------", "---");
    
    for (const auto& result : results) {
        printf("%-25s %12.4f %12.1f %15.1f %10s %8.4f\n",
               result.kernel_name.c_str(),
               result.avg_time_ms,
               result.gflops,
               result.bandwidth_gb_s,
               result.correctness_passed ? "✅" : "❌",
               result.cosine_similarity);
    }
    
    // Calculate speedup
    if (results.size() >= 2) {
        double speedup = results[1].avg_time_ms / results[0].avg_time_ms;
        printf("\n🏆 Speedup vs cuBLAS: %.2fx\n", speedup);
        
        if (speedup > 1.0) {
            printf("🎉 Our kernel is %.1f%% faster than cuBLAS!\n", (speedup - 1.0) * 100);
        } else {
            printf("📈 cuBLAS is %.1f%% faster. Room for optimization!\n", (1.0 / speedup - 1.0) * 100);
        }
    }
    
    // NCU profiling suggestions
    printf("\n🔍 For detailed analysis, run with NCU:\n");
    printf("ncu --set full --target-processes all ./marlin_gemv_benchmark\n");
    printf("ncu --metrics smsp__cycles_elapsed.avg,dram__bytes.sum,smsp__inst_executed.sum ./marlin_gemv_benchmark\n");
    
    // Memory analysis
    printf("\n💾 Memory Analysis:\n");
    double theoretical_bandwidth = prop.memoryClockRate * 2 * prop.memoryBusWidth / 8 / 1e6; // GB/s
    printf("Theoretical Bandwidth: %.1f GB/s\n", theoretical_bandwidth);
    if (results.size() > 0 && results[0].bandwidth_gb_s > 0) {
        printf("Achieved Bandwidth: %.1f GB/s (%.1f%% of peak)\n", 
               results[0].bandwidth_gb_s, 
               100.0 * results[0].bandwidth_gb_s / theoretical_bandwidth);
    }
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B_fp16);
    cudaFree(d_C_cublas);
    
    delete[] h_A;
    delete[] h_B_fp16;
    delete[] h_B_quantized;
    delete[] h_scales;
    delete[] h_C_ref;
    
    printf("\n✨ Benchmark complete! Check results above.\n");
    return 0;
}