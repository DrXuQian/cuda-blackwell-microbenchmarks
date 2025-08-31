#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

/*
 * Marlin Tutorial Step 2: Basic 4-bit GEMV
 * 
 * LEARNING OBJECTIVES:
 * 1. Build a complete 4-bit GEMV kernel from scratch
 * 2. Understand GEMV structure: Y = A * X + bias
 * 3. Master the load → dequantize → multiply → accumulate pattern
 * 4. Learn scale factor application techniques
 * 5. Establish baseline performance for future optimizations
 * 
 * KEY CONCEPTS:
 * - GEMV: General Matrix-Vector multiplication (Y = A*X + bias)
 * - Weight Matrix A: [M x N] stored as packed 4-bit values
 * - Input Vector X: [N] in FP16 format
 * - Output Vector Y: [M] in FP16 format
 * - Scales: Per-row or per-group quantization scales
 * - One thread per output element approach
 */

// CUDA error checking
#define CUDA_CHECK(status) \
    { \
        cudaError_t error = status; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// cuBLAS error checking
#define CUBLAS_CHECK(status) \
    { \
        cublasStatus_t error = status; \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error: " << error \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// 4-bit utilities (from Step 1)
namespace bit_utils {
    
    __host__ __device__ inline uint8_t extract_4bit(uint32_t packed, int index) {
        return (packed >> (index * 4)) & 0xF;
    }
    
    __host__ __device__ inline half dequantize_4bit_to_fp16(uint8_t quantized, half scale, half zero_point = __float2half(7.5f)) {
        // Convert 4-bit [0,15] to symmetric range around zero
        float q_f = float(quantized);
        float zero_f = __half2float(zero_point);
        float scale_f = __half2float(scale);
        
        float dequantized = (q_f - zero_f) * scale_f;
        return __float2half(dequantized);
    }
    
    __host__ __device__ inline uint8_t quantize_fp16_to_4bit(half value, half scale, half zero_point = __float2half(7.5f)) {
        float val_f = __half2float(value);
        float scale_f = __half2float(scale);
        float zero_f = __half2float(zero_point);
        
        float quantized_f = val_f / scale_f + zero_f;
        int quantized = __float2int_rn(fmaxf(0.0f, fminf(15.0f, quantized_f)));
        return (uint8_t)quantized;
    }
}

// GEMV kernel implementations
namespace gemv_kernels {

    // Naive 4-bit GEMV: One thread per output element
    __global__ void naive_4bit_gemv(
        const uint32_t* __restrict__ weights_packed,    // [M x N/8] packed 4-bit weights
        const half* __restrict__ input,                 // [N] input vector
        const half* __restrict__ scales,                // [M] per-row scale factors
        const half* __restrict__ bias,                  // [M] bias vector (optional)
        half* __restrict__ output,                      // [M] output vector
        int M, int N) {
        
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= M) return;
        
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        
        // Process each column (input element)
        for (int col = 0; col < N; col++) {
            // Calculate which packed int32 contains this weight
            int pack_idx = row * (N / 8) + (col / 8);  // Which packed int32
            int sub_idx = col % 8;                      // Position within packed int32
            
            // Load packed weights and extract 4-bit value
            uint32_t packed = weights_packed[pack_idx];
            uint8_t weight_4bit = bit_utils::extract_4bit(packed, sub_idx);
            
            // Dequantize 4-bit weight to FP16
            half weight_fp16 = bit_utils::dequantize_4bit_to_fp16(weight_4bit, scale);
            
            // Multiply and accumulate
            half input_val = input[col];
            accumulator = __hfma(weight_fp16, input_val, accumulator);
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        // Store result
        output[row] = accumulator;
        
        // Debug output for first few elements
        if (row < 8 && blockIdx.x == 0) {
            printf("Row %d: scale=%.3f, output=%.3f\n", 
                   row, __half2float(scale), __half2float(accumulator));
        }
    }

    // Reference FP16 GEMV for comparison
    __global__ void reference_fp16_gemv(
        const half* __restrict__ weights,               // [M x N] FP16 weights
        const half* __restrict__ input,                 // [N] input vector
        const half* __restrict__ bias,                  // [M] bias vector (optional)
        half* __restrict__ output,                      // [M] output vector
        int M, int N) {
        
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= M) return;
        
        half accumulator = __float2half(0.0f);
        
        // Process each column
        for (int col = 0; col < N; col++) {
            half weight = weights[row * N + col];
            half input_val = input[col];
            accumulator = __hfma(weight, input_val, accumulator);
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }

    // Optimized version: Process multiple elements per thread
    __global__ void improved_4bit_gemv(
        const uint32_t* __restrict__ weights_packed,
        const half* __restrict__ input,
        const half* __restrict__ scales,
        const half* __restrict__ bias,
        half* __restrict__ output,
        int M, int N) {
        
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= M) return;
        
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 8);
        
        // Process 8 elements at a time (one packed int32)
        for (int pack = 0; pack < N / 8; pack++) {
            uint32_t packed = weights_packed[row_offset + pack];
            
            // Unroll the loop for 8 elements
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t weight_4bit = bit_utils::extract_4bit(packed, i);
                half weight_fp16 = bit_utils::dequantize_4bit_to_fp16(weight_4bit, scale);
                half input_val = input[pack * 8 + i];
                accumulator = __hfma(weight_fp16, input_val, accumulator);
            }
        }
        
        // Add bias
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }
}

// Host utilities
class GemvBenchmark {
private:
    int M, N;
    uint32_t* d_weights_packed;
    half* d_weights_fp16;
    half* d_input;
    half* d_scales;
    half* d_bias;
    half* d_output_4bit;
    half* d_output_fp16;
    half* d_output_reference;
    
    cudaEvent_t start_event, stop_event;

public:
    GemvBenchmark(int M_, int N_) : M(M_), N(N_) {
        // Ensure N is divisible by 8 for packing
        if (N % 8 != 0) {
            throw std::runtime_error("N must be divisible by 8 for 4-bit packing");
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_weights_packed, M * (N / 8) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_weights_fp16, M * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_bias, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_4bit, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_fp16, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_reference, M * sizeof(half)));
        
        // Create events
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        initialize_data();
    }
    
    ~GemvBenchmark() {
        CUDA_CHECK(cudaFree(d_weights_packed));
        CUDA_CHECK(cudaFree(d_weights_fp16));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_output_4bit));
        CUDA_CHECK(cudaFree(d_output_fp16));
        CUDA_CHECK(cudaFree(d_output_reference));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }
    
private:
    void initialize_data() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> weight_dist(0.0f, 1.0f);
        std::normal_distribution<float> input_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> scale_dist(0.1f, 0.5f);
        
        // Generate FP16 weights
        std::vector<half> h_weights_fp16(M * N);
        std::vector<half> h_scales(M);
        
        for (int i = 0; i < M; i++) {
            h_scales[i] = __float2half(scale_dist(gen));
            for (int j = 0; j < N; j++) {
                h_weights_fp16[i * N + j] = __float2half(weight_dist(gen));
            }
        }
        
        // Quantize to 4-bit and pack
        std::vector<uint32_t> h_weights_packed(M * (N / 8));
        
        for (int i = 0; i < M; i++) {
            half scale = h_scales[i];
            for (int j = 0; j < N / 8; j++) {
                uint8_t values[8];
                for (int k = 0; k < 8; k++) {
                    half weight = h_weights_fp16[i * N + j * 8 + k];
                    values[k] = bit_utils::quantize_fp16_to_4bit(weight, scale);
                }
                
                // Pack 8 values into uint32
                uint32_t packed = 0;
                for (int k = 0; k < 8; k++) {
                    packed |= ((uint32_t)values[k] & 0xF) << (k * 4);
                }
                h_weights_packed[i * (N / 8) + j] = packed;
            }
        }
        
        // Generate input and bias
        std::vector<half> h_input(N);
        std::vector<half> h_bias(M);
        
        for (int i = 0; i < N; i++) {
            h_input[i] = __float2half(input_dist(gen));
        }
        
        for (int i = 0; i < M; i++) {
            h_bias[i] = __float2half(weight_dist(gen) * 0.1f);
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_weights_packed, h_weights_packed.data(), 
                             M * (N / 8) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights_fp16, h_weights_fp16.data(), 
                             M * N * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                             N * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), 
                             M * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 
                             M * sizeof(half), cudaMemcpyHostToDevice));
    }

public:
    float benchmark_naive_4bit() {
        dim3 block_size(256);
        dim3 grid_size((M + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            gemv_kernels::naive_4bit_gemv<<<grid_size, block_size>>>(
                d_weights_packed, d_input, d_scales, d_bias, d_output_4bit, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            gemv_kernels::naive_4bit_gemv<<<grid_size, block_size>>>(
                d_weights_packed, d_input, d_scales, d_bias, d_output_4bit, M, N);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms / iterations;
    }
    
    float benchmark_improved_4bit() {
        dim3 block_size(256);
        dim3 grid_size((M + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            gemv_kernels::improved_4bit_gemv<<<grid_size, block_size>>>(
                d_weights_packed, d_input, d_scales, d_bias, d_output_4bit, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            gemv_kernels::improved_4bit_gemv<<<grid_size, block_size>>>(
                d_weights_packed, d_input, d_scales, d_bias, d_output_4bit, M, N);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms / iterations;
    }
    
    float benchmark_fp16_reference() {
        dim3 block_size(256);
        dim3 grid_size((M + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            gemv_kernels::reference_fp16_gemv<<<grid_size, block_size>>>(
                d_weights_fp16, d_input, d_bias, d_output_fp16, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            gemv_kernels::reference_fp16_gemv<<<grid_size, block_size>>>(
                d_weights_fp16, d_input, d_bias, d_output_fp16, M, N);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms / iterations;
    }
    
    float benchmark_cublas() {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            CUBLAS_CHECK(cublasHgemv(handle, CUBLAS_OP_N, M, N, &alpha, 
                                    d_weights_fp16, M, d_input, 1, &beta, d_output_reference, 1));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            CUBLAS_CHECK(cublasHgemv(handle, CUBLAS_OP_N, M, N, &alpha, 
                                    d_weights_fp16, M, d_input, 1, &beta, d_output_reference, 1));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        
        CUBLAS_CHECK(cublasDestroy(handle));
        return time_ms / iterations;
    }
    
    void verify_correctness() {
        // Run kernels
        dim3 block_size(256);
        dim3 grid_size((M + block_size.x - 1) / block_size.x);
        
        gemv_kernels::naive_4bit_gemv<<<grid_size, block_size>>>(
            d_weights_packed, d_input, d_scales, d_bias, d_output_4bit, M, N);
        
        gemv_kernels::reference_fp16_gemv<<<grid_size, block_size>>>(
            d_weights_fp16, d_input, d_bias, d_output_fp16, M, N);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results to host
        std::vector<half> h_output_4bit(M);
        std::vector<half> h_output_fp16(M);
        
        CUDA_CHECK(cudaMemcpy(h_output_4bit.data(), d_output_4bit, M * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output_fp16.data(), d_output_fp16, M * sizeof(half), cudaMemcpyDeviceToHost));
        
        // Calculate error statistics
        double total_error = 0.0;
        double max_error = 0.0;
        int error_count = 0;
        
        std::cout << "\nCorrectness Verification (first 10 elements):" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Row | 4-bit Output | FP16 Reference | Error    | Error %" << std::endl;
        std::cout << "----+-------------+---------------+----------+--------" << std::endl;
        
        for (int i = 0; i < std::min(10, M); i++) {
            float val_4bit = __half2float(h_output_4bit[i]);
            float val_fp16 = __half2float(h_output_fp16[i]);
            float error = std::abs(val_4bit - val_fp16);
            float error_pct = (std::abs(val_fp16) > 1e-6f) ? (error / std::abs(val_fp16) * 100.0f) : 0.0f;
            
            std::cout << std::setw(3) << i << " | " 
                      << std::setw(11) << val_4bit << " | "
                      << std::setw(13) << val_fp16 << " | "
                      << std::setw(8) << error << " | "
                      << std::setw(6) << error_pct << "%" << std::endl;
            
            total_error += error;
            max_error = std::max(max_error, (double)error);
            if (error_pct > 5.0f) error_count++;
        }
        
        double avg_error = total_error / std::min(10, M);
        
        std::cout << "\nError Statistics:" << std::endl;
        std::cout << "  Average error: " << avg_error << std::endl;
        std::cout << "  Maximum error: " << max_error << std::endl;
        std::cout << "  High error count (>5%): " << error_count << std::endl;
        
        if (avg_error < 0.1 && error_count < 3) {
            std::cout << "  ✅ PASS: 4-bit quantization maintains reasonable accuracy" << std::endl;
        } else {
            std::cout << "  ⚠️  WARNING: High quantization error detected" << std::endl;
        }
    }
    
    void print_performance_analysis(float naive_time, float improved_time, float fp16_time, float cublas_time) {
        // Calculate throughput
        long long ops = 2LL * M * N; // GEMV: M*N multiply-adds
        double naive_gflops = (double)ops / (naive_time / 1000.0) / 1e9;
        double improved_gflops = (double)ops / (improved_time / 1000.0) / 1e9;
        double fp16_gflops = (double)ops / (fp16_time / 1000.0) / 1e9;
        double cublas_gflops = (double)ops / (cublas_time / 1000.0) / 1e9;
        
        // Calculate memory usage
        size_t weights_4bit_mb = M * (N / 8) * sizeof(uint32_t) / (1024 * 1024);
        size_t weights_fp16_mb = M * N * sizeof(half) / (1024 * 1024);
        size_t input_mb = N * sizeof(half) / (1024 * 1024);
        size_t output_mb = M * sizeof(half) / (1024 * 1024);
        
        std::cout << "\n📊 Performance Analysis:" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "┌─────────────────┬─────────────┬─────────────┬──────────────┐" << std::endl;
        std::cout << "│ Implementation  │ Time (ms)   │ GFLOPS      │ Speedup      │" << std::endl;
        std::cout << "├─────────────────┼─────────────┼─────────────┼──────────────┤" << std::endl;
        std::cout << "│ Naive 4-bit     │ " << std::setw(11) << naive_time << " │ " 
                  << std::setw(11) << naive_gflops << " │ " << std::setw(12) << "1.00x" << " │" << std::endl;
        std::cout << "│ Improved 4-bit  │ " << std::setw(11) << improved_time << " │ " 
                  << std::setw(11) << improved_gflops << " │ " << std::setw(12) << (naive_time/improved_time) << "x │" << std::endl;
        std::cout << "│ FP16 Reference  │ " << std::setw(11) << fp16_time << " │ " 
                  << std::setw(11) << fp16_gflops << " │ " << std::setw(12) << (naive_time/fp16_time) << "x │" << std::endl;
        std::cout << "│ cuBLAS FP16     │ " << std::setw(11) << cublas_time << " │ " 
                  << std::setw(11) << cublas_gflops << " │ " << std::setw(12) << (naive_time/cublas_time) << "x │" << std::endl;
        std::cout << "└─────────────────┴─────────────┴─────────────┴──────────────┘" << std::endl;
        
        std::cout << "\n💾 Memory Usage Analysis:" << std::endl;
        std::cout << "  Weights (4-bit): " << weights_4bit_mb << " MB" << std::endl;
        std::cout << "  Weights (FP16):  " << weights_fp16_mb << " MB" << std::endl;
        std::cout << "  Input vector:    " << input_mb << " MB" << std::endl;
        std::cout << "  Output vector:   " << output_mb << " MB" << std::endl;
        std::cout << "  Memory saving:   " << (double)weights_fp16_mb / weights_4bit_mb << "x reduction" << std::endl;
    }
};

int main() {
    std::cout << "=== Marlin Tutorial Step 2: Basic 4-bit GEMV ===" << std::endl;
    
    // Check GPU capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    
    /*
     * PART 1: Understanding GEMV Structure
     */
    std::cout << "\n📐 PART 1: GEMV Structure and Implementation\n" << std::endl;
    
    std::cout << "🎯 GEMV Operation: Y = A * X + bias" << std::endl;
    std::cout << "• Weight Matrix A: [M x N] stored as packed 4-bit values" << std::endl;
    std::cout << "• Input Vector X: [N] in FP16 format" << std::endl;
    std::cout << "• Output Vector Y: [M] in FP16 format" << std::endl;
    std::cout << "• Scale factors: Per-row quantization scales for dequantization" << std::endl;
    std::cout << "• Bias vector: Optional bias addition" << std::endl;
    
    std::cout << "\n🔄 Processing Flow:" << std::endl;
    std::cout << "1. Load packed 4-bit weights from global memory" << std::endl;
    std::cout << "2. Extract individual 4-bit values using bit operations" << std::endl;
    std::cout << "3. Dequantize 4-bit values to FP16 using scale factors" << std::endl;
    std::cout << "4. Multiply dequantized weights with input vector elements" << std::endl;
    std::cout << "5. Accumulate products to compute dot product" << std::endl;
    std::cout << "6. Add bias and store final result" << std::endl;
    
    /*
     * PART 2: Kernel Implementation and Benchmarking
     */
    std::cout << "\n⚡ PART 2: Kernel Implementation and Performance\n" << std::endl;
    
    // Test different problem sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {1024, 1024},    // Small
        {2048, 2048},    // Medium
        {4096, 4096},    // Large
        {8192, 2048}     // Rectangular
    };
    
    for (auto [M, N] : test_sizes) {
        std::cout << "\n🧪 Testing GEMV size: " << M << " x " << N << std::endl;
        
        try {
            GemvBenchmark benchmark(M, N);
            
            // Run benchmarks
            std::cout << "Running benchmarks..." << std::endl;
            float naive_time = benchmark.benchmark_naive_4bit();
            float improved_time = benchmark.benchmark_improved_4bit();
            float fp16_time = benchmark.benchmark_fp16_reference();
            float cublas_time = benchmark.benchmark_cublas();
            
            // Verify correctness
            benchmark.verify_correctness();
            
            // Print performance analysis
            benchmark.print_performance_analysis(naive_time, improved_time, fp16_time, cublas_time);
            
        } catch (const std::exception& e) {
            std::cout << "Error with size " << M << "x" << N << ": " << e.what() << std::endl;
        }
    }
    
    /*
     * PART 3: Key Insights and Learnings
     */
    std::cout << "\n🧠 PART 3: Key Insights from Basic GEMV\n" << std::endl;
    
    std::cout << "📈 Performance Observations:" << std::endl;
    std::cout << "• Naive 4-bit implementation is typically slower than FP16" << std::endl;
    std::cout << "  → Reason: Overhead of dequantization not amortized" << std::endl;
    std::cout << "• Improved version shows better performance through loop optimization" << std::endl;
    std::cout << "• Memory savings are significant (4x reduction)" << std::endl;
    std::cout << "• Accuracy is maintained with proper scale factors" << std::endl;
    
    std::cout << "\n🚧 Current Limitations:" << std::endl;
    std::cout << "• One thread per output element → Poor GPU utilization" << std::endl;
    std::cout << "• No memory coalescing optimization" << std::endl;
    std::cout << "• Sequential processing within each thread" << std::endl;
    std::cout << "• No shared memory utilization" << std::endl;
    std::cout << "• No warp-level cooperation" << std::endl;
    
    std::cout << "\n💡 Optimization Opportunities:" << std::endl;
    std::cout << "• Vectorized memory access (float4/int4)" << std::endl;
    std::cout << "• Warp-level reductions and shuffles" << std::endl;
    std::cout << "• Shared memory for data reuse" << std::endl;
    std::cout << "• Better memory access patterns" << std::endl;
    std::cout << "• Specialized warp roles (producer/consumer)" << std::endl;
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Step 2 Summary: Basic GEMV Mastered ===\n" << std::endl;
    std::cout << "✅ You learned:" << std::endl;
    std::cout << "   • Complete 4-bit GEMV implementation from scratch" << std::endl;
    std::cout << "   • Load → Dequantize → Multiply → Accumulate pattern" << std::endl;
    std::cout << "   • Scale factor application for quantization schemes" << std::endl;
    std::cout << "   • Performance comparison with FP16 baselines" << std::endl;
    std::cout << "   • Correctness verification techniques" << std::endl;
    
    std::cout << "\n🎯 Key Achievements:" << std::endl;
    std::cout << "   • Working 4-bit GEMV kernel with quantization support" << std::endl;
    std::cout << "   • Baseline performance measurements established" << std::endl;
    std::cout << "   • Understanding of current limitations identified" << std::endl;
    std::cout << "   • Foundation for advanced optimizations prepared" << std::endl;
    
    std::cout << "\n🚀 Next: Step 3 will add vectorized memory access for better performance!" << std::endl;
    
    return 0;
}