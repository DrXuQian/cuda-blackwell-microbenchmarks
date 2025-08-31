#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

/*
 * Marlin Tutorial Step 3: Vectorized Memory Access
 * 
 * LEARNING OBJECTIVES:
 * 1. Master vectorized memory operations using float4/int4
 * 2. Understand coalesced memory access patterns
 * 3. Learn to process multiple elements per thread efficiently
 * 4. Optimize memory bandwidth utilization
 * 5. Compare different vectorization strategies
 * 
 * KEY CONCEPTS:
 * - Vectorized Loads: Using float4, int4, uint4 for 128-bit transactions
 * - Memory Coalescing: 32 threads in warp access consecutive addresses
 * - Memory Alignment: Data must be properly aligned for vectorized access
 * - Bandwidth Utilization: Maximizing effective memory throughput
 * - Thread Efficiency: More work per thread reduces launch overhead
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

// 4-bit utilities with vectorized support
namespace vectorized_utils {
    
    // Extract 4-bit values from uint4 (4 packed int32s)
    __device__ inline void unpack_uint4_to_4bit(uint4 packed, uint8_t* values) {
        // Each uint32 contains 8 x 4-bit values
        // uint4 contains 4 uint32s = 32 x 4-bit values total
        
        uint32_t* packed_ptr = reinterpret_cast<uint32_t*>(&packed);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t p = packed_ptr[i];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                values[i * 8 + j] = (p >> (j * 4)) & 0xF;
            }
        }
    }
    
    // Vectorized dequantization: 32 x 4-bit values → 32 x FP16 values
    __device__ inline void dequantize_32_4bit_to_fp16(
        const uint8_t* quantized, half scale, half2* output) {
        
        half zero_point = __float2half(7.5f);
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {  // 32 values = 16 x half2
            uint8_t q0 = quantized[i * 2];
            uint8_t q1 = quantized[i * 2 + 1];
            
            // Dequantize both values
            float f0 = (float(q0) - __half2float(zero_point)) * __half2float(scale);
            float f1 = (float(q1) - __half2float(zero_point)) * __half2float(scale);
            
            output[i] = __floats2half2_rn(f0, f1);
        }
    }
    
    // Vectorized multiply-accumulate using half2
    __device__ inline half2 multiply_accumulate_half2(half2 a, half2 b, half2 c) {
        return __hfma2(a, b, c);
    }
    
    // Horizontal sum of half2 vector
    __device__ inline half horizontal_sum_half2(half2 val) {
        return __hadd(__low2half(val), __high2half(val));
    }
}

// Vectorized GEMV kernel implementations
namespace vectorized_kernels {

    // Baseline: scalar implementation for comparison (from Step 2)
    __global__ void scalar_4bit_gemv(
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
        
        for (int pack = 0; pack < N / 8; pack++) {
            uint32_t packed = weights_packed[row_offset + pack];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t weight_4bit = (packed >> (i * 4)) & 0xF;
                half weight_fp16 = __float2half(
                    (float(weight_4bit) - 7.5f) * __half2float(scale));
                accumulator = __hfma(weight_fp16, input[pack * 8 + i], accumulator);
            }
        }
        
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }

    // Level 1: Vectorized weight loading with uint4
    __global__ void vectorized_weight_load_gemv(
        const uint4* __restrict__ weights_packed,  // Now uint4 for 128-bit loads
        const half* __restrict__ input,
        const half* __restrict__ scales,
        const half* __restrict__ bias,
        half* __restrict__ output,
        int M, int N) {
        
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M) return;
        
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 32);  // 32 weights per uint4
        
        // Process 32 weights at a time
        for (int vec = 0; vec < N / 32; vec++) {
            // Vectorized load: 128-bit transaction
            uint4 packed_vec = weights_packed[row_offset + vec];
            
            // Unpack to 32 x 4-bit values
            uint8_t weight_4bits[32];
            vectorized_utils::unpack_uint4_to_4bit(packed_vec, weight_4bits);
            
            // Process all 32 weights
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                half weight_fp16 = __float2half(
                    (float(weight_4bits[i]) - 7.5f) * __half2float(scale));
                accumulator = __hfma(weight_fp16, input[vec * 32 + i], accumulator);
            }
        }
        
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }

    // Level 2: Vectorized input loading with half2 + vectorized weights
    __global__ void vectorized_input_weight_gemv(
        const uint4* __restrict__ weights_packed,
        const half2* __restrict__ input_vec,      // Input as half2 vectors
        const half* __restrict__ scales,
        const half* __restrict__ bias,
        half* __restrict__ output,
        int M, int N) {
        
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M) return;
        
        half2 accumulator2 = __float2half2_rn(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 32);
        
        // Process 32 weights at a time
        for (int vec = 0; vec < N / 32; vec++) {
            uint4 packed_vec = weights_packed[row_offset + vec];
            
            // Unpack and dequantize to half2 format
            uint8_t weight_4bits[32];
            vectorized_utils::unpack_uint4_to_4bit(packed_vec, weight_4bits);
            
            half2 weight_vec[16];  // 32 weights = 16 x half2
            vectorized_utils::dequantize_32_4bit_to_fp16(weight_4bits, scale, weight_vec);
            
            // Vectorized multiply-accumulate
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                half2 input_pair = input_vec[vec * 16 + i];
                accumulator2 = vectorized_utils::multiply_accumulate_half2(
                    weight_vec[i], input_pair, accumulator2);
            }
        }
        
        // Reduce half2 to scalar
        half accumulator = vectorized_utils::horizontal_sum_half2(accumulator2);
        
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }

    // Level 3: Full vectorization with multiple elements per thread
    __global__ void fully_vectorized_gemv(
        const uint4* __restrict__ weights_packed,
        const half2* __restrict__ input_vec,
        const half* __restrict__ scales,
        const half* __restrict__ bias,
        half* __restrict__ output,
        int M, int N, int elements_per_thread = 4) {
        
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        int start_row = thread_id * elements_per_thread;
        
        // Each thread processes multiple output elements
        for (int elem = 0; elem < elements_per_thread; elem++) {
            int row = start_row + elem;
            if (row >= M) continue;
            
            half2 accumulator2 = __float2half2_rn(0.0f);
            half scale = scales[row];
            int row_offset = row * (N / 32);
            
            // Vectorized inner loop
            for (int vec = 0; vec < N / 32; vec++) {
                uint4 packed_vec = weights_packed[row_offset + vec];
                
                uint8_t weight_4bits[32];
                vectorized_utils::unpack_uint4_to_4bit(packed_vec, weight_4bits);
                
                half2 weight_vec[16];
                vectorized_utils::dequantize_32_4bit_to_fp16(weight_4bits, scale, weight_vec);
                
                #pragma unroll 8  // Partial unroll for better register usage
                for (int i = 0; i < 16; i++) {
                    half2 input_pair = input_vec[vec * 16 + i];
                    accumulator2 = vectorized_utils::multiply_accumulate_half2(
                        weight_vec[i], input_pair, accumulator2);
                }
            }
            
            half accumulator = vectorized_utils::horizontal_sum_half2(accumulator2);
            
            if (bias != nullptr) {
                accumulator = __hadd(accumulator, bias[row]);
            }
            
            output[row] = accumulator;
        }
    }

    // Level 4: Coalesced memory access optimization
    __global__ void coalesced_vectorized_gemv(
        const uint4* __restrict__ weights_packed,
        const half2* __restrict__ input_vec,
        const half* __restrict__ scales,
        const half* __restrict__ bias,
        half* __restrict__ output,
        int M, int N) {
        
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int block_row_start = blockIdx.x * blockDim.x;
        
        // Coalesced access pattern: threads in same warp access consecutive rows
        int row = block_row_start + warp_id * 32 + lane_id;
        
        if (row >= M) return;
        
        half2 accumulator2 = __float2half2_rn(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 32);
        
        // Inner loop with better memory access pattern
        for (int vec = 0; vec < N / 32; vec++) {
            // All threads in warp load consecutive memory locations
            uint4 packed_vec = weights_packed[row_offset + vec];
            
            uint8_t weight_4bits[32];
            vectorized_utils::unpack_uint4_to_4bit(packed_vec, weight_4bits);
            
            half2 weight_vec[16];
            vectorized_utils::dequantize_32_4bit_to_fp16(weight_4bits, scale, weight_vec);
            
            #pragma unroll 4
            for (int i = 0; i < 16; i++) {
                half2 input_pair = input_vec[vec * 16 + i];
                accumulator2 = vectorized_utils::multiply_accumulate_half2(
                    weight_vec[i], input_pair, accumulator2);
            }
        }
        
        half accumulator = vectorized_utils::horizontal_sum_half2(accumulator2);
        
        if (bias != nullptr) {
            accumulator = __hadd(accumulator, bias[row]);
        }
        
        output[row] = accumulator;
    }
}

// Host utility class for vectorized benchmarking
class VectorizedGemvBenchmark {
private:
    int M, N;
    uint4* d_weights_uint4;
    uint32_t* d_weights_uint32;
    half2* d_input_half2;
    half* d_input_scalar;
    half* d_scales;
    half* d_bias;
    half* d_output;
    cudaEvent_t start_event, stop_event;

public:
    VectorizedGemvBenchmark(int M_, int N_) : M(M_), N(N_) {
        // Ensure N is divisible by 32 for vectorization
        if (N % 32 != 0) {
            throw std::runtime_error("N must be divisible by 32 for vectorized access");
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_weights_uint4, M * (N / 32) * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&d_weights_uint32, M * (N / 8) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_input_half2, (N / 2) * sizeof(half2)));
        CUDA_CHECK(cudaMalloc(&d_input_scalar, N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_bias, M * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output, M * sizeof(half)));
        
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        initialize_data();
    }
    
    ~VectorizedGemvBenchmark() {
        CUDA_CHECK(cudaFree(d_weights_uint4));
        CUDA_CHECK(cudaFree(d_weights_uint32));
        CUDA_CHECK(cudaFree(d_input_half2));
        CUDA_CHECK(cudaFree(d_input_scalar));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_output));
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
        
        // Generate and pack weights
        std::vector<uint4> h_weights_uint4(M * (N / 32));
        std::vector<uint32_t> h_weights_uint32(M * (N / 8));
        std::vector<half> h_scales(M);
        
        for (int i = 0; i < M; i++) {
            h_scales[i] = __float2half(scale_dist(gen));
        }
        
        for (int i = 0; i < M; i++) {
            half scale = h_scales[i];
            
            // Pack into uint32 format (for scalar kernel)
            for (int j = 0; j < N / 8; j++) {
                uint32_t packed = 0;
                for (int k = 0; k < 8; k++) {
                    float weight_f = weight_dist(gen);
                    half weight_h = __float2half(weight_f);
                    uint8_t quantized = (uint8_t)fmaxf(0.0f, fminf(15.0f, 
                        weight_f / __half2float(scale) + 7.5f));
                    packed |= ((uint32_t)quantized & 0xF) << (k * 4);
                }
                h_weights_uint32[i * (N / 8) + j] = packed;
            }
            
            // Pack into uint4 format (for vectorized kernel)
            for (int j = 0; j < N / 32; j++) {
                uint4 vec_packed;
                uint32_t* packed_ptr = reinterpret_cast<uint32_t*>(&vec_packed);
                
                for (int k = 0; k < 4; k++) {
                    packed_ptr[k] = h_weights_uint32[i * (N / 8) + j * 4 + k];
                }
                
                h_weights_uint4[i * (N / 32) + j] = vec_packed;
            }
        }
        
        // Generate input data
        std::vector<half> h_input_scalar(N);
        std::vector<half2> h_input_half2(N / 2);
        
        for (int i = 0; i < N; i++) {
            h_input_scalar[i] = __float2half(input_dist(gen));
        }
        
        for (int i = 0; i < N / 2; i++) {
            h_input_half2[i] = __halves2half2(h_input_scalar[i * 2], h_input_scalar[i * 2 + 1]);
        }
        
        // Generate bias
        std::vector<half> h_bias(M);
        for (int i = 0; i < M; i++) {
            h_bias[i] = __float2half(weight_dist(gen) * 0.1f);
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_weights_uint4, h_weights_uint4.data(), 
                             M * (N / 32) * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights_uint32, h_weights_uint32.data(), 
                             M * (N / 8) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_half2, h_input_half2.data(), 
                             (N / 2) * sizeof(half2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_scalar, h_input_scalar.data(), 
                             N * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), 
                             M * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 
                             M * sizeof(half), cudaMemcpyHostToDevice));
    }

public:
    float benchmark_kernel(const std::string& name, std::function<void()> launch_func) {
        // Warmup
        for (int i = 0; i < 3; i++) {
            launch_func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            launch_func();
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms / iterations;
    }
    
    void run_all_benchmarks() {
        dim3 block_size(256);
        dim3 grid_size((M + block_size.x - 1) / block_size.x);
        
        std::cout << "\n🚀 Running Vectorization Benchmarks:" << std::endl;
        
        // Scalar baseline
        float scalar_time = benchmark_kernel("Scalar Baseline", [&]() {
            vectorized_kernels::scalar_4bit_gemv<<<grid_size, block_size>>>(
                d_weights_uint32, d_input_scalar, d_scales, d_bias, d_output, M, N);
        });
        
        // Vectorized weight loading
        float vec_weight_time = benchmark_kernel("Vectorized Weights", [&]() {
            vectorized_kernels::vectorized_weight_load_gemv<<<grid_size, block_size>>>(
                d_weights_uint4, d_input_scalar, d_scales, d_bias, d_output, M, N);
        });
        
        // Vectorized input and weights
        float vec_both_time = benchmark_kernel("Vectorized Input+Weights", [&]() {
            vectorized_kernels::vectorized_input_weight_gemv<<<grid_size, block_size>>>(
                d_weights_uint4, d_input_half2, d_scales, d_bias, d_output, M, N);
        });
        
        // Fully vectorized with multiple elements per thread
        dim3 full_vec_grid((M + block_size.x * 4 - 1) / (block_size.x * 4));
        float full_vec_time = benchmark_kernel("Fully Vectorized", [&]() {
            vectorized_kernels::fully_vectorized_gemv<<<full_vec_grid, block_size>>>(
                d_weights_uint4, d_input_half2, d_scales, d_bias, d_output, M, N, 4);
        });
        
        // Coalesced access optimization
        float coalesced_time = benchmark_kernel("Coalesced Access", [&]() {
            vectorized_kernels::coalesced_vectorized_gemv<<<grid_size, block_size>>>(
                d_weights_uint4, d_input_half2, d_scales, d_bias, d_output, M, N);
        });
        
        print_performance_comparison(scalar_time, vec_weight_time, vec_both_time, 
                                   full_vec_time, coalesced_time);
    }
    
private:
    void print_performance_comparison(float scalar, float vec_weight, float vec_both,
                                    float full_vec, float coalesced) {
        long long ops = 2LL * M * N;
        
        auto calc_gflops = [&](float time) {
            return (double)ops / (time / 1000.0) / 1e9;
        };
        
        auto calc_speedup = [&](float time) {
            return scalar / time;
        };
        
        std::cout << "\n📊 Vectorization Performance Analysis:" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "┌──────────────────────┬──────────────┬──────────────┬──────────────┐" << std::endl;
        std::cout << "│ Implementation       │ Time (ms)    │ GFLOPS       │ Speedup      │" << std::endl;
        std::cout << "├──────────────────────┼──────────────┼──────────────┼──────────────┤" << std::endl;
        std::cout << "│ Scalar Baseline      │ " << std::setw(12) << scalar << " │ " 
                  << std::setw(12) << calc_gflops(scalar) << " │ " 
                  << std::setw(12) << "1.00x" << " │" << std::endl;
        std::cout << "│ Vectorized Weights   │ " << std::setw(12) << vec_weight << " │ " 
                  << std::setw(12) << calc_gflops(vec_weight) << " │ " 
                  << std::setw(12) << calc_speedup(vec_weight) << "x │" << std::endl;
        std::cout << "│ Vec Input+Weights    │ " << std::setw(12) << vec_both << " │ " 
                  << std::setw(12) << calc_gflops(vec_both) << " │ " 
                  << std::setw(12) << calc_speedup(vec_both) << "x │" << std::endl;
        std::cout << "│ Fully Vectorized     │ " << std::setw(12) << full_vec << " │ " 
                  << std::setw(12) << calc_gflops(full_vec) << " │ " 
                  << std::setw(12) << calc_speedup(full_vec) << "x │" << std::endl;
        std::cout << "│ Coalesced Access     │ " << std::setw(12) << coalesced << " │ " 
                  << std::setw(12) << calc_gflops(coalesced) << " │ " 
                  << std::setw(12) << calc_speedup(coalesced) << "x │" << std::endl;
        std::cout << "└──────────────────────┴──────────────┴──────────────┴──────────────┘" << std::endl;
    }
};

int main() {
    std::cout << "=== Marlin Tutorial Step 3: Vectorized Memory Access ===" << std::endl;
    
    // Check GPU capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    
    /*
     * PART 1: Understanding Vectorized Memory Access
     */
    std::cout << "\n💾 PART 1: Vectorized Memory Access Fundamentals\n" << std::endl;
    
    std::cout << "🎯 Why Vectorization Matters:" << std::endl;
    std::cout << "• GPU memory is optimized for wide transactions (128-bit optimal)" << std::endl;
    std::cout << "• Single thread accessing 4 bytes wastes 75% of memory bandwidth" << std::endl;
    std::cout << "• Vectorized loads: float4, int4, uint4 utilize full transaction width" << std::endl;
    std::cout << "• Coalesced access: 32 threads in warp access consecutive addresses" << std::endl;
    
    std::cout << "\n🏗️ Memory Transaction Sizes:" << std::endl;
    std::cout << "• 32-bit scalar:  32-bit transaction (poor utilization)" << std::endl;
    std::cout << "• 64-bit vector:  64-bit transaction (half utilization)" << std::endl;
    std::cout << "• 128-bit vector: 128-bit transaction (optimal utilization)" << std::endl;
    std::cout << "• Alignment: Data must be aligned to transaction size" << std::endl;
    
    std::cout << "\n⚡ Vectorization Strategy:" << std::endl;
    std::cout << "1. Use uint4 for 4-bit weight loading (128-bit = 32 weights)" << std::endl;
    std::cout << "2. Use half2 for input vector loading (32-bit = 2 FP16 values)" << std::endl;
    std::cout << "3. Process multiple elements per thread" << std::endl;
    std::cout << "4. Ensure coalesced access patterns within warps" << std::endl;
    
    /*
     * PART 2: Progressive Vectorization Benchmarks
     */
    std::cout << "\n📈 PART 2: Progressive Vectorization Performance\n" << std::endl;
    
    // Test different problem sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {2048, 2048},    // Medium square
        {4096, 2048},    // Rectangular
        {8192, 4096}     // Large
    };
    
    for (auto [M, N] : test_sizes) {
        std::cout << "\n🧪 Testing GEMV size: " << M << " x " << N << std::endl;
        
        try {
            VectorizedGemvBenchmark benchmark(M, N);
            benchmark.run_all_benchmarks();
            
        } catch (const std::exception& e) {
            std::cout << "Error with size " << M << "x" << N << ": " << e.what() << std::endl;
        }
    }
    
    /*
     * PART 3: Memory Access Pattern Analysis
     */
    std::cout << "\n🔍 PART 3: Memory Access Pattern Analysis\n" << std::endl;
    
    std::cout << "📊 Memory Transaction Analysis:" << std::endl;
    std::cout << "• Scalar Access Pattern:" << std::endl;
    std::cout << "  - Each thread loads 1 uint32 (4 bytes)" << std::endl;
    std::cout << "  - Memory transactions: 32 threads × 4 bytes = 128 bytes per warp" << std::endl;
    std::cout << "  - Utilization: Good if coalesced" << std::endl;
    
    std::cout << "• Vectorized Access Pattern:" << std::endl;
    std::cout << "  - Each thread loads 1 uint4 (16 bytes)" << std::endl;
    std::cout << "  - Memory transactions: 32 threads × 16 bytes = 512 bytes per warp" << std::endl;
    std::cout << "  - Utilization: Excellent if aligned and coalesced" << std::endl;
    
    std::cout << "\n⚖️ Trade-offs:" << std::endl;
    std::cout << "Benefits:" << std::endl;
    std::cout << "  + Higher memory bandwidth utilization" << std::endl;
    std::cout << "  + Fewer memory transactions per element" << std::endl;
    std::cout << "  + Better cache line utilization" << std::endl;
    std::cout << "  + More work per thread (amortizes overheads)" << std::endl;
    
    std::cout << "Costs:" << std::endl;
    std::cout << "  - Higher register pressure per thread" << std::endl;
    std::cout << "  - More complex kernel code" << std::endl;
    std::cout << "  - Alignment requirements" << std::endl;
    std::cout << "  - Potentially lower occupancy" << std::endl;
    
    /*
     * PART 4: Optimization Techniques Demonstrated
     */
    std::cout << "\n🛠️ PART 4: Optimization Techniques\n" << std::endl;
    
    std::cout << "🔧 Level 1 - Vectorized Weight Loading:" << std::endl;
    std::cout << "• Use uint4 instead of uint32 for weight loading" << std::endl;
    std::cout << "• Process 32 weights per load instead of 8" << std::endl;
    std::cout << "• Reduces memory transactions by 4x" << std::endl;
    
    std::cout << "\n🔧 Level 2 - Vectorized Input Loading:" << std::endl;
    std::cout << "• Use half2 for input vector elements" << std::endl;
    std::cout << "• SIMD operations with __hfma2 for parallel MAC" << std::endl;
    std::cout << "• Better arithmetic intensity" << std::endl;
    
    std::cout << "\n🔧 Level 3 - Multiple Elements Per Thread:" << std::endl;
    std::cout << "• Each thread processes 4 output elements" << std::endl;
    std::cout << "• Reduces kernel launch overhead" << std::endl;
    std::cout << "• Better amortization of setup costs" << std::endl;
    
    std::cout << "\n🔧 Level 4 - Coalesced Access Patterns:" << std::endl;
    std::cout << "• Organize threads for optimal memory access" << std::endl;
    std::cout << "• Warp-aware thread indexing" << std::endl;
    std::cout << "• Maximize cache line utilization" << std::endl;
    
    /*
     * PART 5: Future Optimizations Preview
     */
    std::cout << "\n🚀 PART 5: What's Next?\n" << std::endl;
    
    std::cout << "🎯 Remaining Optimization Opportunities:" << std::endl;
    std::cout << "• Warp-level reductions for cross-thread cooperation" << std::endl;
    std::cout << "• Shared memory for data reuse and staging" << std::endl;
    std::cout << "• Producer-consumer warp specialization" << std::endl;
    std::cout << "• Advanced memory layouts and swizzling" << std::endl;
    std::cout << "• Double/triple buffering for latency hiding" << std::endl;
    
    std::cout << "\n📈 Expected Performance Trajectory:" << std::endl;
    std::cout << "• Step 2 (Naive): ~10-20% of peak performance" << std::endl;
    std::cout << "• Step 3 (Vectorized): ~30-50% of peak performance" << std::endl;
    std::cout << "• Step 4 (Warp Ops): ~50-70% of peak performance" << std::endl;
    std::cout << "• Step 5+ (Advanced): ~80-95% of peak performance" << std::endl;
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Step 3 Summary: Vectorization Mastered ===\n" << std::endl;
    std::cout << "✅ You learned:" << std::endl;
    std::cout << "   • Vectorized memory operations using uint4, half2" << std::endl;
    std::cout << "   • Coalesced memory access patterns within warps" << std::endl;
    std::cout << "   • Progressive optimization from scalar to fully vectorized" << std::endl;
    std::cout << "   • Memory bandwidth optimization principles" << std::endl;
    std::cout << "   • Trade-offs between vectorization and occupancy" << std::endl;
    
    std::cout << "\n🎯 Key Achievements:" << std::endl;
    std::cout << "   • Significant performance improvement over naive implementation" << std::endl;
    std::cout << "   • Better GPU resource utilization" << std::endl;
    std::cout << "   • Foundation for advanced optimization techniques" << std::endl;
    std::cout << "   • Understanding of GPU memory architecture" << std::endl;
    
    std::cout << "\n⚡ Next: Step 4 will add warp-level operations and shared memory!" << std::endl;
    
    return 0;
}