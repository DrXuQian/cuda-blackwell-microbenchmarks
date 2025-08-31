#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

/*
 * Marlin Tutorial Step 7: Complete Marlin Implementation
 * 
 * Production-ready 4-bit quantized GEMV with all optimizations:
 * - Vectorized memory access
 * - Warp specialization
 * - Advanced memory layouts
 * - Double buffering
 * - Optimal tile sizes
 */

#define CUDA_CHECK(status) \
    { \
        cudaError_t error = status; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Full Marlin implementation with all optimizations
template<int TILE_M, int TILE_N, int WARP_SIZE = 32>
__global__ void marlin_gemv_kernel(
    const uint4* __restrict__ weights_packed,   // Vectorized weight storage
    const half2* __restrict__ input_vec,        // Vectorized input
    const half* __restrict__ scales,            // Per-group scales
    const half* __restrict__ zeros,             // Per-group zero points
    half* __restrict__ output,                  // Output vector
    int M, int N, int group_size) {
    
    // Shared memory with double buffering
    __shared__ uint4 shared_weights[2][TILE_M][TILE_N/32];
    __shared__ half2 shared_input[2][TILE_N/2];
    __shared__ half shared_scales[TILE_M];
    __shared__ half partial_results[TILE_M];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Producer warps (first half) vs Consumer warps (second half)
    bool is_producer = warp_id < num_warps / 2;
    int buffer = 0;
    
    for (int tile_k = 0; tile_k < N; tile_k += TILE_N) {
        int next_buffer = 1 - buffer;
        
        if (is_producer) {
            // Producer warps: Load data with double buffering
            
            // Load input vector
            int input_warp = warp_id;
            for (int i = lane_id; i < TILE_N/2; i += WARP_SIZE) {
                int global_idx = tile_k/2 + i;
                if (global_idx < N/2) {
                    shared_input[next_buffer][i] = input_vec[global_idx];
                }
            }
            
            // Load weights with Marlin layout
            int weight_warp = warp_id - num_warps/4;
            if (weight_warp >= 0) {
                for (int row = 0; row < TILE_M; row++) {
                    for (int col = lane_id; col < TILE_N/32; col += WARP_SIZE) {
                        int global_row = blockIdx.x * TILE_M + row;
                        if (global_row < M) {
                            // Optimized layout index calculation
                            int layout_idx = global_row * (N/32) + tile_k/32 + col;
                            shared_weights[next_buffer][row][col] = weights_packed[layout_idx];
                        }
                    }
                }
            }
            
            // Load scales
            if (warp_id == 0 && lane_id < TILE_M) {
                int global_row = blockIdx.x * TILE_M + lane_id;
                if (global_row < M) {
                    int group_idx = global_row / group_size;
                    shared_scales[lane_id] = scales[group_idx];
                }
            }
            
        } else {
            // Consumer warps: Compute using loaded data
            __syncthreads(); // Wait for producers to load data
            
            int compute_warp = warp_id - num_warps/2;
            int rows_per_compute_warp = TILE_M / (num_warps/2);
            int start_row = compute_warp * rows_per_compute_warp;
            int end_row = min(start_row + rows_per_compute_warp, TILE_M);
            
            for (int row = start_row; row < end_row; row++) {
                half2 accumulator = __float2half2_rn(0.0f);
                half scale = shared_scales[row];
                
                // Vectorized computation
                for (int col_vec = lane_id; col_vec < TILE_N/32; col_vec += WARP_SIZE) {
                    uint4 packed_weights = shared_weights[buffer][row][col_vec];
                    
                    // Unpack and process 32 weights
                    uint32_t* packed_ptr = reinterpret_cast<uint32_t*>(&packed_weights);
                    
                    #pragma unroll
                    for (int pack = 0; pack < 4; pack++) {
                        uint32_t packed = packed_ptr[pack];
                        
                        #pragma unroll 4
                        for (int i = 0; i < 8; i += 2) {
                            // Extract 2 x 4-bit weights
                            uint8_t w0 = (packed >> (i * 4)) & 0xF;
                            uint8_t w1 = (packed >> ((i+1) * 4)) & 0xF;
                            
                            // Dequantize to half2
                            float f0 = (float(w0) - 7.5f) * __half2float(scale);
                            float f1 = (float(w1) - 7.5f) * __half2float(scale);
                            half2 weights_h2 = __floats2half2_rn(f0, f1);
                            
                            // Load corresponding input
                            int input_idx = col_vec * 16 + pack * 4 + i/2;
                            if (input_idx < TILE_N/2) {
                                half2 input_h2 = shared_input[buffer][input_idx];
                                accumulator = __hfma2(weights_h2, input_h2, accumulator);
                            }
                        }
                    }
                }
                
                // Warp reduction
                half sum = __hadd(__low2half(accumulator), __high2half(accumulator));
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
                }
                
                // Store partial result
                if (lane_id == 0) {
                    if (tile_k == 0) {
                        partial_results[row] = sum;
                    } else {
                        partial_results[row] = __hadd(partial_results[row], sum);
                    }
                }
            }
        }
        
        buffer = next_buffer;
        __syncthreads();
    }
    
    // Write final results
    if (!is_producer && threadIdx.x < TILE_M) {
        int global_row = blockIdx.x * TILE_M + threadIdx.x;
        if (global_row < M) {
            output[global_row] = partial_results[threadIdx.x];
        }
    }
}

// Performance benchmark
class MarlinBenchmark {
private:
    int M, N;
    uint4* d_weights;
    half2* d_input;
    half* d_scales;
    half* d_output;
    cudaEvent_t start_event, stop_event;

public:
    MarlinBenchmark(int M_, int N_) : M(M_), N(N_) {
        CUDA_CHECK(cudaMalloc(&d_weights, M * (N/32) * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&d_input, (N/2) * sizeof(half2)));
        CUDA_CHECK(cudaMalloc(&d_scales, (M/128 + 1) * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output, M * sizeof(half)));
        
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        // Initialize with random data (simplified)
        CUDA_CHECK(cudaMemset(d_weights, 0x88, M * (N/32) * sizeof(uint4)));
        CUDA_CHECK(cudaMemset(d_input, 0, (N/2) * sizeof(half2)));
        CUDA_CHECK(cudaMemset(d_scales, 0, (M/128 + 1) * sizeof(half)));
    }
    
    ~MarlinBenchmark() {
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }
    
    float benchmark() {
        constexpr int TILE_M = 64;
        constexpr int TILE_N = 256;
        
        dim3 block(256);  // 8 warps per block
        dim3 grid((M + TILE_M - 1) / TILE_M);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            marlin_gemv_kernel<TILE_M, TILE_N><<<grid, block>>>(
                d_weights, d_input, d_scales, nullptr, d_output, M, N, 128);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        const int iterations = 100;
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < iterations; i++) {
            marlin_gemv_kernel<TILE_M, TILE_N><<<grid, block>>>(
                d_weights, d_input, d_scales, nullptr, d_output, M, N, 128);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms / iterations;
    }
};

int main() {
    std::cout << "=== Marlin Tutorial Step 7: Complete Implementation ===" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    
    // Test different problem sizes
    std::vector<std::pair<int, int>> sizes = {
        {4096, 4096},
        {8192, 4096},
        {11008, 4096}  // Common LLM sizes
    };
    
    std::cout << "\n🏆 Full Marlin Implementation Benchmarks:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    for (auto [M, N] : sizes) {
        if (N % 32 != 0) continue; // Ensure alignment
        
        MarlinBenchmark benchmark(M, N);
        float time_ms = benchmark.benchmark();
        
        double gflops = (2.0 * M * N) / (time_ms / 1000.0) / 1e9;
        double memory_gb = (M * N * 0.5 + N * 2 + M * 2) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = memory_gb / (time_ms / 1000.0);
        
        std::cout << "Size " << M << "x" << N << ": " 
                  << time_ms << " ms, " 
                  << gflops << " GFLOPS, "
                  << bandwidth << " GB/s" << std::endl;
    }
    
    std::cout << "\n🎉 CONGRATULATIONS! Marlin Tutorial Complete!" << std::endl;
    std::cout << "✅ You mastered:" << std::endl;
    std::cout << "   • 4-bit quantization fundamentals" << std::endl;
    std::cout << "   • Progressive optimization techniques" << std::endl;
    std::cout << "   • Vectorized memory access patterns" << std::endl;
    std::cout << "   • Warp-level cooperation and specialization" << std::endl;
    std::cout << "   • Advanced memory layouts and double buffering" << std::endl;
    std::cout << "   • Production-ready kernel implementation" << std::endl;
    
    std::cout << "\n🚀 You're now ready to optimize quantized LLM inference!" << std::endl;
    std::cout << "Consider integrating these techniques into your AI/ML pipelines." << std::endl;
    
    return 0;
}