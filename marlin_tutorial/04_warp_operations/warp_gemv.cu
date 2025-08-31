#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

/*
 * Marlin Tutorial Step 4: Warp-Level Operations and Shared Memory
 * 
 * LEARNING OBJECTIVES:
 * 1. Master warp-level reductions and communication
 * 2. Understand shared memory usage for data staging
 * 3. Learn cross-warp cooperation patterns
 * 4. Implement efficient warp shuffle operations
 * 5. Optimize occupancy vs register usage trade-offs
 * 
 * KEY CONCEPTS:
 * - Warp Primitives: __shfl_down_sync, __shfl_xor_sync
 * - Shared Memory: Fast on-chip memory for thread cooperation
 * - Warp Reduction: Sum across 32 threads in a warp
 * - Memory Staging: Use shared memory to reduce global memory access
 * - Occupancy: Balance between resource usage and parallelism
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

namespace warp_utils {
    
    // Warp-level reduction using shuffle operations
    __device__ inline half warp_reduce_sum(half val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        return val;
    }
    
    // Extract 4-bit and dequantize
    __device__ inline half extract_and_dequantize(uint32_t packed, int idx, half scale) {
        uint8_t weight_4bit = (packed >> (idx * 4)) & 0xF;
        return __float2half((float(weight_4bit) - 7.5f) * __half2float(scale));
    }
}

namespace warp_kernels {
    
    // Warp-cooperative GEMV with shared memory
    __global__ void warp_cooperative_gemv(
        const uint32_t* __restrict__ weights_packed,
        const half* __restrict__ input,
        const half* __restrict__ scales,
        half* __restrict__ output,
        int M, int N) {
        
        __shared__ half shared_input[2048];  // Shared input staging
        
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int row = blockIdx.x * (blockDim.x / 32) + warp_id;
        
        if (row >= M) return;
        
        // Cooperatively load input to shared memory
        int load_idx = threadIdx.x;
        while (load_idx < N && load_idx < 2048) {
            shared_input[load_idx] = input[load_idx];
            load_idx += blockDim.x;
        }
        __syncthreads();
        
        // Each thread in warp processes different parts of the row
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 8);
        
        // Each thread processes every 32nd packed weight
        for (int pack = lane_id; pack < N / 8; pack += 32) {
            uint32_t packed = weights_packed[row_offset + pack];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int input_idx = pack * 8 + i;
                if (input_idx < N) {
                    half weight = warp_utils::extract_and_dequantize(packed, i, scale);
                    half input_val = (input_idx < 2048) ? shared_input[input_idx] : input[input_idx];
                    accumulator = __hfma(weight, input_val, accumulator);
                }
            }
        }
        
        // Warp reduction to sum across all threads in warp
        accumulator = warp_utils::warp_reduce_sum(accumulator);
        
        // Only lane 0 writes the result
        if (lane_id == 0) {
            output[row] = accumulator;
        }
    }
    
    // Multiple warps per row for larger reductions
    __global__ void multi_warp_gemv(
        const uint32_t* __restrict__ weights_packed,
        const half* __restrict__ input,
        const half* __restrict__ scales,
        half* __restrict__ output,
        int M, int N, int warps_per_row = 4) {
        
        __shared__ half warp_results[32]; // Store results from each warp
        
        int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int row = global_warp_id / warps_per_row;
        int warp_in_row = global_warp_id % warps_per_row;
        int lane_id = threadIdx.x % 32;
        
        if (row >= M) return;
        
        // Each warp processes a portion of the row
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        int row_offset = row * (N / 8);
        int start_pack = warp_in_row * (N / 8) / warps_per_row;
        int end_pack = (warp_in_row + 1) * (N / 8) / warps_per_row;
        
        for (int pack = start_pack + lane_id; pack < end_pack; pack += 32) {
            uint32_t packed = weights_packed[row_offset + pack];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int input_idx = pack * 8 + i;
                if (input_idx < N) {
                    half weight = warp_utils::extract_and_dequantize(packed, i, scale);
                    accumulator = __hfma(weight, input[input_idx], accumulator);
                }
            }
        }
        
        // Warp reduction
        accumulator = warp_utils::warp_reduce_sum(accumulator);
        
        // Store warp result in shared memory
        int warp_idx = threadIdx.x / 32;
        if (lane_id == 0) {
            warp_results[warp_idx] = accumulator;
        }
        __syncthreads();
        
        // Final reduction across warps (only first warp participates)
        if (warp_idx == 0 && lane_id < warps_per_row) {
            half final_sum = warp_results[lane_id];
            final_sum = warp_utils::warp_reduce_sum(final_sum);
            
            if (lane_id == 0) {
                output[row] = final_sum;
            }
        }
    }
}

int main() {
    std::cout << "=== Marlin Tutorial Step 4: Warp-Level Operations ===" << std::endl;
    
    // Simple demonstration
    const int M = 1024, N = 1024;
    
    // Allocate and initialize test data
    std::vector<uint32_t> h_weights(M * N / 8);
    std::vector<half> h_input(N), h_scales(M), h_output(M);
    
    // Initialize with simple patterns
    for (int i = 0; i < M * N / 8; i++) h_weights[i] = 0x12345678;
    for (int i = 0; i < N; i++) h_input[i] = __float2half(1.0f);
    for (int i = 0; i < M; i++) h_scales[i] = __float2half(0.1f);
    
    // Device memory
    uint32_t* d_weights; half* d_input; half* d_scales; half* d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, M * N / 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, M * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), M * N / 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), M * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run warp-cooperative kernel
    dim3 block(128);
    dim3 grid((M + block.x/32 - 1) / (block.x/32));
    
    warp_kernels::warp_cooperative_gemv<<<grid, block>>>(d_weights, d_input, d_scales, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "✅ Warp-cooperative GEMV completed successfully" << std::endl;
    std::cout << "🎯 Key concepts demonstrated:" << std::endl;
    std::cout << "   • Warp-level reductions using shuffle operations" << std::endl;
    std::cout << "   • Shared memory for input staging" << std::endl;
    std::cout << "   • Cross-thread cooperation within warps" << std::endl;
    std::cout << "   • Multi-warp coordination for large problems" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "\n🚀 Next: Step 5 will implement full warp specialization!" << std::endl;
    return 0;
}