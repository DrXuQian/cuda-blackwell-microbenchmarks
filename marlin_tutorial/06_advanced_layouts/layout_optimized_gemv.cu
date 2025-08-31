#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

/*
 * Marlin Tutorial Step 6: Advanced Memory Layouts
 * 
 * Optimized 4-bit weight storage for maximal coalescing
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

// Marlin-style weight layout: interleaved for optimal memory access
__global__ void marlin_layout_gemv(
    const uint32_t* __restrict__ weights_marlin_layout,
    const half* __restrict__ input,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M, int N) {
    
    __shared__ uint32_t tile_weights[16][32]; // 16 rows x 32 packed weights
    __shared__ half tile_scales[16];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tile_row = blockIdx.x * 16 + threadIdx.x / 32;
    
    // Cooperative loading with optimal layout
    if (warp_id < 16 && tile_row < M) {
        // Load scales
        if (lane_id == 0) {
            tile_scales[warp_id] = scales[tile_row];
        }
        
        // Load weights in Marlin layout (interleaved pattern)
        for (int i = lane_id; i < 32 && i < N / 8; i += 32) {
            // Marlin layout: weights for consecutive rows are interleaved
            int layout_idx = (tile_row / 16) * (16 * N / 8) + i * 16 + (tile_row % 16);
            tile_weights[warp_id][i] = weights_marlin_layout[layout_idx];
        }
    }
    __syncthreads();
    
    // Each warp computes one output
    if (warp_id < 16 && tile_row < M) {
        half accumulator = __float2half(0.0f);
        half scale = tile_scales[warp_id];
        
        for (int pack = lane_id; pack < N / 8; pack += 32) {
            uint32_t packed = tile_weights[warp_id][pack];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t weight_4bit = (packed >> (i * 4)) & 0xF;
                half weight = __float2half((float(weight_4bit) - 7.5f) * __half2float(scale));
                accumulator = __hfma(weight, input[pack * 8 + i], accumulator);
            }
        }
        
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            accumulator = __hadd(accumulator, __shfl_down_sync(0xffffffff, accumulator, offset));
        }
        
        if (lane_id == 0) {
            output[tile_row] = accumulator;
        }
    }
}

int main() {
    std::cout << "=== Marlin Tutorial Step 6: Advanced Memory Layouts ===" << std::endl;
    
    const int M = 512, N = 512;
    
    uint32_t* d_weights; half* d_input; half* d_scales; half* d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, M * N / 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, M * sizeof(half)));
    
    dim3 block(512);  // 16 warps per block
    dim3 grid((M + 15) / 16);
    
    marlin_layout_gemv<<<grid, block>>>(d_weights, d_input, d_scales, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "✅ Marlin-style memory layout optimization completed" << std::endl;
    std::cout << "🎯 Key concepts: Interleaved layouts for optimal coalescing" << std::endl;
    
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "\n🚀 Next: Step 7 will combine all optimizations!" << std::endl;
    return 0;
}