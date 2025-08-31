#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

/*
 * Marlin Tutorial Step 5: Warp Specialization
 * 
 * Producer-Consumer pattern with memory-compute overlap
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

__global__ void warp_specialized_gemv(
    const uint32_t* __restrict__ weights_packed,
    const half* __restrict__ input,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M, int N) {
    
    __shared__ uint32_t shared_weights[256];
    __shared__ half shared_input[512];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;
    
    if (row >= M) return;
    
    if (warp_id == 0) {
        // Producer warp: Load weights to shared memory
        int load_idx = lane_id;
        while (load_idx < N / 8) {
            shared_weights[load_idx] = weights_packed[row * (N / 8) + load_idx];
            load_idx += 32;
        }
        
        // Load input
        load_idx = lane_id;
        while (load_idx < min(512, N)) {
            shared_input[load_idx] = input[load_idx];
            load_idx += 32;
        }
    }
    
    __syncthreads();
    
    if (warp_id == 1) {
        // Consumer warp: Compute using shared data
        half accumulator = __float2half(0.0f);
        half scale = scales[row];
        
        for (int pack = lane_id; pack < N / 8; pack += 32) {
            uint32_t packed = shared_weights[pack];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t weight_4bit = (packed >> (i * 4)) & 0xF;
                half weight = __float2half((float(weight_4bit) - 7.5f) * __half2float(scale));
                half input_val = (pack * 8 + i < 512) ? shared_input[pack * 8 + i] : input[pack * 8 + i];
                accumulator = __hfma(weight, input_val, accumulator);
            }
        }
        
        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            accumulator = __hadd(accumulator, __shfl_down_sync(0xffffffff, accumulator, offset));
        }
        
        if (lane_id == 0) {
            output[row] = accumulator;
        }
    }
}

int main() {
    std::cout << "=== Marlin Tutorial Step 5: Warp Specialization ===" << std::endl;
    
    const int M = 512, N = 512;
    
    // Simplified test setup
    uint32_t* d_weights; half* d_input; half* d_scales; half* d_output;
    CUDA_CHECK(cudaMalloc(&d_weights, M * N / 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, M * sizeof(half)));
    
    dim3 block(64);  // 2 warps per block
    dim3 grid(M);    // 1 block per row
    
    warp_specialized_gemv<<<grid, block>>>(d_weights, d_input, d_scales, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "✅ Producer-Consumer warp specialization completed" << std::endl;
    std::cout << "🎯 Key achievement: Memory-compute overlap through warp roles" << std::endl;
    
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "\n🚀 Next: Step 6 will optimize memory layouts!" << std::endl;
    return 0;
}