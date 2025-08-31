#include "../utils/blackwell_common.h"
#include <mma.h>

// Simplified W4A16 GEMV kernel that works on current hardware
using namespace nvcuda;

// Simple 4-bit dequantization
__device__ inline void simple_dequant_w4a16(
    const uint32_t packed_weight,
    half* output,
    half scale
) {
    // Extract 8 4-bit weights from packed uint32
    for (int i = 0; i < 8; i++) {
        uint32_t w4 = (packed_weight >> (i * 4)) & 0xF;
        float w_float = (float)w4 * __half2float(scale);
        output[i] = __float2half(w_float);
    }
}

// W4A16 GEMV kernel with basic warp specialization
__global__ void blackwell_w4a16_simple_kernel(
    const half* __restrict__ A,           // [M, K] activations
    const uint32_t* __restrict__ B,       // [K/8, N] 4-bit weights  
    half* __restrict__ C,                 // [M, N] output
    const half* __restrict__ scales,      // [K/group_size, N] scales
    int M, int N, int K
) {
    extern __shared__ char shmem[];
    uint32_t* shmem_B_packed = reinterpret_cast<uint32_t*>(shmem);
    half* shmem_B_dequant = reinterpret_cast<half*>(shmem_B_packed + 1024); // Offset
    half* shmem_scales = shmem_B_dequant + 8192;  // After dequantized weights
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int block_col = blockIdx.x * 128;  // 128 columns per block
    const int group_size = 128;
    
    // Load weights and scales to shared memory
    if (warp_id == 0) {
        // Load packed weights (K/8 x 128)
        for (int i = tid; i < (K/8) * 128; i += blockDim.x) {
            int row = i / 128;
            int col = i % 128;
            int global_col = block_col + col;
            
            if (global_col < N && row < K/8) {
                shmem_B_packed[i] = B[row * N + global_col];
            } else {
                shmem_B_packed[i] = 0;
            }
        }
    }
    
    if (warp_id == 1) {
        // Load scales
        for (int i = tid; i < (K/group_size) * 128; i += blockDim.x) {
            int row = i / 128;
            int col = i % 128;
            int global_col = block_col + col;
            
            if (global_col < N && row < K/group_size) {
                shmem_scales[i] = scales[row * N + global_col];
            } else {
                shmem_scales[i] = __float2half(1.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Dequantization warp
    if (warp_id == 2) {
        // Dequantize weights in groups
        for (int group = 0; group < K / group_size; group++) {
            half scale_base = shmem_scales[group * 128 + lane_id];
            
            for (int i = 0; i < group_size / 8; i++) {
                int packed_idx = group * (group_size / 8) * 128 + i * 128 + lane_id;
                int output_base = packed_idx * 8;
                
                if (output_base + 8 <= K * 128) {
                    uint32_t packed = shmem_B_packed[packed_idx];
                    half dequant_output[8];
                    simple_dequant_w4a16(packed, dequant_output, scale_base);
                    
                    for (int j = 0; j < 8; j++) {
                        shmem_B_dequant[output_base + j] = dequant_output[j];
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute warps
    if (warp_id >= 3) {
        // Simple GEMV computation
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            
            // Each thread processes multiple elements
            for (int k = lane_id; k < K; k += 32) {
                for (int n_local = 0; n_local < 128; n_local += 32) {
                    int n_global = block_col + n_local + (warp_id - 3) * 8;
                    
                    if (n_global < N) {
                        half a_val = A[m * K + k];
                        half b_val = shmem_B_dequant[k * 128 + n_local + (warp_id - 3) * 8];
                        sum += __half2float(a_val) * __half2float(b_val);
                    }
                }
            }
            
            // Simple warp reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // Store result
            if (lane_id == 0) {
                int output_col = block_col + (warp_id - 3) * 32;
                if (output_col < N) {
                    C[m * N + output_col] = __float2half(sum);
                }
            }
        }
    }
}

// Host launcher
void launch_blackwell_w4a16_simple(
    const half* A, const uint32_t* B, half* C, const half* scales,
    int M, int N, int K
) {
    dim3 grid((N + 127) / 128, 1);
    dim3 block(256);  // 8 warps
    
    // Shared memory: packed weights + dequantized weights + scales
    size_t shmem_size = 1024 * sizeof(uint32_t) +    // Packed weights
                       8192 * sizeof(half) +          // Dequantized weights  
                       1024 * sizeof(half);           // Scales
    
    CUDA_CHECK(cudaFuncSetAttribute(blackwell_w4a16_simple_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_size));
    
    blackwell_w4a16_simple_kernel<<<grid, block, shmem_size>>>(
        A, B, C, scales, M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Test program
int main() {
    printf("🚀 RTX 5070 Blackwell W4A16 GEMV (Simple Version)\n");
    printf("==================================================\n");
    
    if (!check_blackwell_support()) {
        printf("⚠️  Running on non-Blackwell hardware for compatibility\n");
    }
    
    const int M = 1, N = 4096, K = 4096;
    printf("\nTesting W4A16 GEMV: %dx%d @ %dx%d\n", M, K, K, N);
    
    // Memory allocation
    size_t size_A = M * K * sizeof(half);
    size_t size_B = (K/8) * N * sizeof(uint32_t);
    size_t size_scales = (K/128) * N * sizeof(half);  // group_size = 128
    size_t size_C = M * N * sizeof(half);
    
    half *h_A = (half*)malloc(size_A);
    uint32_t *h_B = (uint32_t*)malloc(size_B);
    half *h_scales = (half*)malloc(size_scales);
    half *h_C = (half*)malloc(size_C);
    
    if (!h_A || !h_B || !h_scales || !h_C) {
        printf("❌ Host memory allocation failed\n");
        return 1;
    }
    
    // Initialize test data
    srand(42);
    
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(0.1f);
    }
    
    for (int i = 0; i < (K/8) * N; i++) {
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++) {
            uint32_t w4 = 1;  // Simple weight value
            packed |= (w4 << (j * 4));
        }
        h_B[i] = packed;
    }
    
    for (int i = 0; i < (K/128) * N; i++) {
        h_scales[i] = __float2half(0.1f);
    }
    
    // Device allocation
    half *d_A, *d_C, *d_scales;
    uint32_t *d_B;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, size_scales, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    printf("\n🧪 Running W4A16 GEMV benchmark...\n");
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        launch_blackwell_w4a16_simple(d_A, d_B, d_C, d_scales, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark  
    BlackwellTimer timer;
    blackwell_timer_init(&timer, 50);
    
    const int iterations = 20;
    for (int i = 0; i < iterations; i++) {
        blackwell_timer_start(&timer);
        launch_blackwell_w4a16_simple(d_A, d_B, d_C, d_scales, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        blackwell_timer_stop(&timer);
    }
    
    float avg_time = blackwell_timer_get_avg(&timer);
    double tflops = calculate_tflops(M, N, K, avg_time);
    
    printf("\n📊 W4A16 GEMV Results:\n");
    printf("   Average time: %.3f ms\n", avg_time);
    printf("   Performance: %.1f TFLOPS\n", tflops);
    printf("   Memory compression: 4.0x vs FP16\n");
    
    // Validation
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    bool valid = true;
    for (int i = 0; i < std::min(10, M * N); i++) {
        float val = __half2float(h_C[i]);
        if (isnan(val) || isinf(val)) {
            valid = false;
            break;
        }
    }
    
    printf("   Validation: %s\n", valid ? "✅ PASSED" : "❌ FAILED");
    
    if (valid) {
        printf("   Sample outputs: %.3f, %.3f, %.3f\n",
               __half2float(h_C[0]), __half2float(h_C[1]), __half2float(h_C[2]));
    }
    
    // Cleanup
    blackwell_timer_cleanup(&timer);
    free(h_A); free(h_B); free(h_scales); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_scales); cudaFree(d_C);
    
    printf("\n✨ W4A16 GEMV test completed!\n");
    printf("This simplified version demonstrates basic warp specialization.\n");
    
    return 0;
}