#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

using namespace cute;
using namespace nvcuda;

// Utility functions
__host__ __device__ inline int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// MMA → Multiply → LayerNorm → MMA Fusion Pipeline
// Target: 9600×1024 → MMA → 9600×2730 → Multiply → LayerNorm → MMA → 9600×1024

template<int TILE_M = 256, int TILE_N = 128, int TILE_K = 32>
struct MMAMultiplyLayerNormConfig {
    static constexpr int kTileM = TILE_M;
    static constexpr int kTileN = TILE_N;
    static constexpr int kTileK = TILE_K;
    static constexpr int kThreads = 256;
    static constexpr int kWarps = kThreads / 32;
    static constexpr int kStages = 4;
    
    // Stage dimensions
    static constexpr int kFirstGemmN = 2730;   // First GEMM output width
    static constexpr int kSecondGemmN = 1024;  // Second GEMM output width
    
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t; 
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
};

// Stage 1: MMA kernel (9600×1024 @ 1024×2730 → 9600×2730)
template<typename Config>
__device__ void stage1_mma(
    const half* input,          // [M, K1] = [9600, 1024]
    const half* weights1,       // [K1, N1] = [1024, 2730]
    half* intermediate,         // [M, N1] = [9600, 2730]
    int M, int K1, int N1,
    int tile_m_start, int tile_n_start,
    volatile half* shared_memory
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Tensor core fragments for first MMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C1;
    
    wmma::fill_fragment(frag_C1, __float2half(0.0f));
    
    int warp_row = (warp_id / 2) * 16;
    int warp_col = (warp_id % 2) * 16;
    
    // First GEMM computation: input @ weights1
    for (int k = 0; k < K1; k += 16) {
        if (tile_m_start + warp_row < M && k < K1) {
            const half* a_ptr = input + (tile_m_start + warp_row) * K1 + k;
            wmma::load_matrix_sync(frag_A1, a_ptr, K1);
        }
        
        if (tile_n_start + warp_col < N1 && k < K1) {
            const half* b_ptr = weights1 + k * N1 + (tile_n_start + warp_col);
            wmma::load_matrix_sync(frag_B1, b_ptr, N1);
        }
        
        if (tile_m_start + warp_row < M && tile_n_start + warp_col < N1 && k < K1) {
            wmma::mma_sync(frag_C1, frag_A1, frag_B1, frag_C1);
        }
    }
    
    // Store first MMA results to intermediate buffer
    if (tile_m_start + warp_row < M && tile_n_start + warp_col < N1) {
        half* c_ptr = intermediate + (tile_m_start + warp_row) * N1 + (tile_n_start + warp_col);
        wmma::store_matrix_sync(c_ptr, frag_C1, N1, wmma::mem_row_major);
    }
}

// Stage 2: Element-wise multiply operation
template<typename Config>
__device__ void stage2_multiply(
    const half* intermediate,    // [M, N1] = [9600, 2730]
    const half* multiplier,      // [N1] = [2730] or [M, N1] = [9600, 2730]
    half* multiplied,           // [M, N1] = [9600, 2730]
    int M, int N1,
    int tile_m_start,
    bool broadcast_multiplier = true  // True if multiplier is [N1], false if [M, N1]
) {
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
    
    int tile_size = min(Config::kTileM, M - tile_m_start);
    
    // Process elements in parallel
    for (int idx = tid; idx < tile_size * N1; idx += threads_per_block) {
        int row = idx / N1;
        int col = idx % N1;
        int global_row = tile_m_start + row;
        
        if (global_row < M && col < N1) {
            half input_val = intermediate[global_row * N1 + col];
            half mult_val;
            
            if (broadcast_multiplier) {
                mult_val = multiplier[col];  // Broadcast across rows
            } else {
                mult_val = multiplier[global_row * N1 + col];  // Element-wise
            }
            
            multiplied[global_row * N1 + col] = __hmul(input_val, mult_val);
        }
    }
}

// Stage 3: LayerNorm operation
template<typename Config>
__device__ void stage3_layernorm(
    const half* multiplied,     // [M, N1] = [9600, 2730] 
    half* normalized,           // [M, N1] = [9600, 2730]
    const half* gamma,          // [N1] = [2730]
    const half* beta,           // [N1] = [2730]
    int M, int N1,
    int tile_m_start,
    volatile float* stats_buffer
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warps_per_block = Config::kWarps;
    
    int tile_size = min(Config::kTileM, M - tile_m_start);
    int rows_per_warp = ceildiv(tile_size, warps_per_block);
    
    int row_start = warp_id * rows_per_warp;
    int row_end = min(row_start + rows_per_warp, tile_size);
    
    for (int local_row = row_start; local_row < row_end; local_row++) {
        int global_row = tile_m_start + local_row;
        if (global_row >= M) continue;
        
        const half* input_row = multiplied + global_row * N1;
        half* output_row = normalized + global_row * N1;
        
        // Step 1: Compute mean
        float sum = 0.0f;
        for (int col = lane_id; col < N1; col += 32) {
            sum += __half2float(input_row[col]);
        }
        
        // Warp reduction for mean
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        float mean = sum / N1;
        if (lane_id == 0) stats_buffer[local_row * 2] = mean;
        __syncwarp();
        
        // Step 2: Compute variance
        mean = stats_buffer[local_row * 2];  // Broadcast mean
        float var_sum = 0.0f;
        for (int col = lane_id; col < N1; col += 32) {
            float diff = __half2float(input_row[col]) - mean;
            var_sum += diff * diff;
        }
        
        // Warp reduction for variance
        for (int offset = 16; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
        }
        float inv_std = rsqrtf(var_sum / N1 + 1e-5f);
        if (lane_id == 0) stats_buffer[local_row * 2 + 1] = inv_std;
        __syncwarp();
        
        // Step 3: Apply normalization
        inv_std = stats_buffer[local_row * 2 + 1];
        for (int col = lane_id; col < N1; col += 32) {
            float normalized_val = (__half2float(input_row[col]) - mean) * inv_std;
            float result = normalized_val * __half2float(gamma[col]) + __half2float(beta[col]);
            output_row[col] = __float2half(result);
        }
        __syncwarp();
    }
}

// Stage 4: Second MMA kernel (9600×2730 @ 2730×1024 → 9600×1024)
template<typename Config>
__device__ void stage4_mma(
    const half* normalized,     // [M, N1] = [9600, 2730]
    const half* weights2,       // [N1, N2] = [2730, 1024]
    half* output,              // [M, N2] = [9600, 1024]
    int M, int N1, int N2,
    int tile_m_start, int tile_n_start
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Tensor core fragments for second MMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A2;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C2;
    
    wmma::fill_fragment(frag_C2, __float2half(0.0f));
    
    int warp_row = (warp_id / 2) * 16;
    int warp_col = (warp_id % 2) * 16;
    
    // Second GEMM computation: normalized @ weights2
    for (int k = 0; k < N1; k += 16) {
        if (tile_m_start + warp_row < M && k < N1) {
            const half* a_ptr = normalized + (tile_m_start + warp_row) * N1 + k;
            wmma::load_matrix_sync(frag_A2, a_ptr, N1);
        }
        
        if (tile_n_start + warp_col < N2 && k < N1) {
            const half* b_ptr = weights2 + k * N2 + (tile_n_start + warp_col);
            wmma::load_matrix_sync(frag_B2, b_ptr, N2);
        }
        
        if (tile_m_start + warp_row < M && tile_n_start + warp_col < N2 && k < N1) {
            wmma::mma_sync(frag_C2, frag_A2, frag_B2, frag_C2);
        }
    }
    
    // Store final results
    if (tile_m_start + warp_row < M && tile_n_start + warp_col < N2) {
        half* c_ptr = output + (tile_m_start + warp_row) * N2 + (tile_n_start + warp_col);
        wmma::store_matrix_sync(c_ptr, frag_C2, N2, wmma::mem_row_major);
    }
}

// Main fused kernel: MMA → Multiply → LayerNorm → MMA
template<typename Config>
__global__ void mma_multiply_layernorm_mma_kernel(
    const half* input,          // [M, K1] = [9600, 1024]
    const half* weights1,       // [K1, N1] = [1024, 2730]
    const half* multiplier,     // [N1] = [2730] or [M, N1] = [9600, 2730]
    const half* gamma,          // [N1] = [2730]
    const half* beta,           // [N1] = [2730]
    const half* weights2,       // [N1, N2] = [2730, 1024]
    half* output,              // [M, N2] = [9600, 1024]
    int M = 9600, int K1 = 1024, int N1 = 2730, int N2 = 1024,
    bool broadcast_multiplier = true
) {
    const int tile_m = blockIdx.y * Config::kTileM;
    const int tile_n = blockIdx.x * Config::kTileN;
    
    // Shared memory allocation
    extern __shared__ char shmem[];
    half* s_intermediate = reinterpret_cast<half*>(shmem);
    half* s_multiplied = s_intermediate + Config::kTileM * N1;
    half* s_normalized = s_multiplied + Config::kTileM * N1;
    volatile float* s_stats = reinterpret_cast<volatile float*>(s_normalized + Config::kTileM * N1);
    
    if (tile_m >= M) return;
    
    __syncthreads();
    
    // Stage 1: First MMA (input @ weights1 → intermediate)
    stage1_mma<Config>(
        input, weights1, s_intermediate,
        M, K1, N1, tile_m, 0,  // Process full N1 dimension
        nullptr  // Not used in this stage
    );
    __syncthreads();
    
    // Stage 2: Element-wise multiply
    stage2_multiply<Config>(
        s_intermediate, multiplier, s_multiplied,
        M, N1, tile_m, broadcast_multiplier
    );
    __syncthreads();
    
    // Stage 3: LayerNorm
    stage3_layernorm<Config>(
        s_multiplied, s_normalized, gamma, beta,
        M, N1, tile_m, s_stats
    );
    __syncthreads();
    
    // Stage 4: Second MMA (normalized @ weights2 → output)
    // Process in N2 tiles
    for (int n_tile = 0; n_tile < N2; n_tile += Config::kTileN) {
        int current_tile_n = min(Config::kTileN, N2 - n_tile);
        if (tile_n < current_tile_n) {
            stage4_mma<Config>(
                s_normalized, weights2, output,
                M, N1, N2, tile_m, n_tile + tile_n
            );
        }
        __syncthreads();
    }
}

// High-level launcher
void launch_mma_multiply_layernorm_mma(
    const half* input,          // [9600, 1024]
    const half* weights1,       // [1024, 2730] 
    const half* multiplier,     // [2730] or [9600, 2730]
    const half* gamma,          // [2730]
    const half* beta,           // [2730]
    const half* weights2,       // [2730, 1024]
    half* output,              // [9600, 1024]
    bool broadcast_multiplier = true,
    cudaStream_t stream = 0
) {
    using Config = MMAMultiplyLayerNormConfig<256, 128, 32>;
    
    const int M = 9600, K1 = 1024, N1 = 2730, N2 = 1024;
    
    // Grid configuration
    dim3 grid(ceildiv(N2, Config::kTileN), ceildiv(M, Config::kTileM));
    dim3 block(Config::kThreads);
    
    // Shared memory calculation
    size_t shmem_size = (Config::kTileM * N1 * 3 +      // intermediate + multiplied + normalized
                        Config::kTileM * 2) * sizeof(half) +  // stats buffer
                        Config::kTileM * 2 * sizeof(float);   // stats as float
    
    // Set dynamic shared memory if needed
    if (shmem_size > 48 * 1024) {
        cudaFuncSetAttribute(mma_multiply_layernorm_mma_kernel<Config>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shmem_size);
    }
    
    mma_multiply_layernorm_mma_kernel<Config><<<grid, block, shmem_size, stream>>>(
        input, weights1, multiplier, gamma, beta, weights2, output,
        M, K1, N1, N2, broadcast_multiplier
    );
}

// Alternative streaming approach for better memory efficiency
template<typename Config>
__global__ void streaming_mma_multiply_layernorm_mma_kernel(
    const half* input,          // [M, K1]
    const half* weights1,       // [K1, N1]
    const half* multiplier,     // [N1] or [M, N1]
    const half* gamma,          // [N1]
    const half* beta,           // [N1]
    const half* weights2,       // [N1, N2]
    half* output,              // [M, N2]
    int M, int K1, int N1, int N2,
    bool broadcast_multiplier,
    int tile_batch_start, int tile_batch_size
) {
    const int tile_m = tile_batch_start + blockIdx.y * Config::kTileM;
    const int tile_n = blockIdx.x * Config::kTileN;
    
    if (tile_m >= min(tile_batch_start + tile_batch_size, M)) return;
    
    // Use global memory for intermediate results in streaming approach
    half* g_intermediate = const_cast<half*>(input) + M * K1;  // Reuse space after input
    half* g_multiplied = g_intermediate + M * N1;
    half* g_normalized = g_multiplied + M * N1;
    
    extern __shared__ char shmem[];
    volatile float* s_stats = reinterpret_cast<volatile float*>(shmem);
    
    // Execute pipeline stages
    __syncthreads();
    
    // Stage 1: MMA
    stage1_mma<Config>(input, weights1, g_intermediate, M, K1, N1, tile_m, 0, nullptr);
    __syncthreads();
    
    // Stage 2: Multiply
    stage2_multiply<Config>(g_intermediate, multiplier, g_multiplied, M, N1, tile_m, broadcast_multiplier);
    __syncthreads();
    
    // Stage 3: LayerNorm
    stage3_layernorm<Config>(g_multiplied, g_normalized, gamma, beta, M, N1, tile_m, s_stats);
    __syncthreads();
    
    // Stage 4: Second MMA
    stage4_mma<Config>(g_normalized, weights2, output, M, N1, N2, tile_m, tile_n);
}

void launch_streaming_mma_multiply_layernorm_mma(
    const half* input, const half* weights1, const half* multiplier,
    const half* gamma, const half* beta, const half* weights2, half* output,
    bool broadcast_multiplier = true, cudaStream_t stream = 0
) {
    using Config = MMAMultiplyLayerNormConfig<128, 64, 32>;
    
    const int M = 9600, K1 = 1024, N1 = 2730, N2 = 1024;
    const int batch_size = 2400;  // Process 2400 rows at a time
    
    for (int batch_start = 0; batch_start < M; batch_start += batch_size) {
        int current_batch_size = min(batch_size, M - batch_start);
        
        dim3 grid(ceildiv(N2, Config::kTileN), ceildiv(current_batch_size, Config::kTileM));
        dim3 block(Config::kThreads);
        
        size_t shmem_size = Config::kTileM * 2 * sizeof(float);  // Just stats buffer
        
        streaming_mma_multiply_layernorm_mma_kernel<Config><<<grid, block, shmem_size, stream>>>(
            input, weights1, multiplier, gamma, beta, weights2, output,
            M, K1, N1, N2, broadcast_multiplier, batch_start, current_batch_size
        );
        
        cudaStreamSynchronize(stream);  // Ensure completion before next batch
    }
}

// Main API function with automatic approach selection
void mma_multiply_layernorm_mma_fused(
    const half* input, const half* weights1, const half* multiplier,
    const half* gamma, const half* beta, const half* weights2, half* output,
    bool broadcast_multiplier = true, cudaStream_t stream = 0
) {
    // Calculate memory requirements
    const int M = 9600, K1 = 1024, N1 = 2730, N2 = 1024;
    size_t required_shmem = (256 * N1 * 3 + 256 * 2) * sizeof(half) + 256 * 2 * sizeof(float);
    
    if (required_shmem <= 96 * 1024) {  // Use shared memory approach if fits
        launch_mma_multiply_layernorm_mma(input, weights1, multiplier, gamma, beta, weights2, output, broadcast_multiplier, stream);
    } else {  // Use streaming approach for better memory efficiency
        launch_streaming_mma_multiply_layernorm_mma(input, weights1, multiplier, gamma, beta, weights2, output, broadcast_multiplier, stream);
    }
}