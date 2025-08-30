#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

using namespace cute;

// Utility functions
__host__ __device__ inline int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// LayerNorm + GEMM Fusion using CUTLASS 3.x and CuTe
// Target: 9600x2730 -> LayerNorm -> 9600x2730 @ 2730x1024 -> 9600x1024

template<int TILE_M = 256, int TILE_K = 128>
struct LayerNormGemmConfig {
    static constexpr int kTileM = TILE_M;  // Batch tile size
    static constexpr int kTileK = TILE_K;  // Feature tile size for reduction
    static constexpr int kThreads = 256;
    static constexpr int kWarps = kThreads / 32;
    static constexpr int kStages = 3;      // Pipeline stages
    
    // Memory layout optimization
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor; 
    using LayoutC = cutlass::layout::RowMajor;
    
    // Data types
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;
};

// Efficient LayerNorm implementation with CuTe
template<typename Config>
__device__ void layernorm_tile(
    const half* input,           // Input tile [TILE_M, K]
    half* output,               // Normalized output [TILE_M, K] 
    const half* gamma,          // Scale parameters [K]
    const half* beta,           // Bias parameters [K]
    int M, int K,               // Actual dimensions
    int tile_m_start            // Starting row for this tile
) {
    using namespace cute;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for statistics and intermediate results
    extern __shared__ char shmem[];
    float* s_mean = reinterpret_cast<float*>(shmem);
    float* s_var = s_mean + Config::kTileM;
    half* s_input = reinterpret_cast<half*>(s_var + Config::kTileM);
    
    // Process each row in the tile
    for (int row_offset = 0; row_offset < Config::kTileM; row_offset += Config::kWarps) {
        int row = row_offset + warp_id;
        if (tile_m_start + row >= M) break;
        
        // Step 1: Cooperative load of input row with vectorization
        const half* input_row = input + (tile_m_start + row) * K;
        half* shmem_row = s_input + row * K;
        
        // Vectorized loading with half2
        for (int k = lane_id * 2; k < K; k += 32 * 2) {
            if (k + 1 < K) {
                half2 val = *reinterpret_cast<const half2*>(&input_row[k]);
                *reinterpret_cast<half2*>(&shmem_row[k]) = val;
            } else if (k < K) {
                shmem_row[k] = input_row[k];
            }
        }
        __syncwarp();
        
        // Step 2: Compute mean using warp reduction
        float sum = 0.0f;
        for (int k = lane_id; k < K; k += 32) {
            sum += __half2float(shmem_row[k]);
        }
        
        // Warp-level reduction for mean
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        float mean = sum / K;
        if (lane_id == 0) s_mean[row] = mean;
        __syncwarp();
        
        // Step 3: Compute variance using warp reduction
        float var_sum = 0.0f;
        mean = s_mean[row]; // Broadcast mean to all threads
        
        for (int k = lane_id; k < K; k += 32) {
            float diff = __half2float(shmem_row[k]) - mean;
            var_sum += diff * diff;
        }
        
        // Warp-level reduction for variance
        for (int offset = 16; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
        }
        float variance = var_sum / K;
        if (lane_id == 0) s_var[row] = rsqrtf(variance + 1e-5f); // Store rsqrt for efficiency
        __syncwarp();
        
        // Step 4: Apply normalization with gamma/beta
        float inv_std = s_var[row];
        half* output_row = output + (tile_m_start + row) * K;
        
        for (int k = lane_id; k < K; k += 32) {
            float normalized = (__half2float(shmem_row[k]) - mean) * inv_std;
            float scaled = normalized * __half2float(gamma[k]) + __half2float(beta[k]);
            output_row[k] = __float2half(scaled);
        }
        __syncwarp();
    }
}

// Fused LayerNorm + GEMM kernel using tiled approach
template<typename Config>
__global__ void fused_layernorm_gemm_kernel(
    const half* input,          // [M, K] input matrix  
    const half* weights,        // [K, N] weight matrix
    half* output,              // [M, N] output matrix
    const half* gamma,          // [K] LayerNorm scale
    const half* beta,           // [K] LayerNorm bias
    int M, int K, int N
) {
    using namespace cute;
    
    // Block-level tiling
    const int tile_m = blockIdx.y * Config::kTileM;
    const int tile_n = blockIdx.x * 128; // GEMM N dimension tiling
    
    // Shared memory allocation
    extern __shared__ char shmem[];
    half* s_normalized = reinterpret_cast<half*>(shmem);
    half* s_weights = s_normalized + Config::kTileM * K;
    
    // Stage 1: LayerNorm on current tile
    if (tile_m < M) {
        layernorm_tile<Config>(
            input, s_normalized, gamma, beta, 
            M, K, tile_m
        );
    }
    __syncthreads();
    
    // Stage 2: GEMM computation using normalized data
    // Use CuTe for efficient GEMM tile computation
    auto tensor_A = make_tensor(make_gmem_ptr(s_normalized), 
                               make_shape(min(Config::kTileM, M - tile_m), K),
                               make_stride(K, 1));
    
    auto tensor_B = make_tensor(make_gmem_ptr(weights + tile_n), 
                               make_shape(K, min(128, N - tile_n)),
                               make_stride(N, 1));
    
    auto tensor_C = make_tensor(make_gmem_ptr(output + tile_m * N + tile_n),
                               make_shape(min(Config::kTileM, M - tile_m), min(128, N - tile_n)),
                               make_stride(N, 1));
    
    // Efficient GEMM computation with async MMA + CuTe
    gemm_async_mma_cute(tensor_A, tensor_B, tensor_C);
}

// Tensor core + async MMA GEMM implementation for fusion
template<typename TensorA, typename TensorB, typename TensorC>
__device__ void gemm_async_mma_cute(TensorA const& A, TensorB const& B, TensorC& C) {
    using namespace cute;
    using namespace nvcuda;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Use tensor cores for better performance
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C;
    
    wmma::fill_fragment(frag_C, __float2half(0.0f));
    
    int K_dim = size<1>(A);
    int warp_row = warp_id * 16;
    int warp_col = 0; // Single column for LayerNorm+GEMM fusion
    
    // Main GEMM loop with tensor cores
    for (int k = 0; k < K_dim; k += 16) {
        if (warp_row < size<0>(A) && k < K_dim) {
            // Load A fragment (normalized data)
            half* a_ptr = const_cast<half*>(&A(warp_row, k));
            wmma::load_matrix_sync(frag_A, a_ptr, size<1>(A));
            
            // Load B fragment (weights)
            half* b_ptr = const_cast<half*>(&B(k, warp_col));
            wmma::load_matrix_sync(frag_B, b_ptr, size<1>(B));
            
            // Async MMA computation
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
    }
    
    // Write results back using tensor cores
    if (warp_row < size<0>(C)) {
        half* c_ptr = &C(warp_row, warp_col);
        wmma::store_matrix_sync(c_ptr, frag_C, size<1>(C), wmma::mem_row_major);
    }
}

// High-level launcher with optimal configuration
void launch_fused_layernorm_gemm(
    const half* input,      // [9600, 2730]
    const half* weights,    // [2730, 1024] 
    half* output,          // [9600, 1024]
    const half* gamma,      // [2730]
    const half* beta,       // [2730]
    int M = 9600, int K = 2730, int N = 1024,
    cudaStream_t stream = 0
) {
    using Config = LayerNormGemmConfig<256, 128>;
    
    // Grid configuration for optimal occupancy
    dim3 grid(ceildiv(N, 128), ceildiv(M, Config::kTileM));
    dim3 block(Config::kThreads);
    
    // Shared memory calculation
    size_t shmem_size = (Config::kTileM * K +     // Normalized data
                        128 * K +                  // Weight tile
                        Config::kTileM * 2) *      // Mean + variance
                       sizeof(half);
    
    // Enable dynamic shared memory if needed
    if (shmem_size > 48 * 1024) {
        cudaFuncSetAttribute(fused_layernorm_gemm_kernel<Config>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shmem_size);
    }
    
    fused_layernorm_gemm_kernel<Config><<<grid, block, shmem_size, stream>>>(
        input, weights, output, gamma, beta, M, K, N
    );
}

// Alternative approach: Streaming pipeline with multiple kernels
template<typename Config>
__global__ void streaming_layernorm_kernel(
    const half* input,          // [M, K] 
    half* normalized_output,    // [M, K]
    const half* gamma,          // [K]
    const half* beta,           // [K]
    int M, int K,
    int tile_start, int tile_size
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x + tile_start;
    if (row >= min(tile_start + tile_size, M)) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    
    // Per-row LayerNorm computation
    const half* input_row = input + row * K;
    half* output_row = normalized_output + row * K;
    
    // Compute mean
    float sum = 0.0f;
    for (int k = lane_id; k < K; k += 32) {
        sum += __half2float(input_row[k]);
    }
    
    // Warp reduction
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

// Streaming approach with overlapped execution
void launch_streaming_layernorm_gemm(
    const half* input, const half* weights, half* output,
    const half* gamma, const half* beta,
    int M = 9600, int K = 2730, int N = 1024
) {
    // Create multiple streams for pipelining
    const int num_streams = 4;
    const int tile_size = M / num_streams;
    
    cudaStream_t streams[num_streams];
    half* d_normalized[num_streams];
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_normalized[i], tile_size * K * sizeof(half));
    }
    
    // Use cuBLAS for GEMM operations in streaming approach
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    const half alpha_val = __float2half(1.0f);
    const half beta_val = __float2half(0.0f);
    
    // Pipeline execution
    for (int i = 0; i < num_streams; i++) {
        int tile_start = i * tile_size;
        int current_tile_size = min(tile_size, M - tile_start);
        
        // Stream i: LayerNorm
        dim3 ln_grid(ceildiv(current_tile_size, 256));
        dim3 ln_block(256);
        
        streaming_layernorm_kernel<LayerNormGemmConfig<>><<<ln_grid, ln_block, 0, streams[i]>>>(
            input, d_normalized[i], gamma, beta, M, K, tile_start, current_tile_size
        );
        
        // Stream i: GEMM (depends on LayerNorm completion)
        cublasSetStream(cublas_handle, streams[i]);
        
        cublasHgemm(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   N, current_tile_size, K,
                   &alpha_val,
                   weights, N,
                   d_normalized[i], K, 
                   &beta_val,
                   output + tile_start * N, N);
    }
    
    // Synchronize and cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(d_normalized[i]);
    }
    
    cublasDestroy(cublas_handle);
}

// Utility function to determine best approach based on hardware
bool should_use_fused_kernel(int M, int K, int N) {
    // Use fused kernel for smaller tiles that fit in shared memory
    // Use streaming for larger workloads that benefit from parallelism
    size_t required_shmem = (256 * K + 128 * K + 256 * 2) * sizeof(half);
    return required_shmem <= 64 * 1024; // 64KB shared memory limit
}

// Main API function
void layernorm_gemm_fused(
    const half* input, const half* weights, half* output,
    const half* gamma, const half* beta,
    int M = 9600, int K = 2730, int N = 1024,
    cudaStream_t stream = 0
) {
    if (should_use_fused_kernel(M, K, N)) {
        launch_fused_layernorm_gemm(input, weights, output, gamma, beta, M, K, N, stream);
    } else {
        launch_streaming_layernorm_gemm(input, weights, output, gamma, beta, M, K, N);
    }
}