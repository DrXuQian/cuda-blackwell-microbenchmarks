#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

// Helper function definition for ceildiv if not available
__host__ __device__ inline int ceildiv_naive(int a, int b) {
    return (a + b - 1) / b;
}

// Simple naive GEMM implementation for baseline comparison
// C = A @ B where A is [M, K], B is [K, N], C is [M, N]

using namespace nvcuda;

// Thread-block tiled GEMM using tensor cores
template<int TILE_M = 128, int TILE_N = 128, int TILE_K = 32>
__global__ void naive_tensor_core_gemm_kernel(
    const half* A,      // [M, K] 
    const half* B,      // [K, N]
    half* C,           // [M, N]
    int M, int K, int N
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Shared memory for tiles
    __shared__ half sA[TILE_M * TILE_K];
    __shared__ half sB[TILE_K * TILE_N];
    
    // Warp and thread mapping for 16x16 tensor core tiles
    const int warps_per_block = blockDim.x / 32;
    const int warp_row = (warp_id / (TILE_N / 16)) * 16;
    const int warp_col = (warp_id % (TILE_N / 16)) * 16;
    
    // Tensor core fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C;
    
    // Initialize accumulator
    wmma::fill_fragment(frag_C, __float2half(0.0f));
    
    // Main computation loop
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative loading of A tile
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = by * TILE_M + row;
            int global_col = k_tile + col;
            
            if (global_row < M && global_col < K) {
                sA[row * TILE_K + col] = A[global_row * K + global_col];
            } else {
                sA[row * TILE_K + col] = __float2half(0.0f);
            }
        }
        
        // Cooperative loading of B tile
        for (int i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int global_row = k_tile + row;
            int global_col = bx * TILE_N + col;
            
            if (global_row < K && global_col < N) {
                sB[row * TILE_N + col] = B[global_row * N + global_col];
            } else {
                sB[row * TILE_N + col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Tensor core computation
        if (warp_row + 16 <= TILE_M && warp_col + 16 <= TILE_N) {
            // Load fragments from shared memory
            wmma::load_matrix_sync(frag_A, &sA[warp_row * TILE_K], TILE_K);
            wmma::load_matrix_sync(frag_B, &sB[warp_col], TILE_N);
            
            // Multiply-accumulate
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
    }
    
    // Store results
    int global_warp_row = by * TILE_M + warp_row;
    int global_warp_col = bx * TILE_N + warp_col;
    
    if (global_warp_row + 16 <= M && global_warp_col + 16 <= N) {
        wmma::store_matrix_sync(&C[global_warp_row * N + global_warp_col], frag_C, N, wmma::mem_row_major);
    }
}

// Standard cuBLAS wrapper
void launch_cublas_gemm(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K,
                 &beta,
                 C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cublasDestroy(handle);
}

// Custom tensor core GEMM launcher
void launch_naive_tensor_core_gemm(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    dim3 grid(ceildiv_naive(N, TILE_N), ceildiv_naive(M, TILE_M));
    dim3 block(256);  // 8 warps per block
    
    size_t shmem_size = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(half);
    
    naive_tensor_core_gemm_kernel<TILE_M, TILE_N, TILE_K><<<grid, block, shmem_size, stream>>>(
        A, B, C, M, K, N
    );
}

// Simple CPU-style GEMM (very slow, for correctness testing only)
__global__ void naive_scalar_gemm_kernel(
    const half* A, const half* B, half* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

void launch_naive_scalar_gemm(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    cudaStream_t stream = 0
) {
    dim3 grid(ceildiv_naive(N, 16), ceildiv_naive(M, 16));
    dim3 block(16, 16);
    
    naive_scalar_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

// Utility function to select best naive implementation
void naive_gemm(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    bool use_tensor_cores = true,
    cudaStream_t stream = 0
) {
    if (use_tensor_cores) {
        launch_naive_tensor_core_gemm(A, B, C, M, K, N, stream);
    } else {
        launch_naive_scalar_gemm(A, B, C, M, K, N, stream);
    }
}