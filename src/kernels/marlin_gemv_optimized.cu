#include "common.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Our own utility functions (avoiding conflicts with Marlin)
constexpr int gemv_ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T, int n>
struct GemvVec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

using GemvI4 = GemvVec<int, 4>;
using GemvFragB = GemvVec<half2, 2>;

// LOP3 operation for efficient bit manipulation
template <int lut>
__device__ inline int gemv_lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Efficient 4-bit dequantization from Marlin
__device__ inline GemvFragB gemv_dequant(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    
    int lo = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    
    const int SUB = 0x64086408;
    const int MUL = 0x2c002c00;
    const int ADD = 0xd480d480;
    
    GemvFragB frag_b;
    frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
    frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), 
                       *reinterpret_cast<const half2*>(&ADD));
    return frag_b;
}

// Async copy operations (avoiding Marlin conflicts)
__device__ inline void gemv_cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ inline void gemv_cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ inline void gemv_cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void gemv_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Optimized GEMV kernel for 1xN @ NxM with w4a16f
template<int BLOCK_SIZE, int STAGES>
__global__ void marlin_gemv_w4a16f(
    const half* __restrict__ A,       // 1 x N input vector (fp16)
    const int4* __restrict__ B,       // N x M quantized weight matrix (4-bit packed)
    const half* __restrict__ scales,  // Quantization scales
    half* __restrict__ C,             // 1 x M output vector (fp16)
    int N,                           // Input dimension
    int M                            // Output dimension
) {
    extern __shared__ int4 shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Each block handles BLOCK_SIZE output elements
    const int block_start_m = blockIdx.x * BLOCK_SIZE;
    const int thread_m = block_start_m + threadIdx.x;
    
    // Shared memory layout: [STAGES][A_chunk][B_chunk]
    const int A_chunk_size = 128; // Process 128 elements of A at a time
    const int A_shmem_size = A_chunk_size;
    const int B_shmem_size = A_chunk_size * BLOCK_SIZE / 8; // 4-bit packed
    
    int4* shmem_A = shmem;
    int4* shmem_B = shmem_A + STAGES * (A_shmem_size / 4);
    
    // Output accumulation
    float accum = 0.0f;
    
    // Pipeline stages
    int stage = 0;
    int n_chunks = gemv_ceildiv(N, A_chunk_size);
    
    // Warp specialization: 
    // Warp 0: Load A chunks
    // Warp 1: Load B chunks  
    // Warps 2+: Compute
    
    // Pre-fill pipeline
    for (int prefill = 0; prefill < min(STAGES - 1, n_chunks); prefill++) {
        int chunk_start = prefill * A_chunk_size;
        int chunk_size = min(A_chunk_size, N - chunk_start);
        
        if (warp_id == 0) {
            // Load A chunk - all threads in warp 0 participate
            for (int i = lane_id; i < chunk_size; i += WARP_SIZE) {
                if (chunk_start + i < N) {
                    reinterpret_cast<half*>(&shmem_A[prefill * A_shmem_size / 4])[i] = A[chunk_start + i];
                }
            }
        }
        
        if (warp_id == 1 && thread_m < M) {
            // Load corresponding B column chunk - vectorized 4-bit loads
            for (int i = 0; i < chunk_size; i += 8) { // 8 4-bit values per int32
                if (chunk_start + i < N) {
                    int b_idx = ((chunk_start + i) * M + thread_m) / 8;
                    reinterpret_cast<int*>(&shmem_B[prefill * B_shmem_size / 4])[threadIdx.x * chunk_size / 8 + i / 8] = 
                        reinterpret_cast<const int*>(B)[b_idx];
                }
            }
        }
        
        gemv_cp_async_fence();
    }
    
    // Main computation loop
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int current_stage = chunk % STAGES;
        int next_stage = (chunk + STAGES - 1) % STAGES;
        
        // Wait for current stage to be ready
        gemv_cp_async_wait<STAGES - 2>();
        __syncthreads();
        
        int chunk_start = chunk * A_chunk_size;
        int chunk_size = min(A_chunk_size, N - chunk_start);
        
        // Compute on current data (warps 2+)
        if (warp_id >= 2 && thread_m < M) {
            half* a_data = reinterpret_cast<half*>(&shmem_A[current_stage * A_shmem_size / 4]);
            int* b_data = reinterpret_cast<int*>(&shmem_B[current_stage * B_shmem_size / 4]);
            
            // Vectorized computation with 4-bit dequantization
            for (int i = 0; i < chunk_size; i += 8) {
                if (chunk_start + i < N) {
                    // Load 4 fp16 values from A
                    half2 a_vals[2];
                    a_vals[0] = *reinterpret_cast<half2*>(&a_data[i]);
                    a_vals[1] = *reinterpret_cast<half2*>(&a_data[i + 2]);
                    
                    // Load and dequantize 8 4-bit weights
                    int b_packed = b_data[threadIdx.x * chunk_size / 8 + i / 8];
                    GemvFragB b_deq0 = gemv_dequant(b_packed & 0xFFFF);
                    GemvFragB b_deq1 = gemv_dequant((b_packed >> 16) & 0xFFFF);
                    
                    // Compute dot products
                    accum += __half2float(__hmul(a_vals[0].x, b_deq0[0].x));
                    accum += __half2float(__hmul(a_vals[0].y, b_deq0[0].y));
                    accum += __half2float(__hmul(a_vals[1].x, b_deq0[1].x));
                    accum += __half2float(__hmul(a_vals[1].y, b_deq0[1].y));
                    
                    if (i + 4 < chunk_size) {
                        half2 a_vals2[2];
                        a_vals2[0] = *reinterpret_cast<half2*>(&a_data[i + 4]);
                        a_vals2[1] = *reinterpret_cast<half2*>(&a_data[i + 6]);
                        
                        accum += __half2float(__hmul(a_vals2[0].x, b_deq1[0].x));
                        accum += __half2float(__hmul(a_vals2[0].y, b_deq1[0].y));
                        accum += __half2float(__hmul(a_vals2[1].x, b_deq1[1].x));
                        accum += __half2float(__hmul(a_vals2[1].y, b_deq1[1].y));
                    }
                }
            }
        }
        
        // Load next chunks asynchronously (overlapped with compute)
        int next_chunk = chunk + STAGES - 1;
        if (next_chunk < n_chunks) {
            int next_chunk_start = next_chunk * A_chunk_size;
            int next_chunk_size = min(A_chunk_size, N - next_chunk_start);
            
            if (warp_id == 0) {
                for (int i = lane_id; i < next_chunk_size; i += WARP_SIZE) {
                    if (next_chunk_start + i < N) {
                        gemv_cp_async4_pred(
                            &reinterpret_cast<half*>(&shmem_A[next_stage * A_shmem_size / 4])[i],
                            &A[next_chunk_start + i]
                        );
                    }
                }
            }
            
            if (warp_id == 1 && thread_m < M) {
                for (int i = 0; i < next_chunk_size; i += 8) {
                    if (next_chunk_start + i < N) {
                        int b_idx = ((next_chunk_start + i) * M + thread_m) / 8;
                        gemv_cp_async4_stream(
                            &reinterpret_cast<int*>(&shmem_B[next_stage * B_shmem_size / 4])[threadIdx.x * next_chunk_size / 8 + i / 8],
                            &reinterpret_cast<const int*>(B)[b_idx]
                        );
                    }
                }
            }
            gemv_cp_async_fence();
        }
        
        __syncthreads();
    }
    
    // Final reduction across compute warps
    __shared__ float reduction_buffer[BLOCK_SIZE];
    
    if (warp_id >= 2) {
        reduction_buffer[threadIdx.x] = accum;
    } else {
        reduction_buffer[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Warp 0 does the final reduction
    if (warp_id == 0 && thread_m < M) {
        float final_accum = 0.0f;
        for (int i = lane_id; i < BLOCK_SIZE; i += WARP_SIZE) {
            final_accum += reduction_buffer[i];
        }
        
        // Warp-level reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            final_accum += __shfl_down_sync(0xFFFFFFFF, final_accum, offset);
        }
        
        // Apply scale and write result
        if (lane_id == 0) {
            half scale = scales[thread_m];
            C[thread_m] = __float2half(final_accum * __half2float(scale));
        }
    }
}

// Kernel launch wrapper
void launch_marlin_gemv_w4a16f(
    const half* A, const int4* B, const half* scales, half* C,
    int N, int M, cudaStream_t stream = 0
) {
    const int BLOCK_SIZE = 256;
    const int STAGES = 3;
    
    dim3 grid(gemv_ceildiv(M, BLOCK_SIZE));
    dim3 block(BLOCK_SIZE);
    
    // Calculate shared memory requirement
    const int A_chunk_size = 128;
    const int A_shmem_size = A_chunk_size;
    const int B_shmem_size = A_chunk_size * BLOCK_SIZE / 8;
    size_t shmem_size = STAGES * (A_shmem_size + B_shmem_size) * sizeof(int4) + BLOCK_SIZE * sizeof(float);
    
    marlin_gemv_w4a16f<BLOCK_SIZE, STAGES><<<grid, block, shmem_size, stream>>>(
        A, B, scales, C, N, M
    );
}

// Alternative version with tensor core utilization for larger batches
template<int BLOCK_SIZE>
__global__ void marlin_gemv_tensor_core_w4a16f(
    const half* __restrict__ A,
    const int4* __restrict__ B,
    const half* __restrict__ scales,
    half* __restrict__ C,
    int N, int M, int batch_size = 1
) {
    // When batch_size >= 16, we can utilize tensor cores efficiently
    // by treating multiple input vectors as a small matrix
    
    extern __shared__ int4 shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Use WMMA for 16x16x16 tiles when possible
    if (batch_size >= 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;
        
        wmma::fill_fragment(frag_C, 0.0f);
        
        int block_m = blockIdx.y * 16;
        int block_n = blockIdx.x * 16;
        
        // Process in 16x16x16 chunks
        for (int k = 0; k < N; k += 16) {
            // Load A fragment (input vectors)
            wmma::load_matrix_sync(frag_A, &A[block_m * N + k], N);
            
            // Load and dequantize B fragment (weights)
            // This would require careful 4-bit unpacking into shared memory first
            // ... (implementation details for tensor core path)
            
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        // Convert float accumulator to half before storing
        half temp_result[256];
        wmma::store_matrix_sync(temp_result, frag_C, 16, wmma::mem_row_major);
        
        // Copy to final output
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                if (block_m + i < batch_size && block_n + j < M) {
                    C[(block_m + i) * M + block_n + j] = __float2half(temp_result[i * 16 + j]);
                }
            }
        }
    } else {
        // Fall back to regular GEMV for small batches
        // ... (same implementation as above)
    }
}

// Test harness
__global__ void init_test_data(half* A, int4* B, half* scales, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize A vector
    if (idx < N) {
        A[idx] = __float2half(0.1f * (idx % 100));
    }
    
    // Initialize quantized B matrix (simplified - normally would be properly quantized)
    if (idx < (N * M / 8)) {
        B[idx] = make_int4(0x12345678, 0x9ABCDEF0, 0x11223344, 0x55667788);
    }
    
    // Initialize scales
    if (idx < M) {
        scales[idx] = __float2half(0.01f);
    }
}