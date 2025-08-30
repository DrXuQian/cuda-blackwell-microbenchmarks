#include "common.h"
#include <cuda/barrier>
#include <cuda/pipeline>

// Modern async MMA + TMA implementation for sm_89+
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// TMA descriptor for efficient memory transfers
template<typename T>
struct TMADescriptor {
    CUtensorMap tma_map;
    bool initialized = false;
};

// Async memory copy using TMA (Tensor Memory Accelerator)
__device__ void async_copy_tma(
    void* dst_shmem,
    TMADescriptor<half>* tma_desc,
    int tile_coord_m,
    int tile_coord_k,
    barrier* sync_barrier
) {
    if (threadIdx.x == 0) {
        // Issue async TMA load - only thread 0 in block
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            dst_shmem, &tma_desc->tma_map, tile_coord_m, tile_coord_k, *sync_barrier);
    }
}

__device__ void copy_gmem_to_shmem(half* shmem_ptr, const half* gmem_ptr, 
                                   int rows, int cols, int src_ld) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // All threads participate in copy
    const int total_elements = rows * cols;
    for (int i = tid; i < total_elements; i += block_size) {
        int row = i / cols;
        int col = i % cols;
        shmem_ptr[i] = gmem_ptr[row * src_ld + col];
    }
}

// Modern async MMA kernel with TMA and warp specialization
__global__ void warp_specialized_async_mma_kernel(
    const half* A, const half* B, float* C,
    TMADescriptor<half>* tma_desc_A,
    TMADescriptor<half>* tma_desc_B, 
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    // Shared memory layout with double buffering
    half* shmem_A[2] = {
        reinterpret_cast<half*>(shmem),
        reinterpret_cast<half*>(shmem) + M * K
    };
    half* shmem_B[2] = {
        shmem_A[1] + M * K,
        shmem_A[1] + M * K + K * N
    };
    
    // Async barriers for pipeline stages
    __shared__ barrier barrier_A[2];
    __shared__ barrier barrier_B[2]; 
    
    if (threadIdx.x == 0) {
        init(&barrier_A[0], blockDim.x);
        init(&barrier_A[1], blockDim.x);
        init(&barrier_B[0], blockDim.x);
        init(&barrier_B[1], blockDim.x);
    }
    __syncthreads();
    
    // MMA fragments using async API
    uint32_t frag_A[4]; // 16x16x16 fp16 A fragment
    uint32_t frag_B[2]; // 16x16x16 fp16 B fragment  
    float frag_C[4] = {0.0f}; // 16x16 fp32 accumulator
    
    const int block_row = blockIdx.y * M;
    const int block_col = blockIdx.x * N;
    
    // Main computation loop with async pipeline
    for (int k_tile = 0; k_tile < K_dim; k_tile += K) {
        int stage = k_tile / K % 2; // Alternate between buffers
        
        // Warp specialization: different warps handle different tasks
        if (warp_id == 0) {
            // Warp 0: Issue TMA loads for A matrix
            async_copy_tma(shmem_A[stage], tma_desc_A, 
                          block_row / M, k_tile / K, &barrier_A[stage]);
        }
        else if (warp_id == 1) {
            // Warp 1: Issue TMA loads for B matrix  
            async_copy_tma(shmem_B[stage], tma_desc_B,
                          k_tile / K, block_col / N, &barrier_B[stage]);
        }
        
        // All warps wait for data to arrive
        barrier_A[stage].arrive_and_wait();
        barrier_B[stage].arrive_and_wait();
        
        // Warps 2-7: Compute using async MMA
        if (warp_id >= 2) {
            // Load fragments from shared memory
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(frag_A[0]), "=r"(frag_A[1]), "=r"(frag_A[2]), "=r"(frag_A[3])
                : "r"(__cvta_generic_to_shared(&shmem_A[stage][warp_id * 16]))
            );
            
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(frag_B[0]), "=r"(frag_B[1])
                : "r"(__cvta_generic_to_shared(&shmem_B[stage][warp_id * 16]))
            );
            
            // Async MMA operation
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(frag_C[0]), "+f"(frag_C[1]), "+f"(frag_C[2]), "+f"(frag_C[3])
                : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
                  "r"(frag_B[0]), "r"(frag_B[1])
            );
        }
    }
    
    // Write results back to global memory
    if (warp_id >= 2) {
        int out_row = block_row + (warp_id - 2) * 16 + lane_id / 4;
        int out_col = block_col + (lane_id % 4) * 4;
        
        if (out_row < M_dim && out_col < N_dim) {
            // Store accumulated results
            for (int i = 0; i < 4; i++) {
                if (out_col + i < N_dim) {
                    C[out_row * N_dim + out_col + i] = frag_C[i];
                }
            }
        }
    }
}

// TMA descriptor initialization function
void init_tma_descriptors(
    TMADescriptor<half>* tma_desc_A,
    TMADescriptor<half>* tma_desc_B,
    const half* A, const half* B,
    int M_dim, int N_dim, int K_dim
) {
    // Initialize TMA descriptor for A matrix
    void* global_A = const_cast<half*>(A);
    uint64_t gmem_shape_A[2] = {static_cast<uint64_t>(M_dim), static_cast<uint64_t>(K_dim)};
    uint64_t gmem_stride_A[1] = {static_cast<uint64_t>(K_dim * sizeof(half))};
    uint32_t smem_shape_A[2] = {M, K}; // Tile dimensions
    uint32_t smem_stride_A[1] = {K * sizeof(half)};
    
    CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    cuTensorMapEncodeTiled(
        &tma_desc_A->tma_map, data_type, 2,
        global_A, gmem_shape_A, gmem_stride_A,
        smem_shape_A, smem_stride_A,
        interleave, swizzle, l2promo, oob_fill
    );
    
    // Initialize TMA descriptor for B matrix
    void* global_B = const_cast<half*>(B);
    uint64_t gmem_shape_B[2] = {static_cast<uint64_t>(K_dim), static_cast<uint64_t>(N_dim)};
    uint64_t gmem_stride_B[1] = {static_cast<uint64_t>(N_dim * sizeof(half))};
    uint32_t smem_shape_B[2] = {K, N};
    uint32_t smem_stride_B[1] = {N * sizeof(half)};
    
    cuTensorMapEncodeTiled(
        &tma_desc_B->tma_map, data_type, 2,
        global_B, gmem_shape_B, gmem_stride_B,
        smem_shape_B, smem_stride_B,
        interleave, swizzle, l2promo, oob_fill
    );
    
    tma_desc_A->initialized = true;
    tma_desc_B->initialized = true;
}

// Host launcher function for async MMA kernel
void launch_warp_specialized_async_mma(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    // Initialize TMA descriptors on device
    TMADescriptor<half> *d_tma_desc_A, *d_tma_desc_B;
    cudaMalloc(&d_tma_desc_A, sizeof(TMADescriptor<half>));
    cudaMalloc(&d_tma_desc_B, sizeof(TMADescriptor<half>));
    
    init_tma_descriptors(d_tma_desc_A, d_tma_desc_B, A, B, M_dim, N_dim, K_dim);
    
    // Launch configuration
    dim3 grid((N_dim + N - 1) / N, (M_dim + M - 1) / M);
    dim3 block(8 * 32); // 8 warps × 32 threads
    
    size_t shmem_size = 2 * (M * K + K * N) * sizeof(half); // Double buffering
    
    warp_specialized_async_mma_kernel<<<grid, block, shmem_size>>>(
        A, B, C, d_tma_desc_A, d_tma_desc_B, M_dim, N_dim, K_dim
    );
    
    cudaFree(d_tma_desc_A);
    cudaFree(d_tma_desc_B);
}
        } else if (warp_id == 1) {
            // Warp 1: Perform computation on current data
            wmma::load_matrix_sync(frag_A, shmem_A_buffers[current_buffer], K);
            wmma::load_matrix_sync(frag_B, shmem_B_buffers[current_buffer], N);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
        buffer_idx = next_buffer;
    }
    
    if (warp_id == 1) {
        wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
    }
}

// Improved copy function with better memory coalescing
__device__ void copy_gmem_to_shmem_coalesced(half* shmem_ptr, const half* gmem_ptr, 
                                            int rows, int cols, int src_ld) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Ensure coalesced access patterns
    const int total_elements = rows * cols;
    
    // Each thread handles multiple contiguous elements to improve coalescing
    const int elements_per_thread = 4; // Process 4 elements per thread for better bandwidth
    
    for (int base = tid * elements_per_thread; base < total_elements; base += block_size * elements_per_thread) {
        for (int offset = 0; offset < elements_per_thread && base + offset < total_elements; offset++) {
            int idx = base + offset;
            int row = idx / cols;
            int col = idx % cols;
            shmem_ptr[idx] = gmem_ptr[row * src_ld + col];
        }
    }
}

__device__ void copy_gmem_to_shmem_warp_coalesced(half* shmem_ptr, const half* gmem_ptr, 
                                                 int rows, int cols, int src_ld, int warp_id) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Only specified warp performs the copy
    if (threadIdx.x >= warp_id * WARP_SIZE && threadIdx.x < (warp_id + 1) * WARP_SIZE) {
        const int total_elements = rows * cols;
        
        // Improved coalescing: each lane handles contiguous chunks
        const int elements_per_lane = 2; // Each lane processes 2 consecutive elements
        
        for (int base = lane_id * elements_per_lane; base < total_elements; base += WARP_SIZE * elements_per_lane) {
            for (int offset = 0; offset < elements_per_lane && base + offset < total_elements; offset++) {
                int idx = base + offset;
                int row = idx / cols;
                int col = idx % cols;
                shmem_ptr[idx] = gmem_ptr[row * src_ld + col];
            }
        }
    }
}

__global__ void warp_specialized_mma_kernel_final_opt(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + 2 * M * K;  // Account for double buffering
    
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    int block_row = blockIdx.y * M;
    int block_col = blockIdx.x * N;
    
    // Create double buffer for true async operation
    half* shmem_A_buffers[2] = {shmem_A, shmem_A + M * K};
    half* shmem_B_buffers[2] = {shmem_B, shmem_B + K * N};
    int buffer_idx = 0;
    
    // Pre-load first chunk with coalesced access
    copy_gmem_to_shmem_coalesced(shmem_A_buffers[0], &A[block_row * K_dim], M, K, K_dim);
    copy_gmem_to_shmem_coalesced(shmem_B_buffers[0], &B[block_col], K, N, N_dim);
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current_buffer = buffer_idx;
        int next_buffer = 1 - buffer_idx;
        
        // Optimized warp specialization with better memory coalescing
        if (warp_id == 0 && k + K < K_dim) {
            // Warp 0: Load next chunk with improved coalescing
            copy_gmem_to_shmem_warp_coalesced(shmem_A_buffers[next_buffer], 
                                             &A[block_row * K_dim + k + K], M, K, K_dim, 0);
            copy_gmem_to_shmem_warp_coalesced(shmem_B_buffers[next_buffer], 
                                             &B[(k + K) * N_dim + block_col], K, N, N_dim, 0);
        } else if (warp_id == 1) {
            // Warp 1: Perform computation on current data
            wmma::load_matrix_sync(frag_A, shmem_A_buffers[current_buffer], K);
            wmma::load_matrix_sync(frag_B, shmem_B_buffers[current_buffer], N);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
        buffer_idx = next_buffer;
    }
    
    if (warp_id == 1) {
        wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
    }
}

int main() {
    printf("Final Optimized Warp Specialized MMA Kernel Performance Test\n");
    printf("============================================================\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute Capability %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    const int M_dim = 1024, N_dim = 1024, K_dim = 1024;
    
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    
    if (!h_A || !h_B || !h_C_gpu) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize with small random values
    srand(42);
    for (int i = 0; i < M_dim * K_dim; i++) {
        h_A[i] = __float2half((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    for (int i = 0; i < K_dim * N_dim; i++) {
        h_B[i] = __float2half((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    printf("Testing Final Optimized Warp Specialized MMA Kernel:\n");
    size_t shmem_size = 4 * (M * K + K * N) * sizeof(half); // Double buffer
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel_final_opt, "Final Optimized Warp Specialized", 
                     d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result;
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim, &acc_result)) {
        printf("\n");
        print_accuracy_result(acc_result, "Final Optimized Warp Specialized");
    } else {
        printf("\n");
        print_accuracy_result(acc_result, "Final Optimized Warp Specialized");
    }
    
    printf("\ncuBLAS Reference Performance:\n");
    benchmark_cublas(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    // Compare with original
    printf("\nComparison with Original Warp Specialized Kernel:\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel, "Original Warp Specialized", 
                     d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 Final optimized kernel test completed!\n");
    return 0;
}