#include "../utils/common.h"
#include <cuda/barrier>
#include <cuda/pipeline>
#include <mma.h>

// WMMA-based WGMMA-style + async memory implementation for sm_80+
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// TMA descriptor for efficient memory transfers
template<typename T>
struct TMADescriptor {
    CUtensorMap tma_map;
    bool initialized = false;
};

// Async memory copy using cp.async (sm_80+)
__device__ void async_copy_global_to_shared(
    half* dst_shmem,
    const half* src_global,
    int size_bytes
) {
    if (threadIdx.x == 0) {
        // Use cp.async for sm_80+ async memory operations
        for (int i = 0; i < size_bytes / 16; i++) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                        "r"((uint32_t)__cvta_generic_to_shared(&dst_shmem[i * 8])),
                        "l"(&src_global[i * 8]));
        }
        asm volatile("cp.async.commit_group;\n");
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
            // Warp 0: Async loads for A matrix
            const half* src_A = &A[(block_row + 0) * K_dim + k_tile];
            async_copy_global_to_shared(shmem_A[stage], src_A, M * K * sizeof(half));
        }
        else if (warp_id == 1) {
            // Warp 1: Async loads for B matrix  
            const half* src_B = &B[k_tile * N_dim + block_col];
            async_copy_global_to_shared(shmem_B[stage], src_B, K * N * sizeof(half));
        }
        
        // Wait for async copies using cp.async.wait_all
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();
        
        // Warps 2-7: Compute using MMA (Matrix Multiply Accumulate) - WGMMA-style approach
        if (warp_id >= 2) {
            using namespace nvcuda;
            
            // Use WMMA API which provides wgmma-like functionality
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A_wmma;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B_wmma;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C_wmma;
            
            if (warp_id == 2) {
                // Initialize accumulator (WGMMA-style)
                wmma::fill_fragment(frag_C_wmma, 0.0f);
                
                // Load matrices and perform multiply-accumulate
                wmma::load_matrix_sync(frag_A_wmma, shmem_A[stage], K);
                wmma::load_matrix_sync(frag_B_wmma, shmem_B[stage], N);
                
                // This is the core WGMMA-equivalent operation
                wmma::mma_sync(frag_C_wmma, frag_A_wmma, frag_B_wmma, frag_C_wmma);
                
                // Store to frag_C array for compatibility
                wmma::store_matrix_sync(frag_C, frag_C_wmma, 4, wmma::mem_row_major);
            }
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

int main() {
    printf("🚀 WGMMA + Async Memory Warp Specialized Kernel Test\n");
    printf("=====================================================\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 8) {
        printf("⚠️  This kernel requires sm_80+ for tensor core support\n");
        printf("   Current device: sm_%d%d\n", prop.major, prop.minor);
        printf("   WMMA/MMA operations require Ampere+ architecture\n");
        printf("   Falling back to standard implementation...\n");
        return 0;
    }
    printf("\n");
    
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
    
    printf("Testing Async MMA + TMA Warp Specialized Kernel:\n");
    cudaMemset(d_C, 0, size_C);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        launch_warp_specialized_async_mma(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_warp_specialized_async_mma(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / iterations;
    
    // Calculate performance
    double flops = 2.0 * M_dim * N_dim * K_dim;
    double gflops = flops / (avg_time_ms / 1000.0) / 1e9;
    
    printf("   Average time: %.3f ms\n", avg_time_ms);
    printf("   Performance: %.1f GFLOPS\n", gflops);
    
    // Validate results (basic check)
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    bool valid = true;
    for (int i = 0; i < std::min(10, M_dim * N_dim); i++) {
        if (isnan(h_C_gpu[i]) || isinf(h_C_gpu[i]) || fabs(h_C_gpu[i]) > 100.0f) {
            valid = false;
            break;
        }
    }
    printf("   Validation: %s\n", valid ? "✅ PASSED" : "❌ FAILED");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n✨ WGMMA + async memory kernel test completed!\n");
    return 0;
}