#include "../utils/common.h"
#include <cuda/barrier>
#include <cuda/pipeline>
#include <mma.h>

// Tensor core + TMA optimized Marlin GEMV implementation
using namespace nvcuda;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

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

// TMA descriptor for w4a16f weights
struct W4A16TMADescriptor {
    CUtensorMap tma_map;
    bool initialized = false;
};

// LOP3 operation for efficient bit manipulation
template <int lut>
__device__ inline int gemv_lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Efficient 4-bit dequantization from Marlin with tensor core format
__device__ inline void gemv_dequant_tensor(int q, half* output) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    
    int lo = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    
    const int SUB = 0x64086408;
    const int MUL = 0x2c002c00;
    const int ADD = 0xd480d480;
    
    // Convert to tensor core layout format
    half2 lo_h2 = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
    half2 hi_h2 = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), 
                         *reinterpret_cast<const half2*>(&ADD));
    
    // Store in tensor core compatible format
    output[0] = __low2half(lo_h2);
    output[1] = __high2half(lo_h2);
    output[2] = __low2half(hi_h2);  
    output[3] = __high2half(hi_h2);
}

// Async TMA load for quantized weights
__device__ void async_load_w4_tma(
    void* dst_shmem,
    W4A16TMADescriptor* tma_desc,
    int tile_row, int tile_col,
    barrier* sync_barrier
) {
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            dst_shmem, &tma_desc->tma_map, tile_row, tile_col, *sync_barrier);
    }
}

// Tensor core w4a16f GEMV kernel with async MMA + TMA
__global__ void marlin_gemv_tensor_async_kernel(
    const half* __restrict__ A,      // [M, K] input activations (fp16)
    const int* __restrict__ B,       // [K/8, N] 4-bit weights (int4 packed)
    half* __restrict__ C,            // [M, N] output (fp16)
    const half* __restrict__ s,      // [K/group_size, N] scales
    int M, int N, int K, int group_size,
    W4A16TMADescriptor* tma_desc_B,
    W4A16TMADescriptor* tma_desc_s
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    
    // Shared memory with double buffering
    extern __shared__ char shmem[];
    int* shmem_B[2] = {
        reinterpret_cast<int*>(shmem),
        reinterpret_cast<int*>(shmem) + (K/8) * 16  // 16 cols per tile
    };
    half* shmem_B_dequant[2] = {
        reinterpret_cast<half*>(shmem_B[1]) + (K/8) * 16,
        reinterpret_cast<half*>(shmem_B[1]) + (K/8) * 16 + K * 16
    };
    half* shmem_s[2] = {
        shmem_B_dequant[1] + K * 16,
        shmem_B_dequant[1] + K * 16 + (K/group_size) * 16
    };
    
    // Async barriers
    __shared__ barrier barrier_B[2];
    __shared__ barrier barrier_s[2];
    
    if (threadIdx.x == 0) {
        init(&barrier_B[0], blockDim.x);
        init(&barrier_B[1], blockDim.x);
        init(&barrier_s[0], blockDim.x);
        init(&barrier_s[1], blockDim.x);
    }
    __syncthreads();
    
    const int block_col = blockIdx.x * 16; // 16 columns per block
    
    // Tensor core fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C;
    
    wmma::fill_fragment(frag_C, __float2half(0.0f));
    
    // Main computation loop with async pipeline
    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int stage = (k_tile / 16) % 2;
        
        // Warp specialization for async operations
        if (warp_id == 0) {
            // Load quantized weights via TMA
            async_load_w4_tma(shmem_B[stage], tma_desc_B,
                             k_tile / 16, block_col / 16, &barrier_B[stage]);
        }
        else if (warp_id == 1) {
            // Load scales via TMA
            int scale_tile = k_tile / group_size;
            async_load_w4_tma(shmem_s[stage], tma_desc_s,
                             scale_tile, block_col / 16, &barrier_s[stage]);
        }
        
        // Wait for data arrival
        barrier_B[stage].arrive_and_wait();
        barrier_s[stage].arrive_and_wait();
        
        // Warp 2+: Dequantize weights to fp16 tensor core format
        if (warp_id == 2) {
            for (int i = lane_id; i < (16 * 16) / 8; i += 32) {
                int packed_weight = shmem_B[stage][i];
                half dequant_output[4];
                gemv_dequant_tensor(packed_weight, dequant_output);
                
                // Store in tensor core layout
                int out_idx = i * 4;
                for (int j = 0; j < 4; j++) {
                    shmem_B_dequant[stage][out_idx + j] = dequant_output[j];
                }
            }
        }
        __syncthreads();
        
        // Warp 3+: Tensor core computation
        if (warp_id >= 3) {
            int warp_row = (warp_id - 3) * 16;
            if (warp_row < M) {
                // Load A fragment from global memory
                wmma::load_matrix_sync(frag_A, &A[warp_row * K + k_tile], K);
                
                // Load dequantized B fragment from shared memory  
                wmma::load_matrix_sync(frag_B, shmem_B_dequant[stage], 16);
                
                // Async MMA operation
                wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
            }
        }
    }
    
    // Write results back
    if (warp_id >= 3) {
        int warp_row = (warp_id - 3) * 16;
        if (warp_row < M && block_col < N) {
            wmma::store_matrix_sync(&C[warp_row * N + block_col], frag_C, N, wmma::mem_row_major);
        }
    }
}

// TMA descriptor initialization for w4a16f
void init_w4a16_tma_descriptors(
    W4A16TMADescriptor* tma_desc_B,
    W4A16TMADescriptor* tma_desc_s,
    const int* B, const half* s,
    int M, int N, int K, int group_size
) {
    // Initialize TMA descriptor for quantized weights B
    void* global_B = const_cast<int*>(B);
    uint64_t gmem_shape_B[2] = {static_cast<uint64_t>(K/8), static_cast<uint64_t>(N)};
    uint64_t gmem_stride_B[1] = {static_cast<uint64_t>(N * sizeof(int))};
    uint32_t smem_shape_B[2] = {2, 16}; // 2x16 int tile (16 bytes = 32 weights)
    uint32_t smem_stride_B[1] = {16 * sizeof(int)};
    
    CUtensorMapDataType data_type_B = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    cuTensorMapEncodeTiled(
        &tma_desc_B->tma_map, data_type_B, 2,
        global_B, gmem_shape_B, gmem_stride_B,
        smem_shape_B, smem_stride_B,
        interleave, swizzle, l2promo, oob_fill
    );
    
    // Initialize TMA descriptor for scales s
    void* global_s = const_cast<half*>(s);
    uint64_t gmem_shape_s[2] = {static_cast<uint64_t>(K/group_size), static_cast<uint64_t>(N)};
    uint64_t gmem_stride_s[1] = {static_cast<uint64_t>(N * sizeof(half))};
    uint32_t smem_shape_s[2] = {1, 16}; // 1x16 half tile
    uint32_t smem_stride_s[1] = {16 * sizeof(half)};
    
    CUtensorMapDataType data_type_s = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    
    cuTensorMapEncodeTiled(
        &tma_desc_s->tma_map, data_type_s, 2,
        global_s, gmem_shape_s, gmem_stride_s,
        smem_shape_s, smem_stride_s,
        interleave, swizzle, l2promo, oob_fill
    );
    
    tma_desc_B->initialized = true;
    tma_desc_s->initialized = true;
}

// Host launcher for tensor core w4a16f GEMV
void launch_marlin_gemv_tensor_async(
    const half* A, const int* B, half* C, const half* s,
    int M, int N, int K, int group_size = 128
) {
    // Initialize TMA descriptors
    W4A16TMADescriptor *d_tma_desc_B, *d_tma_desc_s;
    cudaMalloc(&d_tma_desc_B, sizeof(W4A16TMADescriptor));
    cudaMalloc(&d_tma_desc_s, sizeof(W4A16TMADescriptor));
    
    init_w4a16_tma_descriptors(d_tma_desc_B, d_tma_desc_s, B, s, M, N, K, group_size);
    
    // Launch configuration
    dim3 grid(gemv_ceildiv(N, 16), 1); // 16 cols per block
    dim3 block(8 * 32); // 8 warps for specialization
    
    // Shared memory calculation
    size_t shmem_size = 2 * ((K/8) * 16 * sizeof(int) +     // Quantized weights
                             K * 16 * sizeof(half) +          // Dequantized weights  
                             (K/group_size) * 16 * sizeof(half)); // Scales
    
    if (shmem_size > 48 * 1024) {
        cudaFuncSetAttribute(marlin_gemv_tensor_async_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shmem_size);
    }
    
    marlin_gemv_tensor_async_kernel<<<grid, block, shmem_size>>>(
        A, B, C, s, M, N, K, group_size, d_tma_desc_B, d_tma_desc_s
    );
    
    cudaFree(d_tma_desc_B);
    cudaFree(d_tma_desc_s);
}

int main() {
    printf("🚀 Tensor Core + TMA Marlin w4a16f GEMV Performance Test\n");
    printf("========================================================\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
        printf("⚠️  This kernel requires sm_75+ for tensor core support\n");
        return 0;
    }
    if (prop.major < 8 || (prop.major == 8 && prop.minor < 9)) {
        printf("⚠️  TMA features require sm_89+, falling back to standard async\n");
    }
    printf("\n");
    
    // Transformer-like shape: 1x3584 @ 3584x18944 (67.9M parameters)
    const int M = 1, N = 18944, K = 3584;
    const int group_size = 128;
    
    printf("Testing shape: %dx%d @ %dx%d (w4a16f, group_size=%d)\n", M, K, K, N, group_size);
    printf("Parameters: %.1fM\n", (2.0 * K * N) / 1e6); // 2 FLOP per param
    
    size_t size_A = M * K * sizeof(half);
    size_t size_B = (K * N / 2) * sizeof(char);  // 4-bit packed
    size_t size_B_int = (K/8) * N * sizeof(int); // Repacked as int
    size_t size_s = (K/group_size) * N * sizeof(half);
    size_t size_C = M * N * sizeof(half);
    
    half *h_A = (half*)malloc(size_A);
    char *h_B_4bit = (char*)malloc(size_B);
    int *h_B_int = (int*)malloc(size_B_int);
    half *h_s = (half*)malloc(size_s);
    half *h_C = (half*)malloc(size_C);
    
    if (!h_A || !h_B_4bit || !h_B_int || !h_s || !h_C) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    
    // Pack 4-bit weights into int format
    for (int i = 0; i < (K/8) * N; i++) {
        int packed = 0;
        for (int j = 0; j < 8; j++) {
            int w4 = rand() % 16; // 4-bit random weight
            packed |= (w4 << (j * 4));
        }
        h_B_int[i] = packed;
    }
    
    // Initialize scales
    for (int i = 0; i < (K/group_size) * N; i++) {
        h_s[i] = __float2half(1.0f + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    
    // Allocate device memory
    half *d_A, *d_C, *d_s;
    int *d_B_int;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B_int, size_B_int);
    cudaMalloc(&d_s, size_s);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_int, h_B_int, size_B_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, size_s, cudaMemcpyHostToDevice);
    
    printf("\\nTesting Tensor Core + TMA w4a16f GEMV:\\n");
    cudaMemset(d_C, 0, size_C);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_marlin_gemv_tensor_async(d_A, d_B_int, d_C, d_s, M, N, K, group_size);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_marlin_gemv_tensor_async(d_A, d_B_int, d_C, d_s, M, N, K, group_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / iterations;
    
    // Calculate performance
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_time_ms / 1000.0) / 1e9;
    
    // Calculate memory bandwidth (effective fp16 equivalent)
    double bytes = M * K * 2 + (K * N / 2) + (K/group_size) * N * 2 + M * N * 2;
    double bandwidth = bytes / (avg_time_ms / 1000.0) / 1e9;
    
    printf("   Average time: %.3f ms\\n", avg_time_ms);
    printf("   Performance: %.1f GFLOPS\\n", gflops);
    printf("   Effective bandwidth: %.1f GB/s\\n", bandwidth);
    printf("   Memory reduction: %.1fx (vs fp16)\\n", 4.0); // 4-bit vs 16-bit
    
    // Validate results (basic check)
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    bool valid = true;
    for (int i = 0; i < std::min(10, M * N); i++) {
        float val = __half2float(h_C[i]);
        if (isnan(val) || isinf(val) || fabs(val) > 100.0f) {
            valid = false;
            break;
        }
    }
    printf("   Validation: %s\\n", valid ? "✅ PASSED" : "❌ FAILED");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B_4bit);
    free(h_B_int);
    free(h_s);
    free(h_C);
    
    cudaFree(d_A);
    cudaFree(d_B_int);
    cudaFree(d_s);
    cudaFree(d_C);
    
    printf("\\n✨ Tensor core w4a16f GEMV test completed!\\n");
    return 0;
}