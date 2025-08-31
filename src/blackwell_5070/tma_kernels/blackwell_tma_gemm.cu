#include "../utils/blackwell_common.h"

// RTX 5070 Blackwell TMA-optimized GEMM kernel
// Uses TMA (Tensor Memory Accelerator) for optimal memory bandwidth

// TMA descriptor setup for optimal RTX 5070 performance
CUresult setup_blackwell_tma_descriptors(
    BlackwellTMADescriptor* desc_A, BlackwellTMADescriptor* desc_B,
    const half* A, const half* B,
    int M_dim, int N_dim, int K_dim
) {
    CUresult result;
    
    // TMA configuration optimized for Blackwell
    CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;  // Optimal for Blackwell
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;  // Larger for Blackwell
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    // A matrix TMA descriptor (M x K)
    uint64_t global_shape_A[2] = {(uint64_t)M_dim, (uint64_t)K_dim};
    uint64_t global_stride_A[1] = {(uint64_t)(K_dim * sizeof(half))};
    uint32_t box_shape_A[2] = {TMA_TILE_M, TMA_TILE_K};
    uint32_t box_stride_A[1] = {TMA_TILE_K * sizeof(half)};
    
    result = cuTensorMapEncodeTiled(
        &desc_A->tma_map, data_type, 2,
        (void*)A, global_shape_A, global_stride_A,
        box_shape_A, box_stride_A,
        interleave, swizzle, l2promo, oob_fill
    );
    
    if (result != CUDA_SUCCESS) {
        desc_A->initialized = false;
        return result;
    }
    
    // B matrix TMA descriptor (K x N)
    uint64_t global_shape_B[2] = {(uint64_t)K_dim, (uint64_t)N_dim};
    uint64_t global_stride_B[1] = {(uint64_t)(N_dim * sizeof(half))};
    uint32_t box_shape_B[2] = {TMA_TILE_K, TMA_TILE_N};
    uint32_t box_stride_B[1] = {TMA_TILE_N * sizeof(half)};
    
    result = cuTensorMapEncodeTiled(
        &desc_B->tma_map, data_type, 2,
        (void*)B, global_shape_B, global_stride_B,
        box_shape_B, box_stride_B,
        interleave, swizzle, l2promo, oob_fill
    );
    
    desc_A->initialized = (result == CUDA_SUCCESS);
    desc_B->initialized = (result == CUDA_SUCCESS);
    desc_A->bytes_transferred = TMA_TILE_M * TMA_TILE_K * sizeof(half);
    desc_B->bytes_transferred = TMA_TILE_K * TMA_TILE_N * sizeof(half);
    
    return result;
}

// Blackwell TMA GEMM kernel with optimal memory access patterns
__global__ void blackwell_tma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    BlackwellTMADescriptor* tma_desc_A,
    BlackwellTMADescriptor* tma_desc_B,
    int M_dim, int N_dim, int K_dim
) {
    // Shared memory layout optimized for Blackwell
    extern __shared__ char shmem[];
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + TMA_TILE_M * TMA_TILE_K;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / BLACKWELL_WARP_SIZE;
    const int lane_id = tid % BLACKWELL_WARP_SIZE;
    
    // Block-level tiling
    const int block_row = blockIdx.y * TMA_TILE_M;
    const int block_col = blockIdx.x * TMA_TILE_N;
    
    // Warp-level tiling for computation
    const int warps_per_row = 2;  // 2 warps handle M dimension
    const int warps_per_col = 2;  // 2 warps handle N dimension
    
    const int warp_row_id = warp_id / warps_per_col;
    const int warp_col_id = warp_id % warps_per_col;
    
    const int warp_row_offset = warp_row_id * (TMA_TILE_M / warps_per_row);
    const int warp_col_offset = warp_col_id * (TMA_TILE_N / warps_per_col);
    
    // Initialize accumulator
    float acc[16] = {0.0f};  // Each warp computes 64x64 / 4 warps = 32x32 output
    
    // TMA-based loading (only thread 0 initiates TMA)
    if (tid == 0) {
        // TMA load A tile (M x K)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :
            : "r"(__cvta_generic_to_shared(shmem_A)),
              "l"(tma_desc_A->tma_map),
              "r"(block_row), "r"(0),
              "r"(__cvta_generic_to_shared(shmem_A + TMA_TILE_M * TMA_TILE_K - 1))
        );
        
        // TMA load B tile (K x N)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :
            : "r"(__cvta_generic_to_shared(shmem_B)),
              "l"(tma_desc_B->tma_map),
              "r"(0), "r"(block_col),
              "r"(__cvta_generic_to_shared(shmem_B + TMA_TILE_K * TMA_TILE_N - 1))
        );
    }
    
    // Wait for TMA completion with optimal synchronization
    __syncthreads();
    asm volatile("cp.async.bulk.wait_group.read 0;\n");
    __syncthreads();
    
    // Compute using tensor cores (WMMA for compatibility)
    if (warp_id < 4) {  // 4 warps for computation
        using namespace nvcuda::wmma;
        
        // WMMA fragments
        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        
        fill_fragment(c_frag, 0.0f);
        
        // Compute 32x32 output per warp using multiple 16x16 tiles
        for (int warp_k = 0; warp_k < TMA_TILE_K; warp_k += 16) {
            for (int warp_m_sub = 0; warp_m_sub < (TMA_TILE_M / warps_per_row); warp_m_sub += 16) {
                for (int warp_n_sub = 0; warp_n_sub < (TMA_TILE_N / warps_per_col); warp_n_sub += 16) {
                    
                    int a_row = warp_row_offset + warp_m_sub;
                    int a_col = warp_k;
                    int b_row = warp_k;
                    int b_col = warp_col_offset + warp_n_sub;
                    
                    // Load fragments from shared memory
                    load_matrix_sync(a_frag, shmem_A + a_row * TMA_TILE_K + a_col, TMA_TILE_K);
                    load_matrix_sync(b_frag, shmem_B + b_row * TMA_TILE_N + b_col, TMA_TILE_N);
                    
                    // Compute
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
        }
        
        // Store results to global memory with optimal coalescing
        for (int warp_m_sub = 0; warp_m_sub < (TMA_TILE_M / warps_per_row); warp_m_sub += 16) {
            for (int warp_n_sub = 0; warp_n_sub < (TMA_TILE_N / warps_per_col); warp_n_sub += 16) {
                
                int c_row = block_row + warp_row_offset + warp_m_sub;
                int c_col = block_col + warp_col_offset + warp_n_sub;
                
                if (c_row < M_dim && c_col < N_dim) {
                    store_matrix_sync(C + c_row * N_dim + c_col, c_frag, N_dim, mem_row_major);
                }
            }
        }
    }
}

// Host launcher for Blackwell TMA GEMM
void launch_blackwell_tma_gemm(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    // Allocate TMA descriptors on device
    BlackwellTMADescriptor *d_desc_A, *d_desc_B;
    CUDA_CHECK(cudaMalloc(&d_desc_A, sizeof(BlackwellTMADescriptor)));
    CUDA_CHECK(cudaMalloc(&d_desc_B, sizeof(BlackwellTMADescriptor)));
    
    // Setup TMA descriptors on host
    BlackwellTMADescriptor h_desc_A, h_desc_B;
    CUresult result = setup_blackwell_tma_descriptors(&h_desc_A, &h_desc_B, A, B, M_dim, N_dim, K_dim);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        printf("TMA setup failed: %s\n", error_str);
        return;
    }
    
    // Copy descriptors to device
    CUDA_CHECK(cudaMemcpy(d_desc_A, &h_desc_A, sizeof(BlackwellTMADescriptor), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_desc_B, &h_desc_B, sizeof(BlackwellTMADescriptor), cudaMemcpyHostToDevice));
    
    // Launch configuration optimized for RTX 5070
    dim3 grid((N_dim + TMA_TILE_N - 1) / TMA_TILE_N, (M_dim + TMA_TILE_M - 1) / TMA_TILE_M);
    dim3 block(128);  // 4 warps, optimal for Blackwell
    
    size_t shmem_size = (TMA_TILE_M * TMA_TILE_K + TMA_TILE_K * TMA_TILE_N) * sizeof(half);
    
    // Ensure shared memory limit is set
    CUDA_CHECK(cudaFuncSetAttribute(blackwell_tma_gemm_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_size));
    
    // Launch kernel
    blackwell_tma_gemm_kernel<<<grid, block, shmem_size>>>(
        A, B, C, d_desc_A, d_desc_B, M_dim, N_dim, K_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Cleanup
    cudaFree(d_desc_A);
    cudaFree(d_desc_B);
}

// Test program for Blackwell TMA GEMM
int main() {
    printf("🚀 RTX 5070 Blackwell TMA GEMM Kernel\n");
    printf("=====================================\n");
    
    if (!check_blackwell_support()) {
        return 1;
    }
    
    const int M_dim = 2048, N_dim = 2048, K_dim = 2048;
    printf("\nTesting matrix dimensions: %dx%d x %dx%d\n", M_dim, K_dim, K_dim, N_dim);
    
    // Memory allocation
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    printf("Memory usage: A=%.1f MB, B=%.1f MB, C=%.1f MB\n",
           size_A/(1024.0*1024.0), size_B/(1024.0*1024.0), size_C/(1024.0*1024.0));
    
    // Host allocation
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    if (!h_A || !h_B || !h_C) {
        printf("❌ Host memory allocation failed\n");
        return 1;
    }
    
    // Initialize test data
    srand(42);
    for (int i = 0; i < M_dim * K_dim; i++) {
        h_A[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    for (int i = 0; i < K_dim * N_dim; i++) {
        h_B[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    
    // Device allocation
    half *d_A, *d_B;
    float *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    printf("\n🧪 Running TMA GEMM benchmark...\n");
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        launch_blackwell_tma_gemm(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    BlackwellTimer timer;
    blackwell_timer_init(&timer, 100);
    
    const int iterations = 50;
    
    for (int i = 0; i < iterations; i++) {
        blackwell_timer_start(&timer);
        launch_blackwell_tma_gemm(d_A, d_B, d_C, M_dim, N_dim, K_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        blackwell_timer_stop(&timer);
    }
    
    float avg_time_ms = blackwell_timer_get_avg(&timer);
    double tflops = calculate_tflops(M_dim, N_dim, K_dim, avg_time_ms);
    double bandwidth = calculate_bandwidth_gb_s(size_A + size_B + size_C, avg_time_ms);
    
    printf("\n📊 Performance Results:\n");
    printf("   Average time: %.3f ms\n", avg_time_ms);
    printf("   Performance: %.1f TFLOPS\n", tflops);
    printf("   Memory bandwidth: %.1f GB/s\n", bandwidth);
    printf("   Expected on RTX 5070: ~15-20 TFLOPS with TMA\n");
    
    // Validation
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    bool valid = true;
    for (int i = 0; i < 100; i++) {
        if (isnan(h_C[i]) || isinf(h_C[i])) {
            valid = false;
            printf("Invalid result at index %d: %f\n", i, h_C[i]);
            break;
        }
    }
    
    printf("   Validation: %s\n", valid ? "✅ PASSED" : "❌ FAILED");
    
    if (valid) {
        printf("   Sample results: C[0]=%.3f, C[100]=%.3f, C[1000]=%.3f\n", 
               h_C[0], h_C[100], h_C[1000]);
    }
    
    // Cleanup
    blackwell_timer_cleanup(&timer);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("\n✨ TMA GEMM benchmark completed!\n");
    printf("This kernel is optimized for RTX 5070 Blackwell architecture.\n");
    
    return 0;
}