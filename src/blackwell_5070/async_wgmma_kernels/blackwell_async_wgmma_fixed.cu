#include "../utils/blackwell_common.h"
#include <mma.h>

// Simplified RTX 5070 Blackwell Async WGMMA kernel that works on current hardware
using namespace nvcuda;

// Simplified WGMMA-like operation using WMMA for compatibility
__global__ void blackwell_async_wgmma_fixed_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + WGMMA_M * WGMMA_K;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int block_row = blockIdx.y * WGMMA_M;
    const int block_col = blockIdx.x * WGMMA_N;
    
    // Producer warp loads data
    if (warp_id == 0) {
        // Load A tile cooperatively
        for (int i = lane_id; i < WGMMA_M * WGMMA_K; i += 32) {
            int row = i / WGMMA_K;
            int col = i % WGMMA_K;
            int global_row = block_row + row;
            
            shmem_A[i] = (global_row < M_dim && col < K_dim) ? 
                         A[global_row * K_dim + col] : __float2half(0.0f);
        }
    }
    
    // Producer warp loads B data
    if (warp_id == 1) {
        // Load B tile cooperatively
        for (int i = lane_id; i < WGMMA_K * WGMMA_N; i += 32) {
            int row = i / WGMMA_N;
            int col = i % WGMMA_N;
            int global_col = block_col + col;
            
            shmem_B[i] = (row < K_dim && global_col < N_dim) ?
                         B[row * N_dim + global_col] : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Consumer warps do computation using WMMA
    if (warp_id >= 2 && warp_id <= 5) {  // 4 compute warps
        int compute_warp_id = warp_id - 2;
        
        // Each warp computes a 16x16 sub-tile
        int warp_row = (compute_warp_id / 2) * 16;
        int warp_col = (compute_warp_id % 2) * 16;
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;  
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;
        
        wmma::fill_fragment(frag_C, 0.0f);
        
        // Accumulate over K dimension
        for (int k_sub = 0; k_sub < WGMMA_K; k_sub += 16) {
            if (k_sub + 16 <= WGMMA_K) {
                // Load fragments from shared memory
                wmma::load_matrix_sync(frag_A, 
                                     &shmem_A[warp_row * WGMMA_K + k_sub], 
                                     WGMMA_K);
                wmma::load_matrix_sync(frag_B, 
                                     &shmem_B[k_sub * WGMMA_N + warp_col], 
                                     WGMMA_N);
                
                // Compute
                wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
            }
        }
        
        // Store results
        int global_row = block_row + warp_row;
        int global_col = block_col + warp_col;
        
        if (global_row < M_dim && global_col < N_dim) {
            wmma::store_matrix_sync(&C[global_row * N_dim + global_col], 
                                   frag_C, N_dim, wmma::mem_row_major);
        }
    }
}

// Host launcher for fixed async WGMMA
void launch_blackwell_async_wgmma_fixed(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    dim3 grid((N_dim + WGMMA_N - 1) / WGMMA_N, (M_dim + WGMMA_M - 1) / WGMMA_M);
    dim3 block(192);  // 6 warps: 2 producer, 4 compute
    
    size_t shmem_size = (WGMMA_M * WGMMA_K + WGMMA_K * WGMMA_N) * sizeof(half);
    
    CUDA_CHECK(cudaFuncSetAttribute(blackwell_async_wgmma_fixed_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_size));
    
    blackwell_async_wgmma_fixed_kernel<<<grid, block, shmem_size>>>(
        A, B, C, M_dim, N_dim, K_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Test program for fixed async WGMMA
int main() {
    printf("🚀 RTX 5070 Blackwell Async WGMMA (Fixed Version)\n");
    printf("=================================================\n");
    
    if (!check_blackwell_support()) {
        printf("⚠️  Running on non-Blackwell hardware for compatibility\n");
    }
    
    const int M_dim = 512, N_dim = 512, K_dim = 512;
    printf("\nTesting matrix dimensions: %dx%d x %dx%d\n", M_dim, K_dim, K_dim, N_dim);
    
    // Memory allocation
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
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
        h_A[i] = __float2half(0.1f);
    }
    for (int i = 0; i < K_dim * N_dim; i++) {
        h_B[i] = __float2half(0.1f);
    }
    
    // Device allocation
    half *d_A, *d_B;
    float *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    printf("\n🧪 Running Async WGMMA benchmark...\n");
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        launch_blackwell_async_wgmma_fixed(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    BlackwellTimer timer;
    blackwell_timer_init(&timer, 50);
    
    const int iterations = 20;
    for (int i = 0; i < iterations; i++) {
        blackwell_timer_start(&timer);
        launch_blackwell_async_wgmma_fixed(d_A, d_B, d_C, M_dim, N_dim, K_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        blackwell_timer_stop(&timer);
    }
    
    float avg_time = blackwell_timer_get_avg(&timer);
    double tflops = calculate_tflops(M_dim, N_dim, K_dim, avg_time);
    double bandwidth = calculate_bandwidth_gb_s(size_A + size_B + size_C, avg_time);
    
    printf("\n📊 Async WGMMA Results:\n");
    printf("   Average time: %.3f ms\n", avg_time);
    printf("   Performance: %.1f TFLOPS\n", tflops);
    printf("   Memory bandwidth: %.1f GB/s\n", bandwidth);
    
    // Validation
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    bool valid = true;
    float expected = K_dim * 0.1f * 0.1f;
    
    for (int i = 0; i < std::min(100, M_dim * N_dim); i++) {
        if (isnan(h_C[i]) || isinf(h_C[i]) || fabs(h_C[i] - expected) > expected * 0.1f) {
            valid = false;
            printf("   Error at [%d]: expected %.3f, got %.3f\n", i, expected, h_C[i]);
            break;
        }
    }
    
    printf("   Validation: %s", valid ? "✅ PASSED" : "❌ FAILED");
    if (valid) {
        printf(" (expected: %.3f, got: %.3f)", expected, h_C[0]);
    }
    printf("\n");
    
    // Cleanup
    blackwell_timer_cleanup(&timer);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("\n✨ Async WGMMA test completed!\n");
    printf("This version uses WMMA for compatibility with current hardware.\n");
    
    return 0;
}