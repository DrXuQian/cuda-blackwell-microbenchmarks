#include "common.h"

__device__ void copy_gmem_to_shmem_optimized(half* shmem_ptr, const half* gmem_ptr, 
                                            int rows, int cols, int src_ld) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Simple coalesced copy - handle row-major to contiguous conversion
    const int total_elements = rows * cols;
    for (int i = tid; i < total_elements; i += block_size) {
        int row = i / cols;
        int col = i % cols;
        shmem_ptr[i] = gmem_ptr[row * src_ld + col];
    }
}

// Multi-warp specialized kernel - better work distribution
__global__ void warp_specialized_mma_kernel_optimized(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + 2 * M * K;  // Double buffer A
    
    // Create double buffers for overlapped execution
    half* shmem_A_buffers[2] = {shmem_A, shmem_A + M * K};
    half* shmem_B_buffers[2] = {shmem_B, shmem_B + K * N};
    
    // Multiple compute fragments for better parallelism
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A[2];
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B[2];
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C[2];
    
    // Initialize accumulators
    for (int i = 0; i < 2; i++) {
        wmma::fill_fragment(frag_C[i], 0.0f);
    }
    
    int block_row = blockIdx.y * M;
    int block_col = blockIdx.x * N;
    
    int buffer_idx = 0;
    
    // Pre-load first chunk with all warps participating
    copy_gmem_to_shmem_optimized(shmem_A_buffers[0], &A[block_row * K_dim], M, K, K_dim);
    copy_gmem_to_shmem_optimized(shmem_B_buffers[0], &B[block_col], K, N, N_dim);
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current_buffer = buffer_idx;
        int next_buffer = 1 - buffer_idx;
        
        // Improved work distribution:
        // Warps 0-1: Handle data loading (50% of warps)
        // Warps 2-3: Handle computation (50% of warps)  
        if (warp_id < 2 && k + K < K_dim) {
            // Data loading warps - split work between two warps
            if (warp_id == 0) {
                // Warp 0: Load A matrix
                copy_gmem_to_shmem_optimized(shmem_A_buffers[next_buffer], 
                                           &A[block_row * K_dim + k + K], M, K, K_dim);
            } else {
                // Warp 1: Load B matrix  
                copy_gmem_to_shmem_optimized(shmem_B_buffers[next_buffer], 
                                           &B[(k + K) * N_dim + block_col], K, N, N_dim);
            }
        } else if (warp_id == 2) {
            // Single compute warp performs MMA operations
            wmma::load_matrix_sync(frag_A[0], shmem_A_buffers[current_buffer], K);
            wmma::load_matrix_sync(frag_B[0], shmem_B_buffers[current_buffer], N);
            wmma::mma_sync(frag_C[0], frag_A[0], frag_B[0], frag_C[0]);
        }
        
        __syncthreads();
        buffer_idx = next_buffer;
    }
    
    // Only use one compute warp to avoid double computation
    if (warp_id == 2) {
        // Primary compute warp stores results
        wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C[0], N_dim, wmma::mem_row_major);
    }
}

int main() {
    printf("Optimized Warp Specialized MMA Kernel Performance Test\n");
    printf("======================================================\n");
    
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
    
    printf("Testing Optimized Warp Specialized MMA Kernel:\n");
    size_t shmem_size = 4 * (M * K + K * N) * sizeof(half); // Double buffer
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel_optimized, "Optimized Warp Specialized MMA", 
                     d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result;
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim, &acc_result)) {
        printf("\n");
        print_accuracy_result(acc_result, "Optimized Warp Specialized MMA");
    } else {
        printf("\n");
        print_accuracy_result(acc_result, "Optimized Warp Specialized MMA");
    }
    
    printf("\ncuBLAS Reference Performance:\n");
    benchmark_cublas(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 Optimized warp specialized kernel test completed!\n");
    return 0;
}