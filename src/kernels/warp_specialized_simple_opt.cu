#include "common.h"

__device__ void copy_gmem_to_shmem_warp_vectorized(half* shmem_ptr, const half* gmem_ptr, 
                                                  int rows, int cols, int src_ld, int warp_id) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Only specified warp performs the copy with vectorized access
    if (threadIdx.x >= warp_id * WARP_SIZE && threadIdx.x < (warp_id + 1) * WARP_SIZE) {
        // Use vectorized loads when possible (float4 = 8 half values)
        const int total_elements = rows * cols;
        
        // Try vectorized copy first
        if (total_elements >= 8 && total_elements % 8 == 0) {
            const int vec_elements = total_elements / 8;
            float4* shmem_vec = reinterpret_cast<float4*>(shmem_ptr);
            const float4* gmem_vec = reinterpret_cast<const float4*>(gmem_ptr);
            
            for (int i = lane_id; i < vec_elements; i += WARP_SIZE) {
                if (i * 8 < total_elements) {
                    shmem_vec[i] = gmem_vec[i];
                }
            }
        } else {
            // Fallback to element-wise copy
            for (int i = lane_id; i < total_elements; i += WARP_SIZE) {
                int row = i / cols;
                int col = i % cols;
                shmem_ptr[i] = gmem_ptr[row * src_ld + col];
            }
        }
    }
}

__global__ void warp_specialized_mma_kernel_simple_opt(
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
    
    // Pre-load first chunk
    if (warp_id == 0) {
        copy_gmem_to_shmem_warp_vectorized(shmem_A_buffers[0], &A[block_row * K_dim], M, K, K_dim, 0);
        copy_gmem_to_shmem_warp_vectorized(shmem_B_buffers[0], &B[block_col], K, N, N_dim, 0);
    }
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current_buffer = buffer_idx;
        int next_buffer = 1 - buffer_idx;
        
        // Warp specialization: Warp 0 loads next data while Warp 1 computes
        if (warp_id == 0 && k + K < K_dim) {
            // Warp 0: Load next chunk asynchronously with vectorization
            copy_gmem_to_shmem_warp_vectorized(shmem_A_buffers[next_buffer], 
                                              &A[block_row * K_dim + k + K], M, K, K_dim, 0);
            copy_gmem_to_shmem_warp_vectorized(shmem_B_buffers[next_buffer], 
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
    printf("Simple Optimized Warp Specialized MMA Kernel Performance Test\n");
    printf("=============================================================\n");
    
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
    
    printf("Testing Simple Optimized Warp Specialized MMA Kernel:\n");
    size_t shmem_size = 4 * (M * K + K * N) * sizeof(half); // Double buffer
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel_simple_opt, "Simple Optimized Warp Specialized", 
                     d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result;
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim, &acc_result)) {
        printf("\n");
        print_accuracy_result(acc_result, "Simple Optimized Warp Specialized");
    } else {
        printf("\n");
        print_accuracy_result(acc_result, "Simple Optimized Warp Specialized");
    }
    
    printf("\ncuBLAS Reference Performance:\n");
    benchmark_cublas(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 Simple optimized kernel test completed!\n");
    return 0;
}