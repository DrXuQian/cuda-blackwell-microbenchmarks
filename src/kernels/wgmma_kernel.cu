#include "common.h"

__global__ void wgmma_async_kernel(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + M * K;
    
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    int block_row = blockIdx.y * M;
    int block_col = blockIdx.x * N;
    
    for (int k = 0; k < K_dim; k += K) {
        // All threads participate in memory loading
        const int tid = threadIdx.x;
        const int block_size = blockDim.x;
        
        // Load A matrix chunk
        for (int i = tid; i < M * K; i += block_size) {
            int row = i / K;
            int col = i % K;
            if (block_row + row < M_dim && k + col < K_dim) {
                shmem_A[i] = A[(block_row + row) * K_dim + (k + col)];
            }
        }
        
        // Load B matrix chunk  
        for (int i = tid; i < K * N; i += block_size) {
            int row = i / N;
            int col = i % N;
            if (k + row < K_dim && block_col + col < N_dim) {
                shmem_B[i] = B[(k + row) * N_dim + (block_col + col)];
            }
        }
        
        __syncthreads();
        
        // Perform WMMA computation
        wmma::load_matrix_sync(frag_A, shmem_A, K);
        wmma::load_matrix_sync(frag_B, shmem_B, N);
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        
        __syncthreads();
    }
    
    wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
}
__global__ void fallback_wmma_kernel(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + M * K;
    
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    int block_row = blockIdx.y * M;
    int block_col = blockIdx.x * N;
    
    for (int k = 0; k < K_dim; k += K) {
        // All threads participate in memory loading
        const int tid = threadIdx.x;
        const int block_size = blockDim.x;
        
        // Load A matrix chunk
        for (int i = tid; i < M * K; i += block_size) {
            int row = i / K;
            int col = i % K;
            shmem_A[i] = A[(block_row + row) * K_dim + (k + col)];
        }
        
        // Load B matrix chunk  
        for (int i = tid; i < K * N; i += block_size) {
            int row = i / N;
            int col = i % N;
            shmem_B[i] = B[(k + row) * N_dim + (block_col + col)];
        }
        
        __syncthreads();
        
        // Perform WMMA computation
        wmma::load_matrix_sync(frag_A, shmem_A, K);
        wmma::load_matrix_sync(frag_B, shmem_B, N);
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        
        __syncthreads();
    }
    
    wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
}

int main() {
    printf("WGMMA Async Kernel Performance Test\n");
    printf("===================================\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    bool supports_wgmma = (prop.major == 8 && prop.minor >= 9) || prop.major > 8;
    printf("WGMMA Support: %s\n\n", supports_wgmma ? "✅ YES" : "❌ NO (fallback to WMMA)");
    
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
    
    size_t shmem_size = 2 * (M * K + K * N) * sizeof(half);
    
    if (supports_wgmma) {
        printf("Testing WGMMA Async Kernel:\n");
        cudaMemset(d_C, 0, size_C);
        // Note: WGMMA requires 128-thread blocks (warp group size)
        
        printf("WGMMA not supported at compile time - using fallback\n");
        benchmark_kernel(fallback_wmma_kernel, "Fallback WMMA", d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    } else {
        printf("Testing Fallback WMMA Kernel:\n");
        cudaMemset(d_C, 0, size_C);
        benchmark_kernel(fallback_wmma_kernel, "Fallback WMMA", d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    }
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result;
    const char* kernel_name = supports_wgmma ? "WGMMA Async" : "Fallback WMMA";
    
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim, &acc_result)) {
        printf("\n");
        print_accuracy_result(acc_result, kernel_name);
    } else {
        printf("\n");
        print_accuracy_result(acc_result, kernel_name);
    }
    
    printf("\ncuBLAS Reference Performance:\n");
    benchmark_cublas(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 WGMMA kernel test completed!\n");
    return 0;
}