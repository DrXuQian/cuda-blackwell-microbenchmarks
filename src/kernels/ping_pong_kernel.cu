#include "common.h"

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

__global__ void ping_pong_mma_kernel(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    half* shmem_A[2] = {
        reinterpret_cast<half*>(shmem),
        reinterpret_cast<half*>(shmem) + M * K
    };
    half* shmem_B[2] = {
        shmem_A[1] + M * K,
        shmem_A[1] + M * K + K * N
    };
    
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, M, N, K, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    int block_row = blockIdx.y * M;
    int block_col = blockIdx.x * N;
    
    int ping_pong = 0;
    
    // Load first chunk
    copy_gmem_to_shmem(shmem_A[0], &A[block_row * K_dim], M, K, K_dim);
    copy_gmem_to_shmem(shmem_B[0], &B[block_col], K, N, N_dim);
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current = ping_pong;
        int next = 1 - ping_pong;
        
        // Load next chunk while computing current (ping-pong)
        if (k + K < K_dim) {
            copy_gmem_to_shmem(shmem_A[next], &A[block_row * K_dim + k + K], M, K, K_dim);
            copy_gmem_to_shmem(shmem_B[next], &B[(k + K) * N_dim + block_col], K, N, N_dim);
        }
        
        __syncthreads();
        
        // Compute with current chunk
        wmma::load_matrix_sync(frag_A, shmem_A[current], K);
        wmma::load_matrix_sync(frag_B, shmem_B[current], N);
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        
        __syncthreads();
        ping_pong = next;
    }
    
    wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
}

int main() {
    printf("Ping-Pong MMA Kernel Performance Test\n");
    printf("====================================\n");
    
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
    
    printf("Testing Ping-Pong MMA Kernel:\n");
    size_t shmem_size = 2 * (M * K + K * N) * sizeof(half);
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(ping_pong_mma_kernel, "Ping-Pong MMA", d_A, d_B, d_C, M_dim, N_dim, K_dim, shmem_size);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    AccuracyResult acc_result;
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim, &acc_result)) {
        printf("\n");
        print_accuracy_result(acc_result, "Ping-Pong MMA");
    } else {
        printf("\n");
        print_accuracy_result(acc_result, "Ping-Pong MMA");
    }
    
    printf("\ncuBLAS Reference Performance:\n");
    benchmark_cublas(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 Ping-Pong kernel test completed!\n");
    return 0;
}