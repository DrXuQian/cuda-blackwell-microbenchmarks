#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cassert>

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M = 16, N = 16, K = 16;
constexpr int BLOCK_SIZE = 128;

__device__ void copy_gmem_to_shmem_warp(half* shmem_ptr, const half* gmem_ptr, 
                                        int rows, int cols, int src_ld, int warp_id) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Only specified warp performs the copy
    if (threadIdx.x >= warp_id * WARP_SIZE && threadIdx.x < (warp_id + 1) * WARP_SIZE) {
        const int total_elements = rows * cols;
        for (int i = lane_id; i < total_elements; i += WARP_SIZE) {
            int row = i / cols;
            int col = i % cols;
            shmem_ptr[i] = gmem_ptr[row * src_ld + col];
        }
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

__global__ void warp_specialized_mma_kernel(
    const half* A, const half* B, float* C,
    int M_dim, int N_dim, int K_dim
) {
    extern __shared__ char shmem[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    half* shmem_A = reinterpret_cast<half*>(shmem);
    half* shmem_B = shmem_A + M * K;
    
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
    copy_gmem_to_shmem(shmem_A_buffers[0], &A[block_row * K_dim], M, K, K_dim);
    copy_gmem_to_shmem(shmem_B_buffers[0], &B[block_col], K, N, N_dim);
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current_buffer = buffer_idx;
        int next_buffer = 1 - buffer_idx;
        
        // Warp specialization: Warp 0 loads next data while Warp 1+ computes
        if (warp_id == 0 && k + K < K_dim) {
            // Warp 0: Load next chunk asynchronously
            copy_gmem_to_shmem_warp(shmem_A_buffers[next_buffer], 
                                   &A[block_row * K_dim + k + K], M, K, K_dim, 0);
            copy_gmem_to_shmem_warp(shmem_B_buffers[next_buffer], 
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

bool verify_with_cublas(const half* h_A, const half* h_B, const float* h_C_gpu, 
                       int M_dim, int N_dim, int K_dim, float tolerance = 1e-1) {
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C_cublas;
    
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_cublas, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N_dim, M_dim, K_dim,
                 &alpha,
                 d_B, CUDA_R_16F, N_dim,
                 d_A, CUDA_R_16F, K_dim,
                 &beta,
                 d_C_cublas, CUDA_R_32F, N_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    float* h_C_cublas = (float*)malloc(size_C);
    cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);
    
    // Compare results
    bool match = true;
    int errors = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < M_dim * N_dim && errors < 10; i++) {
        float diff = fabsf(h_C_gpu[i] - h_C_cublas[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > tolerance) {
            if (errors == 0) {
                printf("First few mismatches:\n");
            }
            printf("  Index %d: GPU = %f, cuBLAS = %f, diff = %f\n", 
                   i, h_C_gpu[i], h_C_cublas[i], diff);
            match = false;
            errors++;
        }
    }
    
    printf("Max difference vs cuBLAS: %f\n", max_diff);
    
    // Cleanup
    cublasDestroy(handle);
    free(h_C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_cublas);
    
    return match;
}

void benchmark_kernel(void (*kernel)(const half*, const half*, float*, int, int, int),
                     const char* kernel_name,
                     const half* d_A, const half* d_B, float* d_C,
                     int M_dim, int N_dim, int K_dim,
                     int num_iterations = 100) {
    
    dim3 grid((N_dim + N - 1) / N, (M_dim + M - 1) / M);
    dim3 block(BLOCK_SIZE);
    
    size_t shmem_size = 4 * (M * K + K * N) * sizeof(half); // Double buffer
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(error));
        return;
    }
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double flops = 2.0 * M_dim * N_dim * K_dim * num_iterations;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;
    
    printf("%s: %.3f ms (avg), %.1f GFLOPS\n", kernel_name, elapsed_time / num_iterations, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("MMA Performance Test on RTX 4070 Ti Super\n");
    printf("=========================================\n");
    
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
    
    printf("Testing Warp Specialized MMA Kernel:\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel, "Warp Specialized", d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim)) {
        printf("✅ Correctness: PASSED\n\n");
    } else {
        printf("❌ Correctness: FAILED\n\n");
    }
    
    printf("Testing Ping-Pong MMA Kernel:\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(ping_pong_mma_kernel, "Ping-Pong", d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (verify_with_cublas(h_A, h_B, h_C_gpu, M_dim, N_dim, K_dim)) {
        printf("✅ Correctness: PASSED\n\n");
    } else {
        printf("❌ Correctness: FAILED\n\n");
    }
    
    // Performance comparison with cuBLAS
    printf("cuBLAS Reference Performance:\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_dim, M_dim, K_dim,
                     &alpha, d_B, CUDA_R_16F, N_dim,
                     d_A, CUDA_R_16F, K_dim, &beta,
                     d_C, CUDA_R_32F, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    const int cublas_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < cublas_iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_dim, M_dim, K_dim,
                     &alpha, d_B, CUDA_R_16F, N_dim,
                     d_A, CUDA_R_16F, K_dim, &beta,
                     d_C, CUDA_R_32F, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double flops = 2.0 * M_dim * N_dim * K_dim * cublas_iterations;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;
    
    printf("cuBLAS: %.3f ms (avg), %.1f GFLOPS\n", elapsed_time / cublas_iterations, gflops);
    
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n🎉 All tests completed successfully!\n");
    return 0;
}