#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>
#include <cassert>

using namespace nvcuda;
using namespace cooperative_groups;

constexpr int WARP_SIZE = 32;
constexpr int M = 16, N = 16, K = 16;
constexpr int BLOCK_SIZE = 128;

__device__ void async_copy_gmem_to_shmem(void* shmem_ptr, const void* gmem_ptr, size_t bytes) {
    const size_t tid = threadIdx.x;
    const size_t block_size = blockDim.x;
    
    // Use 16-byte aligned copies with cp.async
    const size_t elements_per_16b = bytes / 16;
    const half2* src = reinterpret_cast<const half2*>(gmem_ptr);
    half2* dst = reinterpret_cast<half2*>(shmem_ptr);
    
    for (size_t i = tid; i < elements_per_16b; i += block_size) {
        uint32_t smem_addr = __cvta_generic_to_shared(&dst[i]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem_addr), "l"(&src[i]));
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
    
    for (int k = 0; k < K_dim; k += K) {
        if (warp_id == 0) {
            async_copy_gmem_to_shmem(
                shmem_A,
                &A[block_row * K_dim + k],
                M * K * sizeof(half)
            );
            async_copy_gmem_to_shmem(
                shmem_B,
                &B[k * N_dim + block_col],
                K * N * sizeof(half)
            );
            
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        
        __syncthreads();
        
        if (warp_id == 1) {
            wmma::load_matrix_sync(frag_A, shmem_A, K);
            wmma::load_matrix_sync(frag_B, shmem_B, N);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
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
    
    async_copy_gmem_to_shmem(
        shmem_A[0],
        &A[block_row * K_dim],
        M * K * sizeof(half)
    );
    async_copy_gmem_to_shmem(
        shmem_B[0],
        &B[block_col],
        K * N * sizeof(half)
    );
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    
    for (int k = 0; k < K_dim; k += K) {
        int current = ping_pong;
        int next = 1 - ping_pong;
        
        if (k + K < K_dim) {
            async_copy_gmem_to_shmem(
                shmem_A[next],
                &A[block_row * K_dim + k + K],
                M * K * sizeof(half)
            );
            async_copy_gmem_to_shmem(
                shmem_B[next],
                &B[(k + K) * N_dim + block_col],
                K * N * sizeof(half)
            );
            asm volatile("cp.async.commit_group;\n" ::);
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(frag_A, shmem_A[current], K);
        wmma::load_matrix_sync(frag_B, shmem_B[current], N);
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        
        if (k + K < K_dim) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        
        __syncthreads();
        ping_pong = next;
    }
    
    wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
}

void cpu_reference_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_results(const float* gpu_result, const float* cpu_result, int size, float tolerance = 1e-2) {
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f, diff = %f\n", 
                   i, gpu_result[i], cpu_result[i], diff);
            return false;
        }
    }
    return true;
}

void benchmark_kernel(void (*kernel)(const half*, const half*, float*, int, int, int),
                     const char* kernel_name,
                     const half* d_A, const half* d_B, float* d_C,
                     int M_dim, int N_dim, int K_dim,
                     int num_iterations = 100) {
    
    dim3 grid((N_dim + N - 1) / N, (M_dim + M - 1) / M);
    dim3 block(BLOCK_SIZE);
    
    size_t shmem_size = 2 * (M * K + K * N) * sizeof(half);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double flops = 2.0 * M_dim * N_dim * K_dim * num_iterations;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;
    
    printf("%s: %.2f ms (avg), %.2f GFLOPS\n", kernel_name, elapsed_time / num_iterations, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("Starting program...\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    const int M_dim = 256, N_dim = 256, K_dim = 256;
    
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    printf("Allocating host memory...\n");
    
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    printf("Initializing matrices...\n");
    
    for (int i = 0; i < M_dim * K_dim; i++) {
        h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < K_dim * N_dim; i++) {
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    printf("Allocating device memory...\n");
    
    half *d_A, *d_B;
    float *d_C;
    
    error = cudaMalloc(&d_A, size_A);
    if (error != cudaSuccess) {
        printf("Failed to allocate device memory for A: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    error = cudaMalloc(&d_B, size_B);
    if (error != cudaSuccess) {
        printf("Failed to allocate device memory for B: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        return 1;
    }
    
    error = cudaMalloc(&d_C, size_C);
    if (error != cudaSuccess) {
        printf("Failed to allocate device memory for C: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }
    
    printf("Copying data to device...\n");
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    printf("Computing CPU reference...\n");
    cpu_reference_gemm(h_A, h_B, h_C_cpu, M_dim, N_dim, K_dim);
    
    printf("Testing warp specialized kernel...\n");
    benchmark_kernel(warp_specialized_mma_kernel, "Warp Specialized", 
                    d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (verify_results(h_C_gpu, h_C_cpu, M_dim * N_dim)) {
        printf("Warp specialized kernel: PASSED\n");
    } else {
        printf("Warp specialized kernel: FAILED\n");
    }
    
    printf("Testing ping-pong kernel...\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(ping_pong_mma_kernel, "Ping-Pong", 
                    d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (verify_results(h_C_gpu, h_C_cpu, M_dim * N_dim)) {
        printf("Ping-pong kernel: PASSED\n");
    } else {
        printf("Ping-pong kernel: FAILED\n");
    }
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}