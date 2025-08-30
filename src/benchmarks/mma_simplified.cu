#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <chrono>
#include <cassert>

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M = 16, N = 16, K = 16;
constexpr int BLOCK_SIZE = 128;

__global__ void simple_mma_kernel(
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
        // Simple coalesced memory copy to shared memory
        for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
            if (block_row * K_dim + k + i < M_dim * K_dim) {
                shmem_A[i] = A[block_row * K_dim + k + i];
            }
        }
        
        for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
            int row = i / N;
            int col = i % N;
            int global_row = k + row;
            int global_col = block_col + col;
            if (global_row < K_dim && global_col < N_dim) {
                shmem_B[i] = B[global_row * N_dim + global_col];
            }
        }
        
        __syncthreads();
        
        // Only first warp performs MMA
        if (threadIdx.x < WARP_SIZE) {
            wmma::load_matrix_sync(frag_A, shmem_A, K);
            wmma::load_matrix_sync(frag_B, shmem_B, N);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
    }
    
    if (threadIdx.x < WARP_SIZE) {
        wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
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
            // Warp 0: Memory loading
            for (int i = threadIdx.x % WARP_SIZE; i < M * K; i += WARP_SIZE) {
                if (block_row * K_dim + k + i < M_dim * K_dim) {
                    shmem_A[i] = A[block_row * K_dim + k + i];
                }
            }
            
            for (int i = threadIdx.x % WARP_SIZE; i < K * N; i += WARP_SIZE) {
                int row = i / N;
                int col = i % N;
                int global_row = k + row;
                int global_col = block_col + col;
                if (global_row < K_dim && global_col < N_dim) {
                    shmem_B[i] = B[global_row * N_dim + global_col];
                }
            }
        }
        
        __syncthreads();
        
        if (warp_id == 1) {
            // Warp 1: Compute
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
    
    // Load first chunk
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        if (block_row * K_dim + i < M_dim * K_dim) {
            shmem_A[0][i] = A[block_row * K_dim + i];
        }
    }
    
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        int row = i / N;
        int col = i % N;
        int global_col = block_col + col;
        if (row < K_dim && global_col < N_dim) {
            shmem_B[0][i] = B[row * N_dim + global_col];
        }
    }
    
    __syncthreads();
    
    for (int k = 0; k < K_dim; k += K) {
        int current = ping_pong;
        int next = 1 - ping_pong;
        
        // Async load next chunk while computing current
        if (k + K < K_dim) {
            for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
                if (block_row * K_dim + k + K + i < M_dim * K_dim) {
                    shmem_A[next][i] = A[block_row * K_dim + k + K + i];
                }
            }
            
            for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
                int row = i / N;
                int col = i % N;
                int global_row = k + K + row;
                int global_col = block_col + col;
                if (global_row < K_dim && global_col < N_dim) {
                    shmem_B[next][i] = B[global_row * N_dim + global_col];
                }
            }
        }
        
        // Compute with current chunk
        if (threadIdx.x < WARP_SIZE) {
            wmma::load_matrix_sync(frag_A, shmem_A[current], K);
            wmma::load_matrix_sync(frag_B, shmem_B[current], N);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
        ping_pong = next;
    }
    
    if (threadIdx.x < WARP_SIZE) {
        wmma::store_matrix_sync(&C[block_row * N_dim + block_col], frag_C, N_dim, wmma::mem_row_major);
    }
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

bool verify_results(const float* gpu_result, const float* cpu_result, int size, float tolerance = 1e-1) {
    int errors = 0;
    for (int i = 0; i < size && errors < 10; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f, diff = %f\n", 
                   i, gpu_result[i], cpu_result[i], diff);
            errors++;
        }
    }
    return errors == 0;
}

void benchmark_kernel(void (*kernel)(const half*, const half*, float*, int, int, int),
                     const char* kernel_name,
                     const half* d_A, const half* d_B, float* d_C,
                     int M_dim, int N_dim, int K_dim,
                     int num_iterations = 10) {
    
    dim3 grid((N_dim + N - 1) / N, (M_dim + M - 1) / M);
    dim3 block(BLOCK_SIZE);
    
    size_t shmem_size = 2 * (M * K + K * N) * sizeof(half);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
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
    
    printf("%s: %.2f ms (avg), %.2f GFLOPS\n", kernel_name, elapsed_time / num_iterations, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Starting MMA benchmark..." << std::endl;
    
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
    
    const int M_dim = 512, N_dim = 512, K_dim = 512;
    
    size_t size_A = M_dim * K_dim * sizeof(half);
    size_t size_B = K_dim * N_dim * sizeof(half);
    size_t size_C = M_dim * N_dim * sizeof(float);
    
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize with small values to avoid numerical issues
    srand(42);
    for (int i = 0; i < M_dim * K_dim; i++) {
        h_A[i] = __float2half((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f);
    }
    for (int i = 0; i < K_dim * N_dim; i++) {
        h_B[i] = __float2half((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f);
    }
    
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Test simple kernel first
    printf("Testing simple MMA kernel...\n");
    benchmark_kernel(simple_mma_kernel, "Simple MMA", d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // CPU reference for a smaller subset (too slow for full matrix)
    const int test_size = 64;
    printf("Computing CPU reference for %dx%dx%d...\n", test_size, test_size, test_size);
    cpu_reference_gemm(h_A, h_B, h_C_cpu, test_size, test_size, test_size);
    
    if (verify_results(h_C_gpu, h_C_cpu, test_size * test_size)) {
        printf("Simple MMA kernel: PASSED\n");
    } else {
        printf("Simple MMA kernel: FAILED\n");
    }
    
    printf("Testing warp specialized kernel...\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(warp_specialized_mma_kernel, "Warp Specialized", d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    printf("Testing ping-pong kernel...\n");
    cudaMemset(d_C, 0, size_C);
    benchmark_kernel(ping_pong_mma_kernel, "Ping-Pong", d_A, d_B, d_C, M_dim, N_dim, K_dim);
    
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("All tests completed!\n");
    return 0;
}