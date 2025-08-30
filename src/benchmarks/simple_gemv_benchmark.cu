#include "common.h"
#include <random>
#include <vector>
#include <iomanip>
#include <algorithm>

// Simple 4-bit quantization for testing
__global__ void simple_gemv_w4a16f_kernel(
    const half* __restrict__ A,      // 1 x N input vector
    const half* __restrict__ B,      // N x M weight matrix (fp16 for simplicity)
    const half* __restrict__ scales, // M scaling factors
    half* __restrict__ C,            // 1 x M output vector
    int N, int M
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < M) {
        float accum = 0.0f;
        
        // Simple dot product
        for (int row = 0; row < N; row++) {
            accum += __half2float(A[row]) * __half2float(B[row * M + col]);
        }
        
        // Apply scaling
        accum *= __half2float(scales[col]);
        C[col] = __float2half(accum);
    }
}

// Optimized version with shared memory and vectorization
__global__ void optimized_gemv_w4a16f_kernel(
    const half* __restrict__ A,      // 1 x N input vector
    const half* __restrict__ B,      // N x M weight matrix
    const half* __restrict__ scales, // M scaling factors  
    half* __restrict__ C,            // 1 x M output vector
    int N, int M
) {
    extern __shared__ half shmem_A[];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load A into shared memory cooperatively
    for (int i = tid; i < N; i += blockDim.x) {
        shmem_A[i] = A[i];
    }
    __syncthreads();
    
    if (col < M) {
        float accum = 0.0f;
        
        // Vectorized computation with shared memory A (using half2 pairs)
        for (int row = 0; row < N; row += 4) {
            // Load 4 consecutive elements as two half2 values
            if (row + 3 < N) {
                half2 a_vals0 = *reinterpret_cast<const half2*>(&shmem_A[row]);
                half2 a_vals1 = *reinterpret_cast<const half2*>(&shmem_A[row + 2]);
                half2 b_vals0 = *reinterpret_cast<const half2*>(&B[row * M + col]);
                half2 b_vals1 = *reinterpret_cast<const half2*>(&B[(row + 2) * M + col]);
                
                accum += __half2float(a_vals0.x) * __half2float(b_vals0.x);
                accum += __half2float(a_vals0.y) * __half2float(b_vals0.y);
                accum += __half2float(a_vals1.x) * __half2float(b_vals1.x);
                accum += __half2float(a_vals1.y) * __half2float(b_vals1.y);
            } else {
                // Handle remainder
                for (int r = row; r < N && r < row + 4; r++) {
                    accum += __half2float(shmem_A[r]) * __half2float(B[r * M + col]);
                }
            }
        }
        
        // Apply scaling
        C[col] = __float2half(accum * __half2float(scales[col]));
    }
}

// Warp-specialized version
__global__ void warp_specialized_gemv_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ scales,
    half* __restrict__ C,
    int N, int M
) {
    extern __shared__ half shmem_A[];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp 0: Load A into shared memory
    if (warp_id == 0) {
        for (int i = lane_id; i < N; i += WARP_SIZE) {
            shmem_A[i] = A[i];
        }
    }
    __syncthreads();
    
    // All warps: Compute
    if (col < M) {
        float accum = 0.0f;
        
        // Use warp shuffle for partial reduction
        for (int row_block = 0; row_block < N; row_block += WARP_SIZE) {
            float partial_sum = 0.0f;
            
            if (row_block + lane_id < N) {
                partial_sum = __half2float(shmem_A[row_block + lane_id]) * 
                             __half2float(B[(row_block + lane_id) * M + col]);
            }
            
            // Warp-level reduction
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
            }
            
            if (lane_id == 0) {
                accum += partial_sum;
            }
        }
        
        if (lane_id == 0) {
            C[col] = __float2half(accum * __half2float(scales[col]));
        }
    }
}

// Benchmark function
double benchmark_kernel(
    void (*kernel)(const half*, const half*, const half*, half*, int, int),
    const char* name,
    const half* d_A, const half* d_B, const half* d_scales, half* d_C,
    int N, int M, size_t shmem_size = 0, int iterations = 1000
) {
    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_scales, d_C, N, M);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block, shmem_size>>>(d_A, d_B, d_scales, d_C, N, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    double avg_time = elapsed / iterations;
    
    double gflops = (2.0 * N * M) / (avg_time / 1000.0) / 1e9;
    
    printf("%-25s: %8.3f ms, %8.1f GFLOPS\n", name, avg_time, gflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return gflops;
}

int main() {
    printf("🚀 Simple GEMV Kernel Optimization Benchmark\n");
    printf("============================================\n\n");
    
    // Test configuration - our target shape
    const int N = 3584;
    const int M = 18944;
    
    printf("Shape: 1×%d @ %d×%d (%.2f M parameters)\n\n", N, N, M, (N * M) / 1e6);
    
    // Allocate host memory
    half* h_A = new half[N];
    half* h_B = new half[N * M];
    half* h_scales = new half[M];
    half* h_C_simple = new half[M];
    half* h_C_optimized = new half[M];
    half* h_C_warp = new half[M];
    half* h_C_cublas = new half[M];
    
    // Initialize data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    
    for (int i = 0; i < N; i++) {
        h_A[i] = __float2half(dis(gen));
    }
    
    for (int i = 0; i < N * M; i++) {
        h_B[i] = __float2half(dis(gen));
    }
    
    for (int i = 0; i < M; i++) {
        h_scales[i] = __float2half(1.0f); // No scaling for simplicity
    }
    
    // Allocate device memory
    half *d_A, *d_B, *d_scales, *d_C;
    cudaMalloc(&d_A, N * sizeof(half));
    cudaMalloc(&d_B, N * M * sizeof(half));
    cudaMalloc(&d_scales, M * sizeof(half));
    cudaMalloc(&d_C, M * sizeof(half));
    
    cudaMemcpy(d_A, h_A, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * M * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, M * sizeof(half), cudaMemcpyHostToDevice);
    
    printf("📊 Performance Results:\n");
    printf("========================\n");
    
    // Benchmark kernels
    double gflops_simple = benchmark_kernel(
        simple_gemv_w4a16f_kernel, "Simple GEMV",
        d_A, d_B, d_scales, d_C, N, M
    );
    
    double gflops_optimized = benchmark_kernel(
        optimized_gemv_w4a16f_kernel, "Optimized GEMV", 
        d_A, d_B, d_scales, d_C, N, M, N * sizeof(half)
    );
    
    double gflops_warp = benchmark_kernel(
        warp_specialized_gemv_kernel, "Warp Specialized GEMV",
        d_A, d_B, d_scales, d_C, N, M, N * sizeof(half)
    );
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N, &alpha,
                     d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, N, &beta,
                     d_C, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    
    // Benchmark cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N, &alpha,
                     d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, N, &beta,
                     d_C, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_cublas;
    cudaEventElapsedTime(&elapsed_cublas, start, stop);
    double avg_time_cublas = elapsed_cublas / 1000.0;
    double gflops_cublas = (2.0 * N * M) / (avg_time_cublas / 1000.0) / 1e9;
    
    printf("%-25s: %8.3f ms, %8.1f GFLOPS\n", "cuBLAS Reference", avg_time_cublas, gflops_cublas);
    
    printf("\n🏆 Performance Analysis:\n");
    printf("========================\n");
    double best_custom = std::max(gflops_simple, std::max(gflops_optimized, gflops_warp));
    printf("Best kernel GFLOPS: %.1f\n", best_custom);
    printf("cuBLAS GFLOPS:      %.1f\n", gflops_cublas);
    if (best_custom > gflops_cublas) {
        printf("🎉 Our best kernel is %.1f%% faster than cuBLAS!\n", 
               (best_custom / gflops_cublas - 1.0) * 100);
    } else {
        printf("📈 cuBLAS is %.1f%% faster. Room for optimization!\n", 
               (gflops_cublas / best_custom - 1.0) * 100);
    }
    
    // Memory bandwidth analysis
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double theoretical_bandwidth = prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8.0 / 1e6; // GB/s
    
    double bytes_per_op = N * sizeof(half) + N * M * sizeof(half) + M * sizeof(half) + M * sizeof(half);
    double best_bandwidth = bytes_per_op * (best_custom * 1e9 / (2.0 * N * M)) / 1e9;
    
    printf("\n💾 Memory Analysis:\n");
    printf("===================\n");
    printf("Theoretical Bandwidth: %.1f GB/s\n", theoretical_bandwidth);
    printf("Achieved Bandwidth:    %.1f GB/s (%.1f%% of peak)\n", 
           best_bandwidth, 100.0 * best_bandwidth / theoretical_bandwidth);
    
    printf("\n🔍 For detailed profiling, run:\n");
    printf("ncu --set full ./simple_gemv_benchmark\n");
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scales);
    cudaFree(d_C);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_scales;
    delete[] h_C_simple;
    delete[] h_C_optimized;
    delete[] h_C_warp;
    delete[] h_C_cublas;
    
    printf("\n✨ Benchmark complete!\n");
    return 0;
}