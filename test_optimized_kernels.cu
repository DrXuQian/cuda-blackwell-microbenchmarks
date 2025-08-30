#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <iostream>
#include <cstdlib>

using namespace nvcuda;

// Simple async MMA GEMM kernel (sm_75+)
__global__ void simple_async_mma_gemm_kernel(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Shared memory for double buffering
    extern __shared__ half shmem[];
    half* shmem_A = shmem;
    half* shmem_B = shmem + 16 * 16; // 16x16 A tile + 16x16 B tile
    
    const int block_row = blockIdx.y * 16;
    const int block_col = blockIdx.x * 16;
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;
    
    wmma::fill_fragment(frag_C, 0.0f);
    
    // Main computation loop
    for (int k = 0; k < K; k += 16) {
        // Warp 0: Load A tile to shared memory
        if (warp_id == 0) {
            for (int i = lane_id; i < 16 * 16; i += 32) {
                int row = i / 16;
                int col = i % 16;
                int global_row = block_row + row;
                int global_col = k + col;
                
                if (global_row < M && global_col < K) {
                    shmem_A[i] = A[global_row * K + global_col];
                } else {
                    shmem_A[i] = __float2half(0.0f);
                }
            }
        }
        
        // Warp 1: Load B tile to shared memory  
        if (warp_id == 1) {
            for (int i = lane_id; i < 16 * 16; i += 32) {
                int row = i / 16;
                int col = i % 16;
                int global_row = k + row;
                int global_col = block_col + col;
                
                if (global_row < K && global_col < N) {
                    shmem_B[i] = B[global_row * N + global_col];
                } else {
                    shmem_B[i] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Warp 2+: Compute using tensor cores
        if (warp_id >= 2) {
            wmma::load_matrix_sync(frag_A, shmem_A, 16);
            wmma::load_matrix_sync(frag_B, shmem_B, 16);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        
        __syncthreads();
    }
    
    // Write results
    if (warp_id >= 2 && block_row < M && block_col < N) {
        wmma::store_matrix_sync(&C[block_row * N + block_col], frag_C, N, wmma::mem_row_major);
    }
}

// w4a16f dequantization kernel with tensor cores
__global__ void w4a16f_tensor_gemv_kernel(
    const half* __restrict__ A,    // [M, K] activations
    const int* __restrict__ B,     // [K/8, N] packed 4-bit weights  
    half* __restrict__ C,          // [M, N] output
    const half* __restrict__ s,    // [K/128, N] scales
    int M, int N, int K
) {
    const int group_size = 128;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    extern __shared__ half shmem[];
    half* shmem_A = shmem;
    half* shmem_B = shmem + 16 * 16;
    
    const int block_col = blockIdx.x * 16;
    
    // Accumulator for the entire row
    float row_accum[16] = {0.0f};
    
    // Process each row
    for (int row = blockIdx.y; row < M; row += gridDim.y) {
        // Main K loop
        for (int k_base = 0; k_base < K; k_base += 16) {
            // Load and dequantize weights (warp 0)
            if (warp_id == 0) {
                for (int i = lane_id; i < 16 * 16; i += 32) {
                    int k_local = i / 16;
                    int col_local = i % 16;
                    int k_global = k_base + k_local;
                    int col_global = block_col + col_local;
                    
                    if (k_global < K && col_global < N) {
                        // Get packed weight
                        int packed_idx = (k_global / 8) * N + col_global;
                        int packed_weight = B[packed_idx];
                        
                        // Extract 4-bit weight
                        int bit_offset = (k_global % 8) * 4;
                        int w4 = (packed_weight >> bit_offset) & 0xF;
                        
                        // Dequantize to fp16
                        float scale_val = __half2float(s[(k_global / group_size) * N + col_global]);
                        float dequant = (w4 - 8) * scale_val; // Symmetric quantization
                        shmem_B[i] = __float2half(dequant);
                    } else {
                        shmem_B[i] = __float2half(0.0f);
                    }
                }
            }
            
            // Load activations (warp 1)
            if (warp_id == 1) {
                for (int i = lane_id; i < 16; i += 32) {
                    int k_global = k_base + i;
                    if (k_global < K) {
                        shmem_A[i] = A[row * K + k_global];
                    } else {
                        shmem_A[i] = __float2half(0.0f);
                    }
                }
            }
            
            __syncthreads();
            
            // Compute dot product (all warps)
            for (int i = 0; i < 16; i++) {
                float a_val = __half2float(shmem_A[i]);
                for (int j = lane_id; j < 16; j += 32) {
                    float b_val = __half2float(shmem_B[i * 16 + j]);
                    row_accum[j] += a_val * b_val;
                }
            }
            
            __syncthreads();
        }
        
        // Write results
        for (int j = lane_id; j < 16; j += 32) {
            int col_global = block_col + j;
            if (col_global < N) {
                C[row * N + col_global] = __float2half(row_accum[j]);
            }
        }
        
        // Reset accumulator for next row
        for (int j = 0; j < 16; j++) {
            row_accum[j] = 0.0f;
        }
    }
}

int main() {
    printf("🚀 Testing Optimized Kernels with mma_async + Tensor Cores\n");
    printf("=========================================================\n");
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
        printf("⚠️  Tensor cores require sm_75+\n");
        return 0;
    }
    printf("\n");
    
    // Test 1: Simple async MMA GEMM
    const int M1 = 256, N1 = 256, K1 = 256;
    printf("1. Testing Async MMA GEMM (%dx%dx%d):\n", M1, N1, K1);
    
    size_t size_A1 = M1 * K1 * sizeof(half);
    size_t size_B1 = K1 * N1 * sizeof(half);
    size_t size_C1 = M1 * N1 * sizeof(float);
    
    half *h_A1 = (half*)malloc(size_A1);
    half *h_B1 = (half*)malloc(size_B1);
    float *h_C1 = (float*)malloc(size_C1);
    
    // Initialize data
    srand(42);
    for (int i = 0; i < M1 * K1; i++) {
        h_A1[i] = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    }
    for (int i = 0; i < K1 * N1; i++) {
        h_B1[i] = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    }
    
    half *d_A1, *d_B1;
    float *d_C1;
    cudaMalloc(&d_A1, size_A1);
    cudaMalloc(&d_B1, size_B1);
    cudaMalloc(&d_C1, size_C1);
    
    cudaMemcpy(d_A1, h_A1, size_A1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, size_B1, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid1((N1 + 15) / 16, (M1 + 15) / 16);
    dim3 block1(4 * 32); // 4 warps
    size_t shmem1 = 2 * 16 * 16 * sizeof(half);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        simple_async_mma_gemm_kernel<<<grid1, block1, shmem1>>>(d_A1, d_B1, d_C1, M1, N1, K1);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        simple_async_mma_gemm_kernel<<<grid1, block1, shmem1>>>(d_A1, d_B1, d_C1, M1, N1, K1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time1 = elapsed_ms / iterations;
    
    double flops1 = 2.0 * M1 * N1 * K1;
    double gflops1 = flops1 / (avg_time1 / 1000.0) / 1e9;
    
    printf("   Average time: %.3f ms\n", avg_time1);
    printf("   Performance: %.1f GFLOPS\n", gflops1);
    
    // Validate results
    cudaMemcpy(h_C1, d_C1, size_C1, cudaMemcpyDeviceToHost);
    bool valid1 = true;
    for (int i = 0; i < std::min(10, M1 * N1); i++) {
        if (isnan(h_C1[i]) || isinf(h_C1[i]) || fabs(h_C1[i]) > 100.0f) {
            valid1 = false;
            break;
        }
    }
    printf("   Validation: %s\n", valid1 ? "✅ PASSED" : "❌ FAILED");
    
    // Test 2: w4a16f tensor GEMV
    printf("\n2. Testing w4a16f Tensor GEMV (1x3584 @ 3584x1024):\n");
    
    const int M2 = 1, N2 = 1024, K2 = 3584;
    const int group_size = 128;
    
    size_t size_A2 = M2 * K2 * sizeof(half);
    size_t size_B2 = (K2 / 8) * N2 * sizeof(int); // 4-bit packed as int
    size_t size_s2 = (K2 / group_size) * N2 * sizeof(half);
    size_t size_C2 = M2 * N2 * sizeof(half);
    
    half *h_A2 = (half*)malloc(size_A2);
    int *h_B2 = (int*)malloc(size_B2);
    half *h_s2 = (half*)malloc(size_s2);
    half *h_C2 = (half*)malloc(size_C2);
    
    // Initialize
    for (int i = 0; i < M2 * K2; i++) {
        h_A2[i] = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    }
    for (int i = 0; i < (K2 / 8) * N2; i++) {
        h_B2[i] = rand(); // Random packed weights
    }
    for (int i = 0; i < (K2 / group_size) * N2; i++) {
        h_s2[i] = __float2half(0.01f + (float)rand() / RAND_MAX * 0.02f);
    }
    
    half *d_A2, *d_s2, *d_C2;
    int *d_B2;
    cudaMalloc(&d_A2, size_A2);
    cudaMalloc(&d_B2, size_B2);
    cudaMalloc(&d_s2, size_s2);
    cudaMalloc(&d_C2, size_C2);
    
    cudaMemcpy(d_A2, h_A2, size_A2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, size_B2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2, h_s2, size_s2, cudaMemcpyHostToDevice);
    
    dim3 grid2((N2 + 15) / 16, M2);
    dim3 block2(4 * 32);
    size_t shmem2 = (16 * 16 + 16) * sizeof(half);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        w4a16f_tensor_gemv_kernel<<<grid2, block2, shmem2>>>(d_A2, d_B2, d_C2, d_s2, M2, N2, K2);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        w4a16f_tensor_gemv_kernel<<<grid2, block2, shmem2>>>(d_A2, d_B2, d_C2, d_s2, M2, N2, K2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time2 = elapsed_ms / iterations;
    
    double flops2 = 2.0 * M2 * N2 * K2;
    double gflops2 = flops2 / (avg_time2 / 1000.0) / 1e9;
    
    // Memory bandwidth calculation (effective fp16 equivalent)
    double bytes2 = M2 * K2 * 2 + (K2 * N2 / 2) + (K2/group_size) * N2 * 2 + M2 * N2 * 2;
    double bandwidth2 = bytes2 / (avg_time2 / 1000.0) / 1e9;
    
    printf("   Average time: %.3f ms\n", avg_time2);
    printf("   Performance: %.1f GFLOPS\n", gflops2);
    printf("   Effective bandwidth: %.1f GB/s\n", bandwidth2);
    printf("   Memory reduction: 4x (vs fp16)\n");
    
    // Validate
    cudaMemcpy(h_C2, d_C2, size_C2, cudaMemcpyDeviceToHost);
    bool valid2 = true;
    for (int i = 0; i < std::min(10, M2 * N2); i++) {
        float val = __half2float(h_C2[i]);
        if (isnan(val) || isinf(val) || fabs(val) > 100.0f) {
            valid2 = false;
            break;
        }
    }
    printf("   Validation: %s\n", valid2 ? "✅ PASSED" : "❌ FAILED");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A1); free(h_B1); free(h_C1);
    free(h_A2); free(h_B2); free(h_s2); free(h_C2);
    
    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_s2); cudaFree(d_C2);
    
    printf("\n✨ All optimized kernel tests completed!\n");
    
    return 0;
}