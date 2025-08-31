#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Blackwell RTX 5070 specific constants
#define BLACKWELL_SM_COUNT 56  // RTX 5070 SM count
#define BLACKWELL_WARP_SIZE 32
#define BLACKWELL_MAX_THREADS_PER_BLOCK 1024
#define BLACKWELL_SHARED_MEM_PER_BLOCK (164 * 1024)  // 164KB shared memory

// TMA specific constants for Blackwell
#define TMA_TILE_M 128
#define TMA_TILE_N 128
#define TMA_TILE_K 64

// WGMMA specific constants
#define WGMMA_M 64
#define WGMMA_N 64
#define WGMMA_K 32

// Blackwell TMA descriptor structure
typedef struct {
    CUtensorMap tma_map;
    bool initialized;
    size_t bytes_transferred;
} BlackwellTMADescriptor;

// Performance measurement utilities
typedef struct {
    cudaEvent_t start, stop;
    float* times;
    int capacity;
    int count;
} BlackwellTimer;

// Initialize timer
inline void blackwell_timer_init(BlackwellTimer* timer, int capacity) {
    timer->capacity = capacity;
    timer->count = 0;
    timer->times = (float*)malloc(capacity * sizeof(float));
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
}

// Start timing
inline void blackwell_timer_start(BlackwellTimer* timer) {
    cudaEventRecord(timer->start);
}

// Stop timing and record
inline void blackwell_timer_stop(BlackwellTimer* timer) {
    cudaEventRecord(timer->stop);
    cudaEventSynchronize(timer->stop);
    
    if (timer->count < timer->capacity) {
        cudaEventElapsedTime(&timer->times[timer->count], timer->start, timer->stop);
        timer->count++;
    }
}

// Get average time
inline float blackwell_timer_get_avg(BlackwellTimer* timer) {
    if (timer->count == 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < timer->count; i++) {
        sum += timer->times[i];
    }
    return sum / timer->count;
}

// Cleanup timer
inline void blackwell_timer_cleanup(BlackwellTimer* timer) {
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    free(timer->times);
}

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult error = call; \
    if (error != CUDA_SUCCESS) { \
        const char* str; \
        cuGetErrorString(error, &str); \
        printf("CUDA Driver error at %s:%d - %s\n", __FILE__, __LINE__, str); \
        exit(1); \
    } \
} while(0)

// Device capability check for Blackwell
inline bool check_blackwell_support() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // Blackwell is sm_90+
    if (prop.major < 9) {
        printf("❌ This requires sm_90+ (Blackwell architecture)\n");
        printf("   RTX 5070 should support sm_90+\n");
        printf("   Current device: sm_%d%d\n", prop.major, prop.minor);
        return false;
    }
    
    printf("✅ Blackwell RTX 5070 detected with sm_90+ support!\n");
    return true;
}

// Calculate TFLOPS
inline double calculate_tflops(int M, int N, int K, float time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms / 1000.0) / 1e12;
}

// Memory bandwidth calculation
inline double calculate_bandwidth_gb_s(size_t bytes, float time_ms) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}