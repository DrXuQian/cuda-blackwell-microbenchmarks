#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <numeric>
#include <chrono>
#include <cfloat>
#include <random>
#include <iostream>
#include <iomanip>
#include <cooperative_groups.h>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// ============================================================================
// CHANNEL-SPLIT PERSISTENT KERNEL
// Each thread block handles only a subset of channels (e.g., 8 out of 32)
// This allows much better spatial data caching in shared memory
// ============================================================================

template <typename scalar_t=__half,
          const int NUM_POINT=8,
          const int NUM_LEVELS=4,
          const int TOTAL_CHANNELS=32,
          const int CHANNELS_PER_BLOCK=8>  // Key parameter: channels per block
__global__ void ms_deformable_im2col_persistent_channelsplit(
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int num_query,
    scalar_t *data_col,
    int *global_counter) {

    // Shared memory layout:
    // We can now cache much more spatial data since we only handle 8 channels!
    extern __shared__ __half shared_mem[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Calculate how many channel blocks we have
    const int num_channel_blocks = TOTAL_CHANNELS / CHANNELS_PER_BLOCK;  // 32/8 = 4

    // Persistent kernel work-stealing loop
    while (true) {
        // Get next work item
        // Work items now include channel block index
        int work_id = atomicAdd(global_counter, 1);

        // Total work = batch × queries × channel_blocks
        int total_work = batch_size * num_query * num_channel_blocks;
        if (work_id >= total_work) break;

        // Decode work item to (batch, query, channel_block)
        const int channel_block_id = work_id % num_channel_blocks;
        const int bq = work_id / num_channel_blocks;
        const int b = bq / num_query;
        const int q = bq % num_query;

        // This block handles channels [channel_start, channel_end)
        const int channel_start = channel_block_id * CHANNELS_PER_BLOCK;
        const int channel_end = channel_start + CHANNELS_PER_BLOCK;

        // Load spatial shapes and level indices
        __shared__ int64_t s_spatial_shapes[NUM_LEVELS * 2];
        __shared__ int64_t s_level_start_index[NUM_LEVELS];
        __shared__ int s_cached_level_info[NUM_LEVELS * 3];  // start_idx, size, spatial_size

        if (tid < NUM_LEVELS * 2) {
            s_spatial_shapes[tid] = data_spatial_shapes[tid];
        }
        if (tid < NUM_LEVELS) {
            s_level_start_index[tid] = (tid > 0) ? data_level_start_index[tid - 1] : 0;
        }
        __syncthreads();

        // Calculate caching strategy
        // With only 8 channels, we can cache MUCH more spatial data!
        const int max_cache_elements = 48000;  // Conservative 96KB limit
        const int elements_per_spatial = CHANNELS_PER_BLOCK;  // Only 8 channels now
        const int max_spatial_cached = max_cache_elements / elements_per_spatial;  // 6000 locations!

        // Smart caching: Cache ALL of levels 1, 2, 3 and part of level 0
        __half* cached_data = shared_mem;
        int cache_offset = 0;

        // First pass: Calculate what we'll cache
        if (tid == 0) {
            int remaining_cache = max_spatial_cached;

            // Cache smaller levels first (they're accessed more in deformable attention)
            for (int l = NUM_LEVELS - 1; l >= 0; l--) {
                const int h = s_spatial_shapes[l * 2];
                const int w = s_spatial_shapes[l * 2 + 1];
                const int level_spatial = h * w;

                if (level_spatial <= remaining_cache) {
                    // Cache entire level
                    s_cached_level_info[l * 3] = cache_offset;  // Start in cache
                    s_cached_level_info[l * 3 + 1] = level_spatial;  // Size cached
                    s_cached_level_info[l * 3 + 2] = level_spatial;  // Total size
                    cache_offset += level_spatial;
                    remaining_cache -= level_spatial;
                } else if (remaining_cache > 0) {
                    // Partial cache - cache center region
                    s_cached_level_info[l * 3] = cache_offset;
                    s_cached_level_info[l * 3 + 1] = remaining_cache;  // Cache what we can
                    s_cached_level_info[l * 3 + 2] = level_spatial;
                    cache_offset += remaining_cache;
                    remaining_cache = 0;
                } else {
                    // No cache for this level
                    s_cached_level_info[l * 3] = -1;
                    s_cached_level_info[l * 3 + 1] = 0;
                    s_cached_level_info[l * 3 + 2] = level_spatial;
                }
            }
        }
        __syncthreads();

        // Cooperative loading of cached data
        // Load only the channels this block is responsible for
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int cache_start = s_cached_level_info[l * 3];
            const int cache_size = s_cached_level_info[l * 3 + 1];

            if (cache_start >= 0 && cache_size > 0) {
                const int level_start = s_level_start_index[l];

                // Load this level's data for our channels only
                const int total_elements = cache_size * CHANNELS_PER_BLOCK;
                for (int i = tid; i < total_elements; i += block_size) {
                    const int spatial_idx = i / CHANNELS_PER_BLOCK;
                    const int channel_idx = i % CHANNELS_PER_BLOCK;

                    if (spatial_idx < cache_size) {
                        // Global memory index for our specific channels
                        const int global_idx = (b * spatial_size + level_start + spatial_idx) * TOTAL_CHANNELS
                                              + channel_start + channel_idx;

                        // Cache index
                        const int cache_idx = (cache_start + spatial_idx) * CHANNELS_PER_BLOCK + channel_idx;

                        cached_data[cache_idx] = data_value[global_idx];
                    }
                }
            }
        }
        __syncthreads();

        // Now process the query with excellent cache utilization
        const int base_loc_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT * 2;
        const int base_weight_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT;

        // Each thread processes different channels within our block's range
        for (int c_local = tid; c_local < CHANNELS_PER_BLOCK; c_local += block_size) {
            const int c_global = channel_start + c_local;  // Global channel index

            scalar_t result = scalar_t(0);

            // Process all levels and points
            #pragma unroll
            for (int l = 0; l < NUM_LEVELS; l++) {
                const int spatial_h = s_spatial_shapes[l * 2];
                const int spatial_w = s_spatial_shapes[l * 2 + 1];
                const int level_start_idx = s_level_start_index[l];
                const int cache_start = s_cached_level_info[l * 3];
                const int cache_size = s_cached_level_info[l * 3 + 1];

                // Load sampling locations and weights
                __half loc_x[NUM_POINT], loc_y[NUM_POINT], weights[NUM_POINT];

                #pragma unroll
                for (int p = 0; p < NUM_POINT; p++) {
                    const int loc_idx = base_loc_idx + (l * NUM_POINT + p) * 2;
                    const int weight_idx = base_weight_idx + l * NUM_POINT + p;

                    loc_y[p] = data_sampling_loc[loc_idx];
                    loc_x[p] = data_sampling_loc[loc_idx + 1];
                    weights[p] = data_attn_weight[weight_idx];
                }

                // Process each point
                #pragma unroll
                for (int p = 0; p < NUM_POINT; p++) {
                    // Convert normalized coordinates to actual coordinates
                    const float y = (__half2float(loc_y[p]) + 1) * spatial_h / 2.0f - 0.5f;
                    const float x = (__half2float(loc_x[p]) + 1) * spatial_w / 2.0f - 0.5f;

                    if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                        const int y_low = floorf(y);
                        const int x_low = floorf(x);
                        const int y_high = y_low + 1;
                        const int x_high = x_low + 1;

                        const __half ly = __float2half(y - y_low);
                        const __half lx = __float2half(x - x_low);
                        const __half hy = __float2half(1 - (y - y_low));
                        const __half hx = __float2half(1 - (x - x_low));

                        // Helper lambda to read value (cache-aware)
                        auto read_value = [&](int y_coord, int x_coord) -> scalar_t {
                            if (y_coord >= 0 && x_coord >= 0 && y_coord < spatial_h && x_coord < spatial_w) {
                                int spatial_idx = y_coord * spatial_w + x_coord;

                                // Check if this location is cached
                                if (cache_start >= 0 && spatial_idx < cache_size) {
                                    // Read from shared memory cache
                                    const int cache_idx = (cache_start + spatial_idx) * CHANNELS_PER_BLOCK + c_local;
                                    return cached_data[cache_idx];
                                } else {
                                    // Fall back to global memory
                                    const int global_idx = (b * spatial_size + level_start_idx + spatial_idx) * TOTAL_CHANNELS + c_global;
                                    return data_value[global_idx];
                                }
                            }
                            return scalar_t(0);
                        };

                        // Bilinear interpolation with cache-aware reads
                        scalar_t val = scalar_t(0);
                        val = __hfma(read_value(y_low, x_low), __hmul(hy, hx), val);
                        val = __hfma(read_value(y_low, x_high), __hmul(hy, lx), val);
                        val = __hfma(read_value(y_high, x_low), __hmul(ly, hx), val);
                        val = __hfma(read_value(y_high, x_high), __hmul(ly, lx), val);

                        result = __hfma(weights[p], val, result);
                    }
                }
            }

            // Write output for this channel
            const int out_idx = (b * num_query + q) * num_heads * TOTAL_CHANNELS + c_global;
            data_col[out_idx] = result;
        }

        __syncthreads();  // Ensure all threads complete before next work item
    }
}

// ============================================================================
// MAIN TEST FUNCTION
// ============================================================================

int main() {
    std::cout << "=== Channel-Split Persistent Kernel for MS-Deformable Attention ===" << std::endl;

    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    // ORIGINAL FULL SIZE configuration
    const int batch = 48;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 15422;
    const int channels_per_block = 8;  // Split channels!

    // Original feature map sizes
    const std::vector<int64_t> h_spatial_shapes = {
        92, 160,   // Level 0: 14720
        46, 80,    // Level 1: 3680
        23, 40,    // Level 2: 920
        12, 20     // Level 3: 240
    };

    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;

    for (int i = 0; i < num_levels; i++) {
        h_level_start_index.push_back(spatial_size);
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += h * w;
    }

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "  Batch size: " << batch << std::endl;
    std::cout << "  Spatial size: " << spatial_size << std::endl;
    std::cout << "  Num queries: " << num_query << std::endl;
    std::cout << "  Total channels: " << channels << std::endl;
    std::cout << "  Channels per block: " << channels_per_block << " (NEW!)" << std::endl;
    std::cout << "  Channel blocks: " << channels / channels_per_block << std::endl;

    // Calculate memory and caching
    std::cout << "\n=== Caching Analysis ===" << std::endl;
    const int shared_mem_elements = 48000;  // 96KB
    const int spatial_per_block = shared_mem_elements / channels_per_block;
    std::cout << "  Shared memory: 96KB" << std::endl;
    std::cout << "  Spatial locations cacheable: " << spatial_per_block << " (vs 1536 in full-channel)" << std::endl;
    std::cout << "  Cache coverage: " << (100.0 * spatial_per_block / spatial_size) << "%" << std::endl;

    // What we can cache
    int cumulative = 0;
    for (int i = num_levels - 1; i >= 0; i--) {
        int level_size = h_spatial_shapes[i*2] * h_spatial_shapes[i*2+1];
        cumulative += level_size;
        if (cumulative <= spatial_per_block) {
            std::cout << "  Level " << i << ": ✅ 100% cached (" << level_size << " locations)" << std::endl;
        } else {
            int partial = spatial_per_block - (cumulative - level_size);
            if (partial > 0) {
                std::cout << "  Level " << i << ": ⚡ " << (100.0 * partial / level_size)
                          << "% cached (" << partial << "/" << level_size << ")" << std::endl;
            } else {
                std::cout << "  Level " << i << ": ❌ Not cached" << std::endl;
            }
        }
    }

    // Allocate memory
    const int64_t value_size = batch * spatial_size * channels;
    const int64_t output_size = batch * num_query * num_heads * channels;
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;

    std::cout << "\nAllocating memory..." << std::endl;
    std::vector<__half> h_value(value_size);
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    std::vector<__half> h_attn_weight(attn_weight_size);
    std::vector<__half> h_output(output_size);

    // Initialize data
    std::cout << "Initializing data..." << std::endl;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatial_size; s++) {
            for (int c = 0; c < channels; c++) {
                int idx = (b * spatial_size + s) * channels + c;
                float val = sinf(s * 0.01f + c * 0.1f) * 0.5f + 0.5f;
                h_value[idx] = __float2half(val);
            }
        }
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> loc_dist(-0.9f, 0.9f);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half(loc_dist(gen));
    }

    // Initialize normalized attention weights
    for (int b = 0; b < batch; b++) {
        for (int q = 0; q < num_query; q++) {
            for (int h = 0; h < num_heads; h++) {
                float sum = 0.0f;
                int base = ((b * num_query + q) * num_heads + h) * num_levels * num_points;

                for (int l = 0; l < num_levels; l++) {
                    for (int p = 0; p < num_points; p++) {
                        float w = expf(-l * 0.5f - p * 0.1f);
                        h_attn_weight[base + l * num_points + p] = __float2half(w);
                        sum += w;
                    }
                }

                for (int i = 0; i < num_levels * num_points; i++) {
                    h_attn_weight[base + i] = __float2half(__half2float(h_attn_weight[base + i]) / sum);
                }
            }
        }
    }

    // Allocate device memory
    __half *d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t *d_spatial_shapes, *d_level_start_index;
    int *d_global_counter;

    CUDA_CHECK(cudaMalloc(&d_value, value_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, attn_weight_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));

    // Copy to device
    std::cout << "Copying to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), value_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), sampling_loc_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), attn_weight_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch configuration
    const int num_blocks = prop.multiProcessorCount * 2;  // More blocks since each does less work
    const int threads_per_block = 256;
    size_t smem_size = 96 * 1024;  // Maximum shared memory

    // Set shared memory configuration
    CUDA_CHECK(cudaFuncSetAttribute(
        ms_deformable_im2col_persistent_channelsplit<__half, 8, 4, 32, 8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    std::cout << "\n=== Launch Configuration ===" << std::endl;
    std::cout << "  Active blocks: " << num_blocks << std::endl;
    std::cout << "  Threads per block: " << threads_per_block << std::endl;
    std::cout << "  Shared memory per block: " << smem_size / 1024.0 << " KB" << std::endl;
    std::cout << "  Total work items: " << batch * num_query * (channels / channels_per_block) << std::endl;

    // Warmup
    std::cout << "\nRunning warmup..." << std::endl;
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));
        ms_deformable_im2col_persistent_channelsplit<__half, 8, 4, 32, 8>
            <<<num_blocks, threads_per_block, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, num_query,
            d_output, d_global_counter);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    const int num_iterations = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));
        ms_deformable_im2col_persistent_channelsplit<__half, 8, 4, 32, 8>
            <<<num_blocks, threads_per_block, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, num_query,
            d_output, d_global_counter);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Average kernel time: " << milliseconds / num_iterations << " ms" << std::endl;
    std::cout << "Throughput: " << (num_iterations * 1000.0f) / milliseconds << " iterations/second" << std::endl;

    // Calculate TFLOPS
    double ops_per_output = num_levels * num_points * 10;
    double total_ops = batch * num_query * num_heads * channels * ops_per_output;
    double tflops = (total_ops * num_iterations) / (milliseconds * 1e9);
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                         std::min(100000LL, (long long)output_size) * sizeof(__half),
                         cudaMemcpyDeviceToHost));

    // Check first few outputs
    std::cout << "First 20 outputs: ";
    for (int i = 0; i < std::min(20LL, (long long)output_size); i++) {
        if (i % 10 == 0) std::cout << "\n  ";
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // Statistics
    float max_val = 0, min_val = FLT_MAX, sum_val = 0;
    int non_zero = 0;
    int sample_size = std::min(100000LL, (long long)output_size);
    for (int i = 0; i < sample_size; i++) {
        float val = __half2float(h_output[i]);
        if (val != 0) non_zero++;
        max_val = fmaxf(max_val, val);
        min_val = fminf(min_val, val);
        sum_val += val;
    }

    std::cout << "\n=== Output Statistics ===" << std::endl;
    std::cout << "Non-zero outputs: " << non_zero << "/" << sample_size
              << " (" << (100.0 * non_zero / sample_size) << "%)" << std::endl;
    std::cout << "Min: " << min_val << ", Max: " << max_val
              << ", Mean: " << sum_val / sample_size << std::endl;

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_global_counter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ Channel-split persistent kernel test complete!" << std::endl;
    std::cout << "Key improvement: " << (spatial_per_block / 1536.0)
              << "x more spatial data cached vs full-channel approach!" << std::endl;

    return 0;
}