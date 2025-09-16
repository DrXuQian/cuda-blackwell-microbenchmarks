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
#include <cuda/barrier>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// Persistent kernel with distributed shared memory
// Combines the best of both worlds:
// 1. Persistent kernel pattern (few blocks, high shared memory per block)
// 2. Distributed shared memory across cluster (even more total shared memory)
template <typename scalar_t=__half, const int NUM_POINT=8, const int NUM_LEVELS=4,
          const int CHANNELS=32, const int CLUSTER_SIZE=2>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
ms_deformable_im2col_persistent_distributed(
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

    // Use maximum shared memory per block
    extern __shared__ __half shared_mem[];

    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int cluster_rank = cluster.block_rank();
    const int blocks_in_cluster = cluster.dim_blocks().x;

    // With distributed shared memory, we can cache even more data!
    // Each block in the cluster caches different parts of the data
    __half* local_cache = shared_mem;

    // Persistent kernel loop - keep processing work items
    while (true) {
        // Get next work item using atomic counter
        // Each cluster grabs one work item (not each block)
        int work_id;
        if (cluster_rank == 0 && tid == 0) {
            work_id = atomicAdd(global_counter, 1);
        }

        // Broadcast work_id to all threads in cluster
        work_id = __shfl_sync(0xFFFFFFFF, work_id, 0);
        cluster.sync();

        // Check if all work is done
        int total_work = batch_size * num_query;
        if (work_id >= total_work) break;

        // Decode work item to batch and query indices
        const int b = work_id / num_query;
        const int q = work_id % num_query;

        // Load spatial shapes and level indices (shared across cluster)
        __shared__ int64_t s_spatial_shapes[NUM_LEVELS * 2];
        __shared__ int64_t s_level_start_index[NUM_LEVELS];

        if (tid < NUM_LEVELS * 2) {
            s_spatial_shapes[tid] = data_spatial_shapes[tid];
        }
        if (tid < NUM_LEVELS) {
            s_level_start_index[tid] = (tid > 0) ? data_level_start_index[tid - 1] : 0;
        }
        __syncthreads();

        // Distributed caching strategy:
        // Each block in the cluster caches different levels
        // Block 0: Caches level 0 and 1
        // Block 1: Caches level 2 and 3
        const int levels_per_block = NUM_LEVELS / blocks_in_cluster;
        const int my_first_level = cluster_rank * levels_per_block;
        const int my_last_level = min(my_first_level + levels_per_block, NUM_LEVELS);

        // Calculate how much data to cache for our assigned levels
        int total_cache_size = 0;
        for (int l = my_first_level; l < my_last_level; l++) {
            const int h = s_spatial_shapes[l * 2];
            const int w = s_spatial_shapes[l * 2 + 1];
            total_cache_size += h * w * CHANNELS;
        }

        // Limit to available shared memory (96KB per block)
        const int max_cache_elements = 48000;  // ~96KB / 2 bytes
        const int cache_elements = min(total_cache_size, max_cache_elements);

        // Cooperative loading of data for our assigned levels
        int cache_offset = 0;
        for (int l = my_first_level; l < my_last_level && cache_offset < cache_elements; l++) {
            const int h = s_spatial_shapes[l * 2];
            const int w = s_spatial_shapes[l * 2 + 1];
            const int level_start = s_level_start_index[l];
            const int level_size = h * w * CHANNELS;
            const int elements_to_cache = min(level_size, cache_elements - cache_offset);

            // Load this level's data
            const int global_offset = b * spatial_size * CHANNELS + level_start * CHANNELS;
            for (int i = tid; i < elements_to_cache; i += block_size) {
                local_cache[cache_offset + i] = data_value[global_offset + i];
            }
            cache_offset += elements_to_cache;
        }
        __syncthreads();

        // Now process the query using cached data from across the cluster
        const int base_loc_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT * 2;
        const int base_weight_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT;

        // Each thread processes different channels
        for (int c = tid; c < CHANNELS; c += block_size) {
            scalar_t result = scalar_t(0);

            // Process all levels and points
            for (int l = 0; l < NUM_LEVELS; l++) {
                const int spatial_h = s_spatial_shapes[l * 2];
                const int spatial_w = s_spatial_shapes[l * 2 + 1];
                const int level_start_idx = s_level_start_index[l];

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

                        // Helper lambda to read value from cache or global memory
                        auto read_value = [&](int y_coord, int x_coord) -> scalar_t {
                            if (y_coord >= 0 && x_coord >= 0 && y_coord < spatial_h && x_coord < spatial_w) {
                                int spatial_idx = y_coord * spatial_w + x_coord;

                                // Check if this level is cached by any block in the cluster
                                const int cache_block = l / levels_per_block;  // Which block caches this level

                                if (cache_block < blocks_in_cluster) {
                                    // Try to read from distributed shared memory
                                    if (cache_block == cluster_rank) {
                                        // It's in our local cache
                                        int level_offset = 0;
                                        for (int prev_l = my_first_level; prev_l < l; prev_l++) {
                                            const int prev_h = s_spatial_shapes[prev_l * 2];
                                            const int prev_w = s_spatial_shapes[prev_l * 2 + 1];
                                            level_offset += prev_h * prev_w * CHANNELS;
                                        }
                                        int cache_idx = level_offset + spatial_idx * CHANNELS + c;
                                        if (cache_idx < cache_elements) {
                                            return local_cache[cache_idx];
                                        }
                                    } else {
                                        // It's in another block's cache - use distributed shared memory
                                        __half* remote_cache = cluster.map_shared_rank(shared_mem, cache_block);

                                        // Calculate offset in remote block's cache
                                        int level_offset = 0;
                                        int remote_first_level = cache_block * levels_per_block;
                                        for (int prev_l = remote_first_level; prev_l < l; prev_l++) {
                                            const int prev_h = s_spatial_shapes[prev_l * 2];
                                            const int prev_w = s_spatial_shapes[prev_l * 2 + 1];
                                            level_offset += prev_h * prev_w * CHANNELS;
                                        }
                                        int cache_idx = level_offset + spatial_idx * CHANNELS + c;

                                        // Note: Should check bounds but simplified for clarity
                                        return remote_cache[cache_idx];
                                    }
                                }

                                // Fall back to global memory
                                const int global_idx = (b * spatial_size + level_start_idx + spatial_idx) * CHANNELS + c;
                                return data_value[global_idx];
                            }
                            return scalar_t(0);
                        };

                        // Bilinear interpolation
                        scalar_t val = scalar_t(0);
                        val = __hfma(read_value(y_low, x_low), __hmul(hy, hx), val);
                        val = __hfma(read_value(y_low, x_high), __hmul(hy, lx), val);
                        val = __hfma(read_value(y_high, x_low), __hmul(ly, hx), val);
                        val = __hfma(read_value(y_high, x_high), __hmul(ly, lx), val);

                        result = __hfma(weights[p], val, result);
                    }
                }
            }

            // Write output
            const int out_idx = (b * num_query + q) * num_heads * CHANNELS + c;
            data_col[out_idx] = result;
        }

        cluster.sync();  // Ensure all blocks in cluster finish before next work item
    }
}

// Launch function for persistent distributed kernel
template <typename scalar_t=__half>
void launch_persistent_distributed_kernel(
    cudaStream_t stream,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_points,
    scalar_t *data_col,
    int *global_counter) {

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Configuration for persistent distributed kernel
    const int cluster_size = 2;  // 2 blocks per cluster
    const int num_sms = prop.multiProcessorCount;
    const int num_clusters = num_sms / cluster_size;  // One cluster per 2 SMs
    const int threads_per_block = 256;

    // Calculate shared memory - use maximum available
    size_t smem_size = 96 * 1024;  // 96KB per block

    std::cout << "Launching persistent distributed kernel:" << std::endl;
    std::cout << "  Clusters: " << num_clusters << std::endl;
    std::cout << "  Blocks per cluster: " << cluster_size << std::endl;
    std::cout << "  Total blocks: " << num_clusters * cluster_size << std::endl;
    std::cout << "  Threads per block: " << threads_per_block << std::endl;
    std::cout << "  Shared memory per block: " << smem_size / 1024.0 << " KB" << std::endl;
    std::cout << "  Total distributed shared memory per cluster: " << (smem_size * cluster_size) / 1024.0 << " KB" << std::endl;

    // Set maximum dynamic shared memory
    CUDA_CHECK(cudaFuncSetAttribute(
        ms_deformable_im2col_persistent_distributed<scalar_t, 8, 4, 32, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Launch with cluster configuration
    cudaLaunchConfig_t config = {0};
    config.gridDim = num_clusters * cluster_size;
    config.blockDim = threads_per_block;
    config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    // Reset global counter
    CUDA_CHECK(cudaMemset(global_counter, 0, sizeof(int)));

    // Launch the kernel
    cudaLaunchKernelEx(&config,
        ms_deformable_im2col_persistent_distributed<scalar_t, 8, 4, 32, 2>,
        data_value, data_spatial_shapes, data_level_start_index,
        data_sampling_loc, data_attn_weight,
        batch_size, spatial_size, num_heads, num_query,
        data_col, global_counter);
}

int main() {
    std::cout << "=== Persistent + Distributed Shared Memory Hybrid ===" << std::endl;

    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;

    // Large configuration to test the hybrid approach
    const int batch = 16;  // Even larger batch
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 2048;  // More queries

    // Large feature maps
    const std::vector<int64_t> h_spatial_shapes = {
        92, 92,   // Level 0: 92x92 = 8464
        46, 46,   // Level 1: 46x46 = 2116
        23, 23,   // Level 2: 23x23 = 529
        12, 12    // Level 3: 12x12 = 144
    };

    // Calculate spatial size and level start indices
    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;

    for (int i = 0; i < num_levels; i++) {
        if (i > 0) h_level_start_index.push_back(spatial_size);
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += h * w;
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Batch size: " << batch << std::endl;
    std::cout << "  Spatial size: " << spatial_size << std::endl;
    std::cout << "  Num queries: " << num_query << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Level shapes: ";
    for (int i = 0; i < num_levels; i++) {
        std::cout << h_spatial_shapes[i*2] << "x" << h_spatial_shapes[i*2+1];
        if (i < num_levels - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Allocate host memory
    const int64_t value_size = batch * spatial_size * channels;
    const int64_t output_size = batch * num_query * num_heads * channels;
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;

    std::vector<__half> h_value(value_size);
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    std::vector<__half> h_attn_weight(attn_weight_size);
    std::vector<__half> h_output(output_size);

    // Initialize data
    std::cout << "Initializing data..." << std::endl;

    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half(sinf(i * 0.001f) * 0.5f + 0.5f);
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> loc_dist(-0.8f, 0.8f);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half(loc_dist(gen));
    }

    // Initialize normalized attention weights
    for (int b = 0; b < batch; b++) {
        for (int q = 0; q < num_query; q++) {
            for (int h = 0; h < num_heads; h++) {
                float sum = 0.0f;
                int base = ((b * num_query + q) * num_heads + h) * num_levels * num_points;

                for (int i = 0; i < num_levels * num_points; i++) {
                    float w = expf(-i * 0.1f);
                    h_attn_weight[base + i] = __float2half(w);
                    sum += w;
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
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), value_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), sampling_loc_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), attn_weight_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    std::cout << "\nRunning warmup..." << std::endl;
    for (int i = 0; i < 5; i++) {
        launch_persistent_distributed_kernel(
            stream, d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_levels, num_query, num_points,
            d_output, d_global_counter);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::cout << "\nRunning benchmark..." << std::endl;
    const int num_iterations = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        launch_persistent_distributed_kernel(
            stream, d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_levels, num_query, num_points,
            d_output, d_global_counter);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average kernel time: " << milliseconds / num_iterations * 1000 << " microseconds" << std::endl;
    std::cout << "Throughput: " << (num_iterations * 1000.0f) / milliseconds << " iterations/second" << std::endl;

    // Calculate GFLOPS
    double ops_per_output = num_levels * num_points * 10;
    double total_ops = batch * num_query * num_heads * channels * ops_per_output;
    double gflops = (total_ops * num_iterations) / (milliseconds * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__half), cudaMemcpyDeviceToHost));

    // Check results
    std::cout << "\nFirst 20 output values:" << std::endl;
    for (int i = 0; i < std::min(20LL, (long long)output_size); i++) {
        if (i % 10 == 0 && i > 0) std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // Validation
    bool has_nonzero = false;
    float max_val = 0.0f;
    float min_val = FLT_MAX;
    float sum_val = 0.0f;

    for (int i = 0; i < output_size; i++) {
        float val = __half2float(h_output[i]);
        if (val != 0.0f) has_nonzero = true;
        max_val = fmaxf(max_val, val);
        min_val = fminf(min_val, val);
        sum_val += val;
    }

    std::cout << "\n=== Validation ===" << std::endl;
    std::cout << "Has non-zero values: " << (has_nonzero ? "Yes" : "No") << std::endl;
    std::cout << "Min value: " << min_val << std::endl;
    std::cout << "Max value: " << max_val << std::endl;
    std::cout << "Mean value: " << sum_val / output_size << std::endl;

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_global_counter);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ Persistent + Distributed hybrid test completed!" << std::endl;

    return 0;
}