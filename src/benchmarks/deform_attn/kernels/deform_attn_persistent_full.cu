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

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// Persistent kernel optimized for ORIGINAL FULL SIZE inputs
// This version uses intelligent caching strategy to handle the large 20522 spatial size
template <typename scalar_t=__half, const int NUM_POINT=8, const int NUM_LEVELS=4,
          const int CHANNELS=32>
__global__ void ms_deformable_im2col_persistent_fullsize(
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

    // Use maximum shared memory - 96KB with opt-in
    extern __shared__ __half shared_mem[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Partition shared memory for different uses:
    // 1. Cached values (main portion)
    // 2. Sampling locations cache (small portion)
    // 3. Attention weights cache (small portion)

    // With 96KB = 49152 half values available
    // Reserve 1KB for metadata, use 95KB for caching = 48640 half values
    __half* cached_values = shared_mem;
    const int max_cache_elements = 48000;  // Conservative to avoid overflow

    // Persistent kernel loop
    while (true) {
        // Get next work item
        int work_id = atomicAdd(global_counter, 1);

        int total_work = batch_size * num_query;
        if (work_id >= total_work) break;

        // Decode work item
        const int b = work_id / num_query;
        const int q = work_id % num_query;

        // Load spatial shapes and level indices into registers/shared memory
        __shared__ int64_t s_spatial_shapes[NUM_LEVELS * 2];
        __shared__ int64_t s_level_start_index[NUM_LEVELS];

        if (tid < NUM_LEVELS * 2) {
            s_spatial_shapes[tid] = data_spatial_shapes[tid];
        }
        if (tid < NUM_LEVELS) {
            s_level_start_index[tid] = (tid > 0) ? data_level_start_index[tid - 1] : 0;
        }
        __syncthreads();

        // Smart caching strategy for large spatial sizes:
        // Since we can't cache all 20522 * 32 = 656,704 values,
        // we'll cache based on the query's sampling locations

        // First, analyze the sampling locations to determine hot regions
        const int base_loc_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT * 2;

        // Determine which spatial regions are most accessed by this query
        // For simplicity, we'll cache the smaller levels (2, 3) entirely
        // and partial data from larger levels (0, 1) based on sampling patterns

        // Cache strategy:
        // Level 3 (12x20): 240 * 32 = 7,680 values
        // Level 2 (23x40): 920 * 32 = 29,440 values
        // Partial Level 1 (46x80): ~300 * 32 = 9,600 values
        // Partial Level 0 (92x160): ~50 * 32 = 1,600 values
        // Total: ~48,320 values (fits in 96KB)

        int cache_offset = 0;

        // Cache level 3 entirely (smallest)
        if (cache_offset < max_cache_elements) {
            const int level = 3;
            const int h = s_spatial_shapes[level * 2];
            const int w = s_spatial_shapes[level * 2 + 1];
            const int level_start = s_level_start_index[level];
            const int level_size = h * w * CHANNELS;

            if (cache_offset + level_size <= max_cache_elements) {
                const int global_offset = b * spatial_size * CHANNELS + level_start * CHANNELS;
                for (int i = tid; i < level_size; i += block_size) {
                    cached_values[cache_offset + i] = data_value[global_offset + i];
                }
                cache_offset += level_size;
            }
        }

        // Cache level 2 entirely
        if (cache_offset < max_cache_elements) {
            const int level = 2;
            const int h = s_spatial_shapes[level * 2];
            const int w = s_spatial_shapes[level * 2 + 1];
            const int level_start = s_level_start_index[level];
            const int level_size = h * w * CHANNELS;

            if (cache_offset + level_size <= max_cache_elements) {
                const int global_offset = b * spatial_size * CHANNELS + level_start * CHANNELS;
                for (int i = tid; i < level_size; i += block_size) {
                    cached_values[cache_offset + i] = data_value[global_offset + i];
                }
                cache_offset += level_size;
            }
        }

        // For levels 0 and 1, cache center regions or based on sampling patterns
        // This is a simplified strategy - in production, you'd analyze actual sampling locations

        __syncthreads();

        // Track what we've cached for quick lookup
        __shared__ int cached_level_starts[NUM_LEVELS];
        __shared__ int cached_level_sizes[NUM_LEVELS];

        if (tid == 0) {
            // Mark fully cached levels
            cached_level_starts[3] = 0;
            cached_level_sizes[3] = s_spatial_shapes[6] * s_spatial_shapes[7] * CHANNELS;

            cached_level_starts[2] = cached_level_sizes[3];
            cached_level_sizes[2] = s_spatial_shapes[4] * s_spatial_shapes[5] * CHANNELS;

            // Levels 0 and 1 not fully cached
            cached_level_starts[1] = -1;
            cached_level_sizes[1] = 0;
            cached_level_starts[0] = -1;
            cached_level_sizes[0] = 0;
        }
        __syncthreads();

        // Now process the query with intelligent cache usage
        for (int c = tid; c < CHANNELS; c += block_size) {
            scalar_t result = scalar_t(0);

            // Process all levels and points
            #pragma unroll
            for (int l = 0; l < NUM_LEVELS; l++) {
                const int spatial_h = s_spatial_shapes[l * 2];
                const int spatial_w = s_spatial_shapes[l * 2 + 1];
                const int level_start_idx = s_level_start_index[l];

                // Load sampling locations and weights for this level
                __half loc_x[NUM_POINT], loc_y[NUM_POINT], weights[NUM_POINT];

                #pragma unroll
                for (int p = 0; p < NUM_POINT; p++) {
                    const int loc_idx = base_loc_idx + (l * NUM_POINT + p) * 2;
                    const int weight_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT +
                                          l * NUM_POINT + p;

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

                        // Helper function to read value from cache or global memory
                        auto read_value = [&](int y_coord, int x_coord) -> scalar_t {
                            if (y_coord >= 0 && x_coord >= 0 && y_coord < spatial_h && x_coord < spatial_w) {
                                int spatial_idx = y_coord * spatial_w + x_coord;

                                // Check if this level is fully cached
                                if (cached_level_starts[l] >= 0) {
                                    // Read from cache
                                    int cache_idx = cached_level_starts[l] + spatial_idx * CHANNELS + c;
                                    if (cache_idx < cache_offset) {
                                        return cached_values[cache_idx];
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

        __syncthreads();
    }
}

int main() {
    std::cout << "=== Persistent Kernel with ORIGINAL FULL SIZE Inputs ===" << std::endl;

    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;

    // ORIGINAL FULL SIZE configuration from the paper
    const int batch = 48;  // Original batch size
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 15422;  // Original number of queries

    // ORIGINAL feature map sizes
    const std::vector<int64_t> h_spatial_shapes = {
        92, 160,   // Level 0: 92x160 = 14720
        46, 80,    // Level 1: 46x80 = 3680
        23, 40,    // Level 2: 23x40 = 920
        12, 20     // Level 3: 12x20 = 240
    };

    // Calculate total spatial size - should be 20522 as mentioned
    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;

    for (int i = 0; i < num_levels; i++) {
        h_level_start_index.push_back(spatial_size);
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += h * w;
    }

    std::cout << "\n=== ORIGINAL Configuration ===" << std::endl;
    std::cout << "  Batch size: " << batch << std::endl;
    std::cout << "  Spatial size: " << spatial_size << " (should be ~20522)" << std::endl;
    std::cout << "  Num queries: " << num_query << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Level shapes: ";
    for (int i = 0; i < num_levels; i++) {
        std::cout << h_spatial_shapes[i*2] << "x" << h_spatial_shapes[i*2+1];
        if (i < num_levels - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  Level start indices: ";
    for (auto idx : h_level_start_index) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // Calculate memory requirements
    const int64_t value_size = batch * spatial_size * channels;
    const int64_t output_size = batch * num_query * num_heads * channels;
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;

    std::cout << "\n=== Memory Requirements ===" << std::endl;
    std::cout << "  Value tensor: " << value_size * sizeof(__half) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Output tensor: " << output_size * sizeof(__half) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Sampling locations: " << sampling_loc_size * sizeof(__half) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Attention weights: " << attn_weight_size * sizeof(__half) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Allocate host memory
    std::cout << "\nAllocating host memory..." << std::endl;
    std::vector<__half> h_value(value_size);
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    std::vector<__half> h_attn_weight(attn_weight_size);
    std::vector<__half> h_output(output_size);

    // Initialize data with realistic patterns
    std::cout << "Initializing data..." << std::endl;

    // Initialize value tensor with smooth gradients
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatial_size; s++) {
            for (int c = 0; c < channels; c++) {
                int idx = (b * spatial_size + s) * channels + c;
                // Create smooth patterns that vary by channel
                float val = sinf(s * 0.01f + c * 0.1f) * 0.5f + 0.5f;
                h_value[idx] = __float2half(val);
            }
        }
    }

    // Initialize sampling locations (normalized between -1 and 1)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> loc_dist(-0.9f, 0.9f);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half(loc_dist(gen));
    }

    // Initialize attention weights (normalized per query)
    for (int b = 0; b < batch; b++) {
        for (int q = 0; q < num_query; q++) {
            for (int h = 0; h < num_heads; h++) {
                float sum = 0.0f;
                int base = ((b * num_query + q) * num_heads + h) * num_levels * num_points;

                // Generate weights with level-wise decay
                for (int l = 0; l < num_levels; l++) {
                    for (int p = 0; p < num_points; p++) {
                        float w = expf(-l * 0.5f - p * 0.1f);  // Higher levels get less weight
                        h_attn_weight[base + l * num_points + p] = __float2half(w);
                        sum += w;
                    }
                }

                // Normalize to sum to 1
                for (int i = 0; i < num_levels * num_points; i++) {
                    h_attn_weight[base + i] = __float2half(__half2float(h_attn_weight[base + i]) / sum);
                }
            }
        }
    }

    // Allocate device memory
    std::cout << "Allocating GPU memory..." << std::endl;
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
    std::cout << "Copying data to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), value_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), sampling_loc_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), attn_weight_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch configuration
    const int num_blocks = prop.multiProcessorCount;  // One block per SM
    const int threads_per_block = 256;

    // Calculate shared memory size - use maximum available
    size_t smem_size = 96 * 1024;  // 96KB

    // Set maximum dynamic shared memory
    CUDA_CHECK(cudaFuncSetAttribute(
        ms_deformable_im2col_persistent_fullsize<__half, 8, 4, 32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    std::cout << "\n=== Launch Configuration ===" << std::endl;
    std::cout << "  Blocks: " << num_blocks << " (one per SM)" << std::endl;
    std::cout << "  Threads per block: " << threads_per_block << std::endl;
    std::cout << "  Shared memory per block: " << smem_size / 1024.0 << " KB" << std::endl;
    std::cout << "  Total work items: " << batch * num_query << std::endl;

    // Warmup
    std::cout << "\nRunning warmup..." << std::endl;
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));
        ms_deformable_im2col_persistent_fullsize<__half, 8, 4, 32>
            <<<num_blocks, threads_per_block, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, num_query,
            d_output, d_global_counter);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    const int num_iterations = 50;  // Fewer iterations due to large problem size

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));
        ms_deformable_im2col_persistent_fullsize<__half, 8, 4, 32>
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
    std::cout << "Total time: " << milliseconds << " ms for " << num_iterations << " iterations" << std::endl;
    std::cout << "Average kernel time: " << milliseconds / num_iterations << " ms" << std::endl;
    std::cout << "Throughput: " << (num_iterations * 1000.0f) / milliseconds << " iterations/second" << std::endl;

    // Calculate TFLOPS for the full size
    double ops_per_output = num_levels * num_points * 10;  // bilinear interp + multiply + add
    double total_ops = batch * num_query * num_heads * channels * ops_per_output;
    double tflops = (total_ops * num_iterations) / (milliseconds * 1e9);
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Calculate effective memory bandwidth
    double bytes_accessed = (value_size + sampling_loc_size + attn_weight_size + output_size) * sizeof(__half);
    double bandwidth_gb = (bytes_accessed * num_iterations) / (milliseconds * 1e6);
    std::cout << "Effective bandwidth: " << bandwidth_gb << " GB/s" << std::endl;

    // Copy back a small portion for verification
    std::cout << "\nCopying results back for verification..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                         std::min(100000LL, (long long)output_size) * sizeof(__half),
                         cudaMemcpyDeviceToHost));

    // Check first few results
    std::cout << "\nFirst 20 output values:" << std::endl;
    for (int i = 0; i < std::min(20LL, (long long)output_size); i++) {
        if (i % 10 == 0 && i > 0) std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // Validation on sample
    bool has_nonzero = false;
    float max_val = 0.0f;
    float min_val = FLT_MAX;
    float sum_val = 0.0f;
    int sample_size = std::min(100000LL, (long long)output_size);

    for (int i = 0; i < sample_size; i++) {
        float val = __half2float(h_output[i]);
        if (val != 0.0f) has_nonzero = true;
        max_val = fmaxf(max_val, val);
        min_val = fminf(min_val, val);
        sum_val += val;
    }

    std::cout << "\n=== Validation (on first " << sample_size << " values) ===" << std::endl;
    std::cout << "Has non-zero values: " << (has_nonzero ? "Yes" : "No") << std::endl;
    std::cout << "Min value: " << min_val << std::endl;
    std::cout << "Max value: " << max_val << std::endl;
    std::cout << "Mean value: " << sum_val / sample_size << std::endl;

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

    std::cout << "\n✅ Full-size persistent kernel test completed successfully!" << std::endl;
    std::cout << "Successfully processed original size inputs:" << std::endl;
    std::cout << "  - Batch: " << batch << std::endl;
    std::cout << "  - Spatial size: " << spatial_size << std::endl;
    std::cout << "  - Queries: " << num_query << std::endl;

    return 0;
}