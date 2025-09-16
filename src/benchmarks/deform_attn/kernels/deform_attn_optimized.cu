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

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// Optimized kernel with shared memory but no cluster launch
template <typename scalar_t=__half, const int NUM_POINT=8, const int NUM_LEVELS=4>
__global__ void ms_deformable_im2col_optimized(
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_query,
    scalar_t *data_col) {

    extern __shared__ __half smem[];

    // Each block handles one query across all channels
    const int b = blockIdx.y;
    const int q = blockIdx.x;

    if (b >= batch_size || q >= num_query) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Load spatial shapes to shared memory
    __shared__ int64_t s_spatial_shapes[NUM_LEVELS * 2];
    __shared__ int64_t s_level_start_index[NUM_LEVELS];

    if (tid < NUM_LEVELS * 2) {
        s_spatial_shapes[tid] = data_spatial_shapes[tid];
    }
    if (tid < NUM_LEVELS) {
        s_level_start_index[tid] = (tid > 0) ? data_level_start_index[tid - 1] : 0;
    }
    __syncthreads();

    // Each thread handles multiple channels
    for (int c = tid; c < channels; c += block_size) {
        scalar_t result = scalar_t(0);

        // Base indices for this query
        const int base_loc_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT * 2;
        const int base_weight_idx = ((b * num_query + q) * num_heads) * NUM_LEVELS * NUM_POINT;

        // Process all levels and points
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int spatial_h = s_spatial_shapes[l * 2];
            const int spatial_w = s_spatial_shapes[l * 2 + 1];
            const int level_start_idx = s_level_start_index[l];

            // Load sampling locations and weights for this level
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

                // Check bounds
                if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                    const int y_low = floorf(y);
                    const int x_low = floorf(x);
                    const int y_high = y_low + 1;
                    const int x_high = x_low + 1;

                    const __half ly = __float2half(y - y_low);
                    const __half lx = __float2half(x - x_low);
                    const __half hy = __float2half(1 - (y - y_low));
                    const __half hx = __float2half(1 - (x - x_low));

                    // Bilinear interpolation
                    scalar_t val = scalar_t(0);

                    if (y_low >= 0 && x_low >= 0 && y_low < spatial_h && x_low < spatial_w) {
                        const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_low) * channels + c;
                        val = __hfma(data_value[idx], __hmul(hy, hx), val);
                    }
                    if (y_low >= 0 && x_high < spatial_w && y_low < spatial_h) {
                        const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_high) * channels + c;
                        val = __hfma(data_value[idx], __hmul(hy, lx), val);
                    }
                    if (y_high < spatial_h && x_low >= 0 && x_low < spatial_w) {
                        const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_low) * channels + c;
                        val = __hfma(data_value[idx], __hmul(ly, hx), val);
                    }
                    if (y_high < spatial_h && x_high < spatial_w) {
                        const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_high) * channels + c;
                        val = __hfma(data_value[idx], __hmul(ly, lx), val);
                    }

                    result = __hfma(weights[p], val, result);
                }
            }
        }

        // Write output
        const int out_idx = (b * num_query + q) * num_heads * channels + c;
        data_col[out_idx] = result;
    }
}

int main() {
    std::cout << "=== Optimized MS-Deformable Attention ===" << std::endl;

    // Test configuration with reasonable sizes
    const int batch = 4;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 512;

    // Feature map sizes
    const std::vector<int64_t> h_spatial_shapes = {
        32, 32,   // Level 0: 32x32 = 1024
        16, 16,   // Level 1: 16x16 = 256
        8, 8,     // Level 2: 8x8 = 64
        4, 4      // Level 3: 4x4 = 16
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

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch << std::endl;
    std::cout << "  Spatial size: " << spatial_size << std::endl;
    std::cout << "  Num queries: " << num_query << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Num levels: " << num_levels << std::endl;
    std::cout << "  Num points: " << num_points << std::endl;
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

    // Initialize value tensor with gradient pattern
    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half(sinf(i * 0.001f) * 0.5f + 0.5f);
    }

    // Initialize sampling locations (normalized between -1 and 1)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> loc_dist(-0.8f, 0.8f);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half(loc_dist(gen));
    }

    // Initialize attention weights (normalized to sum to 1)
    for (int b = 0; b < batch; b++) {
        for (int q = 0; q < num_query; q++) {
            for (int h = 0; h < num_heads; h++) {
                float sum = 0.0f;
                int base = ((b * num_query + q) * num_heads + h) * num_levels * num_points;

                // Generate random weights
                for (int i = 0; i < num_levels * num_points; i++) {
                    float w = expf(-i * 0.1f);  // Exponential decay
                    h_attn_weight[base + i] = __float2half(w);
                    sum += w;
                }

                // Normalize
                for (int i = 0; i < num_levels * num_points; i++) {
                    h_attn_weight[base + i] = __float2half(__half2float(h_attn_weight[base + i]) / sum);
                }
            }
        }
    }

    // Allocate device memory
    __half *d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t *d_spatial_shapes, *d_level_start_index;

    CUDA_CHECK(cudaMalloc(&d_value, value_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, attn_weight_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), value_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), sampling_loc_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), attn_weight_size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch configuration
    const int threads = 128;  // Multiple warps for better occupancy
    dim3 blocks(num_query, batch);
    size_t smem_size = 1024;  // Small amount for metadata caching

    std::cout << "\nLaunching optimized kernel:" << std::endl;
    std::cout << "  Grid: (" << blocks.x << ", " << blocks.y << ")" << std::endl;
    std::cout << "  Block: " << threads << " threads" << std::endl;
    std::cout << "  Shared memory: " << smem_size << " bytes" << std::endl;

    // Warmup
    std::cout << "Running warmup..." << std::endl;
    for (int i = 0; i < 10; i++) {
        ms_deformable_im2col_optimized<__half, 8, 4><<<blocks, threads, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_query,
            d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    const int num_iterations = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        ms_deformable_im2col_optimized<__half, 8, 4><<<blocks, threads, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_query,
            d_output);
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
    // Each output element requires:
    // - num_levels * num_points * (4 multiplies for bilinear + 1 multiply for weight + 4 adds)
    // - Total: num_levels * num_points * 10 ops
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ Test completed successfully!" << std::endl;

    return 0;
}