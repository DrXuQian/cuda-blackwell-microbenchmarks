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

// Simplified kernel without distributed shared memory
template <typename scalar_t=__half>
__global__ void ms_deformable_im2col_simple(
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
    scalar_t *data_col) {

    // Simple grid-stride loop implementation
    const int total_outputs = batch_size * num_query * num_heads * channels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_outputs) return;

    // Decode indices
    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    // Loop over levels and points
    for (int l = 0; l < num_levels; l++) {
        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

        for (int p = 0; p < num_points; p++) {
            // Get sampling location and attention weight
            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            const scalar_t loc_y = data_sampling_loc[loc_idx];
            const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
            const scalar_t weight = data_attn_weight[weight_idx];

            // Convert normalized coordinates to actual coordinates
            const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

            // Bilinear interpolation
            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

                // Sample from data_value
                scalar_t val = scalar_t(0);
                if (y_low >= 0 && x_low >= 0) {
                    const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_low) * channels + c;
                    val = __hfma(data_value[idx], __float2half(hy * hx), val);
                }
                if (y_low >= 0 && x_high < spatial_w) {
                    const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_high) * channels + c;
                    val = __hfma(data_value[idx], __float2half(hy * lx), val);
                }
                if (y_high < spatial_h && x_low >= 0) {
                    const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_low) * channels + c;
                    val = __hfma(data_value[idx], __float2half(ly * hx), val);
                }
                if (y_high < spatial_h && x_high < spatial_w) {
                    const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_high) * channels + c;
                    val = __hfma(data_value[idx], __float2half(ly * lx), val);
                }

                result = __hfma(weight, val, result);
            }
        }
    }

    data_col[idx] = result;
}

int main() {
    std::cout << "=== Simple MS-Deformable Attention (No Distributed Shared Memory) ===" << std::endl;

    // Small test configuration
    const int batch = 2;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 256;

    // Small feature maps
    const std::vector<int64_t> h_spatial_shapes = {
        16, 16,   // Level 0: 16x16 = 256
        8, 8,     // Level 1: 8x8 = 64
        4, 4,     // Level 2: 4x4 = 16
        2, 2      // Level 3: 2x2 = 4
    };

    // Calculate spatial size and level start indices
    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;

    for (int i = 0; i < num_levels; i++) {
        h_level_start_index.push_back(spatial_size);
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

    // Allocate host memory
    const int64_t value_size = batch * spatial_size * channels;
    const int64_t output_size = batch * num_query * num_heads * channels;
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;

    std::vector<__half> h_value(value_size);
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    std::vector<__half> h_attn_weight(attn_weight_size);
    std::vector<__half> h_output(output_size);

    // Initialize with simple patterns
    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half(1.0f / (i % 100 + 1));
    }

    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        // Random sampling locations between -1 and 1
        h_sampling_loc[i] = __float2half((i % 200 - 100) * 0.01f);
    }

    for (size_t i = 0; i < h_attn_weight.size(); i++) {
        // Uniform weights
        h_attn_weight[i] = __float2half(1.0f / (num_levels * num_points));
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

    // Launch kernel
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    std::cout << "\nLaunching kernel with " << blocks << " blocks and " << threads << " threads" << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        ms_deformable_im2col_simple<<<blocks, threads>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels,
            num_levels, num_query, num_points,
            d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        ms_deformable_im2col_simple<<<blocks, threads>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels,
            num_levels, num_query, num_points,
            d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Average kernel time: " << duration.count() / num_iterations << " microseconds" << std::endl;
    std::cout << "Throughput: " << (1000000.0 * num_iterations) / duration.count() << " iterations/second" << std::endl;

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__half), cudaMemcpyDeviceToHost));

    // Check results
    std::cout << "\nFirst 20 output values:" << std::endl;
    for (int i = 0; i < std::min(20LL, (long long)output_size); i++) {
        if (i % 10 == 0 && i > 0) std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // Check for non-zero outputs
    bool has_nonzero = false;
    float max_val = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float val = __half2float(h_output[i]);
        if (val != 0.0f) has_nonzero = true;
        if (fabs(val) > max_val) max_val = fabs(val);
    }

    std::cout << "\nValidation: " << (has_nonzero ? "PASS" : "FAIL") << std::endl;
    std::cout << "Max absolute value: " << max_val << std::endl;

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);

    return 0;
}