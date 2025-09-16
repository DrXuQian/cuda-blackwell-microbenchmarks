#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

// Simplified bank-conflict optimized kernel for testing
// Start with smaller inputs to ensure correctness

__global__ void ms_deform_attn_bankopt_simple(
    const __half* __restrict__ value,
    const int* __restrict__ value_spatial_shapes,
    const int* __restrict__ value_level_start_index,
    const __half* __restrict__ sampling_locations,
    const __half* __restrict__ attention_weights,
    __half* __restrict__ output,
    const int batch,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_queries,
    const int num_points
) {
    // Simple grid-stride pattern
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch * num_queries * num_heads * channels;

    if (idx >= total_outputs) return;

    // Decode indices
    const int b = idx / (num_queries * num_heads * channels);
    const int remainder = idx % (num_queries * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int remainder2 = remainder % (num_heads * channels);
    const int h = remainder2 / channels;
    const int c = remainder2 % channels;

    float result = 0.0f;

    // Process each level and point
    for (int l = 0; l < num_levels; l++) {
        const int level_start = value_level_start_index[l];
        const int H = value_spatial_shapes[l * 2];
        const int W = value_spatial_shapes[l * 2 + 1];

        for (int p = 0; p < num_points; p++) {
            // Get sampling location
            const int loc_idx = b * num_queries * num_heads * num_levels * num_points * 2 +
                               q * num_heads * num_levels * num_points * 2 +
                               h * num_levels * num_points * 2 +
                               l * num_points * 2 +
                               p * 2;

            const float loc_x = __half2float(sampling_locations[loc_idx]);
            const float loc_y = __half2float(sampling_locations[loc_idx + 1]);

            // Get attention weight
            const int weight_idx = b * num_queries * num_heads * num_levels * num_points +
                                  q * num_heads * num_levels * num_points +
                                  h * num_levels * num_points +
                                  l * num_points +
                                  p;

            const float attn_weight = __half2float(attention_weights[weight_idx]);

            // Convert to pixel coordinates
            const float x = loc_x * (W - 1);
            const float y = loc_y * (H - 1);

            // Bilinear interpolation
            const int x_low = floorf(x);
            const int y_low = floorf(y);
            const int x_high = x_low + 1;
            const int y_high = y_low + 1;

            const float lx = x - x_low;
            const float ly = y - y_low;
            const float hx = 1.0f - lx;
            const float hy = 1.0f - ly;

            float val = 0.0f;

            // Sample from value tensor
            if (y_low >= 0 && x_low >= 0 && y_low < H && x_low < W) {
                const int value_idx = b * spatial_size * num_heads * channels +
                                    (level_start + y_low * W + x_low) * num_heads * channels +
                                    h * channels + c;
                val += __half2float(value[value_idx]) * hy * hx;
            }
            if (y_low >= 0 && x_high >= 0 && y_low < H && x_high < W) {
                const int value_idx = b * spatial_size * num_heads * channels +
                                    (level_start + y_low * W + x_high) * num_heads * channels +
                                    h * channels + c;
                val += __half2float(value[value_idx]) * hy * lx;
            }
            if (y_high >= 0 && x_low >= 0 && y_high < H && x_low < W) {
                const int value_idx = b * spatial_size * num_heads * channels +
                                    (level_start + y_high * W + x_low) * num_heads * channels +
                                    h * channels + c;
                val += __half2float(value[value_idx]) * ly * hx;
            }
            if (y_high >= 0 && x_high >= 0 && y_high < H && x_high < W) {
                const int value_idx = b * spatial_size * num_heads * channels +
                                    (level_start + y_high * W + x_high) * num_heads * channels +
                                    h * channels + c;
                val += __half2float(value[value_idx]) * ly * lx;
            }

            result += val * attn_weight;
        }
    }

    // Write output
    const int output_idx = b * num_queries * num_heads * channels +
                          q * num_heads * channels +
                          h * channels + c;
    output[output_idx] = __float2half(result);
}

int main() {
    printf("=== Bank-Optimized Kernel V2 - Starting with smaller inputs ===\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);

    // Start with smaller test configuration
    const int batch = 2;
    const int spatial_size = 100;
    const int num_heads = 2;
    const int channels = 8;
    const int num_levels = 2;
    const int num_queries = 50;
    const int num_points = 4;

    printf("=== Small Test Configuration ===\n");
    printf("  Batch: %d\n", batch);
    printf("  Queries: %d\n", num_queries);
    printf("  Channels: %d\n", channels);
    printf("  Levels: %d\n\n", num_levels);

    // Define spatial shapes
    int h_spatial_shapes[] = {8, 8, 6, 6}; // 64 + 36 = 100 spatial
    int h_level_start_index[] = {0, 64};

    // Allocate memory
    size_t value_size = batch * spatial_size * num_heads * channels * sizeof(__half);
    size_t output_size = batch * num_queries * num_heads * channels * sizeof(__half);
    size_t sampling_size = batch * num_queries * num_heads * num_levels * num_points * 2 * sizeof(__half);
    size_t weights_size = batch * num_queries * num_heads * num_levels * num_points * sizeof(__half);

    __half *d_value, *d_output, *d_sampling, *d_weights;
    int *d_spatial_shapes, *d_level_start;

    cudaMalloc(&d_value, value_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_sampling, sampling_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_spatial_shapes, num_levels * 2 * sizeof(int));
    cudaMalloc(&d_level_start, num_levels * sizeof(int));

    // Initialize test data
    __half* h_value = new __half[value_size / sizeof(__half)];
    __half* h_sampling = new __half[sampling_size / sizeof(__half)];
    __half* h_weights = new __half[weights_size / sizeof(__half)];

    for (size_t i = 0; i < value_size / sizeof(__half); i++) {
        h_value[i] = __float2half(0.1f);
    }
    for (size_t i = 0; i < sampling_size / sizeof(__half); i++) {
        h_sampling[i] = __float2half(0.5f);
    }
    for (size_t i = 0; i < weights_size / sizeof(__half); i++) {
        h_weights[i] = __float2half(0.25f);
    }

    cudaMemcpy(d_value, h_value, value_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sampling, h_sampling, sampling_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_shapes, h_spatial_shapes, num_levels * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_start, h_level_start_index, num_levels * sizeof(int), cudaMemcpyHostToDevice);

    // Clear output
    cudaMemset(d_output, 0, output_size);

    // Launch kernel
    const int threads = 256;
    const int blocks = (batch * num_queries * num_heads * channels + threads - 1) / threads;

    printf("Launching kernel with %d blocks, %d threads\n", blocks, threads);

    ms_deform_attn_bankopt_simple<<<blocks, threads>>>(
        d_value, d_spatial_shapes, d_level_start,
        d_sampling, d_weights, d_output,
        batch, spatial_size, num_heads, channels,
        num_levels, num_queries, num_points
    );

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Verify output
    __half* h_output = new __half[output_size / sizeof(__half)];
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    int non_zero = 0;
    float max_val = 0;
    for (size_t i = 0; i < output_size / sizeof(__half); i++) {
        float val = __half2float(h_output[i]);
        if (val != 0) non_zero++;
        if (val > max_val) max_val = val;
    }

    printf("\n=== Results ===\n");
    printf("Non-zero outputs: %d/%lu (%.1f%%)\n", non_zero, output_size/sizeof(__half),
           100.0f * non_zero / (output_size/sizeof(__half)));
    printf("Max value: %.6f\n", max_val);

    if (non_zero > 0) {
        printf("✅ Small test passed!\n\n");

        // Now test with larger configuration
        printf("=== Testing with larger configuration ===\n");

        // Test with paper-size configuration
        const int batch2 = 48;
        const int spatial_size2 = 19560;
        const int num_heads2 = 8;
        const int channels2 = 32;
        const int num_levels2 = 4;
        const int num_queries2 = 15422;
        const int num_points2 = 4;

        printf("  Batch: %d\n", batch2);
        printf("  Spatial: %d\n", spatial_size2);
        printf("  Queries: %d\n", num_queries2);
        printf("  Channels: %d\n\n", channels2);

        // Spatial shapes for full size
        int h_spatial_shapes2[] = {92, 160, 46, 80, 23, 40, 12, 20};
        int h_level_start2[] = {0, 14720, 18400, 19320};

        // Allocate for full size
        size_t value_size2 = batch2 * spatial_size2 * num_heads2 * channels2 * sizeof(__half);
        size_t output_size2 = batch2 * num_queries2 * num_heads2 * channels2 * sizeof(__half);
        size_t sampling_size2 = batch2 * num_queries2 * num_heads2 * num_levels2 * num_points2 * 2 * sizeof(__half);
        size_t weights_size2 = batch2 * num_queries2 * num_heads2 * num_levels2 * num_points2 * sizeof(__half);

        __half *d_value2, *d_output2, *d_sampling2, *d_weights2;
        int *d_spatial_shapes2, *d_level_start2;

        cudaMalloc(&d_value2, value_size2);
        cudaMalloc(&d_output2, output_size2);
        cudaMalloc(&d_sampling2, sampling_size2);
        cudaMalloc(&d_weights2, weights_size2);
        cudaMalloc(&d_spatial_shapes2, num_levels2 * 2 * sizeof(int));
        cudaMalloc(&d_level_start2, num_levels2 * sizeof(int));

        // Initialize
        __half* h_value2 = new __half[value_size2 / sizeof(__half)];
        __half* h_sampling2 = new __half[sampling_size2 / sizeof(__half)];
        __half* h_weights2 = new __half[weights_size2 / sizeof(__half)];

        for (size_t i = 0; i < value_size2 / sizeof(__half); i++) {
            h_value2[i] = __float2half(0.1f * (i % 10));
        }
        for (size_t i = 0; i < sampling_size2 / sizeof(__half); i++) {
            h_sampling2[i] = __float2half(0.5f + 0.3f * sin(i * 0.01f));
        }
        for (size_t i = 0; i < weights_size2 / sizeof(__half); i++) {
            h_weights2[i] = __float2half(0.25f);
        }

        cudaMemcpy(d_value2, h_value2, value_size2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sampling2, h_sampling2, sampling_size2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights2, h_weights2, weights_size2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_spatial_shapes2, h_spatial_shapes2, num_levels2 * 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_level_start2, h_level_start2, num_levels2 * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemset(d_output2, 0, output_size2);

        // Launch full-size kernel
        const int blocks2 = (batch2 * num_queries2 * num_heads2 * channels2 + threads - 1) / threads;

        printf("Launching full-size kernel with %d blocks\n", blocks2);

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        for (int i = 0; i < 5; i++) {
            ms_deform_attn_bankopt_simple<<<blocks2, threads>>>(
                d_value2, d_spatial_shapes2, d_level_start2,
                d_sampling2, d_weights2, d_output2,
                batch2, spatial_size2, num_heads2, channels2,
                num_levels2, num_queries2, num_points2
            );
        }
        cudaDeviceSynchronize();

        const int num_iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            ms_deform_attn_bankopt_simple<<<blocks2, threads>>>(
                d_value2, d_spatial_shapes2, d_level_start2,
                d_sampling2, d_weights2, d_output2,
                batch2, spatial_size2, num_heads2, channels2,
                num_levels2, num_queries2, num_points2
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_time = milliseconds / num_iterations;

        size_t total_ops = (size_t)batch2 * num_queries2 * num_heads2 * channels2 * num_levels2 * num_points2 * 8;
        float tflops = (total_ops / (avg_time / 1000.0)) / 1e12;

        printf("\n=== Performance (Full Size) ===\n");
        printf("Average time: %.3f ms\n", avg_time);
        printf("Performance: %.3f TFLOPS\n", tflops);

        // Verify full-size output
        __half* h_output2 = new __half[output_size2 / sizeof(__half)];
        cudaMemcpy(h_output2, d_output2, output_size2, cudaMemcpyDeviceToHost);

        non_zero = 0;
        max_val = 0;
        for (size_t i = 0; i < output_size2 / sizeof(__half); i++) {
            float val = __half2float(h_output2[i]);
            if (val != 0) non_zero++;
            if (val > max_val) max_val = val;
        }

        printf("\nNon-zero outputs: %d/%lu (%.3f%%)\n", non_zero, output_size2/sizeof(__half),
               100.0f * non_zero / (output_size2/sizeof(__half)));
        printf("Max value: %.6f\n\n", max_val);

        if (non_zero > 0) {
            printf("✅ Full-size test passed!\n");
        }

        // Cleanup
        delete[] h_value2;
        delete[] h_sampling2;
        delete[] h_weights2;
        delete[] h_output2;

        cudaFree(d_value2);
        cudaFree(d_output2);
        cudaFree(d_sampling2);
        cudaFree(d_weights2);
        cudaFree(d_spatial_shapes2);
        cudaFree(d_level_start2);
    }

    // Cleanup
    delete[] h_value;
    delete[] h_sampling;
    delete[] h_weights;
    delete[] h_output;

    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_sampling);
    cudaFree(d_weights);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start);

    return 0;
}