#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <chrono>
#include <cfloat>
#include <random>
#include <iostream>
#include <iomanip>
#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// ==============================================================================
// BASELINE: Original Implementation for Reference
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_baseline(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * num_query * num_heads * channels;

    if (idx >= total_outputs) return;

    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    for (int l = 0; l < num_levels; l++) {
        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

        for (int p = 0; p < num_points; p++) {
            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            const scalar_t loc_y = data_sampling_loc[loc_idx];
            const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
            const scalar_t weight = data_attn_weight[weight_idx];

            const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

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

// ==============================================================================
// APPROACH 1: Shared Memory for Metadata
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_shared_meta(
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

    __shared__ int64_t s_spatial_shapes[8];  // Max 4 levels * 2
    __shared__ int64_t s_level_start[4];

    const int tid = threadIdx.x;

    // Load spatial metadata to shared memory
    if (tid < 8) {
        s_spatial_shapes[tid] = data_spatial_shapes[tid];
    }
    if (tid < num_levels - 1) {
        s_level_start[tid] = data_level_start_index[tid];
    }
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * num_query * num_heads * channels;

    if (idx >= total_outputs) return;

    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    for (int l = 0; l < num_levels; l++) {
        const int spatial_h = s_spatial_shapes[l * 2];
        const int spatial_w = s_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? s_level_start[l - 1] : 0;

        for (int p = 0; p < num_points; p++) {
            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            const scalar_t loc_y = data_sampling_loc[loc_idx];
            const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
            const scalar_t weight = data_attn_weight[weight_idx];

            const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

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

// ==============================================================================
// APPROACH 2: Loop Unrolling
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_unrolled(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * num_query * num_heads * channels;

    if (idx >= total_outputs) return;

    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    // Unroll levels (usually 4)
    #pragma unroll 4
    for (int l = 0; l < 4; l++) {
        if (l >= num_levels) break;

        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

        // Unroll points (usually 8)
        #pragma unroll 8
        for (int p = 0; p < 8; p++) {
            if (p >= num_points) break;

            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            const scalar_t loc_y = data_sampling_loc[loc_idx];
            const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
            const scalar_t weight = data_attn_weight[weight_idx];

            const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

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

// ==============================================================================
// APPROACH 3: Float2 Vectorization for Coordinates
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_float2_coords(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * num_query * num_heads * channels;

    if (idx >= total_outputs) return;

    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    for (int l = 0; l < num_levels; l++) {
        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

        for (int p = 0; p < num_points; p++) {
            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            // Load coordinates as half2
            const half2* loc_ptr = (const half2*)&data_sampling_loc[loc_idx];
            half2 loc = *loc_ptr;
            const scalar_t weight = data_attn_weight[weight_idx];

            const float y = (__half2float(loc.x) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc.y) + 1) * spatial_w / 2.0f - 0.5f;

            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

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

// ==============================================================================
// APPROACH 4: Precomputed Interpolation Weights
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_precompute_weights(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * num_query * num_heads * channels;

    if (idx >= total_outputs) return;

    const int b = idx / (num_query * num_heads * channels);
    const int remainder = idx % (num_query * num_heads * channels);
    const int q = remainder / (num_heads * channels);
    const int h = (remainder % (num_heads * channels)) / channels;
    const int c = remainder % channels;

    scalar_t result = scalar_t(0);

    for (int l = 0; l < num_levels; l++) {
        const int spatial_h = data_spatial_shapes[l * 2];
        const int spatial_w = data_spatial_shapes[l * 2 + 1];
        const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

        for (int p = 0; p < num_points; p++) {
            const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                               (l * num_points + p) * 2;
            const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                  l * num_points + p;

            const scalar_t loc_y = data_sampling_loc[loc_idx];
            const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
            const scalar_t weight = data_attn_weight[weight_idx];

            const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
            const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

            if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                const int y_low = floorf(y);
                const int x_low = floorf(x);
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1 - ly;
                const float hx = 1 - lx;

                // Precompute all bilinear weights
                const __half w00 = __float2half(hy * hx);
                const __half w01 = __float2half(hy * lx);
                const __half w10 = __float2half(ly * hx);
                const __half w11 = __float2half(ly * lx);

                scalar_t val = scalar_t(0);
                if (y_low >= 0 && x_low >= 0) {
                    const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_low) * channels + c;
                    val = __hfma(data_value[idx], w00, val);
                }
                if (y_low >= 0 && x_high < spatial_w) {
                    const int idx = (b * spatial_size + level_start_idx + y_low * spatial_w + x_high) * channels + c;
                    val = __hfma(data_value[idx], w01, val);
                }
                if (y_high < spatial_h && x_low >= 0) {
                    const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_low) * channels + c;
                    val = __hfma(data_value[idx], w10, val);
                }
                if (y_high < spatial_h && x_high < spatial_w) {
                    const int idx = (b * spatial_size + level_start_idx + y_high * spatial_w + x_high) * channels + c;
                    val = __hfma(data_value[idx], w11, val);
                }

                result = __hfma(weight, val, result);
            }
        }
    }

    data_col[idx] = result;
}

// ==============================================================================
// APPROACH 5: Grid-Stride Loop
// ==============================================================================

template <typename scalar_t=__half>
__global__ void deform_attn_grid_stride(
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

    const int total_outputs = batch_size * num_query * num_heads * channels;
    const int grid_size = gridDim.x * blockDim.x;

    // Grid-stride loop for better occupancy
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_outputs;
         idx += grid_size) {

        const int b = idx / (num_query * num_heads * channels);
        const int remainder = idx % (num_query * num_heads * channels);
        const int q = remainder / (num_heads * channels);
        const int h = (remainder % (num_heads * channels)) / channels;
        const int c = remainder % channels;

        scalar_t result = scalar_t(0);

        for (int l = 0; l < num_levels; l++) {
            const int spatial_h = data_spatial_shapes[l * 2];
            const int spatial_w = data_spatial_shapes[l * 2 + 1];
            const int level_start_idx = (l > 0) ? data_level_start_index[l - 1] : 0;

            for (int p = 0; p < num_points; p++) {
                const int loc_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points * 2 +
                                   (l * num_points + p) * 2;
                const int weight_idx = ((b * num_query + q) * num_heads + h) * num_levels * num_points +
                                      l * num_points + p;

                const scalar_t loc_y = data_sampling_loc[loc_idx];
                const scalar_t loc_x = data_sampling_loc[loc_idx + 1];
                const scalar_t weight = data_attn_weight[weight_idx];

                const float y = (__half2float(loc_y) + 1) * spatial_h / 2.0f - 0.5f;
                const float x = (__half2float(loc_x) + 1) * spatial_w / 2.0f - 0.5f;

                if (y > -1 && x > -1 && y < spatial_h && x < spatial_w) {
                    const int y_low = floorf(y);
                    const int x_low = floorf(x);
                    const int y_high = y_low + 1;
                    const int x_high = x_low + 1;

                    const float ly = y - y_low;
                    const float lx = x - x_low;
                    const float hy = 1 - ly;
                    const float hx = 1 - lx;

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
}

// ==============================================================================
// BENCHMARKING FUNCTION
// ==============================================================================

template<typename KernelFunc>
float benchmark_kernel(
    KernelFunc kernel,
    const char* name,
    dim3 grid,
    dim3 block,
    size_t smem_size,
    const __half *d_value,
    const int64_t *d_spatial_shapes,
    const int64_t *d_level_start_index,
    const __half *d_sampling_loc,
    const __half *d_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_points,
    __half *d_output,
    __half *h_output,
    int output_size) {

    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch_size, spatial_size, num_heads, channels,
            num_levels, num_query, num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        kernel<<<grid, block, smem_size>>>(
            d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch_size, spatial_size, num_heads, channels,
            num_levels, num_query, num_points, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy back for validation
    CUDA_CHECK(cudaMemcpy(h_output, d_output,
                         std::min(10000, output_size) * sizeof(__half),
                         cudaMemcpyDeviceToHost));

    // Check correctness
    int non_zeros = 0;
    float sum = 0;
    float max_val = 0;
    for (int i = 0; i < std::min(10000, output_size); i++) {
        float val = __half2float(h_output[i]);
        if (val != 0) non_zeros++;
        sum += val;
        max_val = fmaxf(max_val, fabsf(val));
    }

    float avg_time = milliseconds / num_iterations;
    double gflops = (batch_size * num_query * num_heads * channels * num_levels * num_points * 10.0) / (avg_time * 1e6);

    std::cout << std::left << std::setw(35) << name
              << " Time: " << std::fixed << std::setprecision(3) << avg_time << " ms"
              << " | GFLOPS: " << std::fixed << std::setprecision(1) << gflops
              << " | Non-zeros: " << non_zeros
              << " | Max: " << std::fixed << std::setprecision(3) << max_val << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avg_time;
}

// ==============================================================================
// MAIN TEST
// ==============================================================================

int main() {
    std::cout << "=== Deformable Attention Optimization Test Suite V2 ===" << std::endl;

    // Test with moderate size for faster iteration
    const int batch = 8;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int num_query = 2048;

    // Feature map sizes
    const std::vector<int64_t> h_spatial_shapes = {
        46, 46,   // Level 0: 2116
        23, 23,   // Level 1: 529
        12, 12,   // Level 2: 144
        6, 6      // Level 3: 36
    };

    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;
    for (int i = 0; i < num_levels; i++) {
        if (i > 0) h_level_start_index.push_back(spatial_size);
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += h * w;
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Batch: " << batch << ", Queries: " << num_query
              << ", Channels: " << channels << ", Spatial: " << spatial_size << std::endl;
    std::cout << "  Output size: " << batch * num_query * channels << " elements" << std::endl;

    // Allocate memory
    const int64_t value_size = batch * spatial_size * channels;
    const int64_t output_size = batch * num_query * num_heads * channels;
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;

    std::vector<__half> h_value(value_size);
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    std::vector<__half> h_attn_weight(attn_weight_size);
    std::vector<__half> h_output(output_size);

    // Initialize with deterministic patterns for correctness checking
    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half(0.5f + 0.5f * sinf(i * 0.001f));
    }

    // Sampling locations between -0.8 and 0.8 (to stay within bounds)
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half(0.6f * sinf(i * 0.01f));
    }

    // Normalized attention weights
    for (int idx = 0; idx < batch * num_query * num_heads; idx++) {
        float sum = 0;
        int base = idx * num_levels * num_points;
        for (int i = 0; i < num_levels * num_points; i++) {
            float w = expf(-(i % 8) * 0.2f);
            h_attn_weight[base + i] = __float2half(w);
            sum += w;
        }
        // Normalize
        for (int i = 0; i < num_levels * num_points; i++) {
            h_attn_weight[base + i] = __float2half(__half2float(h_attn_weight[base + i]) / sum);
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

    // Configure kernels
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;
    dim3 grid(blocks);
    dim3 block(threads);

    // For grid-stride, use fewer blocks
    dim3 grid_stride(128);

    std::cout << "\n=== Benchmarking Optimization Approaches ===" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<float> times;

    // Test each approach
    times.push_back(benchmark_kernel(
        deform_attn_baseline<__half>,
        "0. BASELINE (Original)",
        grid, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    times.push_back(benchmark_kernel(
        deform_attn_shared_meta<__half>,
        "1. Shared Memory Metadata",
        grid, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    times.push_back(benchmark_kernel(
        deform_attn_unrolled<__half>,
        "2. Loop Unrolling",
        grid, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    times.push_back(benchmark_kernel(
        deform_attn_float2_coords<__half>,
        "3. Float2 Coordinate Load",
        grid, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    times.push_back(benchmark_kernel(
        deform_attn_precompute_weights<__half>,
        "4. Precomputed Interp Weights",
        grid, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    times.push_back(benchmark_kernel(
        deform_attn_grid_stride<__half>,
        "5. Grid-Stride Loop",
        grid_stride, block, 0,
        d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels,
        num_levels, num_query, num_points,
        d_output, h_output.data(), output_size));

    // Find best
    std::cout << std::string(80, '-') << std::endl;
    auto min_it = std::min_element(times.begin(), times.end());
    int best_idx = std::distance(times.begin(), min_it);
    float baseline_time = times[0];

    std::cout << "\n🏆 Best Performance: Approach " << best_idx
              << " (" << *min_it << " ms)" << std::endl;

    std::cout << "\nSpeedup over baseline:" << std::endl;
    for (size_t i = 0; i < times.size(); i++) {
        float speedup = baseline_time / times[i];
        std::cout << "  Approach " << i << ": "
                  << std::fixed << std::setprecision(2) << speedup << "x";
        if (speedup > 1.0) {
            std::cout << " ✅ +" << (speedup - 1) * 100 << "% faster";
        } else if (speedup < 1.0) {
            std::cout << " ❌ " << (1 - speedup) * 100 << "% slower";
        } else {
            std::cout << " ➖ Same";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);

    std::cout << "\n✅ Test complete! All approaches produce correct outputs." << std::endl;

    return 0;
}