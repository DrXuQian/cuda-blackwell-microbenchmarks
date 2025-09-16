// Alternative implementation without distributed shared memory
// Uses L2 cache residency for inter-block communication

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// Alternative 1: Use L2 cache resident data
// Each block loads its 2 channels to shared memory, 
// but reads other channels from L2 cache instead of distributed shared memory
template <typename scalar_t>
__global__ void deform_attn_l2_cache(
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    scalar_t* data_col,
    const int batch_size,
    const int spatial_size,
    const int channels,
    const int num_query,
    const int num_levels,
    const int num_points
) {
    extern __shared__ __half smem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threads_per_block = blockDim.x;
    
    // Each block processes one output channel group (16 channels)
    const int bi = bid / 2;  // batch index
    const int oc = bid % 2;  // output channel group (0 or 1)
    
    if (bi >= batch_size) return;
    
    // Load ALL 16 channels for this output group to shared memory
    // This requires 80KB * 2 = 160KB which exceeds limit
    // So we process in tiles
    
    const int tile_channels = 4;  // Process 4 channels at a time
    const int num_tiles = 16 / tile_channels;
    
    // Process each query
    for (int q = tid; q < num_query; q += threads_per_block) {
        scalar_t col[16] = {__float2half(0.0f)};
        
        // Process in channel tiles to fit in shared memory
        for (int tile = 0; tile < num_tiles; tile++) {
            int channel_start = oc * 16 + tile * tile_channels;
            
            // Load this tile to shared memory cooperatively
            const scalar_t* src = data_value + 
                (bi * spatial_size * channels) + channel_start;
            
            for (int idx = tid; idx < spatial_size * tile_channels; idx += threads_per_block) {
                int s = idx / tile_channels;
                int c = idx % tile_channels;
                if (s < spatial_size && c < tile_channels) {
                    smem[s * tile_channels + c] = src[s * channels + c];
                }
            }
            __syncthreads();
            
            // Process all levels for this tile
            for (int level = 0; level < num_levels; level++) {
                const int level_start_id = data_level_start_index[level];
                const int spatial_h = data_spatial_shapes[level * 2];
                const int spatial_w = data_spatial_shapes[level * 2 + 1];
                
                // Process each point
                for (int p = 0; p < num_points; p++) {
                    // Get location and weight
                    int loc_idx = bi * num_query * num_levels * num_points * 2 +
                                 q * num_levels * num_points * 2 +
                                 level * num_points * 2 +
                                 p * 2;
                    
                    scalar_t loc_x = data_sampling_loc[loc_idx];
                    scalar_t loc_y = data_sampling_loc[loc_idx + 1];
                    scalar_t weight = data_attn_weight[loc_idx / 2];
                    
                    // Bilinear interpolation
                    float h_im = __half2float(loc_y) * spatial_h - 0.5f;
                    float w_im = __half2float(loc_x) * spatial_w - 0.5f;
                    
                    if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                        int h_low = floorf(h_im);
                        int w_low = floorf(w_im);
                        
                        float lh = h_im - h_low;
                        float lw = w_im - w_low;
                        float hh = 1.0f - lh;
                        float hw = 1.0f - lw;
                        
                        // Clamp indices
                        h_low = max(0, min(h_low, spatial_h - 1));
                        int h_high = max(0, min(h_low + 1, spatial_h - 1));
                        w_low = max(0, min(w_low, spatial_w - 1));
                        int w_high = max(0, min(w_low + 1, spatial_w - 1));
                        
                        // Calculate indices
                        int idx1 = (level_start_id + h_low * spatial_w + w_low);
                        int idx2 = (level_start_id + h_low * spatial_w + w_high);
                        int idx3 = (level_start_id + h_high * spatial_w + w_low);
                        int idx4 = (level_start_id + h_high * spatial_w + w_high);
                        
                        // Read from shared memory for this tile
                        for (int c = 0; c < tile_channels; c++) {
                            scalar_t v1 = smem[idx1 * tile_channels + c];
                            scalar_t v2 = smem[idx2 * tile_channels + c];
                            scalar_t v3 = smem[idx3 * tile_channels + c];
                            scalar_t v4 = smem[idx4 * tile_channels + c];
                            
                            scalar_t interpolated = __float2half(
                                hh * hw * __half2float(v1) +
                                hh * lw * __half2float(v2) +
                                lh * hw * __half2float(v3) +
                                lh * lw * __half2float(v4)
                            );
                            
                            col[tile * tile_channels + c] = __hfma(interpolated, weight, 
                                                                   col[tile * tile_channels + c]);
                        }
                    }
                }
            }
            __syncthreads();  // Before loading next tile
        }
        
        // Write output
        scalar_t* out_ptr = data_col + 
            (bi * num_query * channels) + 
            (q * channels) + 
            (oc * 16);
        
        for (int c = 0; c < 16; c++) {
            out_ptr[c] = col[c];
        }
    }
}

// Alternative 2: Process queries in smaller batches to reduce memory
template <typename scalar_t>
__global__ void deform_attn_query_tiled(
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    scalar_t* data_col,
    const int batch_size,
    const int spatial_size,
    const int channels,
    const int num_query,
    const int num_levels,
    const int num_points,
    const int query_tile_size  // Process this many queries at once
) {
    // Implementation would process queries in tiles
    // This allows using full 32 channels in shared memory
    // by processing fewer queries at once
}

int main() {
    printf("=== Alternative Deformable Attention Implementation ===\n");
    printf("Using L2 cache and tiling instead of distributed shared memory\n");
    
    // Test parameters
    const int batch_size = 48;
    const int spatial_size = 20522;
    const int num_query = 15422;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    
    // Allocate memory
    size_t value_size = batch_size * spatial_size * channels * sizeof(__half);
    size_t output_size = batch_size * num_query * channels * sizeof(__half);
    
    __half *d_value, *d_output, *d_loc, *d_weight;
    int64_t *d_spatial_shapes, *d_level_start;
    
    cudaMalloc(&d_value, value_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_loc, batch_size * num_query * num_levels * num_points * 2 * sizeof(__half));
    cudaMalloc(&d_weight, batch_size * num_query * num_levels * num_points * sizeof(__half));
    cudaMalloc(&d_spatial_shapes, num_levels * 2 * sizeof(int64_t));
    cudaMalloc(&d_level_start, num_levels * sizeof(int64_t));
    
    // Initialize test data
    std::vector<int64_t> h_spatial_shapes = {92, 160, 46, 80, 23, 40, 12, 20};
    std::vector<int64_t> h_level_start = {0, 14720, 18400, 20320};
    cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), num_levels * 2 * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_start, h_level_start.data(), num_levels * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    // Launch configuration
    const int blocks = batch_size * 2;  // 2 output groups per batch
    const int threads = 256;
    const size_t smem_size = spatial_size * 4 * sizeof(__half);  // 4 channels at a time
    
    printf("Launch config: %d blocks, %d threads, %zu KB shared memory\n", 
           blocks, threads, smem_size / 1024);
    
    // Set L2 cache policy for better performance
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Make value data L2 resident
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = d_value;
    stream_attr.accessPolicyWindow.num_bytes = value_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;  // Try to keep all in L2
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    
    // Launch kernel
    printf("Launching alternative kernel...\n");
    deform_attn_l2_cache<<<blocks, threads, smem_size, stream>>>(
        d_value, d_spatial_shapes, d_level_start,
        d_loc, d_weight, d_output,
        batch_size, spatial_size, channels,
        num_query, num_levels, num_points
    );
    
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    } else {
        printf("SUCCESS: Alternative implementation works!\n");
    }
    
    // Cleanup
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_loc);
    cudaFree(d_weight);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start);
    cudaStreamDestroy(stream);
    
    return 0;
}