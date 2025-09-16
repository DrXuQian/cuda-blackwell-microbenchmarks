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
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define HALF2CONST(value) (const_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}


#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

template <typename T>
void random_init_vector(std::vector<T>& vec, float min_val = -1.0f, float max_val = 1.0f) {
    if (vec.empty()) return;
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    if constexpr (std::is_same_v<T, __half>) {
        for (auto& val : vec) {
            val = __float2half(dist(gen));
        }
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<int64_t> int_dist(16, 256);
        for (auto& val : vec) {
            val = int_dist(gen);
        }
    }
    else {
        for (auto& val : vec) {
            val = static_cast<T>(dist(gen));
        }
    }
}

// Simplified TMA copy function for distributed shared memory
__device__ void tma_copy_tensor_3d_device(
    __half* smem_ptr,
    const __half* gmem_ptr,
    int bi_offset,
    int spatial_offset,
    int channel_offset,
    int spatial_size,
    int channels
) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    const int cluster_size = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();

    // Each block copies 2 channels
    const int channels_per_block = 2;
    const int my_channel_start = block_rank * channels_per_block;
    const int my_channel_offset = channel_offset + my_channel_start;

    // Calculate source address for this block's 2 channels
    const __half* src = gmem_ptr +
        (bi_offset * spatial_size * channels) +
        (spatial_offset * channels) +
        my_channel_offset;

    // Copy full spatial dimension for 2 channels
    const int copy_spatial = spatial_size;
    const int copy_channels = channels_per_block;

    // Cooperative copy - all threads in block participate
    int total_elements = copy_spatial * copy_channels;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int s = idx / copy_channels;
        int c = idx % copy_channels;
        if (s < copy_spatial && c < copy_channels) {
            smem_ptr[s * copy_channels + c] = src[s * channels + c];
        }
    }
}

template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32,
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query,
    scalar_t *data_col) {
    extern __shared__ __half smem[];
    namespace cg = cooperative_groups;
    int tid = threadIdx.x;
    constexpr int channels_per_block = 2;
    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_size = cluster.dim_blocks().x;
    const int channel_outer_loops = CHANNELS / (channels_per_block * cluster_size);
    constexpr int channels_per_thread = 16;
    unsigned int block_rank = cluster.block_rank();
    int bi = blockIdx.x / (channel_outer_loops * cluster_size);
    int oc = (blockIdx.x % (channel_outer_loops * cluster_size)) / cluster_size;

    // All threads call the TMA copy function (cooperative copy inside)
    int channel_offset = (block_rank * channels_per_block) + (oc * channels_per_thread);
    int spatial_offset = 0;
    int bi_offset = bi;
    tma_copy_tensor_3d_device(&smem[0], data_value, bi_offset, spatial_offset, channel_offset, spatial_size, CHANNELS);

    // Synchronize after TMA copy
    __syncthreads();
    cluster.sync();

    const __half kONE = __int2half_rz(1);

    scalar_t *data_col_base = data_col + (bi * num_query * CHANNELS);
    int sampling_index = ((num_query + blockDim.x - 1) / blockDim.x) * tid;
    int data_weight_ptr = (sampling_index << (LEVEL_SHIFT + POINT_SHIFT)) + num_query * bi * NUM_LEVELS * NUM_POINT;
    int data_loc_w_ptr = data_weight_ptr << 1;
    scalar_t *data_half = const_cast<scalar_t *>(data_sampling_loc);
    scalar_t *data_attn_weight_half = const_cast<scalar_t *>(data_attn_weight);
    const half2 zp5 = half2(0.5f, 0.5f);

    for (int loc = sampling_index; (loc < num_query) and (loc < sampling_index + blockDim.x); loc ++) {
        scalar_t col[channels_per_thread] = {};
        scalar_t * data_col_ptr = data_col_base + loc * CHANNELS + (oc * channels_per_thread);
        for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const half2 spatail_hw = half2(spatial_w, spatial_h);
            half2 loc_hw_vec[NUM_POINT];
            half  weight_vec[NUM_POINT];
            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4){
                LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
            }
            LDST128BITS(weight_vec[0])      = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr]));
            data_loc_w_ptr += (NUM_POINT << 1);
            data_weight_ptr += NUM_POINT;

            #pragma unroll
            for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
                const half2 loc = loc_hw_vec[p_col];
                const scalar_t weight = weight_vec[p_col];
                half2 weighthalf2 = half2(weight, weight);
                half2 hw_im = __hfma2(loc, spatail_hw, zp5);
                scalar_t h_im = __high2half(hw_im);
                scalar_t w_im = __low2half(hw_im);

                if (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) && h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1)) {
                    int32_t const hLow = __half2int_rd(h_im);
                    int32_t const wLow = __half2int_rd(w_im);
                    const __half lh = __hsub(h_im, __int2half_rd(hLow));
                    const __half lw = __hsub(w_im, __int2half_rd(wLow));
                    const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);
                    int32_t const hLowPtrOffset = hLow * (spatial_w + 2) * channels_per_block;;
                    int32_t const hHighPtrOffset = hLowPtrOffset + (spatial_w + 2) * channels_per_block;
                    int32_t const wLowPtrOffset = wLow * channels_per_block;
                    int32_t const wHighPtrOffset = wLowPtrOffset + channels_per_block;
                    __half pst_lh[4] = {hh, hh, lh, lh};
                    __half pst_rh[4] = {hw, lw, hw, lw};
                    __half wdata[4] ;
                    HALF2(wdata[0]) = __hmul2(HALF2(pst_lh[0]), HALF2(pst_rh[0]));
                    HALF2(wdata[2]) = __hmul2(HALF2(pst_lh[2]), HALF2(pst_rh[2]));
                    __half wdataexp[2];
                    __half vdata2d[2] ;
                    int32_t const ptrs[4] = {hLowPtrOffset + wLowPtrOffset, hLowPtrOffset + wHighPtrOffset,
                                            hHighPtrOffset + wLowPtrOffset, hHighPtrOffset + wHighPtrOffset};

                    HALF2(wdataexp[0]) = __hmul2(half2(wdata[0],  wdata[0]), HALF2(weighthalf2));
                    #pragma unroll
                    for (auto ptr: ptrs){
                        #pragma unroll
                        for (int dst_block_rank = 0; dst_block_rank < cluster_size; dst_block_rank++){
                            __half *data_value_ptr = cluster.map_shared_rank(smem, dst_block_rank);
                            // Each block has only 2 channels, so load them individually
                            vdata2d[0] = data_value_ptr[ptr];
                            vdata2d[1] = data_value_ptr[ptr + 1];
                            HALF2(col[dst_block_rank << 1]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[0]), HALF2(col[dst_block_rank << 1]));
                        }
                    }
                }
            }
            #pragma unroll
            for (int idx = 0; idx < channels_per_thread; idx += 8){
                __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
                data_col_ptr += 8;
            }
        }
    }
}

template <typename scalar_t=__half>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  constexpr int num_threads = 256;  // Reduced thread count for smaller workload
  if (num_heads == 1 and num_point == 8 and num_levels == 4 and channels == 32){
    {
        cudaLaunchConfig_t config = {0};
        int cluster_size = 8;
        config.gridDim = batch_size * cluster_size * 2;
        config.blockDim = num_threads;

        // Calculate actual shared memory required
        const int channels_per_block = 2;
        // Using actual spatial size, not hardcoded value
        config.dynamicSmemBytes = spatial_size * channels_per_block * sizeof(__half);

        std::cout << "Shared memory per block: " << config.dynamicSmemBytes << " bytes ("
                  << config.dynamicSmemBytes / 1024.0 << " KB)" << std::endl;
        std::cout << "Total distributed shared memory: " << config.dynamicSmemBytes * cluster_size
                  << " bytes (" << (config.dynamicSmemBytes * cluster_size) / 1024.0 << " KB)" << std::endl;

        // Check if shared memory fits within limits
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes ("
                  << prop.sharedMemPerBlock / 1024.0 << " KB)" << std::endl;
        std::cout << "Max shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor
                  << " bytes (" << prop.sharedMemPerMultiprocessor / 1024.0 << " KB)" << std::endl;

        if (config.dynamicSmemBytes > 99 * 1024) {
            std::cerr << "ERROR: Shared memory requirement exceeds 99KB opt-in limit!" << std::endl;
            std::cerr << "Consider reducing spatial_size or channels_per_block" << std::endl;
            return;
        }

        // Set the maximum dynamic shared memory size for the kernel
        CUDA_CHECK(::cudaFuncSetAttribute((void *)ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5>,
              data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query,  data_col);
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

int main() {
    // === Using smaller input sizes to reduce shared memory requirements ===
    const int batch = 8;  // Reduced from 48

    // Calculate smaller spatial sizes that fit within shared memory limits
    // Target: ~48KB per block (well within 99KB limit)
    // With channels_per_block = 2 and sizeof(__half) = 2:
    // spatial_size * 2 * 2 = 48KB = 49152 bytes
    // spatial_size = 49152 / 4 = 12288

    // Use smaller feature map resolutions
    const std::vector<int64_t> h_spatial_shapes = {
        64, 64,   // Level 0: 64x64 = 4096
        32, 32,   // Level 1: 32x32 = 1024
        16, 16,   // Level 2: 16x16 = 256
        8, 8      // Level 3: 8x8 = 64
    };

    // Calculate total spatial size
    int spatial_size = 0;
    std::vector<int64_t> h_level_start_index;
    h_level_start_index.push_back(0);

    for (int i = 0; i < h_spatial_shapes.size() / 2; i++) {
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += h * w;
        if (i < h_spatial_shapes.size() / 2 - 1) {
            h_level_start_index.push_back(spatial_size);
        }
    }

    const int num_query = 2048;  // Reduced from 15422
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;

    std::cout << "=== MS-Deformable Attention with Distributed Shared Memory ===" << std::endl;
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
    std::cout << "  Level start indices: ";
    for (auto idx : h_level_start_index) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // Calculate element counts
    long long value_elements = batch * num_query * num_heads * channels;
    const int64_t value_size = batch * spatial_size * channels;

    std::cout << "\nGenerating test data..." << std::endl;
    std::vector<__half> h_value(value_size);
    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half((i % 100) * 0.01f);
    }

    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half((i % 200 - 100) * 0.01f);
    }

    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;
    std::vector<__half> h_attn_weight(attn_weight_size);
    for (size_t i = 0; i < h_attn_weight.size(); i++) {
        h_attn_weight[i] = __float2half((i % 100) * 0.01f);
    }

    // Allocate GPU memory
    std::cout << "Allocating GPU memory..." << std::endl;
    __half* d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t* d_spatial_shapes;
    int64_t* d_level_start_index;

    CUDA_CHECK(cudaMalloc(&d_value, h_value.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, h_sampling_loc.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, h_attn_weight.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, value_elements * sizeof(__half)));

    // Copy data to GPU
    std::cout << "Copying data to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), h_value.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), h_sampling_loc.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), h_attn_weight.size() * sizeof(__half), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    std::cout << "\nRunning warmup..." << std::endl;
    for (int i = 0; i < 5; i++) {
        ms_deformable_im2col_cuda(
                  stream,
                  d_value,
                  d_spatial_shapes,
                  d_level_start_index,
                  d_sampling_loc,
                  d_attn_weight,
                  batch, spatial_size, num_heads, channels, num_levels, num_query,
                  num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::cout << "\nRunning benchmark..." << std::endl;
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        ms_deformable_im2col_cuda(
                  stream,
                  d_value,
                  d_spatial_shapes,
                  d_level_start_index,
                  d_sampling_loc,
                  d_attn_weight,
                  batch, spatial_size, num_heads, channels, num_levels, num_query,
                  num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Average kernel time: " << duration.count() / num_iterations << " microseconds" << std::endl;
    std::cout << "Throughput: " << (1000000.0 * num_iterations) / duration.count() << " iterations/second" << std::endl;

    // Copy output back for verification
    std::vector<__half> h_output(value_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                         value_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    // Print first few output values for verification
    std::cout << "\nFirst 20 output values: ";
    for (int i = 0; i < std::min(20LL, value_elements); i++) {
        if (i % 10 == 0) std::cout << "\n  ";
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // Check for any non-zero outputs (basic sanity check)
    bool has_nonzero = false;
    for (int i = 0; i < value_elements && !has_nonzero; i++) {
        if (__half2float(h_output[i]) != 0.0f) {
            has_nonzero = true;
        }
    }
    std::cout << "\nOutput validation: " << (has_nonzero ? "PASS (has non-zero values)" : "FAIL (all zeros)") << std::endl;

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return 0;
}