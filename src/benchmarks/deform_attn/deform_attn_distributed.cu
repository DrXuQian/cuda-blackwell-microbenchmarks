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
    // 随机数生成器（固定种子确保结果可复现）
    static std::mt19937 gen(42); // 种子=42
    std::uniform_real_distribution<float> dist(min_val, max_val);
    // 对FP16特殊处理（先生成float再转换）
    if constexpr (std::is_same_v<T, __half>) {
        for (auto& val : vec) {
            val = __float2half(dist(gen));
        }
    }     // 对int64_t特殊处理（生成合理范围的整数）
    else if constexpr (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<int64_t> int_dist(16, 256); // 特征图尺寸范围16~256
        for (auto& val : vec) {
            val = int_dist(gen);
        }
    }// 其他类型默认生成float
    else {
        for (auto& val : vec) {
            val = static_cast<T>(dist(gen));
        }
    }
}

/// 使用 PTX 指令在 Hopper 架构下进行 TMA 拷贝（kernel 内使用）
///
/// @param smem_ptr       目标 shared memory 地址
/// @param desc           TMA descriptor（已经在 device 上准备好）
/// @param bi_offset      batch 维度偏移
/// @param spatial_offset spatial 维度偏移
/// @param channel_offset channel 维度偏移
__device__ void tma_copy_tensor_3d_device(
    __half* smem_ptr,
    const __half* gmem_ptr,
    int bi_offset,
    int spatial_offset,
    int channel_offset
) {
    // Each block in cluster handles 2 channels for full spatial dimension
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    const int cluster_size = cluster.dim_blocks().x;  // Should be 8
    const int block_rank = cluster.block_rank();
    const int spatial_size = 20522;
    const int channels = 32;
    
    // Each block copies 2 channels (block 0: ch 0-1, block 1: ch 2-3, etc.)
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


//<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT>
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
    const int channel_outer_loops = CHANNELS / (channels_per_block * cluster_size); // 2
    constexpr int channels_per_thread = 16;  // Maximum channels per thread
    unsigned int block_rank = cluster.block_rank();
    int bi = blockIdx.x / (channel_outer_loops * cluster_size); // 16 = number of cluster * number of outer channels
    int oc = (blockIdx.x % (channel_outer_loops * cluster_size)) / cluster_size; // 2 = number of outer channels
    
    // All threads call the TMA copy function (cooperative copy inside)
    int channel_offset = (block_rank * channels_per_block) + (oc * channels_per_thread);
    int spatial_offset = 0;
    int bi_offset = bi;
    tma_copy_tensor_3d_device(&smem[0], data_value, bi_offset, spatial_offset, channel_offset);
    
    // Synchronize after TMA copy
    __syncthreads();
    cluster.sync();

    const __half kONE = __int2half_rz(1);
   
    scalar_t *data_col_base = data_col + (bi * num_query * CHANNELS); // * CHANNELS
    int sampling_index = ((num_query + blockDim.x - 1) / blockDim.x) * tid;
    int data_weight_ptr = (sampling_index << (LEVEL_SHIFT + POINT_SHIFT)) + num_query * bi * NUM_LEVELS * NUM_POINT; // * NUM_LEVELS * NUM_POINT;
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
            // h -> hight , w -> low 
            const half2 spatail_hw = half2(spatial_w, spatial_h);
            // load data_sampling_loc and  data_attn_weight for NUM_POINT
            // NUM_POINT 4;
            half2 loc_hw_vec[NUM_POINT]; // 8 FP16 = 128 bit  
            half  weight_vec[NUM_POINT]; // 4 FP16 = 64 bit 
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
    // Final sync not needed - all work is complete
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
  // 8 warp, optimal threads for MIG 
  constexpr int num_threads = 512;
  constexpr int THREADS_IN_ONE_BLOCK = 512;
  if (num_heads == 1 and num_point == 8 and num_levels == 4 and channels == 32){
    // Launch via extensible launch
    {
        cudaLaunchConfig_t config = {0};
        int cluster_size = 8; // size 2 is an example here
        config.gridDim = batch_size * cluster_size * 2;
        config.blockDim = num_threads;

        // cluster_size depends on the histogram size.
        // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
        //dynamic shared memory size is per block.
        //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
        // Calculate shared memory more conservatively
        // Each block stores 2 channels for full spatial dimension
        // 20522 * 2 * sizeof(__half) = 20522 * 2 * 2 = 82,088 bytes per block
        // This fits within the 99KB opt-in limit
        const int channels_per_block = 2;
        config.dynamicSmemBytes = spatial_size * channels_per_block * sizeof(__half);  // ~80KB per block
        // Debug output removed for cleaner output
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
    // === 1. 手动定义所有元数据和维度 ===
    // 这些值需要和你 Python 脚本打印出来的一致
    const int batch = 48;
    const int spatial_size = 20522; 
    const int num_query = 15422;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int im2col_step = 64; // 或者你的具体值

    // 根据维度计算元素数量
    long long value_elements = batch * num_query * num_heads * channels;
    long long spatial_shapes_elements = num_levels * 2;
    std::cout << "Loading data from .bin files..." << std::endl;
    std::cout << "Generating test data..." << std::endl;
    const int64_t value_size = batch * spatial_size * channels;
    std::vector<__half> h_value(value_size);
    // Simple initialization for faster testing
    for (size_t i = 0; i < h_value.size(); i++) {
        h_value[i] = __float2half((i % 100) * 0.01f);
    }
    const std::vector<int64_t> h_spatial_shapes = {92, 160, 46, 80, 23, 40, 12, 20};
    const std::vector<int64_t> h_level_start_index = {0, 15228, 19164, 20214};
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    for (size_t i = 0; i < h_sampling_loc.size(); i++) {
        h_sampling_loc[i] = __float2half((i % 200 - 100) * 0.01f);  // Range -1 to 1
    }
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;
    std::vector<__half> h_attn_weight(attn_weight_size);
    for (size_t i = 0; i < h_attn_weight.size(); i++) {
        h_attn_weight[i] = __float2half((i % 100) * 0.01f);  // Range 0 to 1
    }
    
    // === 3. 在 GPU (Device) 上分配内存 ===
    std::cout << "Allocating GPU memory..." << std::endl;
    __half* d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t* d_spatial_shapes;
    int64_t* d_level_start_index;
    CUDA_CHECK(cudaMalloc(&d_value, h_value.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, h_sampling_loc.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, h_attn_weight.size() * sizeof(__half)));
    // 为输出分配内存 (输出维度和 value 一样)
    CUDA_CHECK(cudaMalloc(&d_output, value_elements * sizeof(__half)));
    // === 4. 将数据从 Host 拷贝到 Device ===
    std::cout << "Copying data from Host to Device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), h_value.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), h_sampling_loc.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), h_attn_weight.size() * sizeof(__half), cudaMemcpyHostToDevice));
    // 声明一个 cudaStream_t 变量，通常是一个指针或句柄
    cudaStream_t stream;

    // 调用 cudaStreamCreate 来创建一个新的、非阻塞的异步 Stream
    // 第一个参数是 stream 变量的地址
    cudaError_t status = cudaStreamCreate(&stream);
    // 总是检查 CUDA API 调用的返回值，这是一个好习惯
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(status));
        // 在这里处理错误，例如退出程序
    }
    // === 5. 调用你的独立 CUDA 函数 ===
    // Warmup
    std::cout << "Running warmup..." << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout<< "Warmup iteration " << i+1 << std::endl;
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
    std::cout << "Running benchmark..." << std::endl;
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        std::cout<< "Benchmark iteration " << i+1 << std::endl;
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
    
    std::cout << "Average kernel time: " << duration.count() / num_iterations << " microseconds" << std::endl;
    
    // Copy output back for verification
    std::vector<__half> h_output(value_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 
                         value_elements * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Print first few output values for verification
    std::cout << "\nFirst 20 output values (original): ";
    for (int i = 0; i < std::min(20LL, value_elements); i++) {
        if (i % 10 == 0) std::cout << "\n  ";
        std::cout << std::fixed << std::setprecision(4) << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);

    return 0;
}
