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

template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32, 
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
                                    const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query,
    scalar_t *data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index << NUM_OUTPUT_SHIFT;
    const int c_col = _temp & (CHANNELS -1 ); //_temp % CHANNELS;
    _temp = (_temp >> CHANNELS_SHIFT);
    const int sampling_index = _temp;
    const int b_col = (float)_temp/(float)num_query;
    const __half kZERO = __int2half_rz(0);
    const __half kONE = __int2half_rz(1);
    int32_t const wStride = CHANNELS; // 256 

    scalar_t *data_col_ptr = data_col + (index << NUM_OUTPUT_SHIFT);
    int data_weight_ptr = sampling_index << (LEVEL_SHIFT + POINT_SHIFT); // * NUM_LEVELS * NUM_POINT;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int data_value_ptr_init_offset = (b_col * spatial_size) << CHANNELS_SHIFT;
    scalar_t col[NUM_OUTPUT];
    #pragma unroll
    for (int idx = 0; idx < (NUM_OUTPUT >> 1); idx += 1) {
        reinterpret_cast<__half2*>(col)[idx] = half2(0.0f, 0.0f);
    }
    scalar_t *data_half = const_cast<scalar_t *>(data_sampling_loc);
    scalar_t *data_attn_weight_half = const_cast<scalar_t *>(data_attn_weight);
    const half2 zp5 = half2(0.5f, 0.5f);

    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      int32_t const hStride = (spatial_w + 2) << CHANNELS_SHIFT;

      // h -> hight , w -> low 
      const half2 spatail_hw = half2(spatial_w, spatial_h);
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + (level_start_id << (CHANNELS_SHIFT)));

      // load data_sampling_loc and  data_attn_weight for NUM_POINT
      // NUM_POINT 4;
      half2 loc_hw_vec[NUM_POINT]; // 8 FP16 = 128 bit  
      half  weight_vec[NUM_POINT]; // 4 FP16 = 64 bit 
      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4){
        // LDST128BITS(loc_hw_vec[pack_id]) = LDST128BITS(data_half[data_loc_w_ptr + (pack_id << 1)]);
        LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
      }
      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
        // FLOAT2(weight_vec[pack_id])      = FLOAT2(data_attn_weight_half[data_weight_ptr + pack_id]) ;
        LDST128BITS(weight_vec[pack_id])      = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
      }
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
          int32_t const hHigh = hLow + 1;
          int32_t const wHigh = wLow + 1;
          const __half lh = __hsub(h_im, __int2half_rd(hLow));
          const __half lw = __hsub(w_im, __int2half_rd(wLow));
          const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);
          int32_t const hLowPtrOffset = hLow * hStride;
          int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
          int32_t const wLowPtrOffset = wLow << CHANNELS_SHIFT;
          int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
          __half pst_lh[4] = {hh, hh, lh, lh};
          __half pst_rh[4] = {hw, lw, hw, lw};
          __half wdata[4] ;
          HALF2(wdata[0]) = __hmul2(HALF2(pst_lh[0]), HALF2(pst_rh[0]));
          HALF2(wdata[2]) = __hmul2(HALF2(pst_lh[2]), HALF2(pst_rh[2]));
          __half wdataexp[2];
          __half vdata2d[NUM_OUTPUT] ; 
          int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[0],  wdata[0]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr1 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[1],  wdata[1]), HALF2(weighthalf2));
          int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + c_col;
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr2 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }
          int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[2],  wdata[2]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr3 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }
          int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[3],  wdata[3]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr4 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }
        }
      }
    }
    #pragma unroll
    for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
      // LDST128BITS(*data_col_ptr) = LDST128BITS(col[idx]);
      __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
      data_col_ptr += 8;
    }
  }
}

template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  // 8 warp, optimal threads for MIG 
  const int num_threads = THREADS_IN_ONE_BLOCK;
  if (num_heads == 1 and num_point == 8 and num_levels == 4 and channels == 32){
    ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query,  data_col);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}
// 函数检查 CUDA API 调用的返回值
#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
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
    std::cout << "Random generating data..." << std::endl;
    const int64_t value_size = batch * spatial_size * channels;
    std::vector<__half> h_value(value_size);
    random_init_vector(h_value, -64.0f, 64.0f);
    const std::vector<int64_t> h_spatial_shapes = {92, 160, 46, 80, 23, 40, 12, 20};
    const std::vector<int64_t> h_level_start_index = {0, 15228, 19164, 20214};
    const int64_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    std::vector<__half> h_sampling_loc(sampling_loc_size);
    random_init_vector(h_sampling_loc, -1.0f, 1.0f); 
    const int64_t attn_weight_size = batch * num_query * num_heads * num_levels * num_points;
    std::vector<__half> h_attn_weight(attn_weight_size);
    random_init_vector(h_attn_weight, 0.0f, 1.0f);
    
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
    std::cout << "Launching CUDA kernel..." << std::endl;
    ms_deformable_im2col_cuda(
              stream,
              d_value,
              d_spatial_shapes,
              d_level_start_index,
              d_sampling_loc,
              d_attn_weight,
              batch, spatial_size, num_heads, channels, num_levels, num_query,
              num_points, d_output);

    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 kernel 执行完成

    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);

    return 0;
}
