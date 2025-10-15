/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_CORE_CUDA_SOFTMAX_H_
#define ONEFLOW_CORE_CUDA_SOFTMAX_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace cuda {

namespace softmax {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template<>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
  return exp(x);
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template<typename T>
__inline__ __device__ T Log(T x);

template<>
__inline__ __device__ float Log<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __logf(x);
#else
  return log(x);
#endif
}
template<>
__inline__ __device__ double Log<double>(double x) {
  return log(x);
}

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

#if CUDA_VERSION >= 11000
template<>
struct DefaultComputeType<nv_bfloat16> {
  using type = float;
};
#endif  // CUDA_VERSION >= 11000

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

enum class Algorithm {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_max[row_id] = -Inf<ComputeType>();
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<ComputeType>(); }
        }
      }
    }
    ComputeType warp_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
    }
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
          thread_sum[row_id] += row_buf[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          row_buf[i] -= warp_max[row_id];
          thread_sum[row_id] += Exp(row_buf[i]);
        } else {
          __trap();
        }
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          row_buf[i] -= Log(warp_sum[row_id]);
        } else {
          __trap();
        }
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                         const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                  rows_per_access, padding, algorithm>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, false, algorithm>(
        stream, load, store, rows, cols);
  } else {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, true, algorithm>(
        stream, load, store, rows, cols);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                        \
  else if (cols <= (thread_group_width)*pack_size) {                                               \
    if (rows % 2 == 0) {                                                                           \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 2, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    } else {                                                                                       \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 1, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    }                                                                                              \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                      \
  else if (cols <= (col)*kWarpSize) {                                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                          algorithm>(stream, load, store, rows, cols);            \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                        \
  else if (cols <= (thread_group_width)*pack_size) {                                               \
    if (rows % 2 == 0) {                                                                           \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 2, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    } else {                                                                                       \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,        \
                                            thread_group_width, 1, algorithm>(stream, load, store, \
                                                                              rows, cols);         \
    }                                                                                              \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                      \
  else if (cols <= (col)*kWarpSize) {                                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                          algorithm>(stream, load, store, rows, cols);            \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(stream, load,
                                                                                 store, rows, cols);
    } else {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(stream, load,
                                                                                 store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store,
                                                                                rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
      if (algorithm == Algorithm::kSoftmax) {
        const ComputeType exp_x = Exp(buf[col] - row_max);
        buf[col] = exp_x;
        thread_sum += exp_x;
      } else {
        const ComputeType x = buf[col] - row_max;
        buf[col] = x;
        thread_sum += Exp(x);
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
                                              const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                            STORE store, const int64_t rows,
                                                            const int64_t cols, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2,
                                      algorithm>(stream, load, store, smem, rows, cols);
  }
  *success = true;
  return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1,
                                    algorithm>(stream, load, store, smem, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols, success);
    } else {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                   const int64_t rows, const int64_t cols,
                                                   bool* success) {
  return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols, success);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         Algorithm algorithm>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += Exp(pack[i] - row_max); }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = Div(Exp(pack[i] - row_max), row_sum);
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = (pack[i] - row_max) - Log(row_sum);
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols) {
    if (cols % 2 == 0) {
      return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols);
    } else {
      return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols) {
  if (cols < 1024) {
    return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
        stream, load, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
              stream, load, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
          stream, load, store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
      stream, load, store, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
        stream, load, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
              stream, load, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
          stream, load, store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                   const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(
      stream, load, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         Algorithm algorithm>
__global__ void SoftmaxGradWarpImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
                                    const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int pack_per_thread = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  ComputeType y_buf[rows_per_access][cols_per_thread];
  ComputeType dy_buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load_y.template load<pack_size>(row_y_buf + pack_offset, row + row_id, col);
          load_dy.template load<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            if (algorithm == Algorithm::kSoftmax) {
              thread_sum[row_id] += row_y_buf[pack_offset + i] * row_dy_buf[pack_offset + i];
            } else if (algorithm == Algorithm::kLogSoftmax) {
              thread_sum[row_id] += row_dy_buf[pack_offset + i];
            } else {
              __trap();
            }
          }
        }
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          for (int i = 0; i < pack_size; ++i) {
            if (algorithm == Algorithm::kSoftmax) {
              row_dy_buf[pack_offset + i] =
                  (row_dy_buf[pack_offset + i] - warp_sum[row_id]) * row_y_buf[pack_offset + i];
            } else if (algorithm == Algorithm::kLogSoftmax) {
              row_dy_buf[pack_offset + i] -= Exp(row_y_buf[pack_offset + i]) * warp_sum[row_id];
            } else {
              __trap();
            }
          }
          store.template store<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                             STORE store, const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                      thread_group_width, rows_per_access, padding, algorithm>
      <<<grid_dim_x, block_dim, 0, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                     cols_per_thread, thread_group_width, rows_per_access, false,
                                     algorithm>(stream, load_y, load_dy, store, rows, cols);
  } else {
    return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                     cols_per_thread, thread_group_width, rows_per_access, true,
                                     algorithm>(stream, load_y, load_dy, store, rows, cols);
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 2, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    } else {                                                                                    \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 1, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                              kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                       store, rows, cols);         \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 2, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    } else {                                                                                    \
      return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, \
                                                pack_size, thread_group_width, 1, algorithm>(   \
          stream, load_y, load_dy, store, rows, cols);                                          \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, \
                                              kWarpSize, 1, algorithm>(stream, load_y, load_dy,    \
                                                                       store, rows, cols);         \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct DispatchSoftmaxGradWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                               STORE store, const int64_t rows,
                                               const int64_t cols) {
  return DispatchSoftmaxGradWarpImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(
      stream, load_y, load_dy, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockSMemImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                         const int64_t rows, const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* y_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* dy_buf = y_buf + cols;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        y_buf[i * num_packs + pack_id] = y_pack[i];
        dy_buf[i * num_packs + pack_id] = dy_pack[i];
        if (algorithm == Algorithm::kSoftmax) {
          thread_sum += y_pack[i] * dy_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          thread_sum += dy_pack[i];
        } else {
          __trap();
        }
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          pack[i] = (dy_buf[i * num_packs + pack_id] - row_sum) * y_buf[i * num_packs + pack_id];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          pack[i] = dy_buf[i * num_packs + pack_id] - Exp(y_buf[i * num_packs + pack_id]) * row_sum;
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                  LOAD_DY load_dy, STORE store, int smem,
                                                  const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, algorithm>
      <<<grid_dim_x, block_size, smem, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream, LOAD_Y load_y,
                                                                LOAD_DY load_dy, STORE store,
                                                                const int64_t rows,
                                                                const int64_t cols, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType) * 2;
  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1,
                                 algorithm>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4,
                                 algorithm>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_4, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3,
                                 algorithm>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_3, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2,
                                 algorithm>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_2, algorithm>(stream, load_y, load_dy,
                                                                        store, smem, rows, cols);
  }
  *success = true;
  return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                        block_size_conf_1, algorithm>(stream, load_y, load_dy,
                                                                      store, smem, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 2,
                                                          algorithm>(stream, load_y, load_dy, store,
                                                                     rows, cols, success);
    } else {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 1,
                                                          algorithm>(stream, load_y, load_dy, store,
                                                                     rows, cols, success);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y,
                                                       LOAD_DY load_dy, STORE store,
                                                       const int64_t rows, const int64_t cols,
                                                       bool* success) {
  return TryDispatchSoftmaxGradBlockSMemImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                     algorithm>()(stream, load_y, load_dy, store,
                                                                  rows, cols, success);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockUncachedImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                             const int64_t rows, const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          thread_sum += y_pack[i] * dy_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          thread_sum += dy_pack[i];
        } else {
          __trap();
        }
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          dy_pack[i] = (dy_pack[i] - row_sum) * y_pack[i];
        } else if (algorithm == Algorithm::kLogSoftmax) {
          dy_pack[i] -= Exp(y_pack[i]) * row_sum;
        } else {
          __trap();
        }
      }
      store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                      LOAD_DY load_dy, STORE store,
                                                      const int64_t rows, const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  SoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size,
                               algorithm>
      <<<grid_dim_x, block_size, 0, stream>>>(load_y, load_dy, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
struct DispatchSoftmaxGradBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                        LOAD_DY load_dy, STORE store,
                                                        const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                      algorithm>()(stream, load_y, load_dy, store,
                                                                   rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(
        stream, load_y, load_dy, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                            Algorithm::kSoftmax>(
          stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                  Algorithm::kSoftmax>(stream, load_y, load_dy,
                                                                       store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                    const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                              Algorithm::kSoftmax>(stream, load_y, load_dy, store,
                                                                   rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(
        stream, load_y, load_dy, store, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                            Algorithm::kLogSoftmax>(
          stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                  Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                          store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLogSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                       const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                              Algorithm::kLogSoftmax>(stream, load_y, load_dy,
                                                                      store, rows, cols);
}

}  // namespace softmax

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_SOFTMAX_H_

// ============================================================================
// Test harness
// ============================================================================

#include <iostream>
#include <cstring>
#include <cmath>

using namespace oneflow::cuda::softmax;

// Simple load functor for Softmax
template<typename T, typename ComputeType>
struct SimpleLoad {
    using LoadType = T;
    SimpleLoad(const T* ptr, int64_t cols) : ptr(ptr), cols(cols) {}

    template<int N>
    __device__ void load(ComputeType* dst, int64_t row, int64_t col) const {
        for (int i = 0; i < N; i++) {
            dst[i] = static_cast<ComputeType>(ptr[row * cols + col + i]);
        }
    }

    const T* ptr;
    int64_t cols;
};

// Simple store functor for Softmax
template<typename T, typename ComputeType>
struct SimpleStore {
    SimpleStore(T* ptr, int64_t cols) : ptr(ptr), cols(cols) {}

    template<int N>
    __device__ void store(const ComputeType* src, int64_t row, int64_t col) {
        for (int i = 0; i < N; i++) {
            ptr[row * cols + col + i] = static_cast<T>(src[i]);
        }
    }

    T* ptr;
    int64_t cols;
};

void parse_args(int argc, char** argv, int* batch, int* embed, std::string* dtype) {
    *batch = 32;
    *embed = 768;
    *dtype = "float32";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            *batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--embed") == 0 && i + 1 < argc) {
            *embed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            *dtype = argv[++i];
        }
    }
}

template<typename T>
void cpu_softmax(const T* input, T* output, int batch, int embed) {
    for (int b = 0; b < batch; b++) {
        const T* x = input + b * embed;
        T* y = output + b * embed;

        // Find max
        float max_val = static_cast<float>(x[0]);
        for (int i = 1; i < embed; i++) {
            max_val = fmax(max_val, static_cast<float>(x[i]));
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < embed; i++) {
            float exp_val = expf(static_cast<float>(x[i]) - max_val);
            y[i] = static_cast<T>(exp_val);
            sum += exp_val;
        }

        // Normalize
        for (int i = 0; i < embed; i++) {
            y[i] = static_cast<T>(static_cast<float>(y[i]) / sum);
        }
    }
}

template<typename T>
void run_test(int batch, int embed) {
    const int size = batch * embed;

    // Allocate host memory
    T* h_input = new T[size];
    T* h_output = new T[size];
    T* h_output_ref = new T[size];

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }

    // Allocate device memory
    T *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, size * sizeof(T));

    // Copy data to device
    cudaMemcpy(d_input, h_input, size * sizeof(T), cudaMemcpyHostToDevice);

    // Run GPU kernel
    using ComputeType = float;
    SimpleLoad<T, ComputeType> load(d_input, embed);
    SimpleStore<T, ComputeType> store(d_output, embed);

    DispatchSoftmaxWarpImpl<decltype(load), decltype(store), ComputeType, Algorithm::kSoftmax>(
        cudaStreamDefault, load, store, batch, embed);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_softmax(h_input, h_output_ref, batch, embed);

    // Validate
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabs(static_cast<float>(h_output[i]) - static_cast<float>(h_output_ref[i]));
        max_diff = fmax(max_diff, diff);
    }

    std::cout << "Softmax Test - Batch: " << batch << ", Embed: " << embed << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;

    if (max_diff < 1e-3f) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_ref;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char** argv) {
    int batch, embed;
    std::string dtype;

    parse_args(argc, argv, &batch, &embed, &dtype);

    std::cout << "Running Softmax with:" << std::endl;
    std::cout << "  Batch: " << batch << std::endl;
    std::cout << "  Embed: " << embed << std::endl;
    std::cout << "  Dtype: " << dtype << std::endl;
    std::cout << std::endl;

    if (dtype == "float32") {
        run_test<float>(batch, embed);
    } else if (dtype == "float16") {
        run_test<half>(batch, embed);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        return 1;
    }

    return 0;
}
