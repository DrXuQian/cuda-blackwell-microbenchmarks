/*
 * 独立的 RMSNorm 测试程序
 * 复制 TensorRT-LLM 核心 kernel 代码，与 Qwen2 PyTorch 实现进行精度对比
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// 类型工具 (简化版)
// ============================================================================

template <typename T>
struct num_elems {
    static constexpr int value = 1;
};

template <>
struct num_elems<half2> {
    static constexpr int value = 2;
};

template <>
struct num_elems<float2> {
    static constexpr int value = 2;
};

// Warp reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

// Block reduction
template <typename T, int NUM>
__inline__ __device__ void blockReduceSumV2(T* val) {
    if (blockDim.x <= 32) {
        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
            }
        }
    } else {
        static __shared__ T shared[NUM][32];
        int lane = threadIdx.x & 0x1f;
        int wid = threadIdx.x >> 5;

        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < NUM; i++) {
                shared[i][wid] = val[i];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            val[i] = (threadIdx.x < (blockDim.x >> 5)) ? shared[i][lane] : (T)0.0f;
        }

        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
            }
        }
    }
}

// ============================================================================
// TensorRT-LLM RMSNorm Kernel (简化版，无量化)
// ============================================================================

/*
 * 这是从 TensorRT-LLM rmsnormKernels.cu 简化而来的 kernel
 * 去除了量化相关代码，只保留核心 RMSNorm 逻辑
 *
 * 公式: output = (input / sqrt(mean(input²) + eps)) * weight
 */
template <typename T, bool USE_SHMEM>
__global__ void trtllmRmsNormKernel(
    T const* __restrict__ input,
    T const* __restrict__ gamma,
    T* __restrict__ output,
    float eps,
    int hidden_dim
) {
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_variance;

    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;

    // Step 1: 计算方差 variance = mean(input²)
    float local_var_sum = 0.0f;

    constexpr int num_elems_T = num_elems<T>::value;
    int const n_elems = hidden_dim / num_elems_T;

    for (int i = tidx; i < n_elems; i += blockDim.x) {
        T const val = input[bidx * n_elems + i];
        if (USE_SHMEM) {
            shmem[i] = val;
        }

        float val_f = static_cast<float>(val);
        local_var_sum += val_f * val_f;
    }

    // Reduction
    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);
    float variance = packed[0];

    // Step 2: 计算 rsqrt(variance + eps)
    if (tidx == 0) {
        variance = variance / hidden_dim;  // mean(x²)
        s_variance = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Step 3: 应用归一化和 weight
    for (int i = tidx; i < n_elems; i += blockDim.x) {
        int const index = bidx * n_elems + i;
        float val_f = static_cast<float>(USE_SHMEM ? shmem[i] : input[index]);
        float gamma_f = static_cast<float>(gamma[i]);

        // RMSNorm 公式
        float result = val_f * s_variance * gamma_f;

        output[index] = static_cast<T>(result);
    }
}

// Kernel launcher
template <typename T>
void launchTRTLLMRmsNorm(
    T const* input,
    T const* gamma,
    T* output,
    float eps,
    int tokens,
    int hidden_dim,
    cudaStream_t stream = 0
) {
    dim3 grid(tokens);
    dim3 block(std::min(hidden_dim, 1024));
    block.x = 32 * ((block.x + 31) / 32);

    size_t shmem_size = hidden_dim * sizeof(T);
    bool use_shmem = true;

    if (shmem_size >= (48 << 10)) {
        cudaError_t ret = cudaFuncSetAttribute(
            trtllmRmsNormKernel<T, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        use_shmem = (ret == cudaSuccess);
    }

    if (use_shmem) {
        trtllmRmsNormKernel<T, true><<<grid, block, shmem_size, stream>>>(
            input, gamma, output, eps, hidden_dim
        );
    } else {
        trtllmRmsNormKernel<T, false><<<grid, block, 0, stream>>>(
            input, gamma, output, eps, hidden_dim
        );
    }
}

// ============================================================================
// CPU 参考实现 (模拟 Qwen2RMSNorm)
// ============================================================================

void qwen2RmsNormCPU(
    float const* input,
    float const* weight,
    float* output,
    float eps,
    int tokens,
    int hidden_dim
) {
    for (int token = 0; token < tokens; token++) {
        // 计算方差
        float variance = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float val = input[token * hidden_dim + i];
            variance += val * val;
        }
        variance /= hidden_dim;

        // 计算 rsqrt
        float rsqrt_var = 1.0f / sqrtf(variance + eps);

        // 应用归一化和 weight
        for (int i = 0; i < hidden_dim; i++) {
            int idx = token * hidden_dim + i;
            output[idx] = input[idx] * rsqrt_var * weight[i];
        }
    }
}

// ============================================================================
// 测试程序
// ============================================================================

int main() {
    printf("==================================================\n");
    printf("TensorRT-LLM RMSNorm vs Qwen2 精度验证\n");
    printf("==================================================\n\n");

    // 配置
    const int batch = 2;
    const int seq_len = 4;
    const int hidden_dim = 128;
    const float eps = 1e-6f;
    const int tokens = batch * seq_len;
    const int total = tokens * hidden_dim;

    printf("配置:\n");
    printf("  Batch size: %d\n", batch);
    printf("  Sequence length: %d\n", seq_len);
    printf("  Hidden dimension: %d\n", hidden_dim);
    printf("  Tokens: %d\n", tokens);
    printf("  Epsilon: %.1e\n\n", eps);

    // 分配主机内存
    float *h_input = new float[total];
    float *h_weight = new float[hidden_dim];
    float *h_output_trtllm = new float[total];
    float *h_output_qwen2 = new float[total];

    // 初始化输入数据
    srand(42);
    for (int i = 0; i < total; i++) {
        h_input[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    // 初始化 weight（全1）
    for (int i = 0; i < hidden_dim; i++) {
        h_weight[i] = 1.0f;
    }

    // 分配设备内存
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_weight, hidden_dim * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    // 拷贝到设备
    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 1. 运行 TensorRT-LLM kernel
    printf("运行 TensorRT-LLM RMSNorm kernel...\n");
    launchTRTLLMRmsNorm<float>(d_input, d_weight, d_output, eps, tokens, hidden_dim);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ Kernel 失败: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output_trtllm, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    printf("✓ TensorRT-LLM kernel 完成\n\n");

    // 2. 运行 Qwen2 CPU 参考实现
    printf("运行 Qwen2 CPU 参考实现...\n");
    qwen2RmsNormCPU(h_input, h_weight, h_output_qwen2, eps, tokens, hidden_dim);
    printf("✓ Qwen2 CPU 完成\n\n");

    // 3. 对比结果
    printf("精度对比:\n");
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int max_diff_idx = 0;

    for (int i = 0; i < total; i++) {
        float diff = fabsf(h_output_trtllm[i] - h_output_qwen2[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff;
    }

    float mean_diff = sum_diff / total;

    printf("  最大误差: %.2e (位置: %d)\n", max_diff, max_diff_idx);
    printf("  平均误差: %.2e\n", mean_diff);

    // 显示该位置的详细信息
    int token = max_diff_idx / hidden_dim;
    int dim = max_diff_idx % hidden_dim;
    printf("\n最大误差位置详情:\n");
    printf("  Token: %d, Dim: %d\n", token, dim);
    printf("  Input: %.6f\n", h_input[max_diff_idx]);
    printf("  TensorRT-LLM: %.6f\n", h_output_trtllm[max_diff_idx]);
    printf("  Qwen2: %.6f\n", h_output_qwen2[max_diff_idx]);
    printf("  Weight: %.6f\n", h_weight[dim]);

    // 判断是否通过
    bool passed = (max_diff < 1e-5f);

    printf("\n");
    if (passed) {
        printf("✅ 精度测试通过！TensorRT-LLM kernel 与 Qwen2 实现一致。\n");
    } else {
        printf("⚠️  精度误差较大，可能需要进一步检查。\n");
    }

    // 清理
    delete[] h_input;
    delete[] h_weight;
    delete[] h_output_trtllm;
    delete[] h_output_qwen2;
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    printf("\n==================================================\n");
    printf("测试完成\n");
    printf("==================================================\n");

    return passed ? 0 : 1;
}
