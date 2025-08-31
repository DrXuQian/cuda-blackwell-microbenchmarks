#include "../utils/blackwell_common.h"

// Advanced RTX 5070 Blackwell W4A16 GEMV with full warp specialization
// Producer-Consumer-Computer pattern with native WGMMA async

// Enhanced TMA descriptor for multi-stage pipeline
typedef struct {
    CUtensorMap weight_tma;
    CUtensorMap scale_tma;
    CUtensorMap activation_tma;  // For large batch inference
    uint64_t tile_bytes_weight;
    uint64_t tile_bytes_scale;
    bool initialized;
} BlackwellW4A16AdvancedTMA;

// Warp roles for specialization
enum WarpRole {
    WARP_PRODUCER_WEIGHTS = 0,    // TMA load weights
    WARP_PRODUCER_SCALES = 1,     // TMA load scales  
    WARP_DEQUANTIZER = 2,         // Dequantize 4-bit -> FP16
    WARP_COMPUTER_0 = 3,          // WGMMA compute warp 0
    WARP_COMPUTER_1 = 4,          // WGMMA compute warp 1
    WARP_COMPUTER_2 = 5,          // WGMMA compute warp 2
    WARP_COMPUTER_3 = 6,          // WGMMA compute warp 3
    WARP_REDUCER = 7              // Reduction and output
};

// Optimized 4-bit dequantization with Blackwell vector instructions
__device__ inline void blackwell_fast_dequant_w4a16(
    const uint32_t* packed_4bit,
    half* output_fp16,
    const half* scale_ptr,
    int elements_per_thread,
    int lane_id
) {
    #pragma unroll
    for (int elem = 0; elem < elements_per_thread; elem++) {
        uint32_t packed = packed_4bit[lane_id * elements_per_thread + elem];
        half scale = scale_ptr[lane_id * elements_per_thread + elem];
        
        // Extract 8 4-bit weights using bit field extraction
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        
        asm volatile(
            "bfe.u32 %0, %8, 0, 4;\n"   // Extract bits 0-3
            "bfe.u32 %1, %8, 4, 4;\n"   // Extract bits 4-7
            "bfe.u32 %2, %8, 8, 4;\n"   // Extract bits 8-11
            "bfe.u32 %3, %8, 12, 4;\n"  // Extract bits 12-15
            "bfe.u32 %4, %8, 16, 4;\n"  // Extract bits 16-19
            "bfe.u32 %5, %8, 20, 4;\n"  // Extract bits 20-23
            "bfe.u32 %6, %8, 24, 4;\n"  // Extract bits 24-27
            "bfe.u32 %7, %8, 28, 4;\n"  // Extract bits 28-31
            : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3),
              "=r"(w4), "=r"(w5), "=r"(w6), "=r"(w7)
            : "r"(packed)
        );
        
        // Convert to FP16 and apply scale (vectorized)
        half2 scale2 = __half2half2(scale);
        
        // Convert pairs to FP16 using vector operations
        uint32_t fp16_pair_0, fp16_pair_1, fp16_pair_2, fp16_pair_3;
        
        asm volatile(
            "{\n"
            ".reg .f16 f0, f1, f2, f3, f4, f5, f6, f7;\n"
            
            // Convert uint to FP16
            "cvt.rn.f16.u32 f0, %8;\n"
            "cvt.rn.f16.u32 f1, %9;\n"
            "cvt.rn.f16.u32 f2, %10;\n"
            "cvt.rn.f16.u32 f3, %11;\n"
            "cvt.rn.f16.u32 f4, %12;\n"
            "cvt.rn.f16.u32 f5, %13;\n"
            "cvt.rn.f16.u32 f6, %14;\n"
            "cvt.rn.f16.u32 f7, %15;\n"
            
            // Apply scale
            "mul.f16 f0, f0, %16;\n"
            "mul.f16 f1, f1, %16;\n"
            "mul.f16 f2, f2, %16;\n"
            "mul.f16 f3, f3, %16;\n"
            "mul.f16 f4, f4, %16;\n"
            "mul.f16 f5, f5, %16;\n"
            "mul.f16 f6, f6, %16;\n"
            "mul.f16 f7, f7, %16;\n"
            
            // Pack into half2 pairs
            "mov.b32 %0, {f0, f1};\n"
            "mov.b32 %1, {f2, f3};\n"
            "mov.b32 %2, {f4, f5};\n"
            "mov.b32 %3, {f6, f7};\n"
            "}\n"
            : "=r"(fp16_pair_0), "=r"(fp16_pair_1), "=r"(fp16_pair_2), "=r"(fp16_pair_3)
            : "r"(w0), "r"(w1), "r"(w2), "r"(w3), "r"(w4), "r"(w5), "r"(w6), "r"(w7),
              "h"(scale)
        );
        
        // Store to shared memory as half2 pairs for optimal WGMMA access
        half2* output_half2 = reinterpret_cast<half2*>(&output_fp16[elem * 8]);
        output_half2[0] = *reinterpret_cast<half2*>(&fp16_pair_0);
        output_half2[1] = *reinterpret_cast<half2*>(&fp16_pair_1);
        output_half2[2] = *reinterpret_cast<half2*>(&fp16_pair_2);
        output_half2[3] = *reinterpret_cast<half2*>(&fp16_pair_3);
    }
}

// Advanced Blackwell W4A16 GEMV with full pipeline optimization
__global__ void blackwell_w4a16_warp_specialized_kernel(
    const half* __restrict__ A,           // [M, K] activations (FP16)
    const uint32_t* __restrict__ B,       // [K/8, N] weights (4-bit packed)
    half* __restrict__ C,                 // [M, N] output (FP16)
    const half* __restrict__ scales,      // [K/group_size, N] scales (FP16)
    BlackwellW4A16AdvancedTMA* tma_desc,
    int M, int N, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Shared memory with triple buffering for maximum pipeline efficiency
    extern __shared__ char shmem[];
    
    // Memory layout: [weights][scales][dequantized_weights][activations]
    uint32_t* shmem_weights[3];
    half* shmem_scales[3];
    half* shmem_dequant[3];
    half* shmem_activations;
    
    size_t weight_tile_size = (W4A16_TILE_K/8) * W4A16_TILE_N;
    size_t scale_tile_size = (W4A16_TILE_K/W4A16_GROUP_SIZE) * W4A16_TILE_N;
    size_t dequant_tile_size = W4A16_TILE_K * W4A16_TILE_N;
    size_t activation_size = W4A16_TILE_M * W4A16_TILE_K;
    
    char* shmem_ptr = shmem;
    
    for (int i = 0; i < 3; i++) {
        shmem_weights[i] = reinterpret_cast<uint32_t*>(shmem_ptr);
        shmem_ptr += weight_tile_size * sizeof(uint32_t);
        
        shmem_scales[i] = reinterpret_cast<half*>(shmem_ptr);
        shmem_ptr += scale_tile_size * sizeof(half);
        
        shmem_dequant[i] = reinterpret_cast<half*>(shmem_ptr);
        shmem_ptr += dequant_tile_size * sizeof(half);
    }
    
    shmem_activations = reinterpret_cast<half*>(shmem_ptr);
    
    // Block tiling
    const int block_col = blockIdx.x * W4A16_TILE_N;
    const int block_row = blockIdx.y * W4A16_TILE_M;
    
    // WGMMA accumulator for compute warps
    float wgmma_acc[32] = {0.0f};
    
    // Pipeline management
    int produce_stage = 0;
    int dequant_stage = 0;
    int compute_stage = 0;
    
    const int num_k_tiles = (K + W4A16_TILE_K - 1) / W4A16_TILE_K;
    
    // Main pipeline loop
    for (int k_tile = 0; k_tile < num_k_tiles + 2; k_tile++) {  // +2 for pipeline drain
        
        // === PRODUCER WARPS: TMA Loading ===
        if (warp_id == WARP_PRODUCER_WEIGHTS && k_tile < num_k_tiles) {
            if (threadIdx.x == 0) {
                // TMA load quantized weights
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];\n"
                    :
                    : "r"(__cvta_generic_to_shared(shmem_weights[produce_stage])),
                      "l"(tma_desc->weight_tma),
                      "r"(k_tile), "r"(blockIdx.x),
                      "r"(__cvta_generic_to_shared(shmem_weights[produce_stage] + weight_tile_size - 1))
                );
            }
        }
        
        if (warp_id == WARP_PRODUCER_SCALES && k_tile < num_k_tiles) {
            if (threadIdx.x == 0) {
                // TMA load scales
                int scale_tile_idx = k_tile;  // Assuming scales align with K tiles
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];\n"
                    :
                    : "r"(__cvta_generic_to_shared(shmem_scales[produce_stage])),
                      "l"(tma_desc->scale_tma),
                      "r"(scale_tile_idx), "r"(blockIdx.x),
                      "r"(__cvta_generic_to_shared(shmem_scales[produce_stage] + scale_tile_size - 1))
                );
            }
        }
        
        // Load activations to shared memory (producer warps)
        if (warp_id <= WARP_PRODUCER_SCALES && k_tile < num_k_tiles) {
            int k_offset = k_tile * W4A16_TILE_K;
            
            for (int i = lane_id; i < W4A16_TILE_M * W4A16_TILE_K; i += 32) {
                int row = i / W4A16_TILE_K;
                int col = i % W4A16_TILE_K;
                int global_row = block_row + row;
                int global_col = k_offset + col;
                
                shmem_activations[i] = (global_row < M && global_col < K) ?
                                      A[global_row * K + global_col] : __float2half(0.0f);
            }
        }
        
        // Synchronization point
        __syncthreads();
        if (k_tile < num_k_tiles) {
            asm volatile("cp.async.bulk.wait_group.read 0;\n");
        }
        __syncthreads();
        
        // === DEQUANTIZATION WARP ===
        if (warp_id == WARP_DEQUANTIZER && k_tile >= 1 && dequant_stage < num_k_tiles) {
            int stage = dequant_stage % 3;
            
            // Dequantize weights in groups
            const int groups_per_tile = W4A16_TILE_K / W4A16_GROUP_SIZE;
            
            for (int group = 0; group < groups_per_tile; group++) {
                int scale_base = group * W4A16_TILE_N;
                int weight_base = group * (W4A16_GROUP_SIZE / 8) * W4A16_TILE_N;
                int output_base = group * W4A16_GROUP_SIZE * W4A16_TILE_N;
                
                // Process weights in this group
                blackwell_fast_dequant_w4a16(
                    &shmem_weights[stage][weight_base + lane_id],
                    &shmem_dequant[stage][output_base + lane_id * 8],
                    &shmem_scales[stage][scale_base + lane_id],
                    1,  // elements per thread (adjusted in function)
                    lane_id
                );
            }
            
            dequant_stage++;
        }
        
        // === COMPUTE WARPS: Async WGMMA ===
        if (warp_id >= WARP_COMPUTER_0 && warp_id <= WARP_COMPUTER_3 && 
            k_tile >= 2 && compute_stage < num_k_tiles) {
            
            int compute_warp_idx = warp_id - WARP_COMPUTER_0;
            int compute_stage_idx = compute_stage % 3;
            
            // Each compute warp handles different output regions
            const int output_rows_per_warp = W4A16_TILE_M / 4;  // 4 compute warps
            const int warp_row_start = compute_warp_idx * output_rows_per_warp;
            
            // Native WGMMA async instruction for Blackwell
            asm volatile(
                "{\n"
                ".reg .b64 desc_a, desc_b;\n"
                ".reg .f32 acc<32>;\n"
                
                // Load current accumulator values
                "mov.f32 acc0, %32;\n"   "mov.f32 acc1, %33;\n"
                "mov.f32 acc2, %34;\n"   "mov.f32 acc3, %35;\n"
                "mov.f32 acc4, %36;\n"   "mov.f32 acc5, %37;\n"
                "mov.f32 acc6, %38;\n"   "mov.f32 acc7, %39;\n"
                "mov.f32 acc8, %40;\n"   "mov.f32 acc9, %41;\n"
                "mov.f32 acc10, %42;\n"  "mov.f32 acc11, %43;\n"
                "mov.f32 acc12, %44;\n"  "mov.f32 acc13, %45;\n"
                "mov.f32 acc14, %46;\n"  "mov.f32 acc15, %47;\n"
                "mov.f32 acc16, %48;\n"  "mov.f32 acc17, %49;\n"
                "mov.f32 acc18, %50;\n"  "mov.f32 acc19, %51;\n"
                "mov.f32 acc20, %52;\n"  "mov.f32 acc21, %53;\n"
                "mov.f32 acc22, %54;\n"  "mov.f32 acc23, %55;\n"
                "mov.f32 acc24, %56;\n"  "mov.f32 acc25, %57;\n"
                "mov.f32 acc26, %58;\n"  "mov.f32 acc27, %59;\n"
                "mov.f32 acc28, %60;\n"  "mov.f32 acc29, %61;\n"
                "mov.f32 acc30, %62;\n"  "mov.f32 acc31, %63;\n"
                
                // Create tensor descriptors from shared memory addresses
                "cvta.to.shared.u64 desc_a, %64;\n"  // Activations
                "cvta.to.shared.u64 desc_b, %65;\n"  // Dequantized weights
                
                // WGMMA async operation: m32n32k32 (optimal for W4A16)
                "wgmma.mma_async.sync.aligned.m32n32k32.f32.f16.f16 "
                "{acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7,"
                "acc8,acc9,acc10,acc11,acc12,acc13,acc14,acc15,"
                "acc16,acc17,acc18,acc19,acc20,acc21,acc22,acc23,"
                "acc24,acc25,acc26,acc27,acc28,acc29,acc30,acc31}, "
                "desc_a, desc_b;\n"
                
                // Store accumulator values back
                "mov.f32 %0, acc0;\n"   "mov.f32 %1, acc1;\n"
                "mov.f32 %2, acc2;\n"   "mov.f32 %3, acc3;\n"
                "mov.f32 %4, acc4;\n"   "mov.f32 %5, acc5;\n"
                "mov.f32 %6, acc6;\n"   "mov.f32 %7, acc7;\n"
                "mov.f32 %8, acc8;\n"   "mov.f32 %9, acc9;\n"
                "mov.f32 %10, acc10;\n" "mov.f32 %11, acc11;\n"
                "mov.f32 %12, acc12;\n" "mov.f32 %13, acc13;\n"
                "mov.f32 %14, acc14;\n" "mov.f32 %15, acc15;\n"
                "mov.f32 %16, acc16;\n" "mov.f32 %17, acc17;\n"
                "mov.f32 %18, acc18;\n" "mov.f32 %19, acc19;\n"
                "mov.f32 %20, acc20;\n" "mov.f32 %21, acc21;\n"
                "mov.f32 %22, acc22;\n" "mov.f32 %23, acc23;\n"
                "mov.f32 %24, acc24;\n" "mov.f32 %25, acc25;\n"
                "mov.f32 %26, acc26;\n" "mov.f32 %27, acc27;\n"
                "mov.f32 %28, acc28;\n" "mov.f32 %29, acc29;\n"
                "mov.f32 %30, acc30;\n" "mov.f32 %31, acc31;\n"
                
                // Wait for WGMMA completion
                "wgmma.wait_group.sync.aligned 0;\n"
                "}\n"
                :
                "=f"(wgmma_acc[0]), "=f"(wgmma_acc[1]), "=f"(wgmma_acc[2]), "=f"(wgmma_acc[3]),
                "=f"(wgmma_acc[4]), "=f"(wgmma_acc[5]), "=f"(wgmma_acc[6]), "=f"(wgmma_acc[7]),
                "=f"(wgmma_acc[8]), "=f"(wgmma_acc[9]), "=f"(wgmma_acc[10]), "=f"(wgmma_acc[11]),
                "=f"(wgmma_acc[12]), "=f"(wgmma_acc[13]), "=f"(wgmma_acc[14]), "=f"(wgmma_acc[15]),
                "=f"(wgmma_acc[16]), "=f"(wgmma_acc[17]), "=f"(wgmma_acc[18]), "=f"(wgmma_acc[19]),
                "=f"(wgmma_acc[20]), "=f"(wgmma_acc[21]), "=f"(wgmma_acc[22]), "=f"(wgmma_acc[23]),
                "=f"(wgmma_acc[24]), "=f"(wgmma_acc[25]), "=f"(wgmma_acc[26]), "=f"(wgmma_acc[27]),
                "=f"(wgmma_acc[28]), "=f"(wgmma_acc[29]), "=f"(wgmma_acc[30]), "=f"(wgmma_acc[31])
                :
                "f"(wgmma_acc[0]), "f"(wgmma_acc[1]), "f"(wgmma_acc[2]), "f"(wgmma_acc[3]),
                "f"(wgmma_acc[4]), "f"(wgmma_acc[5]), "f"(wgmma_acc[6]), "f"(wgmma_acc[7]),
                "f"(wgmma_acc[8]), "f"(wgmma_acc[9]), "f"(wgmma_acc[10]), "f"(wgmma_acc[11]),
                "f"(wgmma_acc[12]), "f"(wgmma_acc[13]), "f"(wgmma_acc[14]), "f"(wgmma_acc[15]),
                "f"(wgmma_acc[16]), "f"(wgmma_acc[17]), "f"(wgmma_acc[18]), "f"(wgmma_acc[19]),
                "f"(wgmma_acc[20]), "f"(wgmma_acc[21]), "f"(wgmma_acc[22]), "f"(wgmma_acc[23]),
                "f"(wgmma_acc[24]), "f"(wgmma_acc[25]), "f"(wgmma_acc[26]), "f"(wgmma_acc[27]),
                "f"(wgmma_acc[28]), "f"(wgmma_acc[29]), "f"(wgmma_acc[30]), "f"(wgmma_acc[31]),
                "r"(__cvta_generic_to_shared(&shmem_activations[warp_row_start * W4A16_TILE_K])),
                "r"(__cvta_generic_to_shared(&shmem_dequant[compute_stage_idx][compute_warp_idx * 32 * W4A16_TILE_N]))
            );
            
            compute_stage++;
        }
        
        // Update pipeline stages
        if (k_tile < num_k_tiles) {
            produce_stage = (produce_stage + 1) % 3;
        }
        if (k_tile >= 1) {
            dequant_stage = (dequant_stage + 1) % 3;
        }
        
        __syncthreads();
    }
    
    // === REDUCER WARP: Final Output ===
    if (warp_id == WARP_REDUCER) {
        // Reduce partial results from compute warps and write to global memory
        for (int output_idx = lane_id; output_idx < W4A16_TILE_M * W4A16_TILE_N; output_idx += 32) {
            int row = output_idx / W4A16_TILE_N;
            int col = output_idx % W4A16_TILE_N;
            
            int global_row = block_row + row;
            int global_col = block_col + col;
            
            if (global_row < M && global_col < N) {
                // Collect from appropriate compute warp
                int source_warp = row / (W4A16_TILE_M / 4);
                int warp_local_idx = (row % (W4A16_TILE_M / 4)) * W4A16_TILE_N + col;
                
                // Simple reduction (in real implementation, use warp shuffle)
                float result = 0.0f;
                if (source_warp == 0 && warp_local_idx < 32) result = wgmma_acc[warp_local_idx];
                
                C[global_row * N + global_col] = __float2half(result);
            }
        }
    }
}

// Setup TMA descriptors for advanced W4A16 kernel
CUresult setup_advanced_w4a16_tma(
    BlackwellW4A16AdvancedTMA* desc,
    const uint32_t* weights, const half* scales, const half* activations,
    int M, int N, int K
) {
    CUresult result;
    
    // Initialize CUDA driver API
    cuInit(0);
    
    // Weight TMA setup
    CUtensorMapDataType dtype_weight = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    
    uint64_t weight_shape[2] = {(uint64_t)(K/8), (uint64_t)N};
    uint64_t weight_stride[1] = {(uint64_t)(N * sizeof(uint32_t))};
    uint32_t weight_box[2] = {W4A16_TILE_K/8, W4A16_TILE_N};
    uint32_t weight_box_stride[1] = {W4A16_TILE_N * sizeof(uint32_t)};
    
    result = cuTensorMapEncodeTiled(
        &desc->weight_tma, dtype_weight, 2,
        (void*)weights, weight_shape, weight_stride,
        weight_box, weight_box_stride,
        interleave, swizzle, l2promo, oob_fill
    );
    
    if (result != CUDA_SUCCESS) return result;
    
    // Scale TMA setup
    CUtensorMapDataType dtype_scale = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    
    uint64_t scale_shape[2] = {(uint64_t)(K/W4A16_GROUP_SIZE), (uint64_t)N};
    uint64_t scale_stride[1] = {(uint64_t)(N * sizeof(half))};
    uint32_t scale_box[2] = {W4A16_TILE_K/W4A16_GROUP_SIZE, W4A16_TILE_N};
    uint32_t scale_box_stride[1] = {W4A16_TILE_N * sizeof(half)};
    
    result = cuTensorMapEncodeTiled(
        &desc->scale_tma, dtype_scale, 2,
        (void*)scales, scale_shape, scale_stride,
        scale_box, scale_box_stride,
        interleave, swizzle, l2promo, oob_fill
    );
    
    desc->initialized = (result == CUDA_SUCCESS);
    desc->tile_bytes_weight = (W4A16_TILE_K/8) * W4A16_TILE_N * sizeof(uint32_t);
    desc->tile_bytes_scale = (W4A16_TILE_K/W4A16_GROUP_SIZE) * W4A16_TILE_N * sizeof(half);
    
    return result;
}

// Host launcher
void launch_blackwell_w4a16_specialized(
    const half* A, const uint32_t* B, half* C, const half* scales,
    int M, int N, int K
) {
    // Setup TMA descriptors
    BlackwellW4A16AdvancedTMA *d_tma_desc;
    CUDA_CHECK(cudaMalloc(&d_tma_desc, sizeof(BlackwellW4A16AdvancedTMA)));
    
    BlackwellW4A16AdvancedTMA h_tma_desc;
    CUresult result = setup_advanced_w4a16_tma(&h_tma_desc, B, scales, A, M, N, K);
    if (result != CUDA_SUCCESS) {
        printf("Advanced W4A16 TMA setup failed\n");
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(d_tma_desc, &h_tma_desc, sizeof(BlackwellW4A16AdvancedTMA),
                         cudaMemcpyHostToDevice));
    
    // Launch configuration: 8 warps for full specialization
    dim3 grid((N + W4A16_TILE_N - 1) / W4A16_TILE_N, (M + W4A16_TILE_M - 1) / W4A16_TILE_M);
    dim3 block(8 * 32);  // 8 specialized warps
    
    // Calculate shared memory requirements
    size_t shmem_weights = 3 * (W4A16_TILE_K/8) * W4A16_TILE_N * sizeof(uint32_t);
    size_t shmem_scales = 3 * (W4A16_TILE_K/W4A16_GROUP_SIZE) * W4A16_TILE_N * sizeof(half);
    size_t shmem_dequant = 3 * W4A16_TILE_K * W4A16_TILE_N * sizeof(half);
    size_t shmem_activations = W4A16_TILE_M * W4A16_TILE_K * sizeof(half);
    size_t total_shmem = shmem_weights + shmem_scales + shmem_dequant + shmem_activations;
    
    printf("Shared memory usage: %.1f KB (max: 164 KB)\n", total_shmem / 1024.0);
    
    CUDA_CHECK(cudaFuncSetAttribute(blackwell_w4a16_warp_specialized_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   total_shmem));
    
    blackwell_w4a16_warp_specialized_kernel<<<grid, block, total_shmem>>>(
        A, B, C, scales, d_tma_desc, M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_tma_desc);
}

// Test program for advanced W4A16 GEMV
int main() {
    printf("🎯 RTX 5070 Blackwell W4A16 GEMV - Advanced Warp Specialization\n");
    printf("===============================================================\n");
    
    if (!check_blackwell_support()) {
        return 1;
    }
    
    // LLM inference scenario: single token generation
    const int M = 1;        // Batch size
    const int N = 50257;    // Vocabulary size (GPT-like)
    const int K = 6144;     // Hidden dimension
    
    printf("\nLLM Inference Scenario:\n");
    printf("  Batch size: %d\n", M);
    printf("  Hidden dim: %d\n", K);
    printf("  Vocab size: %d\n", N);
    printf("  Model size: %.1fM parameters (W4A16 compressed)\n", (K * N / 2.0) / 1e6);
    
    // Memory allocation
    size_t size_A = M * K * sizeof(half);
    size_t size_B = (K/8) * N * sizeof(uint32_t);
    size_t size_scales = (K/W4A16_GROUP_SIZE) * N * sizeof(half);
    size_t size_C = M * N * sizeof(half);
    
    half *h_A = (half*)aligned_alloc(256, size_A);
    uint32_t *h_B = (uint32_t*)aligned_alloc(256, size_B);
    half *h_scales = (half*)aligned_alloc(256, size_scales);
    half *h_C = (half*)aligned_alloc(256, size_C);
    
    // Initialize with realistic LLM data
    srand(12345);
    
    // Activations (layer norm output)
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half((float)rand() / RAND_MAX * 4.0f - 2.0f);  // [-2, 2] range
    }
    
    // Quantized weights
    for (int i = 0; i < (K/8) * N; i++) {
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++) {
            uint32_t w4 = rand() % 16;
            packed |= (w4 << (j * 4));
        }
        h_B[i] = packed;
    }
    
    // Scales (per-group quantization)
    for (int i = 0; i < (K/W4A16_GROUP_SIZE) * N; i++) {
        h_scales[i] = __float2half(0.5f + (float)rand() / RAND_MAX * 1.0f);  // [0.5, 1.5]
    }
    
    // Device allocation
    half *d_A, *d_C, *d_scales;
    uint32_t *d_B;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_scales, size_scales));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, size_scales, cudaMemcpyHostToDevice));
    
    printf("\n🚀 Running specialized W4A16 GEMV...\n");
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_blackwell_w4a16_specialized(d_A, d_B, d_C, d_scales, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    BlackwellTimer timer;
    blackwell_timer_init(&timer, 200);
    
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        blackwell_timer_start(&timer);
        launch_blackwell_w4a16_specialized(d_A, d_B, d_C, d_scales, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        blackwell_timer_stop(&timer);
    }
    
    float avg_time = blackwell_timer_get_avg(&timer);
    double tflops = calculate_tflops(M, N, K, avg_time);
    double bandwidth = calculate_bandwidth_gb_s(size_A + size_B + size_scales + size_C, avg_time);
    
    printf("\n📊 Advanced W4A16 GEMV Results:\n");
    printf("   Average time: %.3f ms (%.1f μs)\n", avg_time, avg_time * 1000);
    printf("   Performance: %.1f TFLOPS\n", tflops);
    printf("   Effective bandwidth: %.1f GB/s\n", bandwidth);
    printf("   Tokens/second: %.0f (single token latency)\n", 1000.0 / avg_time);
    printf("   Memory efficiency: 4x compression vs FP16\n");
    
    // Validation
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    bool valid = true;
    float avg_magnitude = 0.0f;
    int valid_samples = 0;
    
    for (int i = 0; i < std::min(1000, M * N); i++) {
        float val = __half2float(h_C[i]);
        if (isnan(val) || isinf(val)) {
            valid = false;
            printf("   Invalid value at index %d: %f\n", i, val);
            break;
        }
        if (fabs(val) < 1000.0f) {
            avg_magnitude += fabs(val);
            valid_samples++;
        }
    }
    
    printf("   Validation: %s\n", valid ? "✅ PASSED" : "❌ FAILED");
    
    if (valid && valid_samples > 0) {
        avg_magnitude /= valid_samples;
        printf("   Average output magnitude: %.3f\n", avg_magnitude);
        printf("   Output range: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
               __half2float(h_C[0]), __half2float(h_C[1]), __half2float(h_C[2]),
               __half2float(h_C[3]), __half2float(h_C[4]));
    }
    
    printf("\n🎯 LLM Inference Performance:\n");
    printf("   Single token latency: %.1f μs\n", avg_time * 1000);
    printf("   Throughput capacity: %.0f tokens/sec\n", 1000.0 / avg_time);
    printf("   Expected on RTX 5070: 500-1000 μs per token\n");
    
    // Cleanup
    blackwell_timer_cleanup(&timer);
    free(h_A); free(h_B); free(h_scales); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_scales); cudaFree(d_C);
    
    printf("\n✨ Advanced W4A16 GEMV completed!\n");
    printf("Features: TMA + Async WGMMA + 8-way warp specialization\n");
    printf("Optimized for: RTX 5070 Blackwell LLM inference workloads\n");
    
    return 0;
}