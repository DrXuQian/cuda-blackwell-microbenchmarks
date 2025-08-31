#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

/*
 * Marlin Tutorial Step 1: 4-bit Fundamentals
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand how 4-bit values are packed into 32-bit integers
 * 2. Master bit manipulation for pack/unpack operations
 * 3. Learn basic quantization and dequantization concepts
 * 4. Understand memory layout implications
 * 5. Explore performance characteristics of packed storage
 * 
 * KEY CONCEPTS:
 * - 4-bit Packing: 8 x 4-bit values per int32 (32 bits total)
 * - Quantization: FP16 → 4-bit integer mapping
 * - Dequantization: 4-bit integer → FP16 reconstruction
 * - Memory Efficiency: 4x reduction in storage requirements
 * - Bit Manipulation: Shifts, masks, and bitwise operations
 */

// CUDA error checking macro
#define CUDA_CHECK(status) \
    { \
        cudaError_t error = status; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// 4-bit packing/unpacking utilities
namespace bit_ops {

    // Pack 8 x 4-bit values into one 32-bit integer
    __host__ __device__ inline uint32_t pack_4bit_8values(
        uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
        uint8_t v4, uint8_t v5, uint8_t v6, uint8_t v7) {
        
        // Each value should be 4 bits (0-15)
        return ((uint32_t(v7) & 0xF) << 28) |
               ((uint32_t(v6) & 0xF) << 24) |
               ((uint32_t(v5) & 0xF) << 20) |
               ((uint32_t(v4) & 0xF) << 16) |
               ((uint32_t(v3) & 0xF) << 12) |
               ((uint32_t(v2) & 0xF) << 8)  |
               ((uint32_t(v1) & 0xF) << 4)  |
               ((uint32_t(v0) & 0xF) << 0);
    }

    // Unpack one 32-bit integer into 8 x 4-bit values
    __host__ __device__ inline void unpack_4bit_8values(
        uint32_t packed, uint8_t* values) {
        
        values[0] = (packed >> 0)  & 0xF;
        values[1] = (packed >> 4)  & 0xF;
        values[2] = (packed >> 8)  & 0xF;
        values[3] = (packed >> 12) & 0xF;
        values[4] = (packed >> 16) & 0xF;
        values[5] = (packed >> 20) & 0xF;
        values[6] = (packed >> 24) & 0xF;
        values[7] = (packed >> 28) & 0xF;
    }

    // Extract single 4-bit value from packed int32
    __host__ __device__ inline uint8_t extract_4bit(uint32_t packed, int index) {
        return (packed >> (index * 4)) & 0xF;
    }

    // Quantize FP16 to 4-bit (simple symmetric quantization)
    __host__ __device__ inline uint8_t quantize_fp16_to_4bit(half value, half scale) {
        float val_f = __half2float(value);
        float scale_f = __half2float(scale);
        
        // Simple symmetric quantization: [-scale, +scale] → [0, 15]
        float normalized = (val_f / scale_f + 1.0f) * 7.5f; // Map [-1,1] to [0,15]
        int quantized = __float2int_rn(fmaxf(0.0f, fminf(15.0f, normalized)));
        return (uint8_t)quantized;
    }

    // Dequantize 4-bit to FP16
    __host__ __device__ inline half dequantize_4bit_to_fp16(uint8_t quantized, half scale) {
        float scale_f = __half2float(scale);
        
        // Reverse the quantization: [0,15] → [-scale, +scale]
        float normalized = (float(quantized) / 7.5f) - 1.0f; // Map [0,15] to [-1,1]
        float dequantized = normalized * scale_f;
        return __float2half(dequantized);
    }
}

// Demonstration kernels
namespace demo_kernels {

    // Kernel to demonstrate basic packing/unpacking
    __global__ void demo_pack_unpack(uint32_t* packed_output, uint8_t* unpacked_output) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx == 0) {
            // Demo: Pack 8 sequential values (0,1,2,3,4,5,6,7)
            uint32_t packed = bit_ops::pack_4bit_8values(0,1,2,3,4,5,6,7);
            packed_output[0] = packed;
            
            printf("GPU: Packed 8 values (0-7) into: 0x%08X\n", packed);
            
            // Demo: Unpack and verify
            uint8_t unpacked[8];
            bit_ops::unpack_4bit_8values(packed, unpacked);
            
            printf("GPU: Unpacked values: ");
            for (int i = 0; i < 8; i++) {
                printf("%d ", unpacked[i]);
                unpacked_output[i] = unpacked[i];
            }
            printf("\n");
        }
    }

    // Kernel to demonstrate quantization/dequantization
    __global__ void demo_quantization(half* input_fp16, uint32_t* packed_4bit, 
                                     half* reconstructed_fp16, half scale, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int pack_idx = idx / 8;  // 8 values per packed int32
        int sub_idx = idx % 8;   // Position within packed int32
        
        if (idx >= size) return;
        
        // Step 1: Quantize FP16 to 4-bit
        uint8_t quantized = bit_ops::quantize_fp16_to_4bit(input_fp16[idx], scale);
        
        // Step 2: Pack into shared int32 (requires coordination)
        __shared__ uint8_t shared_values[256]; // Max 32 warps * 8 values
        shared_values[threadIdx.x] = quantized;
        __syncthreads();
        
        // Only first thread of each group of 8 does the packing
        if (sub_idx == 0 && pack_idx * 8 + 7 < size) {
            uint32_t packed = bit_ops::pack_4bit_8values(
                shared_values[threadIdx.x + 0],
                shared_values[threadIdx.x + 1], 
                shared_values[threadIdx.x + 2],
                shared_values[threadIdx.x + 3],
                shared_values[threadIdx.x + 4],
                shared_values[threadIdx.x + 5],
                shared_values[threadIdx.x + 6],
                shared_values[threadIdx.x + 7]
            );
            packed_4bit[pack_idx] = packed;
        }
        __syncthreads();
        
        // Step 3: Dequantize back to FP16
        uint8_t extracted = bit_ops::extract_4bit(packed_4bit[pack_idx], sub_idx);
        half dequantized = bit_ops::dequantize_4bit_to_fp16(extracted, scale);
        reconstructed_fp16[idx] = dequantized;
    }

    // Performance comparison kernel: FP16 vs 4-bit memory bandwidth
    __global__ void memory_bandwidth_test_fp16(half* data, int size, int iterations) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        half sum = __float2half(0.0f);
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
                sum = __hadd(sum, data[i]);
            }
        }
        
        // Prevent compiler optimization
        if (threadIdx.x == 0 && blockIdx.x == 0 && __half2float(sum) > 1e6f) {
            printf("FP16 sum: %f\n", __half2float(sum));
        }
    }

    __global__ void memory_bandwidth_test_4bit(uint32_t* data, int size, int iterations) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        uint32_t sum = 0;
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = idx; i < size/8; i += blockDim.x * gridDim.x) {
                sum += data[i];
            }
        }
        
        // Prevent compiler optimization
        if (threadIdx.x == 0 && blockIdx.x == 0 && sum > 1000000) {
            printf("4-bit sum: %u\n", sum);
        }
    }
}

// Host utility functions
void print_binary(uint32_t value, const std::string& label) {
    std::cout << label << ": ";
    for (int i = 31; i >= 0; i--) {
        std::cout << ((value >> i) & 1);
        if (i % 4 == 0 && i > 0) std::cout << " ";
    }
    std::cout << " (0x" << std::hex << std::setw(8) << std::setfill('0') << value << std::dec << ")\n";
}

void demonstrate_host_packing() {
    std::cout << "\n🔧 Host-side Packing Demonstration\n" << std::endl;
    
    // Pack example values
    uint8_t values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32_t packed = bit_ops::pack_4bit_8values(
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7]);
    
    std::cout << "Input values: ";
    for (int i = 0; i < 8; i++) {
        std::cout << (int)values[i] << " ";
    }
    std::cout << std::endl;
    
    print_binary(packed, "Packed result");
    
    // Demonstrate bit positions
    std::cout << "\nBit layout explanation:" << std::endl;
    std::cout << "Bits [31:28] = value[7] = " << (int)values[7] << std::endl;
    std::cout << "Bits [27:24] = value[6] = " << (int)values[6] << std::endl;
    std::cout << "Bits [23:20] = value[5] = " << (int)values[5] << std::endl;
    std::cout << "Bits [19:16] = value[4] = " << (int)values[4] << std::endl;
    std::cout << "Bits [15:12] = value[3] = " << (int)values[3] << std::endl;
    std::cout << "Bits [11:8]  = value[2] = " << (int)values[2] << std::endl;
    std::cout << "Bits [7:4]   = value[1] = " << (int)values[1] << std::endl;
    std::cout << "Bits [3:0]   = value[0] = " << (int)values[0] << std::endl;
    
    // Unpack and verify
    uint8_t unpacked[8];
    bit_ops::unpack_4bit_8values(packed, unpacked);
    
    std::cout << "\nUnpacked values: ";
    for (int i = 0; i < 8; i++) {
        std::cout << (int)unpacked[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < 8; i++) {
        if (values[i] != unpacked[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "Verification: " << (correct ? "✅ PASS" : "❌ FAIL") << std::endl;
}

void demonstrate_quantization() {
    std::cout << "\n📊 Quantization/Dequantization Demonstration\n" << std::endl;
    
    // Create test data
    std::vector<float> original_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, -1.5f, 0.75f};
    float scale = 2.0f; // Scale factor
    
    std::cout << "Original → Quantized → Dequantized:" << std::endl;
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    for (size_t i = 0; i < original_values.size(); i++) {
        half orig_half = __float2half(original_values[i]);
        half scale_half = __float2half(scale);
        
        // Quantize
        uint8_t quantized = bit_ops::quantize_fp16_to_4bit(orig_half, scale_half);
        
        // Dequantize
        half dequantized = bit_ops::dequantize_4bit_to_fp16(quantized, scale_half);
        
        float error = std::abs(original_values[i] - __half2float(dequantized));
        
        std::cout << std::setw(8) << original_values[i] 
                  << " → " << std::setw(2) << (int)quantized 
                  << " → " << std::setw(8) << __half2float(dequantized)
                  << " (error: " << std::setw(6) << error << ")" << std::endl;
    }
}

void benchmark_memory_patterns() {
    std::cout << "\n⚡ Memory Bandwidth Comparison\n" << std::endl;
    
    const int size = 1024 * 1024; // 1M elements
    const int iterations = 100;
    
    // Allocate FP16 data
    half* d_fp16_data;
    CUDA_CHECK(cudaMalloc(&d_fp16_data, size * sizeof(half)));
    
    // Allocate 4-bit data (8x smaller)
    uint32_t* d_4bit_data;
    CUDA_CHECK(cudaMalloc(&d_4bit_data, (size / 8) * sizeof(uint32_t)));
    
    // Initialize data
    std::vector<half> h_fp16_data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        h_fp16_data[i] = __float2half(dis(gen));
    }
    
    CUDA_CHECK(cudaMemcpy(d_fp16_data, h_fp16_data.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Initialize 4-bit data (simplified)
    std::vector<uint32_t> h_4bit_data(size / 8);
    for (int i = 0; i < size / 8; i++) {
        h_4bit_data[i] = i; // Simple pattern
    }
    CUDA_CHECK(cudaMemcpy(d_4bit_data, h_4bit_data.data(), (size / 8) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Benchmark FP16 memory bandwidth
    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // FP16 benchmark
    CUDA_CHECK(cudaEventRecord(start));
    demo_kernels::memory_bandwidth_test_fp16<<<grid_size, block_size>>>(d_fp16_data, size, iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float fp16_time;
    CUDA_CHECK(cudaEventElapsedTime(&fp16_time, start, stop));
    
    // 4-bit benchmark
    CUDA_CHECK(cudaEventRecord(start));
    demo_kernels::memory_bandwidth_test_4bit<<<grid_size, block_size>>>(d_4bit_data, size, iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float bit4_time;
    CUDA_CHECK(cudaEventElapsedTime(&bit4_time, start, stop));
    
    // Calculate bandwidth
    size_t fp16_bytes = size * sizeof(half) * iterations;
    size_t bit4_bytes = (size / 8) * sizeof(uint32_t) * iterations;
    
    double fp16_bandwidth = (double)fp16_bytes / (fp16_time / 1000.0) / 1e9;
    double bit4_bandwidth = (double)bit4_bytes / (bit4_time / 1000.0) / 1e9;
    
    std::cout << "FP16 Memory Access:" << std::endl;
    std::cout << "  Time: " << fp16_time << " ms" << std::endl;
    std::cout << "  Bandwidth: " << fp16_bandwidth << " GB/s" << std::endl;
    std::cout << "  Data size: " << fp16_bytes / (1024 * 1024) << " MB" << std::endl;
    
    std::cout << "\n4-bit Memory Access:" << std::endl;
    std::cout << "  Time: " << bit4_time << " ms" << std::endl;
    std::cout << "  Bandwidth: " << bit4_bandwidth << " GB/s" << std::endl;
    std::cout << "  Data size: " << bit4_bytes / (1024 * 1024) << " MB" << std::endl;
    
    std::cout << "\n📈 Storage Efficiency:" << std::endl;
    std::cout << "  FP16 storage: " << size * sizeof(half) / 1024.0 << " KB" << std::endl;
    std::cout << "  4-bit storage: " << (size / 8) * sizeof(uint32_t) / 1024.0 << " KB" << std::endl;
    std::cout << "  Compression ratio: " << (double)(size * sizeof(half)) / ((size / 8) * sizeof(uint32_t)) << ":1" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_fp16_data));
    CUDA_CHECK(cudaFree(d_4bit_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "=== Marlin Tutorial Step 1: 4-bit Fundamentals ===" << std::endl;
    
    // Check GPU capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    
    /*
     * PART 1: Basic Packing/Unpacking
     */
    std::cout << "\n📦 PART 1: Understanding 4-bit Packing\n" << std::endl;
    
    std::cout << "🎯 Key Concept: 8 x 4-bit values pack into 1 x 32-bit integer" << std::endl;
    std::cout << "• Each 4-bit value can represent 0-15" << std::endl;
    std::cout << "• Memory efficiency: 4x reduction compared to FP16" << std::endl;
    std::cout << "• Bit layout: [31:28][27:24][23:20][19:16][15:12][11:8][7:4][3:0]" << std::endl;
    std::cout << "•             val[7]  val[6]  val[5]  val[4]  val[3] val[2] val[1] val[0]" << std::endl;
    
    demonstrate_host_packing();
    
    // GPU demonstration
    std::cout << "\n🚀 GPU Packing Demonstration:" << std::endl;
    
    uint32_t* d_packed;
    uint8_t* d_unpacked;
    CUDA_CHECK(cudaMalloc(&d_packed, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_unpacked, 8 * sizeof(uint8_t)));
    
    demo_kernels::demo_pack_unpack<<<1, 32>>>(d_packed, d_unpacked);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint32_t h_packed;
    uint8_t h_unpacked[8];
    CUDA_CHECK(cudaMemcpy(&h_packed, d_packed, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_unpacked, d_unpacked, 8 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    print_binary(h_packed, "GPU packed result");
    
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_unpacked));
    
    /*
     * PART 2: Quantization and Dequantization
     */
    std::cout << "\n📊 PART 2: Quantization Fundamentals\n" << std::endl;
    
    std::cout << "🎯 Key Concept: FP16 ⟷ 4-bit conversion with scaling" << std::endl;
    std::cout << "• Quantization: Map continuous FP16 range to discrete 4-bit values (0-15)" << std::endl;
    std::cout << "• Dequantization: Reconstruct approximate FP16 from 4-bit values" << std::endl;
    std::cout << "• Scale factor: Determines the range of representable values" << std::endl;
    std::cout << "• Trade-off: Storage efficiency vs numerical precision" << std::endl;
    
    demonstrate_quantization();
    
    /*
     * PART 3: GPU Quantization Demonstration
     */
    std::cout << "\n⚡ PART 3: GPU Quantization Pipeline\n" << std::endl;
    
    const int test_size = 64;
    half* d_input;
    uint32_t* d_packed_quant;
    half* d_reconstructed;
    
    CUDA_CHECK(cudaMalloc(&d_input, test_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_packed_quant, (test_size / 8) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_reconstructed, test_size * sizeof(half)));
    
    // Initialize test data
    std::vector<half> h_input(test_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    for (int i = 0; i < test_size; i++) {
        h_input[i] = __float2half(dis(gen));
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), test_size * sizeof(half), cudaMemcpyHostToDevice));
    
    half scale = __float2half(2.0f);
    
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    
    demo_kernels::demo_quantization<<<grid, block>>>(
        d_input, d_packed_quant, d_reconstructed, scale, test_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verify results
    std::vector<half> h_reconstructed(test_size);
    CUDA_CHECK(cudaMemcpy(h_reconstructed.data(), d_reconstructed, test_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU Quantization Results (first 8 values):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < 8; i++) {
        float error = std::abs(__half2float(h_input[i]) - __half2float(h_reconstructed[i]));
        std::cout << "  [" << i << "] " << std::setw(7) << __half2float(h_input[i]) 
                  << " → " << std::setw(7) << __half2float(h_reconstructed[i])
                  << " (error: " << std::setw(6) << error << ")" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed_quant));
    CUDA_CHECK(cudaFree(d_reconstructed));
    
    /*
     * PART 4: Memory Bandwidth Analysis
     */
    std::cout << "\n💾 PART 4: Memory Efficiency Analysis\n" << std::endl;
    
    std::cout << "🎯 Key Benefit: 4x memory bandwidth improvement" << std::endl;
    std::cout << "• FP16: 2 bytes per weight" << std::endl;
    std::cout << "• 4-bit: 0.5 bytes per weight (8 weights per 4-byte int32)" << std::endl;
    std::cout << "• Memory traffic reduction enables larger models in GPU memory" << std::endl;
    std::cout << "• Higher effective memory bandwidth for weight-dominated operations" << std::endl;
    
    benchmark_memory_patterns();
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Step 1 Summary: 4-bit Fundamentals Mastered ===\n" << std::endl;
    std::cout << "✅ You learned:" << std::endl;
    std::cout << "   • How to pack 8 x 4-bit values into 32-bit integers" << std::endl;
    std::cout << "   • Bit manipulation techniques for pack/unpack operations" << std::endl;
    std::cout << "   • Basic quantization and dequantization concepts" << std::endl;
    std::cout << "   • Memory efficiency benefits of 4-bit storage" << std::endl;
    std::cout << "   • Performance implications and trade-offs" << std::endl;
    
    std::cout << "\n🧠 Key Insights:" << std::endl;
    std::cout << "   • 4-bit quantization provides 4x memory efficiency" << std::endl;
    std::cout << "   • Packing/unpacking operations are computationally lightweight" << std::endl;
    std::cout << "   • Scale factors are crucial for maintaining numerical precision" << std::endl;
    std::cout << "   • Bit manipulation is fundamental to efficient implementation" << std::endl;
    
    std::cout << "\n🎯 Next: Step 2 will build a basic 4-bit GEMV using these fundamentals!" << std::endl;
    
    return 0;
}