#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 4: Thread Organization and Coordination
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand CUTLASS thread hierarchy (Grid → Block → Warp → Thread)
 * 2. Learn about warp specialization and cooperation patterns
 * 3. Explore WMMA instruction mapping to thread organization
 * 4. Understand synchronization and data sharing mechanisms
 * 5. Learn about load balancing and work distribution
 * 6. Compare different thread organization strategies
 * 
 * KEY CONCEPTS:
 * - Thread hierarchy and responsibilities at each level
 * - Warp specialization (Producer, Consumer, Computer warps)
 * - WMMA/MMA instruction execution model
 * - Inter-warp communication and synchronization
 * - Load balancing for irregular workloads
 */

#include <cuda_runtime.h>

// Custom thread organization patterns for demonstration
namespace thread_demo {

    // Helper function to print thread information
    __device__ void print_thread_info(const char* context) {
        int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int warp_id = thread_id / 32;
        int lane_id = thread_id % 32;
        int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        
        if (lane_id == 0) {  // Only lane 0 of each warp prints
            printf("[%s] Block %d, Warp %d, Thread %d\n", context, block_id, warp_id, thread_id);
        }
    }

    // Kernel demonstrating basic thread organization
    __global__ void demonstrate_thread_hierarchy() {
        // Print hierarchy information
        print_thread_info("Thread Hierarchy Demo");
        
        // Demonstrate warp-level operations
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        // Warp-level reduction using __shfl_down_sync
        int value = threadIdx.x;  // Each thread starts with its thread ID
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        // Only lane 0 has the final result
        if (lane_id == 0) {
            printf("Warp %d reduction result: %d\n", warp_id, value);
        }
    }
    
    // Kernel demonstrating warp specialization pattern
    __global__ void demonstrate_warp_specialization(float* A, float* B, float* C, int M, int N, int K) {
        __shared__ float shared_A[128 * 32];  // Shared memory for A tile
        __shared__ float shared_B[32 * 128];  // Shared memory for B tile
        
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        // Different warps have different roles
        if (warp_id == 0) {
            // Producer warp: Load data from global memory
            print_thread_info("Producer Warp");
            
            // Load A tile (simplified)
            int row = lane_id / 4;
            int col = lane_id % 4;
            for (int k = 0; k < 32; k += 4) {
                shared_A[row * 32 + k + col] = A[row * K + k + col];
            }
            
        } else if (warp_id == 1) {
            // Second producer warp: Load B data
            print_thread_info("Producer Warp B");
            
            // Load B tile (simplified)
            int row = lane_id / 4;
            int col = lane_id % 4;
            for (int k = 0; k < 32; k += 4) {
                shared_B[k * 128 + row * 4 + col] = B[k * N + row * 4 + col];
            }
            
        } else {
            // Consumer warps: Compute using loaded data
            print_thread_info("Consumer/Computer Warp");
            
            // Wait for producers to finish loading
            __syncthreads();
            
            // Simplified computation (normally would use WMMA)
            float sum = 0.0f;
            int c_row = (warp_id - 2) * 8 + lane_id / 4;
            int c_col = lane_id % 4;
            
            for (int k = 0; k < 32; k++) {
                sum += shared_A[c_row * 32 + k] * shared_B[k * 128 + c_col];
            }
            
            C[c_row * N + c_col] = sum;
        }
    }
    
    // WMMA demonstration kernel
    #if __CUDA_ARCH__ >= 750
    __global__ void demonstrate_wmma_organization() {
        // WMMA operates on warps - each warp computes a 16x16 tile
        using namespace nvcuda::wmma;
        
        __shared__ half shared_A[16 * 16];
        __shared__ half shared_B[16 * 16]; 
        __shared__ float shared_C[16 * 16];
        
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        if (warp_id == 0) {
            // Declare WMMA fragments
            fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;  
            fragment<accumulator, 16, 16, 16, float> c_frag;
            
            // Initialize accumulator
            fill_fragment(c_frag, 0.0f);
            
            print_thread_info("WMMA Warp");
            
            // Load fragments from shared memory
            load_matrix_sync(a_frag, shared_A, 16);
            load_matrix_sync(b_frag, shared_B, 16);
            
            // Perform matrix multiply-accumulate
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            
            // Store result
            store_matrix_sync(shared_C, c_frag, 16, mem_row_major);
            
            if (lane_id == 0) {
                printf("WMMA computation completed on warp %d\n", warp_id);
            }
        }
    }
    #endif
}

// Function to analyze thread organization of different GEMM configurations
template<typename GemmType>
void analyze_thread_organization(const std::string& name) {
    std::cout << "\n=== Thread Organization: " << name << " ===" << std::endl;
    
    // Extract configuration information
    constexpr int tb_m = GemmType::ThreadblockShape::kM;
    constexpr int tb_n = GemmType::ThreadblockShape::kN;
    constexpr int tb_k = GemmType::ThreadblockShape::kK;
    
    constexpr int warp_m = GemmType::WarpShape::kM;
    constexpr int warp_n = GemmType::WarpShape::kN;
    constexpr int warp_k = GemmType::WarpShape::kK;
    
    // Calculate thread organization
    int warps_m = tb_m / warp_m;
    int warps_n = tb_n / warp_n;
    int total_warps = warps_m * warps_n;
    int total_threads = total_warps * 32;
    
    std::cout << "Thread Block Configuration:" << std::endl;
    std::cout << "  Tile shape: " << tb_m << "x" << tb_n << "x" << tb_k << std::endl;
    std::cout << "  Warp arrangement: " << warps_m << "x" << warps_n << " warps" << std::endl;
    std::cout << "  Total warps per block: " << total_warps << std::endl;
    std::cout << "  Total threads per block: " << total_threads << std::endl;
    
    std::cout << "Warp Specialization:" << std::endl;
    std::cout << "  Each warp handles: " << warp_m << "x" << warp_n << " output elements" << std::endl;
    
    // Analyze work distribution
    long long work_per_warp = 2LL * warp_m * warp_n * tb_k;
    long long work_per_thread = work_per_warp / 32;
    
    std::cout << "Work Distribution:" << std::endl;
    std::cout << "  Work per warp: " << work_per_warp / 1000.0 << " K ops" << std::endl;
    std::cout << "  Work per thread: " << work_per_thread << " ops" << std::endl;
    
    // Analyze instruction utilization
    if (warp_m >= 16 && warp_n >= 16 && warp_k >= 16) {
        std::cout << "  ✅ Suitable for Tensor Core (WMMA) operations" << std::endl;
    } else {
        std::cout << "  ⚠️  May use CUDA cores instead of Tensor Cores" << std::endl;
    }
}

int main() {
    std::cout << "=== CUTLASS Tutorial Chapter 4: Thread Organization ===\n" << std::endl;
    print_device_info();
    
    /*
     * PART 1: Understanding Thread Hierarchy
     */
    std::cout << "\n📚 PART 1: CUTLASS Thread Hierarchy Overview\n" << std::endl;
    
    std::cout << "🏗️  CUDA/CUTLASS Thread Organization:" << std::endl;
    std::cout << "Grid (Device)" << std::endl;
    std::cout << "├── Thread Block 0" << std::endl;
    std::cout << "│   ├── Warp 0 (32 threads: lanes 0-31)" << std::endl;
    std::cout << "│   ├── Warp 1 (32 threads: lanes 0-31)" << std::endl;
    std::cout << "│   └── Warp N..." << std::endl;
    std::cout << "├── Thread Block 1" << std::endl;
    std::cout << "└── Thread Block M..." << std::endl;
    
    std::cout << "\n🎯 Responsibilities at Each Level:" << std::endl;
    std::cout << "• Thread: Individual register-level operations" << std::endl;
    std::cout << "• Warp: SIMT execution, WMMA/MMA instructions" << std::endl;
    std::cout << "• Thread Block: Shared memory cooperation, synchronization" << std::endl;
    std::cout << "• Grid: Covers entire problem, load balancing" << std::endl;
    
    /*
     * PART 2: Demonstrate Thread Hierarchy
     */
    std::cout << "\n🔬 PART 2: Thread Hierarchy Demonstration\n" << std::endl;
    
    std::cout << "Running thread hierarchy demonstration kernel..." << std::endl;
    
    // Launch a simple demonstration
    dim3 block_size(128);  // 4 warps per block
    dim3 grid_size(2);     // 2 blocks
    
    thread_demo::demonstrate_thread_hierarchy<<<grid_size, block_size>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /*
     * PART 3: Warp Specialization Patterns
     */
    std::cout << "\n⚡ PART 3: Warp Specialization Patterns\n" << std::endl;
    
    std::cout << "🎭 Common CUTLASS Warp Specialization Roles:" << std::endl;
    std::cout << "• Producer Warps: Load data from global → shared memory" << std::endl;
    std::cout << "• Consumer Warps: Compute using shared memory data" << std::endl;
    std::cout << "• Mixed Warps: Both load and compute (most common in CUTLASS)" << std::endl;
    
    std::cout << "\n🔄 Producer-Consumer Pattern Benefits:" << std::endl;
    std::cout << "• Hide memory latency behind computation" << std::endl;
    std::cout << "• Maximize memory bandwidth utilization" << std::endl;
    std::cout << "• Enable double/triple buffering" << std::endl;
    
    // Demonstrate warp specialization
    const int demo_M = 128, demo_N = 128, demo_K = 128;
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, demo_M * demo_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, demo_K * demo_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, demo_M * demo_N * sizeof(float)));
    
    std::cout << "\nRunning warp specialization demonstration..." << std::endl;
    thread_demo::demonstrate_warp_specialization<<<1, 128>>>(d_A, d_B, d_C, demo_M, demo_N, demo_K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C));
    
    /*
     * PART 4: WMMA Thread Organization
     */
    std::cout << "\n🔥 PART 4: WMMA/Tensor Core Thread Organization\n" << std::endl;
    
    std::cout << "🧠 WMMA Execution Model:" << std::endl;
    std::cout << "• One warp (32 threads) executes one WMMA instruction" << std::endl;
    std::cout << "• Each thread contributes to matrix fragment data" << std::endl;
    std::cout << "• Fragment distribution across threads is hardware-optimized" << std::endl;
    std::cout << "• Multiple WMMA ops can execute concurrently across warps" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major >= 7 && prop.minor >= 5) {  // Turing+ for WMMA
        std::cout << "\n🚀 Running WMMA demonstration..." << std::endl;
        #if __CUDA_ARCH__ >= 750
        thread_demo::demonstrate_wmma_organization<<<1, 64>>>();  // 2 warps
        CUDA_CHECK(cudaDeviceSynchronize());
        #else
        std::cout << "WMMA demonstration requires compilation for sm_75+" << std::endl;
        #endif
    } else {
        std::cout << "⚠️  WMMA requires Tensor Core capable GPU (sm_75+)" << std::endl;
    }
    
    /*
     * PART 5: Thread Organization Analysis
     */
    std::cout << "\n📊 PART 5: CUTLASS Configuration Analysis\n" << std::endl;
    
    // Define different GEMM configurations to analyze
    using SmallGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>
    >;
    
    using MediumGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>
    >;
    
    using LargeGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor, 
        cutlass::half_t, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>
    >;
    
    analyze_thread_organization<SmallGemm>("Small Thread Block (64x64)");
    analyze_thread_organization<MediumGemm>("Medium Thread Block (128x128)");
    analyze_thread_organization<LargeGemm>("Large Thread Block (256x128)");
    
    /*
     * PART 6: Synchronization and Communication
     */
    std::cout << "\n🔄 PART 6: Thread Synchronization Mechanisms\n" << std::endl;
    
    std::cout << "🔗 CUTLASS Synchronization Primitives:" << std::endl;
    std::cout << "• __syncthreads(): Block-wide barrier" << std::endl;
    std::cout << "• __syncwarp(): Warp-level synchronization" << std::endl;
    std::cout << "• Memory fences: Ensure memory ordering" << std::endl;
    std::cout << "• Atomic operations: Thread-safe shared updates" << std::endl;
    
    std::cout << "\n📡 Inter-Thread Communication:" << std::endl;
    std::cout << "• Warp shuffles: Fast intra-warp data exchange" << std::endl;
    std::cout << "• Shared memory: Inter-warp communication" << std::endl;
    std::cout << "• Global memory: Cross-block communication" << std::endl;
    
    /*
     * PART 7: Load Balancing Strategies
     */
    std::cout << "\n⚖️ PART 7: Load Balancing in CUTLASS\n" << std::endl;
    
    std::cout << "🎯 CUTLASS Load Balancing Techniques:" << std::endl;
    std::cout << "• Dynamic tile assignment: Thread blocks take work as available" << std::endl;
    std::cout << "• Swizzling patterns: Distribute work to avoid hotspots" << std::endl;
    std::cout << "• Cooperative loading: Multiple blocks collaborate on large tiles" << std::endl;
    std::cout << "• Stream compaction: Handle sparse or irregular matrices" << std::endl;
    
    std::cout << "\n🔧 Problem-Specific Optimizations:" << std::endl;
    std::cout << "• Small matrices: Use fewer, larger thread blocks" << std::endl;
    std::cout << "• Large matrices: Use more, smaller thread blocks" << std::endl;
    std::cout << "• Rectangular matrices: Adjust warp organization accordingly" << std::endl;
    std::cout << "• Sparse matrices: Skip empty blocks/warps" << std::endl;
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Chapter 4 Summary: Thread Mastery ===\n" << std::endl;
    std::cout << "✅ You learned about:" << std::endl;
    std::cout << "   • CUTLASS thread hierarchy (Grid → Block → Warp → Thread)" << std::endl;
    std::cout << "   • Warp specialization patterns and their benefits" << std::endl;
    std::cout << "   • WMMA/Tensor Core thread organization and execution model" << std::endl;
    std::cout << "   • Synchronization mechanisms and inter-thread communication" << std::endl;
    std::cout << "   • Load balancing strategies for different problem types" << std::endl;
    std::cout << "   • Thread organization analysis for CUTLASS configurations" << std::endl;
    
    std::cout << "\n🧠 Thread Organization Guidelines:" << std::endl;
    std::cout << "   • Match thread block size to problem characteristics" << std::endl;
    std::cout << "   • Use warp specialization for memory-compute overlap" << std::endl;
    std::cout << "   • Leverage Tensor Cores with proper warp organization" << std::endl;
    std::cout << "   • Balance work per thread vs total parallelism" << std::endl;
    
    std::cout << "\n🎯 Next: Chapter 5 will explore epilogue operations and output fusion!" << std::endl;
    
    return 0;
}