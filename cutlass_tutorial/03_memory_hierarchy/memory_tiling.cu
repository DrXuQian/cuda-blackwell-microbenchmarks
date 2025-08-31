#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 3: Memory Hierarchy Mastery
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand CUTLASS's hierarchical memory management
 * 2. Learn about tile iterators and memory access patterns
 * 3. Explore shared memory usage and optimization
 * 4. Understand register file utilization
 * 5. Learn about memory bandwidth vs compute throughput
 * 6. Compare different tiling strategies
 * 
 * KEY CONCEPTS:
 * - Memory hierarchy: Global → Shared → Register
 * - Tile iterators: How CUTLASS loads data efficiently
 * - Memory coalescing and bank conflicts
 * - Double/triple buffering for latency hiding
 * - Occupancy vs resource utilization trade-offs
 */

// Custom GEMM configurations to demonstrate memory concepts
namespace memory_demo {

    // Configuration 1: Small tiles (less shared memory, higher occupancy)
    using SmallTileGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,    // Small threadblock tile
        cutlass::gemm::GemmShape<32, 32, 32>,    // Warp tile  
        cutlass::gemm::GemmShape<16, 8, 16>      // Instruction tile
    >;

    // Configuration 2: Large tiles (more shared memory, lower occupancy)
    using LargeTileGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 64>,  // Large threadblock tile
        cutlass::gemm::GemmShape<64, 64, 64>,    // Warp tile
        cutlass::gemm::GemmShape<16, 8, 16>      // Instruction tile
    >;

    // Configuration 3: Memory-optimized (balanced approach)
    using BalancedGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,  // Balanced threadblock tile
        cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile
        cutlass::gemm::GemmShape<16, 8, 16>      // Instruction tile
    >;
}

// Function to analyze memory requirements of a GEMM configuration
template<typename GemmType>
void analyze_memory_usage(const std::string& name) {
    std::cout << "\n=== Memory Analysis: " << name << " ===" << std::endl;
    
    // Extract tile dimensions
    constexpr int tb_m = GemmType::ThreadblockShape::kM;
    constexpr int tb_n = GemmType::ThreadblockShape::kN;
    constexpr int tb_k = GemmType::ThreadblockShape::kK;
    
    constexpr int warp_m = GemmType::WarpShape::kM;
    constexpr int warp_n = GemmType::WarpShape::kN;
    constexpr int warp_k = GemmType::WarpShape::kK;
    
    std::cout << "Threadblock Shape: " << tb_m << "x" << tb_n << "x" << tb_k << std::endl;
    std::cout << "Warp Shape: " << warp_m << "x" << warp_n << "x" << warp_k << std::endl;
    
    // Calculate approximate shared memory usage (FP16)
    size_t shared_A = tb_m * tb_k * 2; // 2 bytes per FP16
    size_t shared_B = tb_k * tb_n * 2;
    size_t total_shared = shared_A + shared_B;
    
    // Account for double buffering (common in CUTLASS)
    size_t double_buffered = total_shared * 2;
    
    std::cout << "Shared Memory Usage:" << std::endl;
    std::cout << "  A tile: " << shared_A << " bytes" << std::endl;
    std::cout << "  B tile: " << shared_B << " bytes" << std::endl;
    std::cout << "  Single buffer: " << total_shared << " bytes (" 
              << total_shared / 1024.0f << " KB)" << std::endl;
    std::cout << "  Double buffered: " << double_buffered << " bytes (" 
              << double_buffered / 1024.0f << " KB)" << std::endl;
    
    // Estimate occupancy impact
    constexpr int max_shared_per_sm = 164 * 1024; // RTX 5070 Blackwell
    int max_blocks_shared = max_shared_per_sm / (double_buffered + 1024); // Add some overhead
    
    std::cout << "Occupancy Analysis:" << std::endl;
    std::cout << "  Max blocks limited by shared memory: " << max_blocks_shared << std::endl;
    
    // Calculate work per threadblock
    long long work_per_tb = 2LL * tb_m * tb_n * tb_k; // 2 ops per multiply-add
    std::cout << "  Work per threadblock: " << work_per_tb / 1000000.0 << " M ops" << std::endl;
    
    // Memory arithmetic intensity
    size_t bytes_loaded = (tb_m * tb_k + tb_k * tb_n + tb_m * tb_n) * 2; // FP16
    double arithmetic_intensity = double(work_per_tb) / double(bytes_loaded);
    std::cout << "  Arithmetic intensity: " << arithmetic_intensity << " ops/byte" << std::endl;
    
    if (arithmetic_intensity > 10.0) {
        std::cout << "  → Likely compute-bound (good!)" << std::endl;
    } else if (arithmetic_intensity > 5.0) {
        std::cout << "  → Balanced compute/memory" << std::endl;
    } else {
        std::cout << "  → May be memory-bound" << std::endl;
    }
}

// Function to benchmark and analyze memory bandwidth utilization
template<typename GemmType>
void benchmark_memory_patterns(const std::string& name, int M, int N, int K) {
    std::cout << "\n🔍 Memory Benchmark: " << name << std::endl;
    
    // Allocate tensors
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_C({M, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_D({M, N});
    
    // Initialize
    initialize_matrix_random(tensor_A.host_data(), M, K);
    initialize_matrix_random(tensor_B.host_data(), K, N);
    initialize_matrix_random(tensor_C.host_data(), M, N);
    
    // Sync to device
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    
    // Setup GEMM
    typename GemmType::Arguments arguments{
        {M, N, K},
        tensor_A.device_ref(),
        tensor_B.device_ref(),
        tensor_C.device_ref(),
        tensor_D.device_ref(),
        {1.0f, 1.0f}
    };
    
    GemmType gemm_operator;
    cutlass::Status status = gemm_operator.can_implement(arguments);
    
    if (status != cutlass::Status::kSuccess) {
        std::cout << "   ❌ Cannot implement: " << cutlassGetStatusString(status) << std::endl;
        return;
    }
    
    status = gemm_operator.initialize(arguments);
    CUTLASS_CHECK(status);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_operator();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CutlassTimer timer;
    const int iterations = 10;
    
    for (int i = 0; i < iterations; i++) {
        timer.start();
        status = gemm_operator();
        timer.stop();
        CUTLASS_CHECK(status);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float avg_time = timer.get_average_time();
    double gflops = calculate_gflops(M, N, K, avg_time);
    
    // Calculate memory bandwidth
    size_t bytes_transferred = size_t(M) * K * 2 +  // Read A
                              size_t(K) * N * 2 +   // Read B  
                              size_t(M) * N * 2 +   // Read C
                              size_t(M) * N * 2;    // Write D
    double bandwidth = calculate_memory_bandwidth_gb_s(bytes_transferred, avg_time);
    
    std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "   Memory bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "   Time: " << avg_time << " ms" << std::endl;
}

int main() {
    std::cout << "=== CUTLASS Tutorial Chapter 3: Memory Hierarchy ===\n" << std::endl;
    print_device_info();
    
    /*
     * PART 1: Understanding CUTLASS Memory Organization
     */
    std::cout << "\n📚 PART 1: CUTLASS Memory Hierarchy Overview\n" << std::endl;
    
    std::cout << "🏗️  CUTLASS Memory Architecture:" << std::endl;
    std::cout << "┌─────────────────┐" << std::endl;
    std::cout << "│ Global Memory   │ ← Input/Output matrices (high latency, high bandwidth)" << std::endl;
    std::cout << "└─────────────────┘" << std::endl;
    std::cout << "         ↕" << std::endl;  
    std::cout << "┌─────────────────┐" << std::endl;
    std::cout << "│ Shared Memory   │ ← Tile data shared by thread block (low latency)" << std::endl;
    std::cout << "└─────────────────┘" << std::endl;
    std::cout << "         ↕" << std::endl;
    std::cout << "┌─────────────────┐" << std::endl;
    std::cout << "│ Register File   │ ← Per-thread fragments (ultra-low latency)" << std::endl;
    std::cout << "└─────────────────┘" << std::endl;
    
    std::cout << "\n🎯 Key CUTLASS Memory Concepts:" << std::endl;
    std::cout << "• Tile Iterators: Load data from global → shared memory efficiently" << std::endl;
    std::cout << "• Fragment Iterators: Move data from shared → registers" << std::endl;
    std::cout << "• Double Buffering: Overlap memory loads with computation" << std::endl;
    std::cout << "• Bank Conflict Avoidance: Careful shared memory layout design" << std::endl;
    
    /*
     * PART 2: Analyzing Different Tile Sizes
     */
    std::cout << "\n🔬 PART 2: Tile Size Impact on Memory Usage\n" << std::endl;
    
    analyze_memory_usage<memory_demo::SmallTileGemm>("Small Tiles (64x64x32)");
    analyze_memory_usage<memory_demo::LargeTileGemm>("Large Tiles (128x256x64)");
    analyze_memory_usage<memory_demo::BalancedGemm>("Balanced Tiles (128x128x32)");
    
    /*
     * PART 3: Memory Access Pattern Analysis
     */
    std::cout << "\n⚡ PART 3: Memory Access Pattern Experiments\n" << std::endl;
    
    const int M = 2048, N = 2048, K = 2048;
    
    std::cout << "Testing different tiling strategies on " << M << "x" << N << "x" << K << " GEMM:\n" << std::endl;
    
    benchmark_memory_patterns<memory_demo::SmallTileGemm>("Small Tiles", M, N, K);
    benchmark_memory_patterns<memory_demo::BalancedGemm>("Balanced Tiles", M, N, K);
    benchmark_memory_patterns<memory_demo::LargeTileGemm>("Large Tiles", M, N, K);
    
    /*
     * PART 4: Understanding Memory Coalescing
     */
    std::cout << "\n🚀 PART 4: Memory Coalescing and Layout Impact\n" << std::endl;
    
    std::cout << "🧠 Memory Coalescing Principles:" << std::endl;
    std::cout << "• 32 threads in a warp should access consecutive memory addresses" << std::endl;
    std::cout << "• Optimal: 128-byte aligned, 128-byte transactions" << std::endl;
    std::cout << "• CUTLASS tile iterators are designed for optimal coalescing" << std::endl;
    
    std::cout << "\n📐 Layout Impact on Performance:" << std::endl;
    std::cout << "• RowMajor A + RowMajor B: May require B matrix transpose" << std::endl;
    std::cout << "• RowMajor A + ColumnMajor B: Natural for A*B computation" << std::endl;
    std::cout << "• CUTLASS handles different layout combinations efficiently" << std::endl;
    
    /*
     * PART 5: Shared Memory Deep Dive
     */
    std::cout << "\n🏦 PART 5: Shared Memory Optimization\n" << std::endl;
    
    std::cout << "🔧 CUTLASS Shared Memory Strategies:" << std::endl;
    std::cout << "• Tile Shape Selection: Balance work vs memory usage" << std::endl;
    std::cout << "• Bank Conflict Avoidance: Padding and swizzling patterns" << std::endl;
    std::cout << "• Double Buffering: Pipeline memory transfers with compute" << std::endl;
    std::cout << "• Cross-warp Data Sharing: Maximize data reuse within thread block" << std::endl;
    
    std::cout << "\n💾 RTX 5070 Blackwell Shared Memory Specs:" << std::endl;
    std::cout << "• Total per SM: 164 KB" << std::endl;
    std::cout << "• Banks: 32 banks × 4 bytes = 128 bytes per cycle" << std::endl;
    std::cout << "• Conflict-free access: 32 threads accessing different banks" << std::endl;
    
    /*
     * PART 6: Register File Utilization
     */
    std::cout << "\n⚙️ PART 6: Register File and Occupancy\n" << std::endl;
    
    std::cout << "🎯 Register File Considerations:" << std::endl;
    std::cout << "• Each thread has access to ~65,536 32-bit registers" << std::endl;
    std::cout << "• CUTLASS fragments use registers to hold working data" << std::endl;
    std::cout << "• Higher register usage → Lower occupancy" << std::endl;
    std::cout << "• Trade-off: More registers for better ILP vs more threads for latency hiding" << std::endl;
    
    std::cout << "\n🔄 CUTLASS Fragment Management:" << std::endl;
    std::cout << "• Matrix fragments stored in register arrays" << std::endl;
    std::cout << "• Tensor Core instructions operate on register fragments" << std::endl;
    std::cout << "• CUTLASS automatically manages fragment shapes and layouts" << std::endl;
    
    /*
     * PART 7: Memory Bandwidth vs Compute Analysis
     */
    std::cout << "\n📊 PART 7: Roofline Analysis Concepts\n" << std::endl;
    
    std::cout << "🏗️ Roofline Model for GEMM:" << std::endl;
    std::cout << "• Peak compute: ~165 TFLOPS (RTX 5070 FP16 Tensor Cores)" << std::endl;
    std::cout << "• Peak bandwidth: ~672 GB/s (RTX 5070 GDDR7)" << std::endl;
    std::cout << "• Arithmetic intensity = FLOPs / Bytes" << std::endl;
    std::cout << "• For GEMM: 2*M*N*K FLOPs, (M*K + K*N + M*N)*2 bytes" << std::endl;
    
    std::cout << "\n⚖️ Memory vs Compute Trade-offs:" << std::endl;
    std::cout << "• Small problems: Memory-bound (insufficient work per data loaded)" << std::endl;
    std::cout << "• Large problems: Compute-bound (plenty of work per data)" << std::endl;
    std::cout << "• CUTLASS tiling strategies adapt to problem characteristics" << std::endl;
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Chapter 3 Summary: Memory Mastery ===\n" << std::endl;
    std::cout << "✅ You learned about:" << std::endl;
    std::cout << "   • CUTLASS hierarchical memory management (Global → Shared → Register)" << std::endl;
    std::cout << "   • Tile iterators and their role in efficient data movement" << std::endl;
    std::cout << "   • Impact of tile sizes on shared memory usage and occupancy" << std::endl;
    std::cout << "   • Memory coalescing and layout optimization strategies" << std::endl;
    std::cout << "   • Register file utilization and fragment management" << std::endl;
    std::cout << "   • Roofline analysis for understanding memory vs compute bounds" << std::endl;
    
    std::cout << "\n🧠 Memory Optimization Guidelines:" << std::endl;
    std::cout << "   • Choose tile sizes based on problem size and hardware limits" << std::endl;
    std::cout << "   • Maximize arithmetic intensity to become compute-bound" << std::endl;
    std::cout << "   • Use appropriate layouts for optimal memory access patterns" << std::endl;
    std::cout << "   • Balance occupancy vs work per thread for peak performance" << std::endl;
    
    std::cout << "\n🎯 Next: Chapter 4 will explore thread block organization and warp coordination!" << std::endl;
    
    return 0;
}