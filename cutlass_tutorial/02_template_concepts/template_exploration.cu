#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 2: Template Concepts Deep Dive
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand how CUTLASS templates generate different kernels
 * 2. Explore the impact of different template parameters
 * 3. Learn about tile sizes and how they affect performance
 * 4. Compare different data type combinations
 * 5. Understand operator classes and architectures
 * 
 * KEY CONCEPTS:
 * - Template specialization generates optimized code at compile time
 * - Different combinations create vastly different performance characteristics
 * - Tile sizes control memory access patterns and shared memory usage
 * - Operator classes determine computational pattern (SIMT vs TensorOp)
 */

// Template helper to print GEMM configuration info
template<typename GemmType>
void print_gemm_info(const std::string& name) {
    std::cout << "\n=== " << name << " Configuration ===" << std::endl;
    
    // Extract information from the GEMM type
    using ElementA = typename GemmType::ElementA;
    using ElementB = typename GemmType::ElementB;
    using ElementC = typename GemmType::ElementC;
    using LayoutA = typename GemmType::LayoutA;
    using LayoutB = typename GemmType::LayoutB;
    using LayoutC = typename GemmType::LayoutC;
    using OperatorClass = typename GemmType::OperatorClass;
    using ArchTag = typename GemmType::ArchTag;
    
    std::cout << "Element A: " << typeid(ElementA).name() << std::endl;
    std::cout << "Element B: " << typeid(ElementB).name() << std::endl;
    std::cout << "Element C: " << typeid(ElementC).name() << std::endl;
    std::cout << "Layout A: " << (std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? "RowMajor" : "ColumnMajor") << std::endl;
    std::cout << "Layout B: " << (std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? "RowMajor" : "ColumnMajor") << std::endl;
    std::cout << "Layout C: " << (std::is_same_v<LayoutC, cutlass::layout::RowMajor> ? "RowMajor" : "ColumnMajor") << std::endl;
    
    // Print tile size information if available
    if constexpr (requires { GemmType::ThreadblockShape::kM; }) {
        std::cout << "Threadblock Shape: " << GemmType::ThreadblockShape::kM 
                  << "x" << GemmType::ThreadblockShape::kN 
                  << "x" << GemmType::ThreadblockShape::kK << std::endl;
    }
    
    if constexpr (requires { GemmType::WarpShape::kM; }) {
        std::cout << "Warp Shape: " << GemmType::WarpShape::kM 
                  << "x" << GemmType::WarpShape::kN 
                  << "x" << GemmType::WarpShape::kK << std::endl;
    }
}

// Template function to run and benchmark a GEMM configuration
template<typename GemmType>
float benchmark_gemm_config(const std::string& name, int M, int N, int K, int iterations = 10) {
    std::cout << "\n🚀 Benchmarking: " << name << std::endl;
    
    // Allocate tensors
    cutlass::HostTensor<typename GemmType::ElementA, typename GemmType::LayoutA> tensor_A({M, K});
    cutlass::HostTensor<typename GemmType::ElementB, typename GemmType::LayoutB> tensor_B({K, N});
    cutlass::HostTensor<typename GemmType::ElementC, typename GemmType::LayoutC> tensor_C({M, N});
    cutlass::HostTensor<typename GemmType::ElementC, typename GemmType::LayoutC> tensor_D({M, N});
    
    // Initialize with random data
    initialize_matrix_random(tensor_A.host_data(), M, K);
    initialize_matrix_random(tensor_B.host_data(), K, N);
    initialize_matrix_random(tensor_C.host_data(), M, N);
    
    // Sync to device
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    
    // Set up arguments
    typename GemmType::Arguments arguments{
        {M, N, K},
        tensor_A.device_ref(),
        tensor_B.device_ref(),
        tensor_C.device_ref(),
        tensor_D.device_ref(),
        {1.0f, 1.0f}
    };
    
    // Create and initialize operator
    GemmType gemm_operator;
    cutlass::Status status = gemm_operator.can_implement(arguments);
    
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ Cannot implement this configuration: " << cutlassGetStatusString(status) << std::endl;
        return -1.0f;
    }
    
    status = gemm_operator.initialize(arguments);
    CUTLASS_CHECK(status);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        gemm_operator();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CutlassTimer timer;
    timer.clear();
    
    for (int i = 0; i < iterations; i++) {
        timer.start();
        status = gemm_operator();
        timer.stop();
        CUTLASS_CHECK(status);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float avg_time = timer.get_average_time();
    double gflops = calculate_gflops(M, N, K, avg_time);
    
    std::cout << "   Time: " << avg_time << " ms" << std::endl;
    std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;
    
    return avg_time;
}

int main() {
    std::cout << "=== CUTLASS Tutorial Chapter 2: Template Concepts ===" << std::endl;
    print_device_info();
    
    const int M = 2048;
    const int N = 2048;  
    const int K = 2048;
    
    std::cout << "\nMatrix dimensions: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "We'll compare different template configurations...\n" << std::endl;
    
    /*
     * EXPERIMENT 1: Different Data Type Combinations
     * 
     * This shows how changing data types affects performance and precision
     */
    std::cout << "🧪 EXPERIMENT 1: Data Type Impact\n" << std::endl;
    
    // Configuration 1: FP32 throughout (highest precision, lower performance)
    using GemmFP32 = cutlass::gemm::device::Gemm<
        float,                      // A: FP32
        cutlass::layout::RowMajor,
        float,                      // B: FP32
        cutlass::layout::RowMajor,
        float,                      // C: FP32
        cutlass::layout::RowMajor,
        float,                      // Accumulator: FP32
        cutlass::arch::OpClassSimt, // SIMT operations (no Tensor Cores)
        cutlass::arch::Sm80
    >;
    
    // Configuration 2: FP16 inputs, FP32 accumulation (balanced)
    using GemmMixed = cutlass::gemm::device::Gemm<
        cutlass::half_t,            // A: FP16
        cutlass::layout::RowMajor,
        cutlass::half_t,            // B: FP16
        cutlass::layout::RowMajor,
        cutlass::half_t,            // C: FP16
        cutlass::layout::RowMajor,
        float,                      // Accumulator: FP32
        cutlass::arch::OpClassTensorOp, // Tensor Cores enabled
        cutlass::arch::Sm80
    >;
    
    // Configuration 3: FP16 throughout (highest performance, lower precision)
    using GemmFP16 = cutlass::gemm::device::Gemm<
        cutlass::half_t,            // A: FP16
        cutlass::layout::RowMajor,
        cutlass::half_t,            // B: FP16
        cutlass::layout::RowMajor,
        cutlass::half_t,            // C: FP16
        cutlass::layout::RowMajor,
        cutlass::half_t,            // Accumulator: FP16
        cutlass::arch::OpClassTensorOp, // Tensor Cores enabled
        cutlass::arch::Sm80
    >;
    
    print_gemm_info<GemmFP32>("FP32 Configuration");
    print_gemm_info<GemmMixed>("Mixed Precision Configuration");
    print_gemm_info<GemmFP16>("FP16 Configuration");
    
    // Note: We'll only benchmark the mixed precision version to avoid compilation issues
    // with different data types in this simplified example
    
    float mixed_time = benchmark_gemm_config<GemmMixed>("Mixed Precision (FP16->FP32)", M, N, K);
    
    /*
     * EXPERIMENT 2: Layout Impact
     * 
     * This shows how memory layout affects performance
     */
    std::cout << "\n🧪 EXPERIMENT 2: Memory Layout Impact\n" << std::endl;
    
    // Configuration: Row-major A and B
    using GemmRowRow = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,    // A: Row-major
        cutlass::half_t, 
        cutlass::layout::RowMajor,    // B: Row-major
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;
    
    // Configuration: Row-major A, Column-major B (often optimal for GEMM)
    using GemmRowCol = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,    // A: Row-major
        cutlass::half_t,
        cutlass::layout::ColumnMajor, // B: Column-major
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;
    
    std::cout << "Layout combinations and their typical use cases:" << std::endl;
    std::cout << "• Row-Row: Natural for C/C++ matrices" << std::endl;
    std::cout << "• Row-Col: Often optimal for GEMM performance" << std::endl;
    std::cout << "• Col-Row: Common in scientific computing" << std::endl;
    std::cout << "• Col-Col: Fortran-style matrices" << std::endl;
    
    float row_row_time = benchmark_gemm_config<GemmRowRow>("Row-Row Layout", M, N, K);
    float row_col_time = benchmark_gemm_config<GemmRowCol>("Row-Col Layout", M, N, K);
    
    /*
     * EXPERIMENT 3: Operator Class Impact
     */
    std::cout << "\n🧪 EXPERIMENT 3: Operator Class Impact\n" << std::endl;
    
    std::cout << "Operator Classes:" << std::endl;
    std::cout << "• OpClassSimt: Traditional CUDA cores (more flexible)" << std::endl;
    std::cout << "• OpClassTensorOp: Tensor Cores (higher performance for supported types)" << std::endl;
    std::cout << "• OpClassWmmaTensorOp: WMMA Tensor Cores (broader compatibility)" << std::endl;
    
    // We already benchmarked TensorOp above, so let's compare concepts
    std::cout << "\n💡 Key Insights about Operator Classes:" << std::endl;
    std::cout << "• TensorOp requires specific data types (FP16, BF16, Int8, etc.)" << std::endl;
    std::cout << "• TensorOp provides 4-16x speedup over SIMT for supported operations" << std::endl;
    std::cout << "• SIMT works with any data type but is slower for matrix operations" << std::endl;
    
    /*
     * EXPERIMENT 4: Understanding Template Specialization Effects
     */
    std::cout << "\n🧪 EXPERIMENT 4: Template Specialization Magic\n" << std::endl;
    
    std::cout << "🔍 How CUTLASS Templates Work:" << std::endl;
    std::cout << "1. Compile-time code generation based on template parameters" << std::endl;
    std::cout << "2. Different parameter combinations generate completely different kernels" << std::endl;
    std::cout << "3. Optimal tile sizes, memory access patterns, and instruction selection" << std::endl;
    std::cout << "4. Dead code elimination for unused features" << std::endl;
    
    std::cout << "\n📊 Template Parameter Impact on Performance:" << std::endl;
    if (mixed_time > 0 && row_row_time > 0 && row_col_time > 0) {
        std::cout << "• Layout optimization: " 
                  << ((row_row_time > row_col_time) ? 
                      (row_row_time / row_col_time) : (row_col_time / row_row_time)) 
                  << "x speedup from optimal layout" << std::endl;
    }
    
    /*
     * EXPERIMENT 5: Deep Dive into Tile Sizes
     */
    std::cout << "\n🧪 EXPERIMENT 5: Understanding Tile Sizes\n" << std::endl;
    
    std::cout << "🧩 CUTLASS Tiling Hierarchy:" << std::endl;
    std::cout << "Grid Level      → Multiple thread blocks working on different output tiles" << std::endl;
    std::cout << "Thread Block    → 128x128x32 (typical) threads working on shared tile" << std::endl;  
    std::cout << "Warp Level      → 32x32x32 (typical) threads in a warp" << std::endl;
    std::cout << "Instruction     → 16x16x16 (typical) Tensor Core instruction" << std::endl;
    
    std::cout << "\n📐 Why Tile Sizes Matter:" << std::endl;
    std::cout << "• Larger tiles → Better compute utilization, more shared memory usage" << std::endl;
    std::cout << "• Smaller tiles → Better load balancing, less shared memory usage" << std::endl;
    std::cout << "• Must be multiples of warp/instruction sizes" << std::endl;
    std::cout << "• Hardware limits: shared memory, register count, occupancy" << std::endl;
    
    /*
     * SUMMARY AND KEY TAKEAWAYS
     */
    std::cout << "\n=== Chapter 2 Summary: Template Mastery ===" << std::endl;
    std::cout << "✅ You learned about:" << std::endl;
    std::cout << "   • How template parameters control kernel generation" << std::endl;
    std::cout << "   • Impact of data types on performance and precision" << std::endl;
    std::cout << "   • Memory layout effects on computational efficiency" << std::endl;
    std::cout << "   • Operator classes and their performance characteristics" << std::endl;
    std::cout << "   • Hierarchical tiling and its performance implications" << std::endl;
    
    std::cout << "\n🧠 Template Selection Guidelines:" << std::endl;
    std::cout << "   • Use FP16+TensorOp for maximum performance" << std::endl;
    std::cout << "   • Use FP32 accumulation to maintain numerical stability" << std::endl;
    std::cout << "   • Experiment with layouts for your specific use case" << std::endl;
    std::cout << "   • Choose tile sizes based on problem size and hardware limits" << std::endl;
    
    std::cout << "\n🎯 Next: Chapter 3 will dive deep into memory hierarchy and tiling!" << std::endl;
    
    return 0;
}