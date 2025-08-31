#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 5: Epilogue Operations and Fusion
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand what epilogue operations are and their importance
 * 2. Learn about built-in CUTLASS epilogue functions
 * 3. Explore custom epilogue operation creation
 * 4. Understand fusion benefits for performance and memory
 * 5. Compare fused vs separate operation performance
 * 6. Learn about epilogue visitor patterns
 * 
 * KEY CONCEPTS:
 * - Epilogue operations: Post-GEMM computations applied to output
 * - Kernel fusion: Combining operations to reduce memory traffic
 * - Activation functions: ReLU, GELU, Sigmoid, etc.
 * - Bias addition and scaling operations
 * - Custom epilogue functors
 * - Memory bandwidth savings through fusion
 */

// Namespace for epilogue demonstrations
namespace epilogue_demo {

    // Custom epilogue functor: Leaky ReLU
    template<typename ElementOutput_, typename ElementAccumulator_, typename ElementCompute_>
    struct LeakyReLU {
        using ElementOutput = ElementOutput_;
        using ElementAccumulator = ElementAccumulator_;
        using ElementCompute = ElementCompute_;

        struct Arguments {
            ElementCompute alpha;
            ElementCompute beta;
            ElementCompute leaky_alpha;  // Leaky ReLU parameter
            
            Arguments(ElementCompute alpha_ = ElementCompute(1), 
                     ElementCompute beta_ = ElementCompute(0),
                     ElementCompute leaky_alpha_ = ElementCompute(0.01))
                : alpha(alpha_), beta(beta_), leaky_alpha(leaky_alpha_) {}
        };

        struct Params {
            ElementCompute alpha;
            ElementCompute beta;
            ElementCompute leaky_alpha;
            
            Params() = default;
            Params(Arguments const& args) 
                : alpha(args.alpha), beta(args.beta), leaky_alpha(args.leaky_alpha) {}
        };

        template<typename ConvertOp, int kCount>
        CUTLASS_DEVICE
        cutlass::Array<ElementOutput, kCount> operator()(
            cutlass::Array<ElementAccumulator, kCount> const& accumulator,
            cutlass::Array<ElementOutput, kCount> const& source) const {
            
            ConvertOp convert_op;
            cutlass::Array<ElementOutput, kCount> result;
            
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                ElementCompute intermediate = 
                    alpha * ElementCompute(accumulator[i]) + beta * ElementCompute(source[i]);
                
                // Apply Leaky ReLU: f(x) = x if x > 0, else leaky_alpha * x
                if (intermediate > ElementCompute(0)) {
                    result[i] = convert_op(intermediate);
                } else {
                    result[i] = convert_op(leaky_alpha * intermediate);
                }
            }
            
            return result;
        }
        
        Params params;
    };
}

// Function to benchmark different epilogue configurations
template<typename GemmType>
float benchmark_with_epilogue(const std::string& name, int M, int N, int K, int iterations = 10) {
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
    std::cout << "=== CUTLASS Tutorial Chapter 5: Epilogue Operations ===\n" << std::endl;
    print_device_info();
    
    const int M = 2048;
    const int N = 2048;  
    const int K = 2048;
    
    std::cout << "\nMatrix dimensions: " << M << "x" << N << "x" << K << std::endl;
    
    /*
     * PART 1: Understanding Epilogue Operations
     */
    std::cout << "\n📚 PART 1: What Are Epilogue Operations?\n" << std::endl;
    
    std::cout << "🎯 Epilogue Operations Definition:" << std::endl;
    std::cout << "• Operations applied to GEMM output before storing to global memory" << std::endl;
    std::cout << "• Fused with the GEMM kernel to avoid separate memory round-trips" << std::endl;
    std::cout << "• Common in neural networks: D = activation(α*A*B + β*C)" << std::endl;
    std::cout << "• Examples: bias addition, activation functions, scaling, residual connections" << std::endl;
    
    std::cout << "\n💡 Why Epilogue Fusion Matters:" << std::endl;
    std::cout << "• Reduces memory bandwidth: No intermediate storage required" << std::endl;
    std::cout << "• Improves performance: Eliminates separate kernel launches" << std::endl;
    std::cout << "• Better cache locality: Data processed while still in registers/cache" << std::endl;
    std::cout << "• Lower latency: Fewer synchronization points" << std::endl;
    
    std::cout << "\n🔄 GEMM + Epilogue Flow:" << std::endl;
    std::cout << "1. Load A and B tiles" << std::endl;
    std::cout << "2. Compute matrix multiplication: Accumulator = A * B" << std::endl;  
    std::cout << "3. Load bias/source C" << std::endl;
    std::cout << "4. Apply epilogue: D = epilogue_function(Accumulator, C)" << std::endl;
    std::cout << "5. Store final result D to global memory" << std::endl;
    
    /*
     * PART 2: Built-in Epilogue Operations
     */
    std::cout << "\n🛠️ PART 2: Built-in CUTLASS Epilogue Operations\n" << std::endl;
    
    std::cout << "📋 Common Built-in Epilogues:" << std::endl;
    std::cout << "• LinearCombination: D = α*A*B + β*C (basic GEMM)" << std::endl;
    std::cout << "• LinearCombinationReLU: D = ReLU(α*A*B + β*C)" << std::endl;
    std::cout << "• LinearCombinationGELU: D = GELU(α*A*B + β*C)" << std::endl;
    std::cout << "• LinearCombinationSigmoid: D = Sigmoid(α*A*B + β*C)" << std::endl;
    std::cout << "• LinearCombinationTanh: D = Tanh(α*A*B + β*C)" << std::endl;
    std::cout << "• LinearCombinationClamp: D = Clamp(α*A*B + β*C, min, max)" << std::endl;
    
    /*
     * PART 3: Epilogue Performance Comparison
     */
    std::cout << "\n⚡ PART 3: Epilogue Performance Experiments\n" << std::endl;
    
    // Configuration 1: Basic linear combination (no activation)
    using BasicGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                    // ElementA
        cutlass::layout::RowMajor,          // LayoutA
        cutlass::half_t,                    // ElementB
        cutlass::layout::RowMajor,          // LayoutB
        cutlass::half_t,                    // ElementC
        cutlass::layout::RowMajor,          // LayoutC
        float,                              // ElementAccumulator
        cutlass::arch::OpClassTensorOp,     // OpClass
        cutlass::arch::Sm80                 // ArchTag
    >;
    
    // Configuration 2: With ReLU activation
    using ReLUGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                    // ElementA
        cutlass::layout::RowMajor,          // LayoutA
        cutlass::half_t,                    // ElementB
        cutlass::layout::RowMajor,          // LayoutB
        cutlass::half_t,                    // ElementC
        cutlass::layout::RowMajor,          // LayoutC
        float,                              // ElementAccumulator
        cutlass::arch::OpClassTensorOp,     // OpClass
        cutlass::arch::Sm80,                // ArchTag
        cutlass::gemm::GemmShape<128, 128, 32>,    // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,      // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,       // InstructionShape
        cutlass::epilogue::thread::LinearCombinationRelu<
            cutlass::half_t,                // ElementOutput
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,  // ElementsPerAccess
            float,                          // ElementAccumulator
            float                           // ElementCompute
        >
    >;
    
    float basic_time = benchmark_with_epilogue<BasicGemm>("Basic GEMM (No Activation)", M, N, K);
    float relu_time = benchmark_with_epilogue<ReLUGemm>("GEMM + ReLU Fusion", M, N, K);
    
    /*
     * PART 4: Understanding Epilogue Components
     */
    std::cout << "\n🔧 PART 4: Epilogue Architecture Deep Dive\n" << std::endl;
    
    std::cout << "🏗️ Epilogue Components:" << std::endl;
    std::cout << "• Output Tile Iterator: Manages output memory access patterns" << std::endl;
    std::cout << "• Epilogue Functor: Applies mathematical operations" << std::endl;  
    std::cout << "• Conversion Operators: Handle data type conversions" << std::endl;
    std::cout << "• Vectorized Access: Process multiple elements per thread" << std::endl;
    
    std::cout << "\n⚙️ Epilogue Execution Flow:" << std::endl;
    std::cout << "1. Each thread receives accumulator fragments" << std::endl;
    std::cout << "2. Load corresponding source (bias) data if needed" << std::endl;
    std::cout << "3. Apply epilogue function element-wise" << std::endl;
    std::cout << "4. Convert to output data type" << std::endl;
    std::cout << "5. Store vectorized output to global memory" << std::endl;
    
    /*
     * PART 5: Memory Bandwidth Analysis
     */
    std::cout << "\n📊 PART 5: Memory Bandwidth Analysis\n" << std::endl;
    
    if (basic_time > 0 && relu_time > 0) {
        std::cout << "🔍 Fusion Performance Analysis:" << std::endl;
        std::cout << "   Basic GEMM time: " << basic_time << " ms" << std::endl;
        std::cout << "   GEMM+ReLU time: " << relu_time << " ms" << std::endl;
        
        float overhead = ((relu_time - basic_time) / basic_time) * 100.0f;
        std::cout << "   ReLU fusion overhead: " << overhead << "%" << std::endl;
        
        if (overhead < 10.0f) {
            std::cout << "   ✅ Excellent fusion efficiency!" << std::endl;
        } else if (overhead < 20.0f) {
            std::cout << "   ✅ Good fusion efficiency" << std::endl;
        } else {
            std::cout << "   ⚠️ High fusion overhead (may indicate bottlenecks)" << std::endl;
        }
    }
    
    std::cout << "\n💾 Memory Bandwidth Savings:" << std::endl;
    size_t output_bytes = size_t(M) * N * sizeof(cutlass::half_t);
    std::cout << "   Output tensor size: " << output_bytes / (1024 * 1024) << " MB" << std::endl;
    std::cout << "   Without fusion: Write D, then read D, apply activation, write D" << std::endl;
    std::cout << "   → Total traffic: " << (3 * output_bytes) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "   With fusion: Write D (with activation applied)" << std::endl;
    std::cout << "   → Total traffic: " << output_bytes / (1024 * 1024) << " MB" << std::endl;
    std::cout << "   → Bandwidth savings: 3x reduction!" << std::endl;
    
    /*
     * PART 6: Activation Function Showcase
     */
    std::cout << "\n🎭 PART 6: Activation Function Gallery\n" << std::endl;
    
    std::cout << "📈 Common Activation Functions in Neural Networks:" << std::endl;
    std::cout << "• ReLU: f(x) = max(0, x)" << std::endl;
    std::cout << "  → Fast, simple, widely used" << std::endl;
    std::cout << "  → Problem: Dead neurons when x < 0" << std::endl;
    
    std::cout << "• GELU: f(x) = x * Φ(x)" << std::endl;
    std::cout << "  → Smooth, differentiable everywhere" << std::endl;
    std::cout << "  → Popular in transformers (BERT, GPT)" << std::endl;
    
    std::cout << "• Sigmoid: f(x) = 1/(1 + e^(-x))" << std::endl;
    std::cout << "  → Output range [0,1]" << std::endl;
    std::cout << "  → Used in binary classification" << std::endl;
    
    std::cout << "• Swish: f(x) = x * sigmoid(x)" << std::endl;
    std::cout << "  → Self-gated, smooth" << std::endl;
    std::cout << "  → Good performance in many tasks" << std::endl;
    
    /*
     * PART 7: Custom Epilogue Creation
     */
    std::cout << "\n🛠️ PART 7: Creating Custom Epilogue Operations\n" << std::endl;
    
    std::cout << "🎨 Custom Epilogue Design Pattern:" << std::endl;
    std::cout << "1. Define template parameters (input/output types)" << std::endl;
    std::cout << "2. Implement Arguments structure for runtime parameters" << std::endl;
    std::cout << "3. Implement Params structure for device-side data" << std::endl;
    std::cout << "4. Implement operator() for element-wise computation" << std::endl;
    std::cout << "5. Handle vectorization and data type conversion" << std::endl;
    
    std::cout << "\n💡 Example Custom Operations:" << std::endl;
    std::cout << "• Leaky ReLU: f(x) = x if x > 0, else α*x" << std::endl;
    std::cout << "• Clipped ReLU: f(x) = min(max(0, x), clip_value)" << std::endl;
    std::cout << "• PReLU: f(x) = x if x > 0, else α_i*x (learnable α per channel)" << std::endl;
    std::cout << "• Element-wise operations: add, multiply, max, min with tensors" << std::endl;
    
    /*
     * PART 8: Advanced Fusion Patterns
     */
    std::cout << "\n🚀 PART 8: Advanced Fusion Patterns\n" << std::endl;
    
    std::cout << "🎪 Multi-Stage Fusion Examples:" << std::endl;
    std::cout << "• Residual Connection: D = activation(A*B + C + residual)" << std::endl;
    std::cout << "• Layer Normalization: D = layernorm(A*B + C)" << std::endl;
    std::cout << "• Dropout: D = activation(A*B + C) * dropout_mask" << std::endl;
    std::cout << "• Attention: D = softmax(A*B / √d) * V" << std::endl;
    
    std::cout << "\n🔗 Benefits of Multi-Operation Fusion:" << std::endl;
    std::cout << "• Dramatically reduces memory traffic" << std::endl;
    std::cout << "• Improves arithmetic intensity" << std::endl;
    std::cout << "• Better utilizes high-bandwidth compute units" << std::endl;
    std::cout << "• Enables more complex operations in single kernel" << std::endl;
    
    /*
     * PART 9: Best Practices and Guidelines
     */
    std::cout << "\n📋 PART 9: Epilogue Best Practices\n" << std::endl;
    
    std::cout << "✅ Design Guidelines:" << std::endl;
    std::cout << "• Keep epilogue operations mathematically simple" << std::endl;
    std::cout << "• Avoid complex control flow in epilogue functions" << std::endl;
    std::cout << "• Use vectorized operations when possible" << std::endl;
    std::cout << "• Consider data type precision requirements" << std::endl;
    std::cout << "• Profile fusion overhead vs separate kernel cost" << std::endl;
    
    std::cout << "\n⚠️ Common Pitfalls:" << std::endl;
    std::cout << "• Complex epilogues can reduce overall performance" << std::endl;
    std::cout << "• Memory-bound epilogues negate GEMM compute advantages" << std::endl;
    std::cout << "• Insufficient vectorization leads to poor memory utilization" << std::endl;
    std::cout << "• Forgetting to handle edge cases in custom epilogues" << std::endl;
    
    /*
     * SUMMARY
     */
    std::cout << "\n=== Chapter 5 Summary: Epilogue Mastery ===\n" << std::endl;
    std::cout << "✅ You learned about:" << std::endl;
    std::cout << "   • What epilogue operations are and why they're crucial" << std::endl;
    std::cout << "   • Built-in CUTLASS epilogue functions for common operations" << std::endl;
    std::cout << "   • Performance benefits of kernel fusion vs separate operations" << std::endl;
    std::cout << "   • Memory bandwidth savings through epilogue fusion" << std::endl;
    std::cout << "   • How to create custom epilogue operations" << std::endl;
    std::cout << "   • Advanced fusion patterns for complex workflows" << std::endl;
    
    std::cout << "\n🧠 Epilogue Optimization Guidelines:" << std::endl;
    std::cout << "   • Always consider fusing simple post-GEMM operations" << std::endl;
    std::cout << "   • Use built-in epilogues when available" << std::endl;
    std::cout << "   • Profile custom epilogues to ensure they don't hurt performance" << std::endl;
    std::cout << "   • Design epilogues for vectorized execution" << std::endl;
    
    std::cout << "\n🎯 Next: Chapter 6 will explore advanced kernel fusion techniques!" << std::endl;
    
    return 0;
}