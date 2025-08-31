#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 1: Basic GEMM
 * 
 * LEARNING OBJECTIVES:
 * 1. Understand the basic CUTLASS GEMM API
 * 2. Learn about CUTLASS data types and layouts
 * 3. See how template parameters control kernel generation
 * 4. Compare with cuBLAS reference
 * 
 * KEY CONCEPTS:
 * - cutlass::gemm::device::Gemm: The main GEMM template class
 * - Template parameters: ElementA, ElementB, ElementC, LayoutA, LayoutB, LayoutC
 * - Operator class: Controls the computational pattern
 * - Architecture tag: Specifies target GPU architecture
 */

int main() {
    
    std::cout << "=== CUTLASS Tutorial Chapter 1: Basic GEMM ===" << std::endl;
    print_device_info();
    
    // Matrix dimensions for C = A * B
    int M = 1024;  // Rows of A and C
    int N = 1024;  // Columns of B and C  
    int K = 1024;  // Columns of A, rows of B
    
    std::cout << "\nMatrix dimensions: C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    
    /*
     * STEP 1: Define the CUTLASS GEMM type
     * 
     * This is where CUTLASS's template magic happens. Each template parameter
     * controls how the kernel will be generated:
     */
    using CutlassGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,        // ElementA: Data type of matrix A (FP16)
        cutlass::layout::RowMajor,  // LayoutA: Memory layout of matrix A
        cutlass::half_t,        // ElementB: Data type of matrix B (FP16)  
        cutlass::layout::RowMajor,  // LayoutB: Memory layout of matrix B
        cutlass::half_t,        // ElementC: Data type of matrix C (FP16)
        cutlass::layout::RowMajor,  // LayoutC: Memory layout of matrix C
        float,                  // ElementAccumulator: Data type for intermediate accumulation (FP32)
        cutlass::arch::OpClassTensorOp,  // Operator class: Use Tensor Cores
        cutlass::arch::Sm80     // Architecture: Target Ampere (RTX 30xx, A100)
    >;
    
    /*
     * UNDERSTANDING THE TEMPLATE PARAMETERS:
     * 
     * - ElementA/B/C: Controls data types. half_t = FP16, float = FP32
     * - LayoutA/B/C: Controls memory layout:
     *   * RowMajor: Elements in same row are adjacent in memory (C-style)
     *   * ColumnMajor: Elements in same column are adjacent (Fortran-style)
     * - ElementAccumulator: Internal accumulation precision (usually higher than inputs)
     * - OpClassTensorOp: Use Tensor Cores for maximum performance
     * - Architecture: Must match your GPU (Sm75=Turing, Sm80=Ampere, Sm90=Hopper)
     */
    
    std::cout << "\n=== CUTLASS Configuration ===" << std::endl;
    std::cout << "Data Type: FP16 inputs, FP32 accumulation" << std::endl;
    std::cout << "Layout: Row-major for all matrices" << std::endl;
    std::cout << "Operator: Tensor Core operations" << std::endl;
    std::cout << "Architecture: SM80 (Ampere)" << std::endl;
    
    /*
     * STEP 2: Allocate and initialize host memory
     * 
     * We'll use CUTLASS's HostTensor for convenient memory management:
     */
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_C({M, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_D({M, N}); // Output
    
    // Initialize matrices with random values
    initialize_matrix_random(tensor_A.host_data(), M, K);
    initialize_matrix_random(tensor_B.host_data(), K, N);
    initialize_matrix_random(tensor_C.host_data(), M, N);  // C is used as bias in D = A*B + C
    
    // Display small portion of input matrices
    print_matrix(tensor_A.host_data(), M, K, "Matrix A");
    print_matrix(tensor_B.host_data(), K, N, "Matrix B");
    print_matrix(tensor_C.host_data(), M, N, "Matrix C (bias)");
    
    /*
     * STEP 3: Copy data to GPU
     */
    tensor_A.sync_device();
    tensor_B.sync_device();  
    tensor_C.sync_device();
    tensor_D.sync_device();
    
    /*
     * STEP 4: Set up CUTLASS GEMM arguments
     * 
     * CUTLASS uses a structure to pass all GEMM parameters:
     */
    typename CutlassGemm::Arguments arguments{
        {M, N, K},                    // Problem size
        tensor_A.device_ref(),        // Reference to matrix A on device
        tensor_B.device_ref(),        // Reference to matrix B on device  
        tensor_C.device_ref(),        // Reference to matrix C on device (bias)
        tensor_D.device_ref(),        // Reference to matrix D on device (output)
        {1.0f, 1.0f}                 // Scaling factors: D = alpha * A * B + beta * C
    };
    
    /*
     * UNDERSTANDING THE ARGUMENTS:
     * 
     * - Problem size: {M, N, K} dimensions
     * - Device references: Pointers + stride information
     * - Scaling factors: {alpha, beta} for D = alpha * A*B + beta * C
     *   * alpha=1.0, beta=1.0 means D = A*B + C
     *   * alpha=1.0, beta=0.0 means D = A*B (ignore C)
     */
    
    /*
     * STEP 5: Create and run the CUTLASS GEMM operation
     */
    CutlassGemm gemm_operator;
    
    // Check if the operation is supported on this device
    cutlass::Status status = gemm_operator.can_implement(arguments);
    CUTLASS_CHECK(status);
    std::cout << "\n✅ CUTLASS GEMM can be implemented on this device" << std::endl;
    
    // Initialize the operator
    status = gemm_operator.initialize(arguments);
    CUTLASS_CHECK(status);
    std::cout << "✅ CUTLASS GEMM initialized successfully" << std::endl;
    
    // Run the operation
    std::cout << "\n🚀 Running CUTLASS GEMM..." << std::endl;
    
    CutlassTimer timer;
    const int warmup_iterations = 5;
    const int benchmark_iterations = 10;
    
    // Warmup runs
    for (int i = 0; i < warmup_iterations; i++) {
        status = gemm_operator();
        CUTLASS_CHECK(status);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark runs
    timer.clear();
    for (int i = 0; i < benchmark_iterations; i++) {
        timer.start();
        status = gemm_operator();
        timer.stop();
        CUTLASS_CHECK(status);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /*
     * STEP 6: Analyze results
     */
    tensor_D.sync_host();  // Copy results back to host
    
    float avg_time = timer.get_average_time();
    double gflops = calculate_gflops(M, N, K, avg_time);
    
    std::cout << "\n=== CUTLASS Performance Results ===" << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Display small portion of output
    print_matrix(tensor_D.host_data(), M, N, "Output Matrix D = A*B + C");
    
    /*
     * STEP 7: Verify correctness with cuBLAS reference
     */
    std::cout << "\n🔍 Verifying results against cuBLAS..." << std::endl;
    
    // Allocate reference output
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> reference({M, N});
    reference.sync_device();
    
    // Copy C to reference for cuBLAS computation
    CUDA_CHECK(cudaMemcpy(reference.device_data(), tensor_C.device_data(), 
                         M * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToDevice));
    
    // Run cuBLAS reference
    CutlassTimer cublas_timer;
    cublas_timer.start();
    reference_gemm_cublas(tensor_A.device_data(), tensor_B.device_data(), 
                         reference.device_data(), M, N, K,
                         cutlass::half_t(1.0f), cutlass::half_t(1.0f));
    cublas_timer.stop();
    
    reference.sync_host();
    
    float cublas_time = cublas_timer.get_last_time();
    double cublas_gflops = calculate_gflops(M, N, K, cublas_time);
    
    std::cout << "\n=== cuBLAS Reference Results ===" << std::endl;
    std::cout << "cuBLAS time: " << cublas_time << " ms" << std::endl;
    std::cout << "cuBLAS performance: " << cublas_gflops << " GFLOPS" << std::endl;
    
    // Verify numerical correctness
    bool is_correct = verify_matrix(tensor_D.host_data(), reference.host_data(), M, N, 1e-2f);
    
    std::cout << "\n=== Verification Results ===" << std::endl;
    std::cout << "Correctness: " << (is_correct ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "CUTLASS vs cuBLAS speedup: " << (cublas_time / avg_time) << "x" << std::endl;
    
    /*
     * STEP 8: Key takeaways and next steps
     */
    std::cout << "\n=== Chapter 1 Summary ===" << std::endl;
    std::cout << "✅ You learned how to:" << std::endl;
    std::cout << "   • Use CUTLASS basic GEMM template" << std::endl;
    std::cout << "   • Configure data types and layouts" << std::endl;
    std::cout << "   • Set up problem dimensions and arguments" << std::endl;
    std::cout << "   • Run and benchmark CUTLASS operations" << std::endl;
    std::cout << "   • Verify results against cuBLAS" << std::endl;
    std::cout << "\n🎯 Next: Chapter 2 will explore template concepts in depth!" << std::endl;
    
    return 0;
}