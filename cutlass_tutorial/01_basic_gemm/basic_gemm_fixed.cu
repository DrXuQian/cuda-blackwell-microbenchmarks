#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "../common/utils.h"

/*
 * CUTLASS Tutorial Chapter 1: Basic GEMM (FIXED VERSION)
 * 
 * This version uses the exact configuration that works on RTX 5070
 */

int main() {
    std::cout << "=== CUTLASS Tutorial Chapter 1: Basic GEMM (Fixed) ===" << std::endl;
    print_device_info();
    
    // Matrix dimensions - using smaller size for compatibility
    int M = 256;
    int N = 256; 
    int K = 256;
    
    std::cout << "\nMatrix dimensions: C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    
    /*
     * STEP 1: Define the CUTLASS GEMM type using proven working configuration
     */
    using ElementA = float;
    using ElementB = float; 
    using ElementC = float;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    using MMAOp = cutlass::arch::OpClassSimt;
    using SmArch = cutlass::arch::Sm50;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape
    >;
    
    std::cout << "\n=== CUTLASS Configuration ===" << std::endl;
    std::cout << "Data Type: FP32 inputs and accumulation" << std::endl;
    std::cout << "Layout: Row-major for all matrices" << std::endl;
    std::cout << "Operator: CUDA Core operations" << std::endl;
    std::cout << "Architecture: SM50 (Maxwell) - Maximum Compatibility" << std::endl;
    
    /*
     * STEP 2: Allocate and initialize host memory
     */
    cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_D({M, N});
    
    // Initialize matrices with random values
    initialize_matrix_random(tensor_A.host_data(), M, K);
    initialize_matrix_random(tensor_B.host_data(), K, N);
    initialize_matrix_random(tensor_C.host_data(), M, N);  // C is used as bias
    
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
     */
    typename CutlassGemm::Arguments arguments{
        {M, N, K},                    // problem_size
        tensor_A.device_ref(),        // ref_A
        tensor_B.device_ref(),        // ref_B
        tensor_C.device_ref(),        // ref_C
        tensor_D.device_ref(),        // ref_D
        {1.0f, 1.0f}                  // epilogue: alpha=1.0, beta=1.0 for D = A*B + C
    };
    
    /*
     * STEP 5: Initialize CUTLASS GEMM
     */
    CutlassGemm gemm_operator;
    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ CUTLASS cannot implement this GEMM: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    std::cout << "\n✅ CUTLASS GEMM can be implemented on this device" << std::endl;
    
    status = gemm_operator.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ CUTLASS initialization failed: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    std::cout << "✅ CUTLASS GEMM initialized successfully" << std::endl;
    
    /*
     * STEP 6: Run CUTLASS GEMM with timing
     */
    std::cout << "\n🚀 Running CUTLASS GEMM..." << std::endl;
    
    CutlassTimer timer;
    const int warmup_iterations = 3;
    const int benchmark_iterations = 10;
    
    // Warmup runs
    for (int i = 0; i < warmup_iterations; i++) {
        status = gemm_operator();
        if (status != cutlass::Status::kSuccess) {
            std::cout << "❌ CUTLASS GEMM execution failed in warmup: " << cutlassGetStatusString(status) << std::endl;
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark runs
    timer.clear();
    for (int i = 0; i < benchmark_iterations; i++) {
        timer.start();
        status = gemm_operator();
        timer.stop();
        if (status != cutlass::Status::kSuccess) {
            std::cout << "❌ CUTLASS GEMM execution failed in benchmark: " << cutlassGetStatusString(status) << std::endl;
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    tensor_D.sync_host();
    
    float cutlass_time = timer.get_average_time();
    double cutlass_gflops = calculate_gflops(M, N, K, cutlass_time);
    
    std::cout << "\n=== CUTLASS Results ===" << std::endl;
    std::cout << "CUTLASS time: " << cutlass_time << " ms" << std::endl;
    std::cout << "CUTLASS performance: " << cutlass_gflops << " GFLOPS" << std::endl;
    
    /*
     * STEP 7: Run cuBLAS reference for comparison
     */
    std::cout << "\n🔍 Running cuBLAS reference..." << std::endl;
    
    cutlass::HostTensor<ElementC, LayoutC> reference({M, N});
    
    // Copy bias matrix to reference
    CUDA_CHECK(cudaMemcpy(reference.device_data(), tensor_C.device_data(), 
                         M * N * sizeof(ElementC), cudaMemcpyDeviceToDevice));
    
    // Run cuBLAS reference
    CutlassTimer cublas_timer;
    cublas_timer.start();
    reference_gemm_cublas(tensor_A.device_data(), tensor_B.device_data(), 
                         reference.device_data(), M, N, K, 1.0f, 1.0f);
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
    std::cout << "Speedup vs cuBLAS: " << cublas_time / cutlass_time << "x" << std::endl;
    
    // Show a sample of the output
    print_matrix(tensor_D.host_data(), M, N, "Result Matrix D = A*B + C", 4);
    
    std::cout << "\n🎉 Chapter 1 completed successfully!" << std::endl;
    std::cout << "You've learned how to:" << std::endl;
    std::cout << "  ✅ Configure CUTLASS GEMM templates" << std::endl;
    std::cout << "  ✅ Initialize and run CUTLASS kernels" << std::endl;
    std::cout << "  ✅ Compare with cuBLAS reference" << std::endl;
    std::cout << "  ✅ Verify numerical correctness" << std::endl;
    
    return 0;
}