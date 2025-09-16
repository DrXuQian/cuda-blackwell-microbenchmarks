#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>

// Use the most compatible configuration possible
using ElementA = float;
using ElementB = float; 
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Use SIMT instead of Tensor cores for maximum compatibility
using MMAOp = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm50;  // Very old architecture for compatibility

// Simple tile sizes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using Gemm = cutlass::gemm::device::Gemm<
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

int main() {
    std::cout << "=== Minimal CUTLASS Test ===" << std::endl;
    
    // Small matrices for testing
    int M = 256;
    int N = 256; 
    int K = 256;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    
    // Allocate host tensors
    cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_D({M, N});
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) {
        tensor_A.host_data()[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        tensor_B.host_data()[i] = 1.0f;
    }
    for (int i = 0; i < M * N; i++) {
        tensor_C.host_data()[i] = 0.0f;
    }
    
    // Copy to device
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    
    // Create CUTLASS GEMM arguments
    typename Gemm::Arguments arguments{
        {M, N, K},                    // problem_size
        tensor_A.device_ref(),        // ref_A
        tensor_B.device_ref(),        // ref_B
        tensor_C.device_ref(),        // ref_C
        tensor_D.device_ref(),        // ref_D
        {1.0f, 0.0f}                  // epilogue
    };
    
    // Initialize CUTLASS GEMM
    Gemm gemm_operator;
    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ CUTLASS cannot implement this GEMM: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    status = gemm_operator.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ CUTLASS initialization failed: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    std::cout << "✅ CUTLASS initialized successfully" << std::endl;
    
    // Run the GEMM
    status = gemm_operator();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "❌ CUTLASS GEMM execution failed: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    // Copy result back to host
    tensor_D.sync_host();
    
    // Verify result (should be K since A[i] = B[i] = 1.0)
    bool correct = true;
    for (int i = 0; i < std::min(100, M * N); i++) {
        if (abs(tensor_D.host_data()[i] - K) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at " << i << ": got " << tensor_D.host_data()[i] << ", expected " << K << std::endl;
            break;
        }
    }
    
    std::cout << "✅ CUTLASS GEMM executed successfully" << std::endl;
    std::cout << "Result verification: " << (correct ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "Sample result: " << tensor_D.host_data()[0] << " (expected: " << K << ")" << std::endl;
    
    return 0;
}