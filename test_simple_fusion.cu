#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    std::cout << "🧪 Testing basic CUDA and cuBLAS functionality..." << std::endl;
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices available: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    
    // Test basic memory allocation
    const int size = 1024;
    half* d_test;
    cudaError_t err = cudaMalloc(&d_test, size * sizeof(half));
    if (err != cudaSuccess) {
        std::cerr << "❌ CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Test cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "❌ cuBLAS create failed with status: " << status << std::endl;
        return 1;
    }
    
    std::cout << "✅ Basic CUDA and cuBLAS functionality working!" << std::endl;
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_test);
    
    return 0;
}