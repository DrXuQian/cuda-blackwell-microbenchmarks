#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

// CUDA error checking macro
#define CUTLASS_CHECK(status)                                                    \
  {                                                                             \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)         \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(1);                                                                  \
    }                                                                           \
  }

#define CUDA_CHECK(status)                                                      \
  {                                                                             \
    cudaError_t error = status;                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(1);                                                                  \
    }                                                                           \
  }

// Performance timer class
class CutlassTimer {
private:
    cudaEvent_t start_event, stop_event;
    std::vector<float> times;
    
public:
    CutlassTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CutlassTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        times.push_back(elapsed_time);
    }
    
    float get_last_time() const {
        return times.empty() ? 0.0f : times.back();
    }
    
    float get_average_time() const {
        if (times.empty()) return 0.0f;
        
        float sum = 0.0f;
        for (float time : times) {
            sum += time;
        }
        return sum / times.size();
    }
    
    void clear() {
        times.clear();
    }
    
    size_t get_measurement_count() const {
        return times.size();
    }
};

// Matrix initialization helpers
template<typename T>
void initialize_matrix_random(T* matrix, int rows, int cols, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        if constexpr (std::is_same_v<T, cutlass::half_t>) {
            matrix[i] = cutlass::half_t(dis(gen));
        } else if constexpr (std::is_same_v<T, float>) {
            matrix[i] = dis(gen);
        } else {
            matrix[i] = T(dis(gen));
        }
    }
}

template<typename T>
void initialize_matrix_identity(T* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if constexpr (std::is_same_v<T, cutlass::half_t>) {
                matrix[i * size + j] = cutlass::half_t(i == j ? 1.0f : 0.0f);
            } else {
                matrix[i * size + j] = T(i == j ? 1.0f : 0.0f);
            }
        }
    }
}

template<typename T>
void print_matrix(const T* matrix, int rows, int cols, const std::string& name, int max_display = 8) {
    std::cout << "\n" << name << " (showing " << std::min(max_display, rows) 
              << "x" << std::min(max_display, cols) << "):\n";
    
    for (int i = 0; i < std::min(max_display, rows); i++) {
        for (int j = 0; j < std::min(max_display, cols); j++) {
            if constexpr (std::is_same_v<T, cutlass::half_t>) {
                std::cout << std::fixed << std::setprecision(3) << float(matrix[i * cols + j]) << "\t";
            } else {
                std::cout << std::fixed << std::setprecision(3) << matrix[i * cols + j] << "\t";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Performance calculation helpers
double calculate_gflops(int M, int N, int K, float time_ms) {
    double flops = 2.0 * double(M) * double(N) * double(K);
    return flops / (double(time_ms) / 1000.0) / 1e9;
}

double calculate_memory_bandwidth_gb_s(size_t bytes_transferred, float time_ms) {
    double bytes_per_second = double(bytes_transferred) / (double(time_ms) / 1000.0);
    return bytes_per_second / 1e9;
}

// Matrix verification helpers
template<typename T>
bool verify_matrix(const T* matrix_a, const T* matrix_b, int rows, int cols, 
                   float tolerance = 1e-3f) {
    for (int i = 0; i < rows * cols; i++) {
        float a_val, b_val;
        
        if constexpr (std::is_same_v<T, cutlass::half_t>) {
            a_val = float(matrix_a[i]);
            b_val = float(matrix_b[i]);
        } else {
            a_val = float(matrix_a[i]);
            b_val = float(matrix_b[i]);
        }
        
        if (std::abs(a_val - b_val) > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << b_val 
                     << ", got " << a_val << " (diff: " << std::abs(a_val - b_val) << ")\n";
            return false;
        }
    }
    return true;
}

// Device info helpers
void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "=== CUDA Device Info ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "=========================" << std::endl;
}

// cuBLAS reference implementation for comparison
template<typename T>
void reference_gemm_cublas(const T* A, const T* B, T* C, int M, int N, int K,
                          T alpha = T(1.0f), T beta = T(0.0f)) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha,
                   B, N, A, K, &beta, C, N);
    } else if constexpr (std::is_same_v<T, cutlass::half_t>) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, 
                   reinterpret_cast<const __half*>(&alpha),
                   reinterpret_cast<const __half*>(B), N,
                   reinterpret_cast<const __half*>(A), K,
                   reinterpret_cast<const __half*>(&beta),
                   reinterpret_cast<__half*>(C), N);
    }
    
    cublasDestroy(handle);
}

#include <iomanip>