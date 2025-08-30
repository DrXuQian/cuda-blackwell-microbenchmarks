# 🚀 High-Performance CUDA Kernel Microbenchmarks

A research-oriented project for developing, benchmarking, and analyzing high-performance CUDA kernels with focus on:
- **GEMV/GEMM optimizations** for transformer inference
- **Quantized (w4a16f) kernel implementations** inspired by Marlin
- **Warp specialization techniques** for memory-bound operations
- **Tensor Core utilization** for modern GPU architectures

## 📁 Project Structure

```
microbenchmark/
├── src/                    # Source code organized by purpose
│   ├── kernels/           # CUDA kernel implementations
│   │   ├── ping_pong_kernel.cu           # Ping-pong double buffering
│   │   ├── warp_specialized_*.cu         # Warp specialization variants
│   │   ├── wgmma_kernel.cu              # WGMMA/Tensor Core implementation
│   │   └── marlin_gemv_optimized.cu     # Marlin-inspired GEMV kernels
│   ├── benchmarks/        # Benchmark harnesses and test suites
│   │   ├── simple_gemv_benchmark.cu     # GEMV optimization comparison
│   │   ├── marlin_gemv_benchmark.cu     # Advanced w4a16f benchmarks
│   │   └── mma_*.cu                     # Matrix multiplication tests
│   └── utils/             # Common utilities and headers
│       └── common.h                     # Shared accuracy validation & benchmarking
├── external/              # Third-party dependencies
│   └── marlin/           # Marlin quantization framework
├── docs/                  # Comprehensive documentation
│   ├── kernels/          # Kernel design and optimization guides
│   ├── benchmarks/       # Benchmark analysis and results
│   ├── performance/      # Performance analysis and profiling
│   └── setup/            # Build and environment setup
├── build/                 # Build artifacts and executables
│   └── bin/              # Compiled binaries
├── scripts/              # Build, profiling, and automation scripts
└── tests/                # Validation and correctness tests
```

## 🎯 Key Features

### Kernel Implementations
- **Ping-Pong Memory Patterns** - Cache-friendly data movement (5,137 GFLOPS)
- **Warp-Specialized GEMV** - Async memory loading with compute overlap (4,289 GFLOPS)
- **Marlin-Inspired w4a16f** - 4-bit quantized inference kernels (409 GFLOPS target shape)
- **WGMMA Integration** - Modern tensor core utilization (6,688 GFLOPS)

### Benchmarking Framework
- **Performance Comparison** vs cuBLAS and vendor libraries
- **NCU Profiling Integration** with automated metrics collection
- **Memory Bandwidth Analysis** and optimization guidance
- **Accuracy Verification** using cosine similarity metrics

### Target Workloads
- **Transformer Inference** - Focus on 1×N @ N×M GEMV patterns
- **LLM Serving** - Optimized for batch-1 inference scenarios
- **Quantized Models** - 4-bit weight, 16-bit activation kernels

## 🚀 Quick Start

### Prerequisites
- CUDA 12.0+ with compute capability 8.0+
- cuBLAS and NCU profiler
- Modern C++17 compiler

### Build and Run
```bash
# Build all kernels
make all

# Run GEMV optimization benchmarks
make run-simple-gemv          # GEMV optimization comparison
make run-marlin-gemv          # Marlin-inspired w4a16f benchmarks

# Run original GEMM benchmarks
make run-warp-specialized     # Warp specialization analysis
make run-ping-pong           # Double buffering techniques
make run-wgmma               # Tensor core utilization
make run-all                 # Complete benchmark suite

# Profile with NCU
make profile-simple-gemv      # GEMV kernel profiling
make profile-marlin-gemv      # Advanced w4a16f profiling
ncu --set full ./build/bin/simple_gemv_benchmark
```

## 📊 Performance Highlights

### GEMV Optimization Results
**Target Shape: 1×3584 @ 3584×18944 (67.9M parameters)**
- **409 GFLOPS** achieved across kernel variants
- **61% memory bandwidth utilization** (409/672 GB/s theoretical)
- **Efficient vectorization** with half2 operations
- **NCU-verified metrics** for optimization validation

### GEMM Optimization Results  
**Target Shape: 1024×1024×1024**
| Kernel | Performance | vs cuBLAS | Key Strength |
|--------|-------------|-----------|---------------|
| **cuBLAS Reference** | ~67,000 GFLOPS | 100% | Highly optimized library |
| **WGMMA Fallback** | 6,688 GFLOPS | 10.0% | Simple, effective |
| **Ping-Pong** | 5,137 GFLOPS | 7.7% | Best custom implementation |
| **Optimized Warp Specialized** | 4,289 GFLOPS | 6.4% | Memory-optimized |

## 🧮 Kernel Architecture Deep Dive

### 1. Simple GEMV Kernels (`src/benchmarks/simple_gemv_benchmark.cu`)
Three progressive optimization levels:

```cuda
// 1. Baseline: Simple thread-per-output-element
__global__ void simple_gemv_kernel(const half* A, const half* B, half* C, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < M) {
        float accum = 0.0f;
        for (int row = 0; row < N; row++) {
            accum += __half2float(A[row]) * __half2float(B[row * M + col]);
        }
        C[col] = __float2half(accum);
    }
}

// 2. Optimized: Shared memory + vectorization
__global__ void optimized_gemv_kernel(/* uses shared memory for A, half2 vectorization */) {
    // Load A cooperatively into shared memory
    // Process 4 elements at a time with half2 pairs
}

// 3. Warp Specialized: Warp-level reduction
__global__ void warp_specialized_gemv_kernel(/* warp shuffle reductions */) {
    // Warp 0: Load A into shared memory
    // All warps: Compute with warp-level partial reductions
}
```

### 2. Marlin-Inspired w4a16f (`src/kernels/marlin_gemv_optimized.cu`)
Advanced quantization techniques:

```cuda
// Efficient 4-bit dequantization using LOP3 bit manipulation
__device__ inline GemvFragB gemv_dequant(int q) {
    const int LO = 0x000f000f, HI = 0x00f000f0, EX = 0x64006400;
    int lo = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = gemv_lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // ... symmetric zero-point and scaling
}

// Async pipeline with warp specialization
template<int BLOCK_SIZE, int STAGES>
__global__ void marlin_gemv_w4a16f(
    const half* A, const int4* B_quantized, const half* scales, half* C
) {
    // Warp 0: Load A chunks with cp.async
    // Warp 1: Load quantized B chunks  
    // Warps 2+: Compute with dequantization
    // Multi-stage pipeline for memory/compute overlap
}
```

### 3. WGMMA Tensor Core Integration (`src/kernels/wgmma_kernel.cu`)
Architecture-aware compilation:

```cuda
#if __CUDA_ARCH__ >= 890
    // True WGMMA for sm_90+ (H100, etc.)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 ...");
#else
    // Fallback to WMMA for sm_89 (RTX 4070 Ti Super)
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
#endif
```

## 🎯 Accuracy Validation

### Cosine Distance Methodology
All kernels use **cosine distance** for robust accuracy validation:

```cuda
struct AccuracyResult {
    double cosine_similarity;  // Dot product normalized by magnitudes
    double cosine_distance;    // 1.0 - cosine_similarity  
    float max_abs_error;       // Traditional max error
    float mean_abs_error;      // Traditional mean error
    bool passed;               // cosine_similarity >= 0.9999
};
```

**Why Cosine Distance?**
- ✅ **Scale-invariant**: Robust to magnitude differences
- ✅ **Numerically stable**: Better than relative error for small values
- ✅ **Geometric meaning**: Measures angle between result vectors
- ✅ **Threshold-based**: Clear pass/fail criteria (similarity ≥ 0.9999)

## 🔬 Research Focus

This project explores:
1. **Memory-bound optimization** for GEMV operations
2. **Quantization techniques** for efficient inference (w4a16f)
3. **Warp specialization** patterns for async compute/memory overlap
4. **Tensor core integration** for mixed-precision workloads
5. **Pipeline optimization** with multi-stage async operations

## 📚 Documentation

| Component | Description | Link |
|-----------|-------------|------|
| **Kernel Design** | Architecture and optimization techniques | [docs/kernels/](docs/kernels/) |
| **Benchmark Analysis** | Performance results and comparisons | [docs/benchmarks/](docs/benchmarks/) |
| **Profiling Guide** | NCU setup and analysis workflows | [docs/performance/](docs/performance/) |
| **Build Setup** | Environment and dependency configuration | [docs/setup/](docs/setup/) |

## 🛠️ Build System

### Makefile Targets

```bash
# GEMV-focused targets
make run-simple-gemv           # Simple GEMV optimization comparison
make run-marlin-gemv           # Advanced w4a16f benchmarks  
make profile-simple-gemv       # NCU profiling for GEMV kernels
make profile-marlin-gemv       # NCU profiling for w4a16f kernels

# Original GEMM targets  
make run-ping-pong            # Ping-pong double buffering
make run-warp-specialized     # Warp specialization
make run-wgmma               # Tensor core utilization
make run-all                 # All benchmarks

# Build targets
make all                     # Build primary kernels
make all-experimental        # Build experimental variants
make clean                   # Remove binaries
```

### Compilation Settings

```makefile
NVCC = nvcc
CFLAGS = -std=c++17 -O3 -arch=sm_89 -lcublas -I.

# Optimizations enabled:
# -O3: Aggressive compiler optimizations
# -arch=sm_89: Target Ada Lovelace compute capability  
# -lcublas: Link cuBLAS for reference comparison
# -I.: Include src/utils/ for common.h
```

## 💻 Hardware Requirements

### Tested Platform
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **Compute Capability**: 8.9 (Ada Lovelace)  
- **CUDA Version**: 12.8
- **Memory**: 16GB GDDR6X

### Compatibility
- **Minimum**: sm_75+ (Turing) for tensor core operations
- **Recommended**: sm_89+ (Ada Lovelace) for full feature support  
- **Future**: sm_90+ (Hopper) for true WGMMA operations

## 🏗️ Development Workflow

1. **Implement** new kernels in `src/kernels/`
2. **Create benchmarks** in `src/benchmarks/`
3. **Profile** with NCU using `make profile-*` targets
4. **Document** results in `docs/performance/`
5. **Validate** correctness with cosine distance metrics

## 📈 Performance Tracking

### Current Best Results
- **GEMV (1×3584 @ 3584×18944)**: 409 GFLOPS, 61% bandwidth utilization
- **GEMM (1024³)**: 6,688 GFLOPS (tensor core), 5,137 GFLOPS (ping-pong)
- **Warp Specialization**: +44% improvement with memory coalescing
- **Quantization Ready**: Framework for w4a16f implementations

### NCU Profiling Integration
```bash
# Comprehensive kernel analysis
ncu --set full ./build/bin/simple_gemv_benchmark

# Memory-focused profiling
ncu --section MemoryWorkloadAnalysis ./build/bin/warp_specialized_test

# Custom metric collection
ncu --metrics smsp__cycles_elapsed.avg,dram__bytes.sum.per_second ./build/bin/kernel_test
```

## 🤝 Contributing

Key areas for contribution:
- Novel warp specialization patterns
- Advanced quantization schemes (beyond w4a16f)
- Tensor core utilization strategies
- Memory hierarchy optimizations
- Multi-GPU scaling techniques

### Adding New Kernels

1. Place implementation in `src/kernels/`
2. Create benchmark in `src/benchmarks/`
3. Add Makefile targets
4. Include accuracy validation
5. Document performance characteristics

## 📚 References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Tensor Core Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)  
- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [Marlin: Mixed-Precision Linear Algebra](https://github.com/IST-DASLab/marlin)

---

*Built for CUDA research and high-performance computing optimization* 🚀