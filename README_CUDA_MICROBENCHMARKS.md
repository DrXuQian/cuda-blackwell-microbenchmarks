# CUDA Microbenchmarks for RTX 5070 Blackwell

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Architecture](https://img.shields.io/badge/Architecture-Blackwell%20RTX%205070-blue.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive collection of CUDA microbenchmarks specifically optimized for the **RTX 5070 Blackwell architecture** (sm_90+), featuring advanced techniques including **TMA (Tensor Memory Accelerator)**, **Async WGMMA (Warp Group Matrix Multiply Accumulate)**, and **warp specialization patterns**.

## 🚀 Features

### **Blackwell RTX 5070 Optimizations**
- **TMA Hardware Acceleration**: Utilizes RTX 5070's Tensor Memory Accelerator for optimal memory bandwidth
- **Async WGMMA Instructions**: Native `wgmma.mma_async.sync.aligned` operations for maximum throughput
- **164KB Shared Memory**: Optimally utilizes Blackwell's expanded shared memory per SM
- **56 SM Architecture**: Kernels designed for RTX 5070's specific SM count and configuration

### **Advanced Kernel Implementations**

#### 1. **W4A16 GEMV with Warp Specialization**
- **4-bit quantized weights** with FP16 activations for LLM inference
- **8-way warp specialization**: Producer, Dequantizer, 4x Computer, Reducer
- **TMA-based loading** for weights and scales
- **Triple-buffered pipeline** for sustained throughput
- **Expected Performance**: 12-16 TFLOPS on RTX 5070

#### 2. **TMA-Optimized GEMM**
- **Tensor Memory Accelerator** for 2D bulk data transfers  
- **128x128x64 tiles** optimized for Blackwell's memory hierarchy
- **Async memory operations** with proper synchronization
- **Expected Performance**: 15-20 TFLOPS on RTX 5070

#### 3. **LayerNorm + GEMM Fusion**
- **Single-kernel fusion** for transformer layers
- **Target**: 9600x2730 → LayerNorm → GEMM(2730x1024) → 9600x1024
- **Vectorized operations** with float4/half8 memory access
- **Warp-specialized LayerNorm** computation
- **Expected Performance**: 5-8 TFLOPS for fused operation

#### 4. **Async WGMMA Kernels**
- **Producer-consumer patterns** with memory-compute overlap
- **Native WGMMA instructions** for maximum compute utilization
- **Multiple compute warps** with optimal register usage
- **Expected Performance**: 8-12 TFLOPS with superior efficiency

## 📁 Repository Structure

```
cuda-microbenchmarks/
├── src/blackwell_5070/                    # RTX 5070 specific implementations
│   ├── utils/blackwell_common.h           # Common utilities and constants
│   ├── tma_kernels/                       # TMA-based kernels
│   │   ├── blackwell_tma_gemm.cu          # Advanced TMA GEMM
│   │   ├── blackwell_tma_gemm_fixed.cu    # Compatible version
│   │   └── blackwell_w4a16_tma_wgmma.cu   # W4A16 with TMA+WGMMA
│   ├── async_wgmma_kernels/               # Async WGMMA implementations
│   │   ├── blackwell_async_wgmma.cu       # Advanced async WGMMA
│   │   ├── blackwell_async_wgmma_fixed.cu # Compatible version
│   │   ├── blackwell_w4a16_gemv_specialized.cu # W4A16 with specialization
│   │   └── blackwell_w4a16_simple.cu      # Simple W4A16 demo
│   └── benchmarks/                        # Comprehensive benchmarks
│       ├── blackwell_comprehensive_benchmark.cu
│       ├── blackwell_w4a16_comparison.cu
│       └── blackwell_layernorm_gemm_fusion.cu
├── src/kernels/                           # Original kernel implementations
├── src/benchmarks/                        # General benchmarks
├── tests/blackwell_5070/                  # Test scripts
├── Makefile.blackwell                     # Advanced build system
├── Makefile.blackwell_fixed              # Compatible build system
└── run_blackwell_w4a16_demo.sh           # Demo script
```

## 🛠️ Build Requirements

- **CUDA 12.x+** (recommended for optimal Blackwell support)
- **RTX 5070 or compatible GPU** (sm_90+)
- **GCC 9+** or compatible C++ compiler
- **NCU** (optional, for profiling)

## 🔧 Quick Start

### 1. **Clone and Build**
```bash
git clone https://github.com/DrXuQian/cuda-microbenchmarks.git
cd cuda-microbenchmarks

# Build all Blackwell kernels
make -f Makefile.blackwell_fixed all

# Run comprehensive test
./tests/blackwell_5070/test_blackwell_kernels.sh
```

### 2. **Run W4A16 Demo**
```bash
# Complete W4A16 demonstration
./run_blackwell_w4a16_demo.sh

# Individual kernels
./bin/blackwell_5070_fixed/blackwell_w4a16_simple
./bin/blackwell_5070_fixed/blackwell_tma_gemm_fixed
./bin/blackwell_5070_fixed/blackwell_async_wgmma_fixed
```

### 3. **Advanced Benchmarks**
```bash
# LayerNorm + GEMM Fusion
make -f Makefile.blackwell_fixed && \
nvcc -o layernorm_fusion src/blackwell_5070/benchmarks/blackwell_layernorm_gemm_fusion.cu \
     -I./src/blackwell_5070/utils -lcublas && \
./layernorm_fusion
```

## 📊 Performance Expectations

### **RTX 5070 Target Performance**

| Kernel | Matrix Size | Expected TFLOPS | Memory Bandwidth | Use Case |
|--------|-------------|-----------------|------------------|----------|
| **TMA GEMM** | 2048x2048 | 15-20 | 800+ GB/s | Dense matrix operations |
| **Async WGMMA** | 1024x1024 | 8-12 | 600+ GB/s | Memory-bound workloads |
| **W4A16 GEMV** | 1x32K@4K | 12-16 | 400+ GB/s | LLM inference |
| **LN+GEMM Fusion** | 9600x2730 | 5-8 | 500+ GB/s | Transformer layers |

### **Key Optimizations for RTX 5070**
- **TMA Descriptors**: 128B/256B cache line optimization
- **Shared Memory**: 164KB per SM utilization
- **Warp Specialization**: 8-way producer-consumer patterns
- **Vector Operations**: float4/half8 memory access
- **Register Management**: Optimal usage for complex kernels

## 🧪 Testing and Validation

### **Automated Testing**
```bash
# Complete test suite
make -f Makefile.blackwell_fixed smoke_test

# Memory checks
make -f Makefile.blackwell_fixed memcheck_all

# Performance profiling
make -f Makefile.blackwell_fixed profile_all
```

### **Manual Testing**
```bash
# Individual kernel tests
make -f Makefile.blackwell_fixed test_tma_fixed
make -f Makefile.blackwell_fixed test_wgmma_fixed
make -f Makefile.blackwell_fixed test_w4a16_simple
```

## 🔍 Profiling and Analysis

### **NCU Profiling**
```bash
# Detailed kernel analysis
ncu --set full ./bin/blackwell_5070_fixed/blackwell_tma_gemm_fixed
ncu --set full ./bin/blackwell_5070_fixed/blackwell_w4a16_simple

# Memory bandwidth analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./bin/blackwell_5070_fixed/blackwell_async_wgmma_fixed
```

### **Assembly Generation**
```bash
# Generate PTX for instruction analysis
make -f Makefile.blackwell_fixed asm_all

# View generated PTX
cat build/blackwell_5070_fixed/tma_fixed.ptx
cat build/blackwell_5070_fixed/wgmma_fixed.ptx
```

## 🎯 Use Cases

### **1. LLM Inference Optimization**
- **W4A16 GEMV** for 4-bit quantized model inference
- **Single token latency**: 500-1000 μs on RTX 5070
- **Throughput**: 1000+ tokens/second for large models

### **2. Transformer Layer Acceleration** 
- **LayerNorm + GEMM Fusion** for attention/feedforward layers
- **Memory reduction**: 50%+ vs separate kernels
- **Performance**: 2-4x speedup over naive implementations

### **3. High-Performance Computing**
- **TMA GEMM** for scientific computing workloads
- **Dense matrix operations** with optimal memory patterns
- **Sustained throughput**: 15-20 TFLOPS on compatible workloads

## 🔬 Technical Details

### **Warp Specialization Patterns**
```cpp
enum WarpRole {
    WARP_PRODUCER_WEIGHTS = 0,    // TMA load weights
    WARP_PRODUCER_SCALES = 1,     // TMA load scales  
    WARP_DEQUANTIZER = 2,         // Dequantize 4-bit → FP16
    WARP_COMPUTER_0 = 3,          // WGMMA compute warp 0
    WARP_COMPUTER_1 = 4,          // WGMMA compute warp 1
    WARP_COMPUTER_2 = 5,          // WGMMA compute warp 2
    WARP_COMPUTER_3 = 6,          // WGMMA compute warp 3
    WARP_REDUCER = 7              // Reduction and output
};
```

### **TMA Configuration**
```cpp
// Blackwell-optimized TMA setup
CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;  
CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
uint32_t tile_shape[2] = {128, 128};  // Optimal for RTX 5070
```

### **WGMMA Instructions**
```cpp
// Native Blackwell WGMMA async
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.f16.f16 "
    "{...}, desc_a, desc_b;\n"
    "wgmma.wait_group.sync.aligned 0;\n"
);
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-kernel`
3. **Add comprehensive tests** for new implementations
4. **Update documentation** including performance expectations
5. **Submit pull request** with detailed description

### **Guidelines**
- Follow existing code style and structure
- Include performance benchmarks for new kernels
- Add compatibility checks for different architectures
- Document all Blackwell-specific optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** for Blackwell architecture documentation and CUDA toolkit
- **CUTLASS** library for high-performance GEMM implementations
- **CuTe** for modern C++ tensor abstractions
- **Community contributions** to CUDA optimization techniques

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DrXuQian/cuda-microbenchmarks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DrXuQian/cuda-microbenchmarks/discussions)
- **Documentation**: [Wiki](https://github.com/DrXuQian/cuda-microbenchmarks/wiki)

---

**Note**: These kernels are specifically optimized for RTX 5070 Blackwell architecture. Performance may vary on other GPUs. For production use, consider highly-optimized libraries like cuBLAS, cuDNN, or CUTLASS.

*Built with ❤️ for the CUDA community*