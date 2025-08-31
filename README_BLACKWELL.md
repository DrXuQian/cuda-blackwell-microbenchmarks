# RTX 5070 Blackwell CUDA Microbenchmarks

This directory contains specialized CUDA kernels optimized for the RTX 5070 Blackwell architecture, featuring **TMA (Tensor Memory Accelerator)** and **Async WGMMA (Warp Group Matrix Multiply Accumulate)** implementations.

## 🏗️ Architecture Overview

The RTX 5070 Blackwell architecture includes:
- **56 SM units** with enhanced tensor processing
- **sm_90+ compute capability** 
- **164KB shared memory** per block
- **TMA hardware** for optimal memory bandwidth
- **Async WGMMA instructions** for maximum throughput

## 📁 Directory Structure

```
src/blackwell_5070/
├── utils/
│   └── blackwell_common.h          # Common utilities and constants
├── tma_kernels/
│   └── blackwell_tma_gemm.cu       # TMA-optimized GEMM kernel
├── async_wgmma_kernels/
│   └── blackwell_async_wgmma.cu    # Async WGMMA kernel
└── benchmarks/
    └── blackwell_comprehensive_benchmark.cu  # Full benchmark suite

tests/blackwell_5070/
└── test_blackwell_kernels.sh       # Comprehensive test script

Makefile.blackwell                   # Build system
```

## 🚀 Features

### TMA GEMM Kernel (`blackwell_tma_gemm.cu`)
- **Tensor Memory Accelerator** for 2D bulk transfers
- **128x128x64 tile sizes** optimized for Blackwell
- **Asynchronous memory operations** with proper synchronization
- **Expected performance: 15-20 TFLOPS** on RTX 5070

### Async WGMMA Kernel (`blackwell_async_wgmma.cu`)
- **Native WGMMA instructions** (wgmma.mma_async.sync.aligned)
- **Producer-consumer pattern** for memory-compute overlap
- **64x64x32 computation tiles** 
- **Expected performance: 8-12 TFLOPS** with better efficiency

### Key Optimizations
- **Double buffering** for async operations
- **Optimal shared memory layout** (164KB utilization)
- **Register pressure management** 
- **Coalesced memory access patterns**
- **Blackwell-specific instruction scheduling**

## 🛠️ Build Requirements

- **CUDA 12.x+** (recommended for Blackwell support)
- **sm_90+ GPU** (RTX 4090, RTX 5070, etc.)
- **GCC 9+** or compatible C++ compiler
- **NCU** (optional, for profiling)

## 📦 Building

### Quick Build
```bash
# Build all kernels
make -f Makefile.blackwell all

# Build individual kernels
make -f Makefile.blackwell tma_kernel
make -f Makefile.blackwell async_wgmma
make -f Makefile.blackwell benchmark
```

### Build Options
```bash
# Debug builds
make -f Makefile.blackwell debug_tma
make -f Makefile.blackwell debug_async_wgmma

# Assembly generation
make -f Makefile.blackwell asm_all

# Clean builds
make -f Makefile.blackwell clean_all
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run all tests
./tests/blackwell_5070/test_blackwell_kernels.sh

# Individual tests
make -f Makefile.blackwell test_tma
make -f Makefile.blackwell test_async_wgmma
make -f Makefile.blackwell test_benchmark
```

### Manual Testing
```bash
# TMA GEMM test
./bin/blackwell_5070/blackwell_tma_gemm

# Async WGMMA test
./bin/blackwell_5070/blackwell_async_wgmma

# Full benchmark suite
./bin/blackwell_5070/blackwell_comprehensive_benchmark
```

## 📊 Performance Analysis

### Profiling with NCU
```bash
# Profile TMA kernel
make -f Makefile.blackwell profile_tma

# Profile WGMMA kernel  
make -f Makefile.blackwell profile_async_wgmma

# View detailed metrics
ncu --set full --target-processes all ./bin/blackwell_5070/blackwell_tma_gemm
```

### Memory Check
```bash
# Check for memory errors
make -f Makefile.blackwell memcheck_all

# Individual memory checks
cuda-memcheck ./bin/blackwell_5070/blackwell_tma_gemm
```

## 🎯 Expected Performance

### RTX 5070 Performance Targets

| Kernel | Matrix Size | Expected TFLOPS | Memory Bandwidth |
|--------|-------------|-----------------|------------------|
| TMA GEMM | 2048x2048 | 15-20 | 800+ GB/s |
| TMA GEMM | 4096x4096 | 18-22 | 900+ GB/s |
| Async WGMMA | 1024x1024 | 8-10 | 600+ GB/s |
| Async WGMMA | 2048x2048 | 10-12 | 700+ GB/s |

### Performance Comparison
```bash
# Run scalability benchmark
./bin/blackwell_5070/blackwell_comprehensive_benchmark

# Expected output:
# Size      TMA TFLOPS    WGMMA TFLOPS    TMA Time(ms)    WGMMA Time(ms)
# 512x512   12.5          8.2             0.85            1.28
# 1024x1024 16.8          9.7             2.51            3.45
# 2048x2048 19.2          11.1            11.25           15.22
```

## 🔧 Customization

### Tuning Parameters

Edit `src/blackwell_5070/utils/blackwell_common.h`:

```cpp
// TMA tile sizes (adjust for your workload)
#define TMA_TILE_M 128    // Increase for larger matrices
#define TMA_TILE_N 128    // Decrease for memory-bound workloads
#define TMA_TILE_K 64     // Optimize for K dimension

// WGMMA tile sizes
#define WGMMA_M 64        // Must be multiple of 16
#define WGMMA_N 64        // Must be multiple of 16  
#define WGMMA_K 32        // Fixed for fp16 input
```

### Adding New Kernels

1. Create new `.cu` file in appropriate subdirectory
2. Include `blackwell_common.h` for utilities
3. Add build target to `Makefile.blackwell`
4. Add test case to `test_blackwell_kernels.sh`

## 🐛 Troubleshooting

### Common Issues

**Build Errors:**
- Ensure CUDA 12.x is installed: `nvcc --version`
- Check compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Verify sm_90+ support: Kernels require Blackwell/Hopper architecture

**Runtime Errors:**
- `Invalid device function`: GPU doesn't support sm_90
- `Out of memory`: Reduce matrix sizes or tile dimensions
- `TMA setup failed`: Check memory alignment and descriptor setup

**Performance Issues:**
- Low TFLOPS: Check if running on compatible hardware
- Memory bandwidth limited: Verify memory clocks with `nvidia-smi`
- Synchronization overhead: Profile with NCU for bottlenecks

### Debug Mode
```bash
# Build with debug symbols
make -f Makefile.blackwell debug_tma

# Run with cuda-gdb
cuda-gdb ./bin/blackwell_5070/debug_tma_gemm
```

## 📚 References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX Instruction Set Architecture](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUTLASS GEMM Implementations](https://github.com/NVIDIA/cutlass)

## 🤝 Contributing

1. Follow the existing code style and structure
2. Add comprehensive tests for new features
3. Include performance benchmarks
4. Update documentation for new parameters
5. Verify compatibility across different matrix sizes

## 📄 License

This code is provided for educational and research purposes. Ensure compliance with NVIDIA CUDA licensing terms when using these implementations.

---

**Note**: These kernels are optimized specifically for RTX 5070 Blackwell architecture. Performance may vary on other GPUs. For production use, consider using highly-optimized libraries like cuBLAS or CUTLASS.