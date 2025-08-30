# 🧮 CUDA Kernel Design Guide

This directory contains detailed documentation for all kernel implementations in the microbenchmark project, covering both GEMM and GEMV optimizations with focus on transformer inference workloads.

## 📚 Documentation Structure

| Document | Focus Area | Kernels Covered |
|----------|------------|-----------------|
| [**gemv-optimization.md**](gemv-optimization.md) | GEMV kernel design for transformer inference | Simple, Optimized, Warp-Specialized GEMV |
| [**quantization-techniques.md**](quantization-techniques.md) | w4a16f quantized inference methods | Marlin-inspired dequantization |
| [**warp-specialization.md**](warp-specialization.md) | Warp specialization patterns | Async memory/compute overlap |
| [**tensor-cores.md**](tensor-cores.md) | WMMA and WGMMA utilization | Modern GPU tensor operations |
| [**memory-optimization.md**](memory-optimization.md) | Memory hierarchy optimization | Coalescing, shared memory, async |

## 🎯 Kernel Categories

### GEMV Kernels (Transformer Focus)
**Target: 1×N @ N×M operations for batch-1 inference**

1. **Simple GEMV** - Thread-per-output baseline
2. **Optimized GEMV** - Shared memory + vectorization  
3. **Warp Specialized GEMV** - Warp-level reductions
4. **Marlin-inspired w4a16f** - Quantized inference

### GEMM Kernels (Matrix Multiplication)
**Target: M×N @ N×K operations for training/large-batch inference**

1. **Ping-Pong** - Double buffering for compute/memory overlap
2. **Warp Specialized** - Async loading with dedicated warps
3. **WGMMA** - Modern tensor core utilization
4. **Optimized Variants** - Memory coalescing improvements

## 🏗️ Design Principles

### Memory Optimization
- **Coalescing**: Ensure contiguous memory access patterns
- **Shared Memory**: Cooperative loading and bank conflict avoidance
- **Async Pipeline**: cp.async for compute/memory overlap
- **Vectorization**: half2/half4 operations where applicable

### Compute Optimization  
- **Tensor Cores**: WMMA/WGMMA for mixed-precision operations
- **Warp Primitives**: Shuffle operations for reductions
- **Register Usage**: Minimize to increase occupancy
- **Loop Unrolling**: Compiler optimizations for inner loops

### Quantization Support
- **LOP3 Bit Manipulation**: Efficient 4-bit dequantization
- **Symmetric Zero-Point**: Simplified quantization scheme
- **Group Quantization**: Per-channel vs per-tensor scaling
- **Mixed Precision**: f16 activations, 4-bit weights

## 📊 Performance Characteristics

### GEMV Results (1×3584 @ 3584×18944)
- **Memory Bound**: All variants achieve ~409 GFLOPS
- **Bandwidth Limited**: 61% of theoretical peak (672 GB/s)
- **Vectorization Effective**: half2 operations provide good utilization

### GEMM Results (1024³)
- **Compute Bound**: Higher arithmetic intensity
- **Tensor Core Advantage**: 6,688 GFLOPS with WMMA
- **Memory Coalescing Critical**: +44% improvement with optimization

## 🔬 Analysis Methodology

### Performance Measurement
```cuda
// Benchmark template used across all kernels
cudaEventRecord(start);
for (int i = 0; i < iterations; i++) {
    kernel<<<grid, block, shmem>>>(args);
}
cudaEventRecord(stop);
cudaEventElapsedTime(&elapsed, start, stop);
double gflops = (2.0 * M * N * K * iterations) / (elapsed / 1000.0) / 1e9;
```

### Accuracy Validation
```cuda
// Cosine similarity for robust numerical comparison
AccuracyResult result = verify_with_cosine_distance(gpu_result, reference, size);
bool passed = result.cosine_similarity >= 0.9999; // Very strict threshold
```

### NCU Profiling
```bash
# Memory-focused analysis
ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis kernel

# Custom metrics for optimization
ncu --metrics smsp__cycles_elapsed.avg,dram__bytes.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum kernel
```

## 🚀 Optimization Workflow

### 1. Baseline Implementation
- Simple, correct implementation
- Single optimization focus (e.g., just coalescing)
- Establish performance baseline and correctness

### 2. Memory Optimization
- Analyze memory access patterns with NCU
- Implement coalescing improvements
- Add shared memory where beneficial
- Consider async copy operations

### 3. Compute Optimization
- Profile compute utilization
- Add vectorization (half2, etc.)
- Implement tensor core operations
- Optimize register usage

### 4. Advanced Techniques
- Warp specialization for memory/compute overlap
- Multi-stage async pipelines
- Quantization for reduced memory bandwidth
- Architecture-specific optimizations

## 🔧 Implementation Guidelines

### Code Structure
```cuda
// Kernel template structure
__global__ void optimized_kernel(
    const DataType* __restrict__ input,
    DataType* __restrict__ output,
    int param1, int param2
) {
    // 1. Thread/block mapping
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // 2. Shared memory declaration
    extern __shared__ DataType shmem[];
    
    // 3. Main computation loop with optimizations
    // 4. Result write-back
}
```

### Best Practices
- Always use `__restrict__` for kernel parameters
- Declare shared memory as `extern __shared__`
- Implement bounds checking for irregular sizes
- Use `#pragma unroll` judiciously for known loop counts
- Profile before and after each optimization

### Common Pitfalls
- **Bank conflicts** in shared memory access
- **Warp divergence** from conditional code
- **Register spilling** from complex kernels
- **Incorrect memory coalescing** patterns

## 📈 Future Directions

### sm_90+ Optimizations (H100, etc.)
- True WGMMA warp group operations
- TMA (Tensor Memory Accelerator) integration
- cp.async.bulk for improved memory pipeline
- Thread block clustering

### Quantization Research
- Sub-4-bit quantization (2-bit, 1-bit)
- Dynamic quantization schemes
- Mixed-precision beyond f16/4-bit
- Sparsity-aware quantization

### Multi-GPU Scaling
- NCCL integration for distributed operations
- Pipeline parallelism across GPUs
- Memory-efficient model sharding
- Communication/computation overlap

## 📋 Validation Checklist

Before merging new kernels:
- [ ] Correctness: Cosine similarity ≥ 0.9999 vs cuBLAS
- [ ] Performance: Measured GFLOPS and memory bandwidth
- [ ] Profiling: NCU analysis with key metrics documented
- [ ] Documentation: Optimization techniques explained
- [ ] Code Quality: Clean, well-commented implementation

---

See individual kernel documentation for detailed implementation analysis and optimization techniques.