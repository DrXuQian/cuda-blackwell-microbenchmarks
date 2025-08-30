# 🚀 Getting Started Guide

Complete walkthrough for setting up, building, and running the CUDA kernel microbenchmark project. This guide assumes basic familiarity with CUDA development.

## 📋 Prerequisites Checklist

### Hardware Requirements
- [ ] **NVIDIA GPU** with compute capability 7.5+ (Turing, Ampere, Ada Lovelace, or Hopper)
- [ ] **8GB+ GPU Memory** (16GB recommended for large workloads)
- [ ] **Adequate Cooling** for sustained high-performance operation

### Software Requirements  
- [ ] **CUDA Toolkit 12.0+** (tested with 12.8)
- [ ] **NVIDIA Driver** compatible with CUDA version
- [ ] **GCC 9+** or compatible C++17 compiler
- [ ] **Git** for version control (if cloning)

### Optional Tools
- [ ] **NVIDIA Nsight Compute** for kernel profiling
- [ ] **NVIDIA Nsight Systems** for timeline analysis
- [ ] **Modern IDE** with CUDA syntax highlighting

## 🔧 Environment Setup

### 1. Verify CUDA Installation
```bash
# Check CUDA compiler
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Thu_Sep_12_02:18:05_PDT_2024
# Cuda compilation tools, release 12.8, V12.8.46

# Check GPU detection
nvidia-smi

# Expected output should show your GPU details:
# GPU 0: NVIDIA GeForce RTX 4070 Ti SUPER (UUID: ...)
```

### 2. Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional: Optimize for development
export CUDA_LAUNCH_BLOCKING=0  # Enable async launches
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Consistent device ordering
```

### 3. Verify cuBLAS
```bash
# Test cuBLAS linking (should compile without errors)
echo '#include <cublas_v2.h>
int main() { cublasHandle_t handle; cublasCreate(&handle); return 0; }' > test_cublas.cu
nvcc -lcublas test_cublas.cu -o test_cublas
./test_cublas && echo "✅ cuBLAS working" || echo "❌ cuBLAS issue"
rm test_cublas.cu test_cublas
```

## 📥 Project Setup

### Option 1: Clone from Repository (if available)
```bash
git clone <repository-url> cuda-kernel-microbenchmarks
cd cuda-kernel-microbenchmarks
```

### Option 2: Download and Extract
```bash
# Extract project archive
tar -xzf cuda-kernel-microbenchmarks.tar.gz
cd cuda-kernel-microbenchmarks
```

### Verify Project Structure
```bash
ls -la
# Expected structure:
# src/
# ├── kernels/     # CUDA kernel implementations
# ├── benchmarks/  # Benchmark harnesses  
# └── utils/       # Common utilities
# external/        # Third-party dependencies
# docs/           # Comprehensive documentation
# build/          # Build artifacts (created automatically)
# Makefile        # Build system
# README.md       # Project overview
```

## 🏗️ Building the Project

### Quick Start Build
```bash
# Build and run GEMV benchmark (fastest way to test)
make gemv

# Expected output:
# 🚀 Simple GEMV Kernel Optimization Benchmark
# ============================================
# Shape: 1×3584 @ 3584×18944 (67.90 M parameters)
# ...
# Best kernel GFLOPS: 405.5
# ✨ Benchmark complete!
```

### Build All Kernels
```bash
# Primary kernels (stable, well-tested)
make all

# All kernels including experimental variants
make all-experimental

# Check what was built
ls build/bin/
# Expected binaries:
# ping_pong_test  warp_specialized_test  wgmma_test  simple_gemv_benchmark  ...
```

### Build Individual Components
```bash
# Specific kernels
make build/bin/ping_pong_test         # Double buffering technique
make build/bin/warp_specialized_test  # Warp specialization
make build/bin/wgmma_test            # Tensor core integration
make build/bin/simple_gemv_benchmark # GEMV optimization comparison

# Experimental variants
make build/bin/warp_specialized_optimized_test
```

## 🧪 Running Benchmarks

### GEMV Benchmarks (Transformer Focus)
```bash
# Simple GEMV optimization comparison
make run-simple-gemv

# Advanced w4a16f quantized benchmarks (may have build issues)
make run-marlin-gemv
```

### GEMM Benchmarks (Matrix Multiplication)
```bash
# Individual kernel tests
make run-ping-pong           # Double buffering
make run-warp-specialized    # Warp specialization
make run-warp-optimized      # Optimized warp specialization
make run-wgmma              # Tensor core utilization

# All primary tests
make run-all

# Performance summary
make benchmark
```

### Experimental Kernels
```bash
# Run experimental variants
make run-experimental
```

## 📊 Understanding Results

### Performance Output Format
```
Simple GEMV              :    0.335 ms,    405.5 GFLOPS
Optimized GEMV           :    0.335 ms,    405.5 GFLOPS
Warp Specialized GEMV    :    0.335 ms,    405.5 GFLOPS
cuBLAS Reference         :    0.123 ms,   1105.3 GFLOPS
```

**Key Metrics:**
- **Execution Time**: Lower is better (milliseconds)
- **GFLOPS**: Higher is better (billion floating-point operations per second)
- **Bandwidth Utilization**: Percentage of theoretical GPU memory bandwidth

### Accuracy Validation
```
Cosine Similarity: 0.999987
Cosine Distance:   0.000013
Status: ✅ PASSED
```

**Interpretation:**
- **Cosine Similarity ≥ 0.9999**: Numerically accurate implementation
- **Status: ✅ PASSED**: Results match cuBLAS reference within tolerance

## 🔬 Profiling with NCU

### Prerequisites
```bash
# Install Nsight Compute (if not already installed)
sudo apt install nvidia-nsight-compute

# Verify installation
ncu --version
```

### Automated Profiling
```bash
# Profile GEMV kernels with comprehensive analysis
make profile-simple-gemv

# Profile with specific metrics
make profile-marlin-gemv-metrics

# Profile original GEMM kernels
make profile-warp
make profile-optimized
```

### Manual Profiling
```bash
# Full analysis (slowest, most comprehensive)
ncu --set full ./build/bin/simple_gemv_benchmark

# Memory-focused analysis
ncu --section MemoryWorkloadAnalysis ./build/bin/warp_specialized_test

# Custom metrics for optimization
ncu --metrics smsp__cycles_elapsed.avg,dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum ./build/bin/ping_pong_test
```

### Profiling Output Interpretation
```
Metric Name                              Metric Unit    Metric Value
dram__bytes_read.sum                     Mbyte          135.84
dram__bytes_write.sum                    Mbyte          17.15
smsp__cycles_elapsed.avg                 cycle          850000
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum inst   18944
```

**Key Insights:**
- **Memory Bandwidth**: `dram__bytes_*.sum` indicates memory traffic
- **Execution Time**: `smsp__cycles_elapsed.avg` shows kernel duration
- **Compute Instructions**: `op_f*` counts show mathematical operations
- **Efficiency**: Compare achieved vs theoretical performance

## 🚨 Troubleshooting

### Build Issues

#### Architecture Mismatch
```bash
# Error: ptxas fatal error
# Solution: Update architecture flag
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits

# Edit Makefile CFLAGS line:
# -arch=sm_XX  (where XX is your compute capability × 10)
```

#### Missing cuBLAS
```bash
# Error: cannot find -lcublas
# Solution: Install CUDA development packages
sudo apt install nvidia-cuda-dev libcublas-dev
```

#### Shared Memory Limit
```bash
# Error: too much shared memory
# Solution: Check GPU specifications
nvidia-smi --query-gpu=compute_cap,memory.total --format=csv

# For older GPUs, reduce shared memory usage in kernels
```

### Runtime Issues

#### No GPU Detected
```bash
# Check GPU visibility
nvidia-smi
lspci | grep -i nvidia

# If issues, reinstall NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### Performance Inconsistencies
```bash
# Check thermal throttling
nvidia-smi -l 1  # Monitor temperature
# Normal: <80°C under load
# Throttling: >85°C

# Enable performance mode
sudo nvidia-smi -pm ENABLED
sudo nvidia-smi -ac 1313,2610  # Memory,Graphics clocks (adjust for your GPU)
```

#### Accuracy Failures
```bash
# Check numerical precision
# Cosine similarity < 0.9999 indicates implementation error
# Debug: Compare intermediate results with reference implementation
# Solution: Review algorithm correctness and floating-point handling
```

## 🎯 Next Steps

### Explore Documentation
1. **[Kernel Design Guide](../kernels/README.md)** - Understand optimization techniques
2. **[GEMV Optimization](../kernels/gemv-optimization.md)** - Transformer inference focus
3. **[Performance Analysis](../performance/)** - NCU profiling results

### Experiment with Modifications
```bash
# Create new kernel variant
cp src/kernels/warp_specialized_kernel.cu src/kernels/my_optimization.cu

# Edit kernel implementation
# Update Makefile to add new target
# Test and benchmark changes
```

### Development Workflow
1. **Implement** optimization in `src/kernels/`
2. **Create benchmark** in `src/benchmarks/`  
3. **Test correctness** with accuracy validation
4. **Profile performance** with NCU
5. **Document results** in `docs/performance/`

### Advanced Topics
- **Quantization**: Implement w4a16f techniques from Marlin
- **Multi-GPU**: Scale across multiple devices
- **Custom Shapes**: Adapt to specific model architectures
- **Architecture-Specific**: Optimize for sm_90+ features

## 📚 Additional Resources

### CUDA Learning
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Profiling and Optimization
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

### Research Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Marlin: Mixed-Precision Quantization](https://github.com/IST-DASLab/marlin) - w4a16f techniques

## 🤝 Getting Help

### Common Issues Database
Check `docs/performance/` for known performance characteristics and solutions.

### Community Resources
- NVIDIA Developer Forums
- CUDA GitHub Discussions  
- Academic CUDA optimization papers

### Debugging Strategy
1. **Start Simple**: Begin with basic kernels that work
2. **Incremental Changes**: Add one optimization at a time
3. **Profile Early**: Use NCU to understand bottlenecks
4. **Validate Always**: Ensure correctness before optimizing performance

---

**You're now ready to explore high-performance CUDA kernel development!** 🚀

Start with `make gemv` to see the system in action, then dive into the kernel implementations to understand the optimization techniques.