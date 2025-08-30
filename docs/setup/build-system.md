# 🛠️ Build System and Setup Guide

Complete guide for setting up the development environment, building kernels, and running benchmarks for the CUDA microbenchmark project.

## 🔧 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 7.5+ (Turing or newer)
- **Memory**: 8GB+ GPU memory recommended for large workloads
- **Tested Platform**: RTX 4070 Ti SUPER (Ada Lovelace, CC 8.9)

### Software Dependencies
- **CUDA Toolkit**: 12.0+ (tested with 12.8)
- **NVIDIA Driver**: Latest stable version
- **cuBLAS**: Included with CUDA Toolkit
- **NCU Profiler**: NVIDIA Nsight Compute for performance analysis
- **Compiler**: GCC 9+ or compatible C++17 compiler

### Installation Commands
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
sudo apt install nvidia-nsight-compute

# Verify installation
nvcc --version
nvidia-smi
ncu --version
```

## 📁 Project Structure After Build

```
microbenchmark/
├── build/
│   └── bin/                 # All compiled executables
│       ├── simple_gemv_benchmark       # GEMV optimization comparison
│       ├── marlin_gemv_benchmark      # Advanced w4a16f benchmarks
│       ├── ping_pong_test             # Ping-pong kernel
│       ├── warp_specialized_test      # Warp specialization
│       ├── wgmma_test                 # Tensor core integration
│       └── ...                        # Other kernel binaries
├── src/                     # Source code (never modified by build)
├── Makefile                # Main build system
└── ...
```

## 🏗️ Build System Architecture

### Makefile Structure
The build system is organized into logical target categories:

```makefile
# Source organization
KERNEL_DIR = src/kernels
BENCHMARK_DIR = src/benchmarks  
UTILS_DIR = src/utils

# Build configuration
NVCC = nvcc
CFLAGS = -std=c++17 -O3 -arch=sm_89 -lcublas -I$(UTILS_DIR)

# Target categories
PRIMARY_TARGETS = ping_pong_test warp_specialized_test warp_final_opt_test wgmma_test
EXPERIMENTAL_TARGETS = warp_specialized_optimized_test warp_simple_opt_test  
MARLIN_TARGETS = marlin_gemv_benchmark simple_gemv_benchmark
```

### Compilation Settings Explained

```makefile
CFLAGS = -std=c++17 -O3 -arch=sm_89 -lcublas -I$(UTILS_DIR)
```

| Flag | Purpose | Notes |
|------|---------|-------|
| `-std=c++17` | C++17 standard | Required for modern STL features |
| `-O3` | Aggressive optimization | Maximum compiler optimization |
| `-arch=sm_89` | Target architecture | Ada Lovelace (RTX 4070 Ti Super) |
| `-lcublas` | Link cuBLAS library | For reference comparisons |
| `-I$(UTILS_DIR)` | Include path | Access to common.h utilities |

### Architecture-Specific Compilation
```makefile
# Automatic architecture detection (future enhancement)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d '.')
CFLAGS += -arch=sm_$(GPU_ARCH)
```

## 🚀 Build Commands

### Primary Build Targets

```bash
# Build all primary kernels
make all

# Build individual kernels  
make ping_pong_test           # Ping-pong double buffering
make warp_specialized_test    # Basic warp specialization
make warp_final_opt_test      # Optimized warp specialization
make wgmma_test              # Tensor core integration

# Build experimental variants
make all-experimental         # All experimental kernels
make warp_specialized_optimized_test
make warp_simple_opt_test

# Build GEMV benchmarks
make simple_gemv_benchmark    # GEMV optimization comparison
make marlin_gemv_benchmark    # Advanced w4a16f (may have build issues)
```

### Clean and Maintenance
```bash
# Remove all binaries
make clean

# Remove only experimental binaries  
make clean-experimental

# Show all available targets
make help
```

## 🧪 Running Benchmarks

### Individual Kernel Tests
```bash
# Run specific kernels
make run-ping-pong           # Ping-pong kernel benchmark
make run-warp-specialized    # Warp specialization analysis
make run-warp-optimized      # Optimized warp kernel
make run-wgmma              # Tensor core benchmark

# Run GEMV benchmarks  
make run-simple-gemv         # GEMV optimization comparison
make run-marlin-gemv         # Advanced w4a16f benchmarks

# Run all primary tests
make run-all
```

### Batch Execution
```bash
# Run all experiments sequentially
make run-experimental        # Experimental kernel variants
make benchmark              # Performance summary of primary kernels
```

### Expected Output Format
```
🚀 Simple GEMV Kernel Optimization Benchmark
============================================

Shape: 1×3584 @ 3584×18944 (67.90 M parameters)

📊 Performance Results:
========================
Simple GEMV              :    0.332 ms,    409.2 GFLOPS
Optimized GEMV           :    0.332 ms,    409.2 GFLOPS  
Warp Specialized GEMV    :    0.332 ms,    409.2 GFLOPS
cuBLAS Reference         :    0.123 ms,   1105.3 GFLOPS

🏆 Performance Analysis:
========================
Best kernel GFLOPS: 409.2
cuBLAS GFLOPS:      1105.3
📈 cuBLAS is 2.7x faster. Room for optimization!

💾 Memory Analysis:
===================
Theoretical Bandwidth: 672.1 GB/s
Achieved Bandwidth:    409.4 GB/s (60.9% of peak)
```

## 🔬 Profiling Integration

### NCU Profiling Commands
```bash
# Automated profiling targets
make profile-simple-gemv      # GEMV kernels with full NCU analysis
make profile-marlin-gemv      # w4a16f kernels profiling
make profile-warp            # Warp specialization analysis
make profile-optimized       # Optimized kernels analysis

# Manual profiling
ncu --set full ./build/bin/simple_gemv_benchmark
ncu --section MemoryWorkloadAnalysis ./build/bin/warp_specialized_test

# Custom metrics collection
ncu --metrics smsp__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum ./build/bin/kernel_test
```

### Profiling Output Interpretation
```bash
# Key NCU metrics to analyze
--metrics smsp__cycles_elapsed.avg          # Execution time
--metrics dram__bytes_read.sum               # Memory bandwidth  
--metrics smsp__sass_thread_inst_executed_op_fmul_pred_on.sum  # Compute instructions
--metrics smsp__warps_launched.sum          # Warp utilization
--metrics l1tex__t_sector_hit_rate.pct       # L1 cache efficiency
```

## 🚨 Troubleshooting

### Common Build Issues

#### 1. Architecture Mismatch
```bash
# Error: ptxas fatal error
# Solution: Update arch flag to match your GPU
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
# Use result to set -arch=sm_XX in Makefile CFLAGS
```

#### 2. cuBLAS Linking Issues
```bash
# Error: cannot find -lcublas
# Solution: Ensure CUDA installation includes cuBLAS
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 3. Shared Memory Limit Exceeded
```bash
# Error: too much shared memory in function 'kernel_name'
# Solution: Reduce shared memory usage or increase dynamic allocation
# Check kernel launch with proper shmem_size parameter
```

#### 4. Register Limit Exceeded  
```bash
# Error: too many resources requested for launch
# Solution: Reduce register usage or decrease block size
# Add __launch_bounds__(max_threads_per_block) to kernel
```

### Runtime Issues

#### 1. Incorrect Results (Low Cosine Similarity)
```bash
# Check: Numerical precision issues
# Solution: Increase precision threshold or check algorithm implementation
# Verify: Input data initialization and boundary conditions
```

#### 2. Performance Regression
```bash
# Check: Thermal throttling or GPU utilization
nvidia-smi -l 1  # Monitor GPU temperature and utilization
# Solution: Ensure adequate cooling and power supply
```

#### 3. NCU Profiling Failures
```bash
# Error: No kernels found to profile
# Solution: Ensure binary was compiled with -lineinfo flag
# Check: GPU compute capability supported by NCU version
```

## ⚡ Performance Tuning

### Compile-Time Optimizations
```makefile
# Additional NVCC flags for maximum performance
PERF_FLAGS = -use_fast_math -lineinfo -Xptxas -O3,-v
CFLAGS += $(PERF_FLAGS)

# Debug build for development
DEBUG_FLAGS = -g -G -O0 -DDEBUG
# Use: make EXTRA_FLAGS="$(DEBUG_FLAGS)" target_name
```

### Runtime Configuration
```bash
# GPU performance mode (Linux)
sudo nvidia-smi -pm ENABLED
sudo nvidia-smi -ac 1313,2610  # Memory,Graphics clocks (adjust for your GPU)

# CUDA context optimization
export CUDA_LAUNCH_BLOCKING=0  # Enable async launches
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Consistent device ordering
```

### Memory Optimization
```cpp
// Kernel launch configuration tuning
dim3 grid, block;
size_t shmem_size;

// Rule of thumb for block size selection
int optimal_block_size = 128;  // Start with 128 threads
// Adjust based on register usage and shared memory requirements

// Shared memory size calculation  
shmem_size = N * sizeof(half);  // A vector size
if (shmem_size > 48 * 1024) {  // 48KB limit on most GPUs
    // Use dynamic allocation or reduce problem size
}
```

## 📊 Build System Extension

### Adding New Kernels

1. **Create kernel implementation**
```bash
# Add to src/kernels/
touch src/kernels/my_new_kernel.cu
```

2. **Create benchmark harness**
```bash
# Add to src/benchmarks/  
touch src/benchmarks/my_new_benchmark.cu
```

3. **Update Makefile**
```makefile
# Add target definition
MY_NEW_TARGET = my_new_test
MY_NEW_SRC = src/benchmarks/my_new_benchmark.cu

$(MY_NEW_TARGET): $(MY_NEW_SRC) src/utils/common.h src/kernels/my_new_kernel.cu
	$(NVCC) $(CFLAGS) -o $(MY_NEW_TARGET) $(MY_NEW_SRC)

# Add to appropriate target group
EXPERIMENTAL_TARGETS += $(MY_NEW_TARGET)
```

4. **Add run target**
```makefile
run-my-new: $(MY_NEW_TARGET)
	@echo "🧪 Running My New Kernel Test:"
	@echo "============================="
	@./$(MY_NEW_TARGET)
```

### Multi-Platform Support
```makefile
# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM_FLAGS = -Xlinker -rpath=/usr/local/cuda/lib64
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM_FLAGS = -Xlinker -rpath=/usr/local/cuda/lib
endif

CFLAGS += $(PLATFORM_FLAGS)
```

---

This build system provides a solid foundation for CUDA kernel development with comprehensive profiling integration and extensibility for future optimizations.