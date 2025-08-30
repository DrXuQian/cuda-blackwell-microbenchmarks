# 📁 Project Structure Overview

Complete overview of the reorganized CUDA kernel microbenchmark project structure, designed for maintainable research and development of high-performance GPU kernels.

## 🎯 Design Principles

The project structure follows these key principles:
- **Separation of Concerns**: Source code, documentation, and build artifacts are clearly separated
- **Logical Organization**: Related components are grouped together
- **Scalability**: Easy to add new kernels, benchmarks, and documentation
- **Research-Friendly**: Clear documentation and analysis workflows
- **Build System Integration**: Automated compilation and testing

## 📊 Complete Directory Structure

```
microbenchmark/                     # Project root
├── README.md                       # Main project overview and quick start
├── PROJECT_STRUCTURE.md           # This file - complete structure guide
├── Makefile                        # Build system with comprehensive targets
│
├── src/                            # All source code organized by purpose
│   ├── kernels/                   # CUDA kernel implementations
│   │   ├── ping_pong_kernel.cu           # Double buffering technique
│   │   ├── warp_specialized_kernel.cu    # Basic warp specialization
│   │   ├── warp_specialized_final_opt.cu # Optimized warp specialization
│   │   ├── warp_specialized_optimized.cu # Experimental optimization v1
│   │   ├── warp_specialized_simple_opt.cu # Simplified optimization
│   │   ├── wgmma_kernel.cu              # Tensor core (WMMA/WGMMA)
│   │   └── marlin_gemv_optimized.cu     # Marlin-inspired w4a16f GEMV
│   │
│   ├── benchmarks/                # Benchmark harnesses and test suites
│   │   ├── simple_gemv_benchmark.cu     # GEMV optimization comparison
│   │   ├── marlin_gemv_benchmark.cu     # Advanced w4a16f benchmarks
│   │   ├── debug_mma.cu                 # MMA debugging utilities
│   │   ├── mma_final.cu                 # Final MMA implementation
│   │   ├── mma_kernels.cu               # MMA kernel variants
│   │   ├── mma_simplified.cu            # Simplified MMA test
│   │   └── simple_test.cu               # Basic functionality test
│   │
│   └── utils/                     # Common utilities and headers
│       └── common.h                     # Shared benchmarking and validation
│
├── external/                      # Third-party dependencies
│   └── marlin/                   # Marlin quantization framework
│       └── marlin_cuda_kernel.cu        # Original Marlin implementation
│
├── build/                         # Build artifacts (auto-created)
│   └── bin/                      # Compiled executables
│       ├── simple_gemv_benchmark        # GEMV benchmark suite
│       ├── ping_pong_test               # Ping-pong kernel test
│       ├── warp_specialized_test        # Warp specialization test
│       ├── warp_final_opt_test          # Optimized warp test
│       ├── wgmma_test                   # Tensor core test
│       ├── marlin_gemv_benchmark        # Advanced w4a16f benchmark
│       └── [other compiled binaries]    # Additional test executables
│
├── docs/                          # Comprehensive documentation
│   ├── kernels/                  # Kernel design and optimization guides
│   │   ├── README.md                   # Kernel documentation overview
│   │   ├── gemv-optimization.md        # GEMV techniques for transformers
│   │   ├── quantization-techniques.md  # w4a16f implementation details
│   │   ├── warp-specialization.md      # Async compute/memory patterns
│   │   ├── tensor-cores.md             # WMMA/WGMMA utilization
│   │   └── memory-optimization.md      # Memory hierarchy techniques
│   │
│   ├── benchmarks/               # Benchmark analysis and results
│   │   ├── README.md                   # Performance analysis overview
│   │   ├── gemv-results.md            # GEMV performance breakdown
│   │   ├── gemm-comparison.md          # GEMM optimization results
│   │   ├── memory-analysis.md          # Bandwidth utilization analysis
│   │   └── accuracy-validation.md      # Numerical correctness methods
│   │
│   ├── performance/              # Performance analysis and profiling
│   │   ├── PROFILING_ANALYSIS.md       # NCU profiling results and insights
│   │   ├── benchmarks/                 # Benchmark-specific analysis
│   │   ├── kernels/                    # Kernel-specific performance data
│   │   └── setup/                      # Profiling configuration guides
│   │
│   └── setup/                    # Build and environment setup
│       ├── getting-started.md          # Complete setup walkthrough
│       ├── build-system.md             # Build system documentation
│       ├── environment-setup.md        # Development environment config
│       └── troubleshooting.md          # Common issues and solutions
│
├── scripts/                       # Build, profiling, and automation scripts
│   ├── profile_all.sh            # Automated NCU profiling
│   ├── benchmark_suite.sh         # Complete benchmark runner
│   ├── build_matrix.sh           # Multi-architecture builds
│   └── perf_analysis.py          # Performance data analysis
│
└── tests/                         # Validation and correctness tests
    ├── unit_tests.cu             # Individual kernel validation
    ├── integration_tests.cu      # End-to-end functionality
    ├── accuracy_tests.cu         # Numerical precision validation
    └── performance_regression.cu  # Performance monitoring
```

## 🔧 Build System Architecture

### Target Organization
```makefile
# Primary targets (stable, production-ready)
PRIMARY_TARGETS = ping_pong_test warp_specialized_test warp_final_opt_test wgmma_test

# Experimental targets (research, development)
EXPERIMENTAL_TARGETS = warp_specialized_optimized_test warp_simple_opt_test

# GEMV/quantization targets (transformer focus)
MARLIN_TARGETS = simple_gemv_benchmark marlin_gemv_benchmark
```

### Key Build Commands
```bash
# Quick development cycle
make gemv                        # Build and run GEMV benchmark
make all                        # Build primary kernels
make all-experimental           # Build everything including experimental

# Individual components
make build/bin/simple_gemv_benchmark    # GEMV optimization suite
make build/bin/ping_pong_test          # Double buffering test
make build/bin/wgmma_test             # Tensor core test

# Testing and profiling
make run-simple-gemv            # GEMV performance comparison
make run-all                    # All primary kernel tests
make profile-simple-gemv        # NCU profiling for GEMV
```

## 📚 Documentation Architecture

### Hierarchical Organization
1. **Project Level** (`README.md`) - Overview and quick start
2. **Component Level** (`docs/*/README.md`) - Category overviews  
3. **Detail Level** (`docs/*/*.md`) - Specific techniques and analysis
4. **Code Level** - Inline comments and documentation

### Documentation Categories

**Kernel Design** (`docs/kernels/`)
- Architecture patterns and optimization techniques
- Algorithm explanations and implementation details  
- Performance characteristics and trade-offs

**Benchmark Analysis** (`docs/benchmarks/`)
- Performance results and comparisons
- Statistical analysis and trends
- Hardware-specific optimizations

**Performance Profiling** (`docs/performance/`)
- NCU profiling results and interpretation
- Bottleneck analysis and optimization guidance
- Memory and compute utilization metrics

**Setup and Configuration** (`docs/setup/`)
- Environment setup and dependency management
- Build system configuration and troubleshooting
- Development workflow and best practices

## 🎯 Research Workflow

### Development Cycle
1. **Design**: Research optimization technique
2. **Implement**: Create kernel in `src/kernels/`
3. **Benchmark**: Add test harness in `src/benchmarks/`
4. **Profile**: Use NCU for detailed analysis
5. **Document**: Record results in `docs/performance/`
6. **Iterate**: Refine based on profiling insights

### File Naming Conventions
```
Kernels:          technique_variant.cu
Benchmarks:       technique_benchmark.cu  
Documentation:    technique-analysis.md
Build Targets:    technique_test
```

### Version Control Integration
```bash
# Ignore build artifacts
build/
*.o
*.so

# Include all source and documentation
src/
docs/
external/
Makefile
README.md
```

## 🚀 Usage Examples

### Quick Start for New Users
```bash
# Clone and setup
git clone [repository] cuda-kernel-microbenchmarks
cd cuda-kernel-microbenchmarks

# Test installation
make gemv

# Expected output: GEMV performance comparison with ~405 GFLOPS
```

### Research Development
```bash
# Create new optimization
cp src/kernels/warp_specialized_kernel.cu src/kernels/my_optimization.cu
# Edit implementation...

# Add benchmark
cp src/benchmarks/simple_gemv_benchmark.cu src/benchmarks/my_optimization_benchmark.cu
# Update benchmark to test new kernel...

# Update build system
# Add targets to Makefile...

# Test and profile  
make build/bin/my_optimization_test
make profile-my-optimization
```

### Performance Analysis
```bash
# Comprehensive benchmarking
make run-all > results.txt
make benchmark > summary.txt

# Detailed profiling
make profile-simple-gemv > gemv_profile.txt
make profile-warp > warp_profile.txt

# Compare results across configurations
./scripts/benchmark_suite.sh > full_analysis.txt
```

## 🔍 Key Features

### Maintainability
- **Clear separation** of source, build, and documentation
- **Consistent naming** conventions across all components  
- **Automated builds** with dependency tracking
- **Comprehensive documentation** for all major components

### Extensibility  
- **Modular design** allows easy addition of new kernels
- **Template-based** benchmark structure for consistency
- **Flexible build system** supports various target configurations
- **Documentation framework** scales with project growth

### Research Support
- **Performance tracking** with automated profiling integration
- **Accuracy validation** framework for numerical correctness
- **Comparative analysis** tools for optimization assessment
- **Knowledge preservation** through comprehensive documentation

---

This structure provides a solid foundation for CUDA kernel research and development, supporting both individual learning and collaborative development efforts. The organization scales from simple experiments to comprehensive optimization studies while maintaining clarity and usability.