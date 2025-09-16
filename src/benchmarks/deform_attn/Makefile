# MS-Deformable Attention CUDA Kernels Makefile

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Architecture settings
GPU_ARCH ?= sm_90
NVCC_FLAGS = -std=c++17 -O3 -arch=$(GPU_ARCH)
NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo
NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda

# For Hopper-specific features (sm_90a)
ifeq ($(GPU_ARCH),sm_90a)
    NVCC_FLAGS += -arch=sm_90a
endif

# Directories
KERNEL_DIR = kernels
TEST_DIR = tests
BENCH_DIR = benchmarks
BUILD_DIR = build
DOCS_DIR = docs

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Kernel implementations
KERNELS = \
	$(BUILD_DIR)/deform_attn_simple \
	$(BUILD_DIR)/deform_attn_optimized \
	$(BUILD_DIR)/deform_attn_persistent \
	$(BUILD_DIR)/deform_attn_persistent_full \
	$(BUILD_DIR)/deform_attn_persistent_distributed \
	$(BUILD_DIR)/deform_attn_distributed \
	$(BUILD_DIR)/deform_attn_distributed_small \
	$(BUILD_DIR)/deform_attn_original \
	$(BUILD_DIR)/deform_attn_alternative \
	$(BUILD_DIR)/deform_attn_persistent_channelsplit \
	$(BUILD_DIR)/deform_attn_optimizations_suite \
	$(BUILD_DIR)/deform_attn_optimizations_v2 \
	$(BUILD_DIR)/deform_attn_persistent_bankopt

# Test programs
TESTS = \
	$(BUILD_DIR)/test_cluster \
	$(BUILD_DIR)/test_cluster_large \
	$(BUILD_DIR)/test_cluster_shm

# Benchmark programs
BENCHMARKS = \
	$(BUILD_DIR)/deform_attn_analysis_demo \
	$(BUILD_DIR)/benchmark_tma_vs_cooperative \
	$(BUILD_DIR)/deform_attn_tma_multilevel

# All targets
ALL_TARGETS = $(KERNELS) $(TESTS) $(BENCHMARKS)

# Default target
all: $(BUILD_DIR) $(ALL_TARGETS)
	@echo "✅ All targets built successfully!"
	@echo "📁 Executables are in $(BUILD_DIR)/"

# Main kernels (recommended for use)
main: $(BUILD_DIR) \
	$(BUILD_DIR)/deform_attn_simple \
	$(BUILD_DIR)/deform_attn_optimized \
	$(BUILD_DIR)/deform_attn_persistent \
	$(BUILD_DIR)/deform_attn_persistent_full
	@echo "✅ Main kernels built successfully!"

# Build rules for kernels
$(BUILD_DIR)/deform_attn_simple: $(KERNEL_DIR)/deform_attn_simple.cu | $(BUILD_DIR)
	@echo "🔨 Building simple kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_optimized: $(KERNEL_DIR)/deform_attn_optimized.cu | $(BUILD_DIR)
	@echo "🔨 Building optimized kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_persistent: $(KERNEL_DIR)/deform_attn_persistent.cu | $(BUILD_DIR)
	@echo "🔨 Building persistent kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_persistent_full: $(KERNEL_DIR)/deform_attn_persistent_full.cu | $(BUILD_DIR)
	@echo "🔨 Building persistent full-size kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_persistent_distributed: $(KERNEL_DIR)/deform_attn_persistent_distributed.cu | $(BUILD_DIR)
	@echo "🔨 Building persistent distributed kernel..."
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90a -o $@ $<

$(BUILD_DIR)/deform_attn_distributed: $(KERNEL_DIR)/deform_attn_distributed.cu | $(BUILD_DIR)
	@echo "🔨 Building distributed shared memory kernel..."
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90a -o $@ $<

$(BUILD_DIR)/deform_attn_distributed_small: $(KERNEL_DIR)/deform_attn_distributed_small.cu | $(BUILD_DIR)
	@echo "🔨 Building distributed small kernel..."
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90a -o $@ $<

$(BUILD_DIR)/deform_attn_original: $(KERNEL_DIR)/deform_attn_original.cu | $(BUILD_DIR)
	@echo "🔨 Building original kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_alternative: $(KERNEL_DIR)/deform_attn_alternative.cu | $(BUILD_DIR)
	@echo "🔨 Building alternative kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_persistent_channelsplit: $(KERNEL_DIR)/deform_attn_persistent_channelsplit.cu | $(BUILD_DIR)
	@echo "🔨 Building persistent channel-split kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_optimizations_suite: $(KERNEL_DIR)/deform_attn_optimizations_suite.cu | $(BUILD_DIR)
	@echo "🔨 Building optimization suite kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_optimizations_v2: $(KERNEL_DIR)/deform_attn_optimizations_v2.cu | $(BUILD_DIR)
	@echo "🔨 Building optimization v2 kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_persistent_bankopt: $(KERNEL_DIR)/deform_attn_persistent_bankopt.cu | $(BUILD_DIR)
	@echo "🔨 Building bank-optimized persistent kernel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build rules for tests
$(BUILD_DIR)/test_cluster: $(TEST_DIR)/test_cluster.cu | $(BUILD_DIR)
	@echo "🔨 Building cluster test..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/test_cluster_large: $(TEST_DIR)/test_cluster_large.cu | $(BUILD_DIR)
	@echo "🔨 Building large cluster test..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/test_cluster_shm: $(TEST_DIR)/test_cluster_shm.cu | $(BUILD_DIR)
	@echo "🔨 Building cluster shared memory test..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build rules for benchmarks
$(BUILD_DIR)/deform_attn_analysis_demo: $(BENCH_DIR)/deform_attn_analysis_demo.cu | $(BUILD_DIR)
	@echo "🔨 Building analysis demo..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/benchmark_tma_vs_cooperative: $(BENCH_DIR)/benchmark_tma_vs_cooperative.cu | $(BUILD_DIR)
	@echo "🔨 Building TMA benchmark..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR)/deform_attn_tma_multilevel: $(BENCH_DIR)/deform_attn_tma_multilevel.cu | $(BUILD_DIR)
	@echo "🔨 Building TMA multilevel..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Special targets
.PHONY: benchmark
benchmark: main $(BUILD_DIR)/deform_attn_analysis_demo
	@echo "🚀 Running benchmarks..."
	@echo "=== Simple Baseline ===" > benchmark_results.txt
	@$(BUILD_DIR)/deform_attn_simple >> benchmark_results.txt 2>&1
	@echo "\n=== Optimized ===" >> benchmark_results.txt
	@$(BUILD_DIR)/deform_attn_optimized >> benchmark_results.txt 2>&1
	@echo "\n=== Persistent ===" >> benchmark_results.txt
	@$(BUILD_DIR)/deform_attn_persistent >> benchmark_results.txt 2>&1
	@echo "\n=== Persistent Full Size ===" >> benchmark_results.txt
	@$(BUILD_DIR)/deform_attn_persistent_full >> benchmark_results.txt 2>&1
	@echo "\n=== Analysis ===" >> benchmark_results.txt
	@$(BUILD_DIR)/deform_attn_analysis_demo >> benchmark_results.txt 2>&1
	@echo "✅ Benchmark complete! Results in benchmark_results.txt"

.PHONY: test
test: $(TESTS)
	@echo "🧪 Running tests..."
	@$(BUILD_DIR)/test_cluster
	@echo "✅ All tests passed!"

# Individual kernel targets for convenience
.PHONY: simple optimized persistent persistent_full
simple: $(BUILD_DIR)/deform_attn_simple
	@echo "✅ Simple kernel ready: $(BUILD_DIR)/deform_attn_simple"

optimized: $(BUILD_DIR)/deform_attn_optimized
	@echo "✅ Optimized kernel ready: $(BUILD_DIR)/deform_attn_optimized"

persistent: $(BUILD_DIR)/deform_attn_persistent
	@echo "✅ Persistent kernel ready: $(BUILD_DIR)/deform_attn_persistent"

persistent_full: $(BUILD_DIR)/deform_attn_persistent_full
	@echo "✅ Persistent full-size kernel ready: $(BUILD_DIR)/deform_attn_persistent_full"
	@echo "📊 This kernel handles the original paper input sizes (48×19560×15422)"

# Clean targets
.PHONY: clean clean-all
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f benchmark_results.txt
	@echo "✅ Clean complete!"

clean-all: clean
	@echo "🧹 Cleaning all generated files..."
	rm -f *.o *.exe *.exp *.lib
	@echo "✅ Deep clean complete!"

# Help target
.PHONY: help
help:
	@echo "MS-Deformable Attention CUDA Kernels - Makefile Help"
	@echo "===================================================="
	@echo ""
	@echo "Main targets:"
	@echo "  make all              - Build all kernels, tests, and benchmarks"
	@echo "  make main             - Build main kernels (recommended set)"
	@echo "  make benchmark        - Build and run performance benchmarks"
	@echo "  make test             - Build and run tests"
	@echo ""
	@echo "Individual kernels:"
	@echo "  make simple           - Build baseline implementation"
	@echo "  make optimized        - Build optimized version"
	@echo "  make persistent       - Build persistent kernel (best performance)"
	@echo "  make persistent_full  - Build full-size input handler"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make clean-all        - Deep clean all generated files"
	@echo "  make help             - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  GPU_ARCH=sm_XX        - Set GPU architecture (default: sm_90)"
	@echo "  CUDA_PATH=/path       - Set CUDA toolkit path"
	@echo ""
	@echo "Examples:"
	@echo "  make GPU_ARCH=sm_86   - Build for RTX 3090"
	@echo "  make GPU_ARCH=sm_90a  - Build for H100 with cluster support"
	@echo ""

# Print configuration
.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  CUDA Path: $(CUDA_PATH)"
	@echo "  NVCC: $(NVCC)"
	@echo "  GPU Architecture: $(GPU_ARCH)"
	@echo "  Build Directory: $(BUILD_DIR)"
	@echo "  NVCC Flags: $(NVCC_FLAGS)"