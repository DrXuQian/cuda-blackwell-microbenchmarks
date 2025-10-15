NVCC = nvcc
CFLAGS = -std=c++17 -O3 -arch=sm_89 -lcublas -Isrc/utils -I/home/qianxu/cutlass/include

# Directory structure
KERNEL_DIR = src/kernels
BENCHMARK_DIR = src/benchmarks
UTILS_DIR = src/utils
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin

# Primary kernel targets
PING_PONG_TARGET = $(BIN_DIR)/ping_pong_test
WARP_SPEC_TARGET = $(BIN_DIR)/warp_specialized_test
WARP_FINAL_OPT_TARGET = $(BIN_DIR)/warp_final_opt_test
WGMMA_TARGET = $(BIN_DIR)/wgmma_test

# Experimental targets
WARP_OPT_TARGET = $(BIN_DIR)/warp_specialized_optimized_test
WARP_SIMPLE_OPT_TARGET = $(BIN_DIR)/warp_simple_opt_test

# Marlin GEMV targets
MARLIN_GEMV_TARGET = $(BIN_DIR)/marlin_gemv_benchmark
SIMPLE_GEMV_TARGET = $(BIN_DIR)/simple_gemv_benchmark

# Fusion targets
LAYERNORM_GEMM_FUSION_TARGET = $(BIN_DIR)/layernorm_gemm_fusion_benchmark
ENHANCED_LAYERNORM_GEMM_TARGET = $(BIN_DIR)/enhanced_layernorm_gemm_benchmark
SIMPLE_ASYNC_TARGET = $(BIN_DIR)/simple_async_benchmark

# All targets
PRIMARY_TARGETS = $(PING_PONG_TARGET) $(WARP_SPEC_TARGET) $(WARP_FINAL_OPT_TARGET) $(WGMMA_TARGET)
EXPERIMENTAL_TARGETS = $(WARP_OPT_TARGET) $(WARP_SIMPLE_OPT_TARGET)
MARLIN_TARGETS = $(MARLIN_GEMV_TARGET) $(SIMPLE_GEMV_TARGET)
FUSION_TARGETS = $(LAYERNORM_GEMM_FUSION_TARGET) $(ENHANCED_LAYERNORM_GEMM_TARGET) $(SIMPLE_ASYNC_TARGET)
ALL_TARGETS = $(PRIMARY_TARGETS) $(EXPERIMENTAL_TARGETS) $(MARLIN_TARGETS) $(FUSION_TARGETS)

# Kernel sources
PING_PONG_SRC = $(KERNEL_DIR)/ping_pong_kernel.cu
WARP_SPEC_SRC = $(KERNEL_DIR)/warp_specialized_kernel.cu
WARP_FINAL_OPT_SRC = $(KERNEL_DIR)/warp_specialized_final_opt.cu
WGMMA_SRC = $(KERNEL_DIR)/wgmma_kernel.cu

# Experimental sources
WARP_OPT_SRC = $(KERNEL_DIR)/warp_specialized_optimized.cu
WARP_SIMPLE_OPT_SRC = $(KERNEL_DIR)/warp_specialized_simple_opt.cu

# Benchmark sources
MARLIN_GEMV_SRC = $(BENCHMARK_DIR)/marlin_gemv_benchmark.cu
SIMPLE_GEMV_SRC = $(BENCHMARK_DIR)/simple_gemv_benchmark.cu

# Fusion sources
LAYERNORM_GEMM_FUSION_SRC = $(BENCHMARK_DIR)/layernorm_gemm_fusion_benchmark.cu
ENHANCED_LAYERNORM_GEMM_SRC = $(BENCHMARK_DIR)/enhanced_layernorm_gemm_benchmark.cu
SIMPLE_ASYNC_SRC = $(BENCHMARK_DIR)/simple_async_benchmark.cu

# Common utilities
COMMON_DEPS = $(UTILS_DIR)/common.h

# Create build directories
$(shell mkdir -p $(BIN_DIR))

# Default target - build primary kernels
all: $(PRIMARY_TARGETS)

# Build all including experimental and GEMV benchmarks
all-experimental: $(ALL_TARGETS)

# Primary kernel builds
$(PING_PONG_TARGET): $(PING_PONG_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(PING_PONG_TARGET) $(PING_PONG_SRC)

$(WARP_SPEC_TARGET): $(WARP_SPEC_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(WARP_SPEC_TARGET) $(WARP_SPEC_SRC)

$(WARP_FINAL_OPT_TARGET): $(WARP_FINAL_OPT_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(WARP_FINAL_OPT_TARGET) $(WARP_FINAL_OPT_SRC)

$(WGMMA_TARGET): $(WGMMA_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(WGMMA_TARGET) $(WGMMA_SRC)

# Experimental kernel builds
$(WARP_OPT_TARGET): $(WARP_OPT_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(WARP_OPT_TARGET) $(WARP_OPT_SRC)

$(WARP_SIMPLE_OPT_TARGET): $(WARP_SIMPLE_OPT_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(WARP_SIMPLE_OPT_TARGET) $(WARP_SIMPLE_OPT_SRC)

# GEMV benchmark builds
$(SIMPLE_GEMV_TARGET): $(SIMPLE_GEMV_SRC) $(COMMON_DEPS)
	$(NVCC) $(CFLAGS) -o $(SIMPLE_GEMV_TARGET) $(SIMPLE_GEMV_SRC)

$(MARLIN_GEMV_TARGET): $(MARLIN_GEMV_SRC) $(COMMON_DEPS) $(KERNEL_DIR)/marlin_gemv_optimized.cu external/marlin/marlin_cuda_kernel.cu
	$(NVCC) $(CFLAGS) -o $(MARLIN_GEMV_TARGET) $(MARLIN_GEMV_SRC)

# Fusion benchmark builds
$(LAYERNORM_GEMM_FUSION_TARGET): $(LAYERNORM_GEMM_FUSION_SRC) $(COMMON_DEPS) $(KERNEL_DIR)/layernorm_gemm_fusion.cu
	$(NVCC) $(CFLAGS) -o $(LAYERNORM_GEMM_FUSION_TARGET) $(LAYERNORM_GEMM_FUSION_SRC)

$(ENHANCED_LAYERNORM_GEMM_TARGET): $(ENHANCED_LAYERNORM_GEMM_SRC) $(COMMON_DEPS) $(KERNEL_DIR)/layernorm_gemm_fusion.cu $(KERNEL_DIR)/layernorm_gemm_async_pipeline.cu $(KERNEL_DIR)/naive_gemm.cu
	$(NVCC) $(CFLAGS) -o $(ENHANCED_LAYERNORM_GEMM_TARGET) $(ENHANCED_LAYERNORM_GEMM_SRC)

$(SIMPLE_ASYNC_TARGET): $(SIMPLE_ASYNC_SRC) $(COMMON_DEPS) $(KERNEL_DIR)/layernorm_gemm_async_pipeline.cu $(KERNEL_DIR)/naive_gemm.cu
	$(NVCC) $(CFLAGS) -lcurand -o $(SIMPLE_ASYNC_TARGET) $(SIMPLE_ASYNC_SRC)

# Run primary kernel tests
run-ping-pong: $(PING_PONG_TARGET)
	@echo "🔄 Running Ping-Pong Kernel Test:"
	@echo "================================="
	@./$(PING_PONG_TARGET)

run-warp-specialized: $(WARP_SPEC_TARGET)
	@echo "🔄 Running Warp Specialized Kernel Test:"
	@echo "========================================"
	@./$(WARP_SPEC_TARGET)

run-warp-optimized: $(WARP_FINAL_OPT_TARGET)
	@echo "🔄 Running Final Optimized Warp Specialized Test:"
	@echo "================================================="
	@./$(WARP_FINAL_OPT_TARGET)

run-wgmma: $(WGMMA_TARGET)
	@echo "🔄 Running WGMMA Kernel Test:"
	@echo "============================"
	@./$(WGMMA_TARGET)

# Run experimental kernels
run-experimental: $(EXPERIMENTAL_TARGETS)
	@echo "🧪 Running Experimental Kernels:"
	@echo "================================"
	@echo "Warp Specialized Optimized (v1):"
	@./$(WARP_OPT_TARGET)
	@echo
	@echo "Warp Specialized Simple Optimized:"
	@./$(WARP_SIMPLE_OPT_TARGET)

# Run GEMV benchmarks
run-simple-gemv: $(SIMPLE_GEMV_TARGET)
	@echo "🚀 Running Simple GEMV Optimization Benchmark:"
	@echo "=============================================="
	@./$(SIMPLE_GEMV_TARGET)

run-marlin-gemv: $(MARLIN_GEMV_TARGET)
	@echo "🚀 Running Marlin GEMV w4a16f Benchmark:"
	@echo "======================================="
	@./$(MARLIN_GEMV_TARGET)

# Run fusion benchmarks
run-layernorm-gemm-fusion: $(LAYERNORM_GEMM_FUSION_TARGET)
	@echo "🚀 Running LayerNorm+GEMM Fusion Benchmark:"
	@echo "=========================================="
	@./$(LAYERNORM_GEMM_FUSION_TARGET)

run-enhanced-layernorm-gemm: $(ENHANCED_LAYERNORM_GEMM_TARGET)
	@echo "🚀 Running Enhanced LayerNorm+GEMM Async Pipeline Benchmark:"
	@echo "==========================================================="
	@./$(ENHANCED_LAYERNORM_GEMM_TARGET)

run-simple-async: $(SIMPLE_ASYNC_TARGET)
	@echo "🚀 Running Simple Async Pipeline Performance Test:"
	@echo "================================================="
	@./$(SIMPLE_ASYNC_TARGET)

# Run primary tests
run-all: $(PRIMARY_TARGETS)
	@echo "🚀 Running All Primary Kernel Tests:"
	@echo "===================================="
	@./$(PING_PONG_TARGET)
	@echo
	@./$(WARP_SPEC_TARGET)
	@echo
	@./$(WARP_FINAL_OPT_TARGET)
	@echo
	@./$(WGMMA_TARGET)
	@echo "✅ All primary tests completed!"

# Performance comparison
benchmark: $(PRIMARY_TARGETS)
	@echo "📊 Performance Benchmark Summary:"
	@echo "================================="
	@echo "Testing 1024x1024x1024 FP16 GEMM on RTX 4070 Ti Super..."
	@echo
	@./$(PING_PONG_TARGET) | grep "GFLOPS\|Status"
	@./$(WARP_SPEC_TARGET) | grep "GFLOPS\|Status"
	@./$(WARP_FINAL_OPT_TARGET) | grep "GFLOPS\|Status"
	@./$(WGMMA_TARGET) | grep "GFLOPS\|Status"

# NCU Profiling targets
profile-simple-gemv: $(SIMPLE_GEMV_TARGET)
	@echo "🔬 Profiling Simple GEMV kernels with NCU..."
	ncu --set full --target-processes all --launch-count 1 ./$(SIMPLE_GEMV_TARGET)

profile-enhanced-layernorm-gemm: $(ENHANCED_LAYERNORM_GEMM_TARGET)
	@echo "🔬 Profiling Enhanced LayerNorm+GEMM Async Pipeline with NCU..."
	ncu --set full --target-processes all --launch-count 1 ./$(ENHANCED_LAYERNORM_GEMM_TARGET)

profile-layernorm-gemm-metrics: $(ENHANCED_LAYERNORM_GEMM_TARGET)
	@echo "🔬 Profiling LayerNorm+GEMM with detailed metrics..."
	ncu --metrics smsp__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__warps_launched.sum,smsp__warp_issue_stalled_barrier.avg,smsp__warp_issue_stalled_membar.avg --launch-count 10 ./$(ENHANCED_LAYERNORM_GEMM_TARGET)

profile-marlin-gemv: $(MARLIN_GEMV_TARGET)
	@echo "🔬 Profiling Marlin GEMV with NCU..."
	ncu --set full --target-processes all --launch-count 1 ./$(MARLIN_GEMV_TARGET)

profile-marlin-gemv-metrics: $(MARLIN_GEMV_TARGET)
	@echo "🔬 Profiling Marlin GEMV with specific metrics..."
	ncu --metrics smsp__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__warps_launched.sum --launch-count 10 ./$(MARLIN_GEMV_TARGET)

profile-warp: $(WARP_SPEC_TARGET)
	@echo "🔬 Profiling Warp Specialized Kernel with NCU..."
	ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --launch-count 1 ./$(WARP_SPEC_TARGET)

profile-optimized: $(WARP_FINAL_OPT_TARGET)
	@echo "🔬 Profiling Optimized Warp Kernel with NCU..."
	ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --launch-count 1 ./$(WARP_FINAL_OPT_TARGET)

# Clean up
clean:
	rm -rf $(BUILD_DIR)
	@echo "🧹 Cleaned all build artifacts"

clean-experimental:
	rm -f $(EXPERIMENTAL_TARGETS)
	@echo "🧹 Cleaned experimental binaries"

clean-gemv:
	rm -f $(MARLIN_TARGETS)
	@echo "🧹 Cleaned GEMV benchmark binaries"

# Development helpers
compile-commands:
	@echo "Generating compile_commands.json for IDE integration..."
	@echo '[' > compile_commands.json
	@for src in $(KERNEL_DIR)/*.cu $(BENCHMARK_DIR)/*.cu; do \
		echo '  {' >> compile_commands.json; \
		echo '    "directory": "'$(PWD)'",' >> compile_commands.json; \
		echo '    "command": "$(NVCC) $(CFLAGS) -c $$src",' >> compile_commands.json; \
		echo '    "file": "$$src"' >> compile_commands.json; \
		echo '  },' >> compile_commands.json; \
	done
	@sed -i '$$s/,$$//' compile_commands.json
	@echo ']' >> compile_commands.json

# Help target
help:
	@echo "📖 Available targets:"
	@echo "===================="
	@echo "  all                     - Build primary kernels"
	@echo "  all-experimental        - Build all kernels (including experimental)"
	@echo "  run-ping-pong           - Test ping-pong kernel"
	@echo "  run-warp-specialized    - Test warp specialized kernel"
	@echo "  run-warp-optimized      - Test optimized warp specialized kernel"
	@echo "  run-wgmma               - Test WGMMA kernel"
	@echo "  run-all                 - Test all primary kernels"
	@echo "  run-experimental        - Test experimental kernels"
	@echo "  run-simple-gemv         - Run Simple GEMV optimization benchmark"
	@echo "  run-marlin-gemv         - Run Marlin GEMV w4a16f benchmark"
	@echo "  run-layernorm-gemm-fusion - Run LayerNorm+GEMM fusion benchmark"
	@echo "  benchmark               - Performance summary of primary kernels"
	@echo "  profile-simple-gemv     - Profile Simple GEMV kernels with NCU"
	@echo "  profile-marlin-gemv     - Profile Marlin GEMV with full NCU analysis"
	@echo "  profile-marlin-gemv-metrics - Profile Marlin GEMV with specific metrics"
	@echo "  profile-warp            - Profile warp specialized kernel with NCU"
	@echo "  profile-optimized       - Profile optimized kernel with NCU"
	@echo "  clean                   - Remove all build artifacts"
	@echo "  clean-experimental      - Remove experimental binaries"
	@echo "  clean-gemv              - Remove GEMV benchmark binaries"
	@echo "  compile-commands        - Generate IDE integration files"
	@echo "  help                    - Show this help"

# Quick development shortcuts
gemv: $(SIMPLE_GEMV_TARGET)
	@echo "🚀 Quick GEMV build and run:"
	@./$(SIMPLE_GEMV_TARGET)

# Legacy compatibility
run: run-all

.PHONY: all all-experimental clean clean-experimental clean-gemv run run-ping-pong run-warp-specialized run-warp-optimized run-wgmma run-all run-experimental run-simple-gemv run-marlin-gemv run-layernorm-gemm-fusion benchmark profile-simple-gemv profile-marlin-gemv profile-marlin-gemv-metrics profile-warp profile-optimized compile-commands gemv help
# OneFlow kernel targets
ONEFLOW_DIR = $(BENCHMARK_DIR)/oneflow_kernels
ONEFLOW_LAYERNORM_TARGET = $(BIN_DIR)/test_layernorm
ONEFLOW_SOFTMAX_TARGET = $(BIN_DIR)/test_softmax

# OneFlow sources
ONEFLOW_LAYERNORM_SRC = $(ONEFLOW_DIR)/test_layernorm.cu
ONEFLOW_SOFTMAX_SRC = $(ONEFLOW_DIR)/test_softmax.cu

# Build OneFlow kernels
$(ONEFLOW_LAYERNORM_TARGET): $(ONEFLOW_LAYERNORM_SRC)
	$(NVCC) $(CFLAGS) -o $(ONEFLOW_LAYERNORM_TARGET) $(ONEFLOW_LAYERNORM_SRC)

$(ONEFLOW_SOFTMAX_TARGET): $(ONEFLOW_SOFTMAX_SRC)
	$(NVCC) $(CFLAGS) -o $(ONEFLOW_SOFTMAX_TARGET) $(ONEFLOW_SOFTMAX_SRC)

# Run OneFlow kernel tests
run-test-layernorm: $(ONEFLOW_LAYERNORM_TARGET)
	@echo "🔄 Running OneFlow LayerNorm Test (default params):"
	@echo "==================================================="
	@./$(ONEFLOW_LAYERNORM_TARGET)

run-test-softmax: $(ONEFLOW_SOFTMAX_TARGET)
	@echo "🔄 Running OneFlow Softmax Test (default params):"
	@echo "================================================="
	@./$(ONEFLOW_SOFTMAX_TARGET)

run-oneflow-tests: $(ONEFLOW_LAYERNORM_TARGET) $(ONEFLOW_SOFTMAX_TARGET)
	@echo "🚀 Running All OneFlow Kernel Tests:"
	@echo "===================================="
	@echo
	@./$(ONEFLOW_LAYERNORM_TARGET)
	@echo
	@echo "---------------------------------------------------"
	@echo
	@./$(ONEFLOW_SOFTMAX_TARGET)
	@echo
	@echo "✅ All OneFlow tests completed!"

.PHONY: run-test-layernorm run-test-softmax run-oneflow-tests
