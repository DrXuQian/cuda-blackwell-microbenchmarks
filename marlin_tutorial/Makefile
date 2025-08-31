# Marlin Tutorial Makefile
# Progressive 4-bit Quantized GEMV Learning Experience

# ============================================================================
# Configuration
# ============================================================================

# CUDA toolkit path
CUDA_PATH ?= /usr/local/cuda

# Compiler
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Architecture targets (focus on modern GPUs for Marlin)
ARCHS = 75 80 86 89 90

# CUDA architecture flags
CUDA_ARCH_FLAGS = $(foreach arch,$(ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

# Include directories
INCLUDES = -I$(CUDA_PATH)/include

# Compiler flags
NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3
NVCC_FLAGS += $(CUDA_ARCH_FLAGS)
NVCC_FLAGS += $(INCLUDES)
NVCC_FLAGS += -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1

# Linker flags
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand

# Output directory
BIN_DIR = bin
OBJ_DIR = obj

# Create directories
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# ============================================================================
# Tutorial steps and targets
# ============================================================================

# Tutorial chapters
STEPS = 01_4bit_fundamentals \
        02_naive_gemv \
        03_vectorized_access \
        04_warp_operations \
        05_warp_specialization \
        06_advanced_layouts \
        07_full_marlin

# Executable names
EXECUTABLES = $(BIN_DIR)/01_4bit_fundamentals \
              $(BIN_DIR)/02_naive_gemv \
              $(BIN_DIR)/03_vectorized_access \
              $(BIN_DIR)/04_warp_operations \
              $(BIN_DIR)/05_warp_specialization \
              $(BIN_DIR)/06_advanced_layouts \
              $(BIN_DIR)/07_full_marlin

# Source files
STEP1_SRC = 01_4bit_fundamentals/bit_packing.cu
STEP2_SRC = 02_naive_gemv/simple_gemv.cu
STEP3_SRC = 03_vectorized_access/vectorized_gemv.cu
STEP4_SRC = 04_warp_operations/warp_gemv.cu
STEP5_SRC = 05_warp_specialization/specialized_gemv.cu
STEP6_SRC = 06_advanced_layouts/layout_optimized_gemv.cu
STEP7_SRC = 07_full_marlin/marlin_complete.cu

# ============================================================================
# Build rules
# ============================================================================

.PHONY: all clean tutorial help test quick

# Default target
all: $(EXECUTABLES)

# Help target
help:
	@echo "Marlin Tutorial Build System"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build all tutorial steps"
	@echo "  tutorial  - Run complete interactive tutorial"
	@echo "  test      - Quick functionality test"
	@echo "  quick     - Build and test all steps quickly"
	@echo "  clean     - Clean build artifacts"
	@echo ""
	@echo "Individual steps:"
	@echo "  step1     - 4-bit fundamentals (bit packing/unpacking)"
	@echo "  step2     - Naive 4-bit GEMV implementation"
	@echo "  step3     - Vectorized memory access optimization"
	@echo "  step4     - Warp-level operations and shared memory"
	@echo "  step5     - Producer-consumer warp specialization"
	@echo "  step6     - Advanced memory layout optimization"
	@echo "  step7     - Complete Marlin implementation"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  Target architectures: $(ARCHS)"

# ============================================================================
# Individual step build rules
# ============================================================================

# Step 1: 4-bit Fundamentals
$(BIN_DIR)/01_4bit_fundamentals: $(STEP1_SRC)
	@echo "🔨 Building Step 1: 4-bit Fundamentals..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP1_SRC) $(LDFLAGS)
	@echo "✅ Step 1 built successfully"

step1: $(BIN_DIR)/01_4bit_fundamentals

# Step 2: Naive GEMV
$(BIN_DIR)/02_naive_gemv: $(STEP2_SRC)
	@echo "🔨 Building Step 2: Naive 4-bit GEMV..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP2_SRC) $(LDFLAGS)
	@echo "✅ Step 2 built successfully"

step2: $(BIN_DIR)/02_naive_gemv

# Step 3: Vectorized Access
$(BIN_DIR)/03_vectorized_access: $(STEP3_SRC)
	@echo "🔨 Building Step 3: Vectorized Memory Access..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP3_SRC) $(LDFLAGS)
	@echo "✅ Step 3 built successfully"

step3: $(BIN_DIR)/03_vectorized_access

# Step 4: Warp Operations
$(BIN_DIR)/04_warp_operations: $(STEP4_SRC)
	@echo "🔨 Building Step 4: Warp-Level Operations..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP4_SRC) $(LDFLAGS)
	@echo "✅ Step 4 built successfully"

step4: $(BIN_DIR)/04_warp_operations

# Step 5: Warp Specialization
$(BIN_DIR)/05_warp_specialization: $(STEP5_SRC)
	@echo "🔨 Building Step 5: Warp Specialization..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP5_SRC) $(LDFLAGS)
	@echo "✅ Step 5 built successfully"

step5: $(BIN_DIR)/05_warp_specialization

# Step 6: Advanced Layouts
$(BIN_DIR)/06_advanced_layouts: $(STEP6_SRC)
	@echo "🔨 Building Step 6: Advanced Memory Layouts..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP6_SRC) $(LDFLAGS)
	@echo "✅ Step 6 built successfully"

step6: $(BIN_DIR)/06_advanced_layouts

# Step 7: Full Marlin
$(BIN_DIR)/07_full_marlin: $(STEP7_SRC)
	@echo "🔨 Building Step 7: Complete Marlin Implementation..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(STEP7_SRC) $(LDFLAGS)
	@echo "✅ Step 7 built successfully"

step7: $(BIN_DIR)/07_full_marlin

# ============================================================================
# Tutorial execution
# ============================================================================

# Run complete interactive tutorial
tutorial: all
	@echo ""
	@echo "🧠 Starting Complete Marlin Tutorial"
	@echo "===================================="
	@echo ""
	@echo "This tutorial will teach you to master 4-bit quantized GEMV optimization."
	@echo "Each step builds upon the previous, from fundamentals to production-ready kernels."
	@echo ""
	@read -p "Press Enter to begin Step 1: 4-bit Fundamentals..."
	@echo ""
	$(BIN_DIR)/01_4bit_fundamentals
	@echo ""
	@read -p "Press Enter to continue to Step 2: Naive 4-bit GEMV..."
	@echo ""
	$(BIN_DIR)/02_naive_gemv
	@echo ""
	@read -p "Press Enter to continue to Step 3: Vectorized Memory Access..."
	@echo ""
	$(BIN_DIR)/03_vectorized_access
	@echo ""
	@read -p "Press Enter to continue to Step 4: Warp-Level Operations..."
	@echo ""
	$(BIN_DIR)/04_warp_operations
	@echo ""
	@read -p "Press Enter to continue to Step 5: Warp Specialization..."
	@echo ""
	$(BIN_DIR)/05_warp_specialization
	@echo ""
	@read -p "Press Enter to continue to Step 6: Advanced Memory Layouts..."
	@echo ""
	$(BIN_DIR)/06_advanced_layouts
	@echo ""
	@read -p "Press Enter to begin final Step 7: Complete Marlin Implementation..."
	@echo ""
	$(BIN_DIR)/07_full_marlin
	@echo ""
	@echo "🎉 CONGRATULATIONS! 🎉"
	@echo "You have mastered 4-bit quantized GEMV optimization!"
	@echo "You're now ready to optimize quantized LLM inference pipelines."

# Quick test of all steps
quick: all
	@echo "🧪 Quick Test: Running all tutorial steps..."
	@echo "Step 1: 4-bit Fundamentals"
	@$(BIN_DIR)/01_4bit_fundamentals > /dev/null && echo "✅ Step 1 PASS" || echo "❌ Step 1 FAIL"
	@echo "Step 2: Naive GEMV"  
	@$(BIN_DIR)/02_naive_gemv > /dev/null && echo "✅ Step 2 PASS" || echo "❌ Step 2 FAIL"
	@echo "Step 3: Vectorized Access"
	@$(BIN_DIR)/03_vectorized_access > /dev/null && echo "✅ Step 3 PASS" || echo "❌ Step 3 FAIL"
	@echo "Step 4: Warp Operations"
	@$(BIN_DIR)/04_warp_operations > /dev/null && echo "✅ Step 4 PASS" || echo "❌ Step 4 FAIL"
	@echo "Step 5: Warp Specialization"
	@$(BIN_DIR)/05_warp_specialization > /dev/null && echo "✅ Step 5 PASS" || echo "❌ Step 5 FAIL"
	@echo "Step 6: Advanced Layouts"
	@$(BIN_DIR)/06_advanced_layouts > /dev/null && echo "✅ Step 6 PASS" || echo "❌ Step 6 FAIL"
	@echo "Step 7: Full Marlin"
	@$(BIN_DIR)/07_full_marlin > /dev/null && echo "✅ Step 7 PASS" || echo "❌ Step 7 FAIL"
	@echo "🎯 Quick test completed!"

# Quick functionality test
test: $(BIN_DIR)/01_4bit_fundamentals
	@echo "🧪 Running functionality test..."
	$(BIN_DIR)/01_4bit_fundamentals
	@echo "✅ Test completed successfully"

# ============================================================================
# Utility targets
# ============================================================================

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BIN_DIR) $(OBJ_DIR)
	@echo "✅ Clean completed"

# Build for specific architectures
sm_75: CUDA_ARCH_FLAGS = -gencode arch=compute_75,code=sm_75
sm_75: all

sm_80: CUDA_ARCH_FLAGS = -gencode arch=compute_80,code=sm_80
sm_80: all

sm_86: CUDA_ARCH_FLAGS = -gencode arch=compute_86,code=sm_86
sm_86: all

sm_89: CUDA_ARCH_FLAGS = -gencode arch=compute_89,code=sm_89
sm_89: all

sm_90: CUDA_ARCH_FLAGS = -gencode arch=compute_90,code=sm_90
sm_90: all

# Debug build
debug: NVCC_FLAGS += -g -G -lineinfo
debug: all
	@echo "✅ Debug build completed"

# Profile build
profile: NVCC_FLAGS += -lineinfo --ptxas-options=-v
profile: all
	@echo "✅ Profile build completed"

# Memory check
memcheck: $(BIN_DIR)/01_4bit_fundamentals
	@echo "🔍 Running memory check..."
	cuda-memcheck $(BIN_DIR)/01_4bit_fundamentals

# Show build configuration
config:
	@echo "Marlin Tutorial Build Configuration:"
	@echo "==================================="
	@echo "CUDA_PATH:    $(CUDA_PATH)"
	@echo "NVCC:         $(NVCC)"
	@echo "Architectures: $(ARCHS)"
	@echo "NVCC_FLAGS:   $(NVCC_FLAGS)"
	@echo "LDFLAGS:      $(LDFLAGS)"

# Performance benchmark
benchmark: all
	@echo "🏃 Running performance benchmarks..."
	@echo "Step 2 vs Step 7 comparison:"
	@echo "Naive GEMV:"
	@$(BIN_DIR)/02_naive_gemv | grep -E "(GFLOPS|Time)" || true
	@echo "Full Marlin:"
	@$(BIN_DIR)/07_full_marlin | grep -E "(GFLOPS|Time)" || true

# Create source package
package:
	@echo "📦 Creating Marlin tutorial package..."
	tar czf marlin-tutorial.tar.gz --exclude=bin --exclude=obj --exclude=*.tar.gz .
	@echo "✅ Package created: marlin-tutorial.tar.gz"

# Version information
version:
	@echo "Marlin Tutorial Build System v1.0"
	@echo "==================================="
	@$(NVCC) --version | head -n 5
	@echo ""
	@echo "Tutorial steps: $(words $(STEPS)) progressive steps"
	@echo "Target architectures: $(ARCHS)"

# Tutorial structure
structure:
	@echo "Marlin Tutorial Structure:"
	@echo "=========================="
	@find . -name "*.cu" -o -name "*.md" -o -name "Makefile" | sort | sed 's/^/  /'

# End of Makefile