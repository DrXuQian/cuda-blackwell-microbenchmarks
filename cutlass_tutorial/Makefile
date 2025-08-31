# CUTLASS Tutorial Makefile
# Complete step-by-step CUTLASS learning experience

# ============================================================================
# Configuration
# ============================================================================

# CUDA toolkit path
CUDA_PATH ?= /usr/local/cuda

# CUTLASS path (adjust to your CUTLASS installation)
CUTLASS_PATH ?= /opt/nvidia/cutlass

# Compiler
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Architecture targets
ARCHS = 75 80 86 89 90

# CUDA architecture flags
CUDA_ARCH_FLAGS = $(foreach arch,$(ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

# Include directories
INCLUDES = -I$(CUTLASS_PATH)/include \
           -I$(CUTLASS_PATH)/tools/util/include \
           -I$(CUDA_PATH)/include \
           -Icommon

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
# Source files and targets
# ============================================================================

# Tutorial chapters
CHAPTERS = 01_basic_gemm \
           02_template_concepts \
           03_memory_hierarchy \
           04_thread_organization \
           05_epilogue_operations \
           06_advanced_fusion \
           07_cute_introduction

# Executable names
EXECUTABLES = $(BIN_DIR)/01_basic_gemm \
              $(BIN_DIR)/02_template_concepts \
              $(BIN_DIR)/03_memory_hierarchy \
              $(BIN_DIR)/04_thread_organization \
              $(BIN_DIR)/05_epilogue_operations \
              $(BIN_DIR)/06_advanced_fusion \
              $(BIN_DIR)/07_cute_introduction

# Source files
CH1_SRC = 01_basic_gemm/basic_gemm.cu
CH2_SRC = 02_template_concepts/template_exploration.cu
CH3_SRC = 03_memory_hierarchy/memory_tiling.cu
CH4_SRC = 04_thread_organization/thread_coordination.cu
CH5_SRC = 05_epilogue_operations/epilogue_fusion.cu
CH6_SRC = 06_advanced_fusion/advanced_patterns.cu
CH7_SRC = 07_cute_introduction/cute_basics.cu

# ============================================================================
# Build rules
# ============================================================================

.PHONY: all clean test tutorial help install-deps check-deps

# Default target
all: check-deps $(EXECUTABLES)

# Help target
help:
	@echo "CUTLASS Tutorial Build System"
	@echo "============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build all tutorial chapters"
	@echo "  tutorial      - Run complete tutorial (builds and executes all chapters)"
	@echo "  test          - Quick test of core functionality"
	@echo "  clean         - Clean build artifacts"
	@echo "  check-deps    - Verify dependencies are available"
	@echo "  install-deps  - Install/update dependencies (requires sudo)"
	@echo ""
	@echo "Individual chapters:"
	@echo "  chapter1      - Basic GEMM operations"
	@echo "  chapter2      - Template concepts exploration"
	@echo "  chapter3      - Memory hierarchy optimization"
	@echo "  chapter4      - Thread organization patterns"
	@echo "  chapter5      - Epilogue fusion techniques"
	@echo "  chapter6      - Advanced fusion patterns"
	@echo "  chapter7      - CuTe introduction"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  CUTLASS_PATH=$(CUTLASS_PATH)"
	@echo "  Target architectures: $(ARCHS)"

# Dependency checking
check-deps:
	@echo "🔍 Checking dependencies..."
	@if [ ! -f "$(CUDA_PATH)/bin/nvcc" ]; then \
		echo "❌ NVCC not found at $(CUDA_PATH)/bin/nvcc"; \
		echo "   Please set CUDA_PATH or install CUDA toolkit"; \
		exit 1; \
	fi
	@if [ ! -d "$(CUTLASS_PATH)/include" ]; then \
		echo "❌ CUTLASS headers not found at $(CUTLASS_PATH)/include"; \
		echo "   Please set CUTLASS_PATH or install CUTLASS"; \
		echo "   You can download CUTLASS from: https://github.com/NVIDIA/cutlass"; \
		exit 1; \
	fi
	@echo "✅ All dependencies found"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "📦 Installing CUTLASS tutorial dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake git
	@echo "✅ Base dependencies installed"
	@echo "📥 Please manually install:"
	@echo "   1. CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
	@echo "   2. CUTLASS: git clone https://github.com/NVIDIA/cutlass.git"

# ============================================================================
# Chapter build rules
# ============================================================================

# Chapter 1: Basic GEMM
$(BIN_DIR)/01_basic_gemm: $(CH1_SRC) common/utils.h
	@echo "🔨 Building Chapter 1: Basic GEMM..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH1_SRC) $(LDFLAGS)
	@echo "✅ Chapter 1 built successfully"

chapter1: $(BIN_DIR)/01_basic_gemm

# Chapter 2: Template Concepts
$(BIN_DIR)/02_template_concepts: $(CH2_SRC) common/utils.h
	@echo "🔨 Building Chapter 2: Template Concepts..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH2_SRC) $(LDFLAGS)
	@echo "✅ Chapter 2 built successfully"

chapter2: $(BIN_DIR)/02_template_concepts

# Chapter 3: Memory Hierarchy
$(BIN_DIR)/03_memory_hierarchy: $(CH3_SRC) common/utils.h
	@echo "🔨 Building Chapter 3: Memory Hierarchy..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH3_SRC) $(LDFLAGS)
	@echo "✅ Chapter 3 built successfully"

chapter3: $(BIN_DIR)/03_memory_hierarchy

# Chapter 4: Thread Organization
$(BIN_DIR)/04_thread_organization: $(CH4_SRC) common/utils.h
	@echo "🔨 Building Chapter 4: Thread Organization..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH4_SRC) $(LDFLAGS)
	@echo "✅ Chapter 4 built successfully"

chapter4: $(BIN_DIR)/04_thread_organization

# Chapter 5: Epilogue Operations
$(BIN_DIR)/05_epilogue_operations: $(CH5_SRC) common/utils.h
	@echo "🔨 Building Chapter 5: Epilogue Operations..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH5_SRC) $(LDFLAGS)
	@echo "✅ Chapter 5 built successfully"

chapter5: $(BIN_DIR)/05_epilogue_operations

# Chapter 6: Advanced Fusion
$(BIN_DIR)/06_advanced_fusion: $(CH6_SRC) common/utils.h
	@echo "🔨 Building Chapter 6: Advanced Fusion..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH6_SRC) $(LDFLAGS)
	@echo "✅ Chapter 6 built successfully"

chapter6: $(BIN_DIR)/06_advanced_fusion

# Chapter 7: CuTe Introduction
$(BIN_DIR)/07_cute_introduction: $(CH7_SRC) common/utils.h
	@echo "🔨 Building Chapter 7: CuTe Introduction..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CH7_SRC) $(LDFLAGS)
	@echo "✅ Chapter 7 built successfully"

chapter7: $(BIN_DIR)/07_cute_introduction

# ============================================================================
# Tutorial execution
# ============================================================================

# Run complete tutorial
tutorial: all
	@echo ""
	@echo "🎓 Starting Complete CUTLASS Tutorial"
	@echo "===================================="
	@echo ""
	@echo "This tutorial will take you through 7 chapters of CUTLASS mastery."
	@echo "Each chapter builds upon the previous one."
	@echo ""
	@read -p "Press Enter to begin Chapter 1: Basic GEMM..."
	@echo ""
	$(BIN_DIR)/01_basic_gemm
	@echo ""
	@read -p "Press Enter to continue to Chapter 2: Template Concepts..."
	@echo ""
	$(BIN_DIR)/02_template_concepts
	@echo ""
	@read -p "Press Enter to continue to Chapter 3: Memory Hierarchy..."
	@echo ""
	$(BIN_DIR)/03_memory_hierarchy
	@echo ""
	@read -p "Press Enter to continue to Chapter 4: Thread Organization..."
	@echo ""
	$(BIN_DIR)/04_thread_organization
	@echo ""
	@read -p "Press Enter to continue to Chapter 5: Epilogue Operations..."
	@echo ""
	$(BIN_DIR)/05_epilogue_operations
	@echo ""
	@read -p "Press Enter to continue to Chapter 6: Advanced Fusion..."
	@echo ""
	$(BIN_DIR)/06_advanced_fusion
	@echo ""
	@read -p "Press Enter to begin final Chapter 7: CuTe Introduction..."
	@echo ""
	$(BIN_DIR)/07_cute_introduction
	@echo ""
	@echo "🎉 CONGRATULATIONS! 🎉"
	@echo "You have completed the entire CUTLASS tutorial!"
	@echo "You are now ready to build high-performance GPU kernels."

# Quick test
test: $(BIN_DIR)/01_basic_gemm
	@echo "🧪 Running quick functionality test..."
	$(BIN_DIR)/01_basic_gemm
	@echo "✅ Test completed successfully"

# ============================================================================
# Utility targets
# ============================================================================

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BIN_DIR) $(OBJ_DIR)
	@echo "✅ Clean completed"

# Show build configuration
config:
	@echo "Build Configuration:"
	@echo "==================="
	@echo "CUDA_PATH:    $(CUDA_PATH)"
	@echo "CUTLASS_PATH: $(CUTLASS_PATH)"
	@echo "NVCC:         $(NVCC)"
	@echo "Architectures: $(ARCHS)"
	@echo "NVCC_FLAGS:   $(NVCC_FLAGS)"
	@echo "LDFLAGS:      $(LDFLAGS)"

# Check CUDA device
device-info:
	@echo "🔍 CUDA Device Information:"
	@$(CUDA_PATH)/bin/nvidia-smi || echo "nvidia-smi not found"
	@$(CUDA_PATH)/extras/demo_suite/deviceQuery || echo "deviceQuery not found"

# Debug build (with debug symbols)
debug: NVCC_FLAGS += -g -G -lineinfo
debug: all
	@echo "✅ Debug build completed"

# Profile build (optimized for profiling)
profile: NVCC_FLAGS += -lineinfo --ptxas-options=-v
profile: all
	@echo "✅ Profile build completed"

# ============================================================================
# Advanced targets
# ============================================================================

# Build for specific architecture only
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

# Build with different optimization levels
O0: NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O0 -g
O0: all

O2: NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O2
O2: all

# Memory debugging
memcheck: all
	@echo "🔍 Running memory check on Chapter 1..."
	cuda-memcheck $(BIN_DIR)/01_basic_gemm

# Install tutorial system-wide (optional)
install: all
	@echo "📦 Installing CUTLASS tutorial..."
	sudo mkdir -p /usr/local/share/cutlass-tutorial
	sudo cp -r . /usr/local/share/cutlass-tutorial/
	sudo chmod -R 755 /usr/local/share/cutlass-tutorial
	@echo "✅ Tutorial installed to /usr/local/share/cutlass-tutorial"

# Uninstall tutorial
uninstall:
	@echo "🗑️  Uninstalling CUTLASS tutorial..."
	sudo rm -rf /usr/local/share/cutlass-tutorial
	@echo "✅ Tutorial uninstalled"

# Create source package
package:
	@echo "📦 Creating source package..."
	tar czf cutlass-tutorial.tar.gz --exclude=bin --exclude=obj --exclude=*.tar.gz .
	@echo "✅ Package created: cutlass-tutorial.tar.gz"

# ============================================================================
# Information targets
# ============================================================================

.PHONY: version
version:
	@echo "CUTLASS Tutorial Build System v1.0"
	@echo "===================================="
	@$(NVCC) --version | head -n 5
	@echo ""
	@echo "Tutorial chapters: $(words $(CHAPTERS)) chapters"
	@echo "Target architectures: $(ARCHS)"

# Show makefile structure
structure:
	@echo "Tutorial Structure:"
	@echo "=================="
	@find . -name "*.cu" -o -name "*.h" -o -name "Makefile*" | sort | sed 's/^/  /'

# End of Makefile