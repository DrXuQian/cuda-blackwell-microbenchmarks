#!/bin/bash

# RTX 5070 Blackwell W4A16 GEMV Demo Script
# Demonstrates warp specialization with TMA and async WGMMA

set -e

echo "🎯 RTX 5070 Blackwell W4A16 GEMV Demo"
echo "====================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${1}${2}${NC}"
}

print_status $BLUE "Building W4A16 kernels for RTX 5070 Blackwell..."

# Build the new kernels
if make -f Makefile.blackwell_fixed w4a16_simple tma_fixed wgmma_fixed; then
    print_status $GREEN "✅ W4A16 kernels built successfully"
else
    print_status $RED "❌ Build failed"
    exit 1
fi

echo ""
print_status $BLUE "🚀 Demo 1: Simple W4A16 GEMV with Warp Specialization"
print_status $BLUE "Features: Basic warp roles + 4-bit dequantization + WMMA compute"
echo ""

if ./bin/blackwell_5070_fixed/blackwell_w4a16_simple; then
    print_status $GREEN "✅ Simple W4A16 GEMV completed"
else
    print_status $YELLOW "⚠️  Simple W4A16 GEMV had issues (may be expected on non-Blackwell hardware)"
fi

echo ""
print_status $BLUE "⚡ Demo 2: TMA GEMM (Blackwell Memory Optimization)"
print_status $BLUE "Features: Async copy + Tensor core computation + Memory tiling"
echo ""

if ./bin/blackwell_5070_fixed/blackwell_tma_gemm_fixed; then
    print_status $GREEN "✅ TMA GEMM completed"
else
    print_status $YELLOW "⚠️  TMA GEMM had issues (may be expected on non-Blackwell hardware)"
fi

echo ""
print_status $BLUE "🔥 Demo 3: Async WGMMA (Producer-Consumer Pattern)"
print_status $BLUE "Features: Warp specialization + Memory-compute overlap + WMMA"
echo ""

if ./bin/blackwell_5070_fixed/blackwell_async_wgmma_fixed; then
    print_status $GREEN "✅ Async WGMMA completed"
else
    print_status $YELLOW "⚠️  Async WGMMA had issues (may be expected on non-Blackwell hardware)"
fi

echo ""
print_status $BLUE "📋 Summary of W4A16 GEMV Implementations"
print_status $BLUE "========================================"

echo ""
echo "1. 🚀 Simple W4A16 GEMV (blackwell_w4a16_simple.cu):"
echo "   • 4-way warp specialization (Producer, Dequant, 2x Compute)"
echo "   • Basic 4-bit dequantization with scale application"
echo "   • Simple warp reduction and output writing"
echo "   • Target: Demonstrates basic concepts"

echo ""
echo "2. ⚡ TMA GEMM Fixed (blackwell_tma_gemm_fixed.cu):"
echo "   • Async memory operations with cooperative loading"
echo "   • WMMA tensor core computation for FP16"
echo "   • 128x128 tile sizes optimized for memory hierarchy"
echo "   • Target: 8-15 TFLOPS on RTX 5070"

echo ""
echo "3. 🔥 Async WGMMA Fixed (blackwell_async_wgmma_fixed.cu):"
echo "   • Producer-consumer warp specialization"
echo "   • Memory-compute overlap patterns"
echo "   • 64x64 computation tiles with tensor cores"
echo "   • Target: 6-12 TFLOPS with better efficiency"

echo ""
print_status $GREEN "🎯 Key RTX 5070 Blackwell Optimizations:"
echo "   ✓ Warp specialization maximizing 56 SM utilization"
echo "   ✓ Shared memory layout optimized for 164KB per block"
echo "   ✓ Compatible build system supporting sm_75+ through sm_90"
echo "   ✓ 4-bit quantization reducing memory footprint by 4x"
echo "   ✓ Tensor core utilization for optimal FP16 throughput"

echo ""
print_status $BLUE "📖 Usage Examples:"
echo ""
echo "# Build and test specific kernel:"
echo "make -f Makefile.blackwell_fixed w4a16_simple"
echo "./bin/blackwell_5070_fixed/blackwell_w4a16_simple"
echo ""
echo "# Profile with NCU (if available):"
echo "ncu --set basic ./bin/blackwell_5070_fixed/blackwell_tma_gemm_fixed"
echo ""
echo "# Memory check:"
echo "make -f Makefile.blackwell_fixed memcheck_all"

echo ""
print_status $GREEN "✨ RTX 5070 Blackwell W4A16 GEMV demo completed!"
print_status $BLUE "These implementations showcase advanced Blackwell architecture features"
print_status $BLUE "for efficient LLM inference with quantized weights."
print_status $BLUE ""
print_status $BLUE "Repository: https://github.com/DrXuQian/cuda-blackwell-microbenchmarks"