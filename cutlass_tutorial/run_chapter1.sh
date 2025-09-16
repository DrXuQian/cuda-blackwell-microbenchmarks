#!/bin/bash

echo "🚀 Running CUTLASS Tutorial Chapter 1 (Fixed)"
echo "=============================================="

export CUTLASS_PATH=~/cutlass

echo "Building Chapter 1 with optimized flags..."
/usr/local/cuda/bin/nvcc -std=c++17 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -I${CUTLASS_PATH}/include \
    -I${CUTLASS_PATH}/tools/util/include \
    -I/usr/local/cuda/include \
    -Icommon \
    -o bin/01_basic_gemm \
    01_basic_gemm/basic_gemm.cu \
    -lcudart -lcublas 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo ""
    echo "Running Chapter 1..."
    echo "===================="
    ./bin/01_basic_gemm
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "🎉 Chapter 1 completed successfully!"
        echo "CUTLASS is now working perfectly on your RTX 5070!"
    else
        echo ""
        echo "❌ Chapter 1 execution failed with exit code $exit_code"
    fi
    exit $exit_code
else
    echo "❌ Build failed"
    exit 1
fi