#!/bin/bash

set -e

echo "🔧 CUTLASS Setup and Build Script"
echo "================================="
echo ""

CUTLASS_DIR="$HOME/cutlass"
BUILD_DIR="$CUTLASS_DIR/build"

# Check if CUTLASS exists
if [ ! -d "$CUTLASS_DIR" ]; then
    echo "📥 Cloning CUTLASS repository..."
    git clone https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
else
    echo "✅ CUTLASS repository found at $CUTLASS_DIR"
    echo "📡 Updating to latest version..."
    cd "$CUTLASS_DIR"
    git fetch origin
    git checkout main
    git pull origin main
fi

# Create build directory
echo "📁 Setting up build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "⚙️ Configuring CUTLASS with CMake..."
cmake .. \
    -DCUTLASS_NVCC_ARCHS="50;75;80;86;89;90" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    2>/dev/null || {
        echo "❌ CMake configuration failed"
        exit 1
    }

echo "✅ CMake configuration successful"
echo ""

# Build core examples to verify compilation
echo "🔨 Building CUTLASS core examples..."
echo "Building 00_basic_gemm..."
make 00_basic_gemm -j$(nproc) 2>/dev/null || {
    echo "❌ Failed to build 00_basic_gemm"
    exit 1
}

echo "Building 01_cutlass_utilities..."
make 01_cutlass_utilities -j$(nproc) 2>/dev/null || {
    echo "❌ Failed to build 01_cutlass_utilities"
    exit 1
}

echo "✅ Core examples built successfully"
echo ""

# Test the examples
echo "🧪 Testing CUTLASS examples..."
echo -n "Testing 00_basic_gemm: "
if ./examples/00_basic_gemm/00_basic_gemm &>/dev/null; then
    echo "✅ PASSED"
else
    echo "❌ FAILED"
fi

echo -n "Testing 01_cutlass_utilities: "
if ./examples/01_cutlass_utilities/01_cutlass_utilities &>/dev/null; then
    echo "✅ PASSED"
else
    echo "❌ FAILED"
fi

echo ""
echo "🎯 CUTLASS Setup Complete!"
echo "========================="
echo "CUTLASS Version: $(cd $CUTLASS_DIR && git describe --tags 2>/dev/null || echo 'main')"
echo "Install Path: $CUTLASS_DIR"
echo "Build Path: $BUILD_DIR"
echo "Architecture Support: SM50, SM75, SM80, SM86, SM89, SM90"
echo ""
echo "✅ CUTLASS is ready for use!"
echo "You can now build and run CUTLASS applications."
echo ""
echo "To use CUTLASS in your projects:"
echo "  export CUTLASS_PATH=$CUTLASS_DIR"
echo "  Include path: $CUTLASS_DIR/include"
echo "  Utils path: $CUTLASS_DIR/tools/util/include"