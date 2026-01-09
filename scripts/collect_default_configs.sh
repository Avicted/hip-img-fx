#!/bin/bash
# Collect default autotuning configurations for current GPU
# This script runs validation tests to generate optimal configs

set -e

echo "========================================="
echo "  Collecting Default Configurations"
echo "========================================="
echo ""

# Get GPU architecture
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep "Name:" | head -1 | awk '{print $2}')
    echo "Detected GPU: $GPU_ARCH"
else
    echo "Error: rocminfo not found. Is ROCm installed?"
    exit 1
fi

# Verify build exists
if [ ! -f ./build/validate-grayscale-migration ]; then
    echo "Error: validate-grayscale-migration not found"
    echo "Please run: ninja -C build"
    exit 1
fi

# Clean slate (force fresh tuning)
echo "Removing existing cache..."
rm -f .autotune_cache.json

# Run validation to generate optimal configs
echo ""
echo "Running benchmarks (this may take 30-60 seconds)..."
echo ""
./build/validate-grayscale-migration > /dev/null 2>&1

# Check if cache was created
if [ -f .autotune_cache.json ]; then
    OUTPUT_FILE="default_configs_${GPU_ARCH}.json"
    cp .autotune_cache.json "$OUTPUT_FILE"
    
    echo ""
    echo "========================================="
    echo "  ✓ Success!"
    echo "========================================="
    echo ""
    echo "Saved configs to: $OUTPUT_FILE"
    echo ""
    echo "Contents:"
    cat "$OUTPUT_FILE"
    echo ""
else
    echo ""
    echo "Error: No cache file generated!"
    exit 1
fi
