#!/bin/bash
# Collect default autotuning configurations for current GPU
# Uses the benchmark tool to generate optimal configs

set -e

echo "========================================="
echo "  Collecting Default Configurations"
echo "========================================="
echo ""

# Get GPU architecture
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
    echo "Detected GPU: $GPU_ARCH"
else
    echo "Error: rocminfo not found. Is ROCm installed?"
    exit 1
fi

# Verify build exists
if [ ! -f ./build/hip-img-fx-bench ]; then
    echo "Error: hip-img-fx-bench not found"
    echo "Please run: ninja -C build"
    exit 1
fi

# Clean slate (force fresh tuning)
echo "Removing existing cache and embedded cache..."
rm -f .autotune_cache.json

# Temporarily blank the embedded cache to force full retuning
EMBEDDED_CACHE="include/hip-img-fx/autotune/embedded_cache.h"
BACKUP_CACHE="${EMBEDDED_CACHE}.backup"
if [ -f "$EMBEDDED_CACHE" ]; then
    cp "$EMBEDDED_CACHE" "$BACKUP_CACHE"
    cat > "$EMBEDDED_CACHE" << 'EOF'
#pragma once
namespace imgfx::core::autotune {
inline const char* EMBEDDED_DEFAULT_CACHE_V2 = R"(
{"version": "2.0", "entries": []}
)";
}
EOF
    echo "Temporarily disabled embedded cache for clean benchmark"
    
    # Rebuild with empty cache
    echo "Rebuilding..."
    ninja -C build > /dev/null 2>&1
fi

# Run benchmarks to generate optimal configs
echo ""
echo "Running tests to generate configs for all filters and sizes..."
echo ""

# Create test images of different sizes to trigger all cache contexts
for size in 512 2048 4096; do
    convert -size ${size}x${size} xc:white /tmp/test_${size}.jpg 2>/dev/null || {
        echo "Warning: imagemagick 'convert' not available, using existing images"
        break
    }
done

# Test each filter at each size to populate cache
# This ensures all 9 entries (3 filters × 3 sizes) are generated
for filter in grayscale negative gaussian-blur; do
    for size in 512 2048 4096; do
        IMG="/tmp/test_${size}.jpg"
        if [ ! -f "$IMG" ]; then
            # Fallback to example images if test images weren't created
            IMG="examples/output_filter_${filter/_/-}/example_01.jpg"
        fi
        
        echo "Testing ${filter} at ${size}x${size}..."
        ./build/hip-img-fx --input "$IMG" --output /tmp/out.jpg --filter "$filter" > /dev/null 2>&1 || {
            echo "Warning: Failed to test $filter at $size"
        }
    done
done

echo ""
echo "Tuning complete!"

# Restore embedded cache
if [ -f "$BACKUP_CACHE" ]; then
    mv "$BACKUP_CACHE" "$EMBEDDED_CACHE"
    echo "Restored embedded cache"
fi

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
    echo "To embed this in the binary, run:"
    echo "  ./scripts/merge_default_caches.py $OUTPUT_FILE > include/hip-img-fx/autotune/embedded_cache.h"
    echo ""
else
    echo ""
    echo "Error: No cache file generated!"
    exit 1
fi
