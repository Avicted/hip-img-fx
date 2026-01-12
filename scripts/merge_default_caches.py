#!/usr/bin/env python3
"""
Merge multiple cache files into single embedded cache for C++.
Usage: ./merge_default_caches.py default_configs_*.json > embedded_cache_data.h
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def merge_caches(input_files):
    """Merge multiple cache files into single embedded cache."""
    merged = {
        "version": "2.0",
        "entries": []
    }
    
    # Track unique entries (gpu_arch, kernel_name, context)
    seen = set()
    
    for file_path in input_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            for entry in data.get("entries", []):
                # Create unique key
                key = (
                    entry.get("gpu_arch", ""),
                    entry.get("kernel_name", ""),
                    entry.get("context", "")
                )
                
                if key not in seen and all(key):
                    seen.add(key)
                    merged["entries"].append(entry)
                    
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}", file=sys.stderr)
            continue
    
    return merged

def format_as_cpp_header(cache_data):
    """Format JSON as C++ header with embedded string."""
    json_str = json.dumps(cache_data, indent=2)
    
    header = f"""#pragma once

/**
 * @file embedded_cache.h
 * @brief Embedded default autotuning cache
 * 
 * Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 * 
 * To regenerate:
 *   1. Run ./scripts/collect_default_configs.sh on each target GPU
 *   2. Run ./scripts/merge_default_caches.py default_configs_*.json > embedded_cache.h
 * 
 * Included GPUs:
"""
    
    # List unique GPU architectures
    gpus = sorted(set(entry["gpu_arch"] for entry in cache_data["entries"]))
    for gpu in gpus:
        header += f" *   - {gpu}\n"
    
    header += """ */

namespace imgfx::core::autotune {{

const char* EMBEDDED_DEFAULT_CACHE_V2 = R"(
{json_data}
)";

}}  // namespace imgfx::core::autotune
""".format(json_data=json_str)
    
    return header

def main():
    if len(sys.argv) < 2:
        print("Usage: merge_default_caches.py <input_files...>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  ./merge_default_caches.py default_configs_*.json > embedded_cache.h", file=sys.stderr)
        sys.exit(1)
    
    # Merge all input files
    input_files = sys.argv[1:]
    merged = merge_caches(input_files)
    
    if not merged["entries"]:
        print("Error: No valid cache entries found!", file=sys.stderr)
        sys.exit(1)
    
    # Output as C++ header
    header = format_as_cpp_header(merged)
    print(header)
    
    # Statistics to stderr
    print(f"", file=sys.stderr)
    print(f"✓ Merged {len(merged['entries'])} entries from {len(input_files)} files", file=sys.stderr)
    
    # Show breakdown by GPU
    from collections import Counter
    gpu_counts = Counter(entry["gpu_arch"] for entry in merged["entries"])
    for gpu, count in sorted(gpu_counts.items()):
        print(f"  {gpu}: {count} entries", file=sys.stderr)

if __name__ == "__main__":
    main()
