#pragma once

/**
 * @file embedded_cache.h
 * @brief Embedded default autotuning cache
 * 
 * Generated: 2026-01-11 13:47:01
 * 
 * To regenerate:
 *   1. Run ./scripts/collect_default_configs.sh on each target GPU
 *   2. Run ./scripts/merge_default_caches.py default_configs_*.json > embedded_cache.h
 * 
 * Included GPUs:
 *   - gfx1030
 */

namespace imgfx::core::autotune {

inline const char* EMBEDDED_DEFAULT_CACHE_V2 = R"(
{
  "version": "2.0",
  "entries": [
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "gaussian_blur",
      "context": "large_blur_large",
      "config": "block_x=32,block_y=16",
      "benchmark_time_ms": 4.36126,
      "timestamp": "2026-01-11T13:46:26"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "gaussian_blur",
      "context": "small_blur_large",
      "config": "block_x=32,block_y=16",
      "benchmark_time_ms": 0.0830922,
      "timestamp": "2026-01-11T13:46:25"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale",
      "context": "large",
      "config": "block_x=16,block_y=4",
      "benchmark_time_ms": 0.123892,
      "timestamp": "2026-01-11T13:46:24"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale",
      "context": "small",
      "config": "block_x=16,block_y=4",
      "benchmark_time_ms": 0.010348,
      "timestamp": "2026-01-11T13:46:23"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "negative",
      "context": "medium",
      "config": "block_x=16,block_y=4",
      "benchmark_time_ms": 0.0320441,
      "timestamp": "2026-01-11T13:46:24"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale",
      "context": "medium",
      "config": "block_x=16,block_y=8",
      "benchmark_time_ms": 0.0404881,
      "timestamp": "2026-01-11T13:46:24"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "negative",
      "context": "small",
      "config": "block_x=32,block_y=8",
      "benchmark_time_ms": 0.0097361,
      "timestamp": "2026-01-11T13:46:24"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "gaussian_blur",
      "context": "medium_blur_large",
      "config": "block_x=16,block_y=32",
      "benchmark_time_ms": 1.08833,
      "timestamp": "2026-01-11T13:46:25"
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "negative",
      "context": "large",
      "config": "block_x=16,block_y=4",
      "benchmark_time_ms": 0.0914443,
      "timestamp": "2026-01-11T13:46:25"
    }
  ]
}
)";

}  // namespace imgfx::core::autotune

