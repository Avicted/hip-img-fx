#pragma once

/**
 * @file embedded_cache.h
 * @brief Embedded default autotuning cache
 *
 * Generated: 2026-01-10 00:08:42
 *
 * To regenerate:
 *   1. Run ./scripts/collect_default_configs.sh on each target GPU
 *   2. Run ./scripts/merge_default_caches.py default_configs_*.json > embedded_cache.h
 *
 * Included GPUs:
 *   - gfx1030
 */

namespace imgfx::core::autotune
{

  inline const char *EMBEDDED_DEFAULT_CACHE_V2 = R"(
{
  "version": "2.0",
  "entries": [
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale_v2",
      "context": "large",
      "config": "block_x=16,block_y=8",
      "benchmark_time_ms": 0,
      "timestamp": ""
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale_v2",
      "context": "medium",
      "config": "block_x=16,block_y=4",
      "benchmark_time_ms": 0,
      "timestamp": ""
    },
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale_v2",
      "context": "small",
      "config": "block_x=32,block_y=4",
      "benchmark_time_ms": 0,
      "timestamp": ""
    }
  ]
}
)";

} // namespace imgfx::core::autotune
