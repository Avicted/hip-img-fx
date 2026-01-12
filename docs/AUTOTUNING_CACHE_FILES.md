# Autotuning Cache Files

## File Overview

The autotuning system uses two types of cache files:

### 1. Runtime Cache: `.autotune_cache.json`
- **Purpose**: Stores optimal configurations discovered during program execution
- **Location**: Project root directory (gitignored)
- **Created by**: Main application (`hip-img-fx`)
- **Lifecycle**: 
  - Loads on startup (embedded cache used as fallback if missing)
  - Saves on program exit (only if new configs were added)
  - Accumulates entries from all filters

### 2. Default Config: `default_configs_<gpu_arch>.json`
- **Purpose**: Source file for embedding optimal configs into the binary
- **Location**: Project root directory (gitignored)
- **Created by**: `scripts/collect_default_configs.sh`
- **Example**: `default_configs_gfx1030.json` for AMD RX 6900 XT
- **Contents**: 9 entries (3 filters × 3 size contexts)

## Why Multiple Files?

- **`.autotune_cache.json`**: Dynamic runtime cache that grows as you use different filters/sizes
- **`default_configs_gfx1030.json`**: Static snapshot for a specific GPU, used to generate embedded defaults

## Generating Embedded Configs

1. Run collection script:
   ```bash
   ./scripts/collect_default_configs.sh
   ```
   - Detects GPU architecture (e.g., gfx1030)
   - Creates `default_configs_gfx1030.json`

2. Embed in binary:
   ```bash
   ./scripts/merge_default_caches.py default_configs_gfx1030.json > include/hip-img-fx/autotune/embedded_cache.h
   sed -i 's/^const char\*/inline const char*/' include/hip-img-fx/autotune/embedded_cache.h
   ```

3. Rebuild:
   ```bash
   ninja -C build
   ```

## Benchmark Tool Behavior

The benchmark tool (`hip-img-fx-bench`) **does not save** autotuning configs because:
- It creates multiple static orchestrators (one per filter)
- They interfere with each other during destruction
- This is controlled by `HIP_IMG_FX_NO_CACHE_SAVE=1` environment variable

The `scripts/run_benchmark.sh` script automatically sets this variable.

## Summary

- Use **collection script** to generate configs for embedding
- **Benchmark tool** for performance testing (doesn't save cache)
- **Main app** for normal usage (saves cache)
- Only one default config file per GPU architecture
