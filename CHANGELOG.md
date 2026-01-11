# Changelog

All notable changes to HIP Image FX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-11

###  First Stable Release

HIP Image FX v1.0.0 introduces a production-ready autotuning framework for GPU kernels with comprehensive documentation and a stable public API.

### Added

#### Autotuning Framework
- **TuningOrchestrator** - Main autotuning API with three-tier caching system
  - Thread-local cache for instant lookup
  - Persistent JSON cache (`.autotune_cache.json`)
  - Automatic benchmarking with statistical analysis
- **Compile-time safety** - C++20 concepts enforce correct kernel traits implementation
- **Type-safe kernel integration** - ValidKernelTraits concept validates at compile time
- **Flexible tuning options** - Presets: default, quiet, conservative, aggressive
- **GPU architecture detection** - Automatic per-GPU configuration

#### Framework Features
- **TuningBenchmarker** - Sophisticated benchmarking engine
  - HIP event-based precise timing
  - Statistical analysis (mean, stddev, min, max)
  - Early-exit optimization (skip slow candidates)
  - Configurable warmup and timing runs
- **CacheStore** - Persistent configuration cache
  - JSON format for human-readable configs
  - GPU architecture + kernel name + context keys
  - Automatic save/load on orchestrator lifecycle
- **Embedded cache initialization** - Pre-seed optimal configurations for production
- **HIPEvent RAII wrapper** - Safe event management for custom kernels

#### Public API
- Installable headers in `include/hip-img-fx/autotune/`
  - `orchestrator.h` - Main TuningOrchestrator template
  - `tuning_config.h` - Configuration representation
  - `types.h` - TuningOptions and utilities
  - `kernel_traits_concepts.h` - ValidKernelTraits concept
  - `benchmarker.h` - Benchmarking engine
  - `cache_store.h` - Cache management
  - `embedded_cache.h` - Production cache initialization
  - `hip_event.h` - HIP event utilities
- **pkg-config support** - `hip-img-fx.pc` for easy integration

#### Application Features
- **Automatic GPU tuning** - Zero-configuration optimal block sizes
- **Three production filters** - Grayscale, negative, gaussian blur
- **Batch processing** - Configurable batch size for throughput optimization
- **CPU fallback** - OpenMP-accelerated CPU implementations
- **Fine-grained profiling** - H2D/Kernel/D2H timing breakdown

### Performance

- **12-18% speedup** from optimal configurations vs default 256-thread blocks
- **582× speedup** on Gaussian blur vs single-threaded CPU (4096×4096)
- **41× speedup** on Gaussian blur vs OpenMP (32 threads, 1024×1024)
- **~100-200ms** first-run tuning overhead (one-time cost)
- **Zero overhead** for cached configurations

### Documentation

- **[Autotuning Guide](docs/AUTOTUNING_GUIDE.md)** - Comprehensive framework guide
  - Quick start (5-minute intro)
  - Core concepts and architecture
  - Complete API reference
  - Advanced topics (custom tuning, multi-GPU)
  - Performance tuning strategies
  - Troubleshooting guide
- **[Documentation Hub](docs/README.md)** - Central navigation
- Updated main README with library usage instructions
- Archived detailed benchmarks and quick reference to docs/archive/

### Changed

- **API naming** - Removed experimental `_v2` suffix from all functions
  - `apply_grayscale_autotuned_v2` → `apply_grayscale_autotuned`
  - `apply_negative_autotuned_v2` → `apply_negative_autotuned`
  - `apply_gaussian_blur_autotuned_v2` → `apply_gaussian_blur_autotuned`
- **Kernel names** - Removed `_v2` suffix
  - `"grayscale_v2"` → `"grayscale"`
  - `"negative_v2"` → `"negative"`
  - `"gaussian_blur_v2"` → `"gaussian_blur"`
- **Header organization** - Moved public API to `include/` directory
- **Build system** - Enabled warning level 2, zero warnings
- **Documentation** - Consolidated 23 docs → 6 active docs + archive

### Removed

- **Old AutoTuner system** - Replaced with TuningOrchestrator (~900 lines removed)
- **Commented debug code** - Cleaned up experimental code paths
- **Historical documentation** - Moved 12 development docs to `docs/archive/`
  - Migration reports, phase reports, implementation notes
  - Preserved for reference but not needed for framework users

### Stability Guarantees (v1.0)

#### Stable & Supported
-  `TuningOrchestrator<KernelTraits>` template interface
-  `ValidKernelTraits` concept requirements
-  `TuningConfig` get/set/iteration API
-  `TuningOptions` configuration structure
-  Cache file format (JSON schema)
-  Compile-time validation behavior

#### Not Yet Stable
-  Benchmarking heuristics (early-exit thresholds may be tuned)
-  Cache pruning strategies (future enhancement)
-  Specific candidate generation patterns (kernel-specific)
-  Error message wording (can improve without breaking)

#### Explicitly Not Guaranteed
- Performance characteristics (hardware-dependent)
- Cache file locations (may become configurable)
- Verbose output formatting

### Build & Installation

**Requirements:**
- AMD ROCm 5.0+
- HIP-enabled AMD GPU
- Meson build system
- C++20 compiler

**Building:**
```bash
meson setup build --native-file native/hip.ini
ninja -C build
```

**Installing:**
```bash
meson install -C build
```

**Using in projects:**
```bash
pkg-config --cflags hip-img-fx
```

### Migration Guide

For users of the experimental v0.2.0 autotuning:

1. **Function names** - Remove `_v2` suffix from all function calls
2. **Include paths** - Update to `<hip-img-fx/autotune/orchestrator.h>`
3. **Cache invalidation** - Delete `.autotune_cache.json` (will retune once)

### Credits

Developed as a demonstration of production-grade GPU performance engineering with HIP.

### License

MIT License - See [LICENSE](LICENSE) for details.

---

## [0.2.0] - 2025-12-XX

### Added
- New TuningOrchestrator-based autotuning system
- Three-tier caching (thread-local, persistent, tuning)
- Compile-time safety with C++20 concepts
- Comprehensive benchmarking infrastructure

### Changed
- Improved kernel launch patterns
- Enhanced timing accuracy

---

## [0.1.0] - 2025-11-XX

### Added
- Initial HIP image processing framework
- Grayscale, negative, and Gaussian blur filters
- Basic GPU vs CPU benchmarking
- Batch processing support
