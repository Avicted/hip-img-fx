# HIP Kernel Autotuning Framework Guide

A complete guide to the HIP Image FX autotuning framework that automatically finds optimal GPU kernel configurations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Using the Framework](#using-the-framework)
4. [API Reference](#api-reference)
5. [Advanced Topics](#advanced-topics)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### What is Autotuning?

Autotuning automatically finds the fastest GPU kernel launch configuration for your specific hardware. Instead of using fixed block sizes, the framework tests multiple configurations and selects the optimal one.

### No Code Changes Required!

For existing filters (grayscale, negative, gaussian_blur), autotuning works automatically:

```bash
./build/hip-img-fx --input photo.jpg --output result.jpg --filter grayscale
```

**First run**: Autotuning benchmarks ~20 configurations (~100-200ms overhead)  
**Subsequent runs**: Uses cached optimal configuration (instant, zero overhead)

### Example Output

```
[AutoTune] Tuning 'grayscale' on gfx1030...
  [64x1]   = 0.0382 ms
  [128x1]  = 0.0371 ms
  [256x1]  = 0.0377 ms
  [16x8]   = 0.0337 ms  ← Best!
[AutoTune] Selected [16x8], avg: 0.0337ms
[AutoTune] Saved to .autotune_cache.json
```

---

## Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
├─────────────────────────────────────────────────────┤
│              apply_grayscale_autotuned()             │
│              apply_negative_autotuned()              │
│              apply_gaussian_blur_autotuned()         │
├─────────────────────────────────────────────────────┤
│          TuningOrchestrator<KernelTraits>            │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  Three-Tier Caching System                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────┐ │  │
│  │  │ Thread-    │→│ Persistent │→│ Tuning │ │  │
│  │  │ Local      │  │ Cache      │  │ Run    │ │  │
│  │  │ (instant)  │  │ (JSON)     │  │ (bench)│ │  │
│  │  └────────────┘  └────────────┘  └────────┘ │  │
│  └──────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│              TuningBenchmarker                       │
│         (HIP event timing, statistics)               │
└─────────────────────────────────────────────────────┘
```

### Three-Tier Caching

1. **Thread-Local Cache** (L1)
   - In-memory `std::unordered_map`
   - Per-thread, instant lookup
   - Lifetime: Single program run

2. **Persistent Cache** (L2)
   - `.autotune_cache.json` file
   - GPU architecture + kernel name + context → config
   - Lifetime: Persists across runs

3. **Tuning Run** (L3 - Cache Miss)
   - Benchmark candidate configurations
   - Test 5-30 configs (depends on search space)
   - Select fastest, save to persistent cache

### Configuration Format

Configurations are 2D block dimensions:
- `[256x1]` = 256 threads in 1D layout
- `[16x8]` = 128 threads in 2D layout (16×8)
- `[16x16]` = 256 threads in 2D layout (16×16)

Different layouts affect:
- Memory coalescing
- Register usage
- Cache hit rates
- Wavefront utilization (AMD: 64 threads/wavefront)

---

## Using the Framework

### For Filter Users (No Code Changes)

Just run the application. Autotuning is transparent:

```bash
# First run: autotuning
./hip-img-fx --input photo.jpg --output result.jpg --filter grayscale
# Output: Tuning... Selected [16x8]

# Subsequent runs: cached
./hip-img-fx --input photo.jpg --output result.jpg --filter grayscale
# Output: Using cached config [16x8]
```

### For Kernel Authors (Implementing Custom Kernels)

#### Step 1: Define Kernel Traits

```cpp
struct MyKernelTraits {
    static constexpr const char* name() { return "my_kernel"; }
    
    // Arguments structure
    struct Args {
        const float* input;
        float* output;
        int size;
    };
    
    // Context for caching (must be comparable)
    struct Context {
        int size_category;  // e.g., small/medium/large
        
        bool operator==(const Context& other) const {
            return size_category == other.size_category;
        }
    };
    
    // Candidate generation
    static std::vector<imgfx::core::autotune::TuningConfig> 
    generate_candidates(const Context& ctx) {
        std::vector<imgfx::core::autotune::TuningConfig> configs;
        for (int bx : {64, 128, 256}) {
            for (int by : {1, 2, 4}) {
                configs.push_back(
                    imgfx::core::autotune::TuningConfig::create(bx, by)
                );
            }
        }
        return configs;
    }
    
    // Validation
    static bool is_valid_config(
        const imgfx::core::autotune::TuningConfig& cfg,
        const Args& args)
    {
        int threads = cfg.block_x() * cfg.block_y();
        return threads >= 64 && threads <= 1024 && threads % 64 == 0;
    }
    
    // Kernel launch
    static void launch(
        const imgfx::core::autotune::TuningConfig& cfg,
        const Args& args,
        hipStream_t stream)
    {
        dim3 block(cfg.block_x(), cfg.block_y(), 1);
        dim3 grid((args.size + block.x - 1) / block.x, 1, 1);
        
        hipLaunchKernelGGL(
            my_kernel,
            grid, block, 0, stream,
            args.input, args.output, args.size
        );
    }
};
```

#### Step 2: Use TuningOrchestrator

```cpp
#include <hip-img-fx/autotune/orchestrator.h>

void apply_my_filter_autotuned(
    const float* input,
    float* output,
    int size,
    hipStream_t stream)
{
    using namespace imgfx::core::autotune;
    
    // Static orchestrator (initialized once)
    static TuningOrchestrator<MyKernelTraits> orchestrator;
    
    // Prepare arguments
    MyKernelTraits::Args args;
    args.input = input;
    args.output = output;
    args.size = size;
    
    // Prepare context for caching
    MyKernelTraits::Context ctx;
    ctx.size_category = (size < 1000000) ? 0 : 1;
    
    // Get optimal config and launch
    orchestrator.launch_tuned(ctx, args, stream);
}
```

That's it! The orchestrator handles:
- Checking thread-local cache
- Loading from persistent cache
- Running tuning benchmarks if needed
- Saving results

---

## API Reference

### Core Classes

#### `TuningOrchestrator<KernelTraits>`

Main entry point for autotuning.

**Methods:**
```cpp
void launch_tuned(
    const typename KernelTraits::Context& ctx,
    const typename KernelTraits::Args& args,
    hipStream_t stream = 0,
    const TuningOptions& opts = TuningOptions::default_options()
);
```

**Parameters:**
- `ctx`: Context for cache lookup (e.g., image size category)
- `args`: Kernel arguments
- `stream`: HIP stream for execution
- `opts`: Tuning behavior options

#### `TuningConfig`

Represents a kernel launch configuration.

**Factory Methods:**
```cpp
static TuningConfig create(int block_x, int block_y);
static TuningConfig create_1d(int threads);
```

**Accessors:**
```cpp
int block_x() const;
int block_y() const;
bool is_valid() const;
```

#### `TuningOptions`

Controls tuning behavior.

**Presets:**
```cpp
static TuningOptions default_options();   // Balanced
static TuningOptions quiet();             // Minimal output
static TuningOptions conservative();      // More warmup/timing runs
static TuningOptions aggressive();        // Fast tuning
```

**Fields:**
```cpp
bool verbose = true;              // Print tuning progress
int num_warmup_runs = 5;          // Warmup iterations
int num_timing_runs = 10;         // Timing iterations
float early_exit_threshold = 1.5; // Stop if X times slower
```

### Kernel Traits Requirements

Your `KernelTraits` struct must provide:

**Required static members:**
```cpp
static constexpr const char* name();  // Unique kernel identifier
```

**Required nested types:**
```cpp
struct Args { /* kernel arguments */ };
struct Context { 
    /* cache key data */
    bool operator==(const Context&) const;
};
```

**Required static methods:**
```cpp
static std::vector<TuningConfig> generate_candidates(const Context&);
static bool is_valid_config(const TuningConfig&, const Args&);
static void launch(const TuningConfig&, const Args&, hipStream_t);
```

**Compile-time validation:**
The framework uses C++20 concepts to enforce requirements at compile-time. Invalid traits produce clear error messages.

---

## Advanced Topics

### Custom Tuning Options

Control tuning aggressiveness:

```cpp
auto opts = imgfx::core::autotune::TuningOptions::default_options();
opts.num_warmup_runs = 10;     // More warmup
opts.num_timing_runs = 20;     // More timing samples
opts.early_exit_threshold = 2.0; // Be more patient
opts.verbose = false;           // Silent mode

orchestrator.launch_tuned(ctx, args, stream, opts);
```

### Context-Aware Tuning

Different configurations may be optimal for different problem sizes:

```cpp
struct Context {
    enum class SizeClass { SMALL, MEDIUM, LARGE };
    SizeClass size_class;
    
    static Context from_size(int n) {
        if (n < 10000) return {SizeClass::SMALL};
        if (n < 1000000) return {SizeClass::MEDIUM};
        return {SizeClass::LARGE};
    }
    
    bool operator==(const Context& other) const {
        return size_class == other.size_class;
    }
};
```

The cache will store separate optimal configurations for each size class.

### Embedded Cache Initialization

For production deployments, embed pre-tuned configurations:

```cpp
#include <hip-img-fx/autotune/embedded_cache.h>

// Precomputed configurations
static const char* embedded_cache_json = R"({
  "version": "1.0",
  "entries": [
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "my_kernel",
      "block_x": 128,
      "block_y": 2
    }
  ]
})";

// Initialize before first use
imgfx::core::autotune::EmbeddedCacheInitializer<MyKernelTraits>::initialize(
    embedded_cache_json
);
```

This avoids tuning overhead in production.

### Multi-GPU Considerations

Each GPU architecture (e.g., `gfx1030`, `gfx90a`) stores separate cache entries. The framework automatically detects GPU architecture and uses the appropriate configuration.

For multi-GPU applications:
```cpp
// Set device before launching
hipSetDevice(gpu_id);
orchestrator.launch_tuned(ctx, args, stream);
```

---

## Performance Tuning

### Candidate Generation Strategy

**Too few candidates**: May miss optimal configuration  
**Too many candidates**: Longer tuning time

**Recommended approach:**
```cpp
static std::vector<TuningConfig> generate_candidates(const Context& ctx) {
    std::vector<TuningConfig> configs;
    
    // 1D configurations (simple)
    for (int bx : {64, 128, 256, 512, 1024}) {
        configs.push_back(TuningConfig::create_1d(bx));
    }
    
    // 2D configurations (better memory patterns)
    for (int bx : {16, 32, 64, 128}) {
        for (int by : {2, 4, 8, 16}) {
            if (bx * by <= 1024 && bx * by >= 64) {
                configs.push_back(TuningConfig::create(bx, by));
            }
        }
    }
    
    return configs;
}
```

**Target**: 15-30 candidates for comprehensive search

### Validation Guidelines

**Always validate**:
- Total threads ≥ 64 (minimum wavefront)
- Total threads ≤ 1024 (hardware limit)
- Total threads % 64 == 0 (wavefront aligned)

**Kernel-specific**:
- Shared memory requirements
- Register usage limits
- Minimum work per thread

```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    int threads = cfg.block_x() * cfg.block_y();
    
    // Basic AMD constraints
    if (threads < 64 || threads > 1024 || threads % 64 != 0)
        return false;
    
    // Kernel-specific: need at least 4 pixels per thread
    int work_per_thread = args.image_size / threads;
    if (work_per_thread < 4)
        return false;
    
    return true;
}
```

### Cache File Management

**Location**: `.autotune_cache.json` (project root)

**Format**:
```json
{
  "version": "1.0",
  "entries": [
    {
      "gpu_arch": "gfx1030",
      "kernel_name": "grayscale",
      "image_size_cat": "large",
      "block_x": 16,
      "block_y": 8
    }
  ]
}
```

**Management**:
```bash
# Force retune (delete cache)
rm .autotune_cache.json

# Backup optimal configs
cp .autotune_cache.json configs/optimal_gfx1030.json

# Share configs across team
git add .autotune_cache.json
```

---

## Troubleshooting

### Problem: "Empty candidate list" assertion

**Cause**: `generate_candidates()` returned empty vector

**Solution**: Check candidate generation logic
```cpp
static std::vector<TuningConfig> generate_candidates(const Context& ctx) {
    std::vector<TuningConfig> configs;
    // Must add at least one config!
    configs.push_back(TuningConfig::create_1d(256));
    return configs;
}
```

### Problem: "No valid candidates" assertion

**Cause**: All candidates rejected by `is_valid_config()`

**Solution**: Relax validation or add more candidates
```cpp
static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
    // Too strict: threads < 256
    // Better: threads >= 64 && threads <= 1024
    return cfg.block_x() * cfg.block_y() >= 64;
}
```

### Problem: "All benchmarks failed" assertion

**Cause**: Kernel crashes or returns errors for all configs

**Solution**: 
1. Test kernel launch manually with known-good config
2. Check for race conditions or memory errors
3. Enable verbose mode to see which configs fail
4. Add error checking in `launch()` method

### Problem: Slow tuning (>1 second)

**Cause**: Too many candidates or insufficient early exit

**Solution**: Reduce candidates or adjust options
```cpp
auto opts = TuningOptions::aggressive();  // Faster tuning
opts.num_warmup_runs = 3;
opts.num_timing_runs = 5;
opts.early_exit_threshold = 1.2;  // More aggressive pruning
```

### Problem: Suboptimal configuration selected

**Cause**: Insufficient timing samples or noisy measurements

**Solution**: Use conservative options
```cpp
auto opts = TuningOptions::conservative();
opts.num_warmup_runs = 10;
opts.num_timing_runs = 20;
```

### Debug Mode

For development, enable verbose output:

```cpp
auto opts = TuningOptions::default_options();
opts.verbose = true;  // See detailed tuning progress
```

**Disable in production**:
```cpp
// Set NDEBUG for release builds
// AUTOTUNE_ASSERT becomes no-op
// Silent operation (unless verbose=true)
```

---

## Examples

### Example: Simple Grayscale Filter

See [src/filters/grayscale_autotune.hip.cpp](../src/filters/grayscale_autotune.hip.cpp) for a complete working example.

### Example: Custom Reduction Kernel

```cpp
struct ReductionKernelTraits {
    static constexpr const char* name() { return "reduction"; }
    
    struct Args {
        const float* input;
        float* output;
        int n;
    };
    
    struct Context {
        enum { SMALL, LARGE } size_class;
        bool operator==(const Context& o) const { 
            return size_class == o.size_class; 
        }
    };
    
    static std::vector<TuningConfig> generate_candidates(const Context& ctx) {
        std::vector<TuningConfig> configs;
        if (ctx.size_class == Context::SMALL) {
            configs.push_back(TuningConfig::create_1d(256));
        } else {
            for (int threads : {256, 512, 1024}) {
                configs.push_back(TuningConfig::create_1d(threads));
            }
        }
        return configs;
    }
    
    static bool is_valid_config(const TuningConfig& cfg, const Args& args) {
        int threads = cfg.block_x() * cfg.block_y();
        return threads >= 64 && threads <= 1024 && threads % 64 == 0;
    }
    
    static void launch(
        const TuningConfig& cfg,
        const Args& args,
        hipStream_t stream)
    {
        int threads = cfg.block_x();
        int blocks = (args.n + threads - 1) / threads;
        
        hipLaunchKernelGGL(
            reduction_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            args.input, args.output, args.n
        );
    }
};
```

---

## Additional Resources

- **[Quick Reference Card](COMPILE_TIME_SAFETY_QUICK_REFERENCE.md)** - API cheat sheet
- **[Benchmark Results](BENCHMARK_RESULTS.md)** - Empirical performance data
- **[Examples](../examples/)** - Sample implementations
- **[Source Code](../include/hip-img-fx/autotune/)** - Framework implementation

---

## Version History

- **v0.2.0** (2026-01): Production-ready autotuning framework
  - Three-tier caching system
  - Compile-time safety with C++20 concepts
  - Comprehensive benchmarking engine
  - Persistent JSON cache

---

*For more information, see the main [README](../README.md).*
