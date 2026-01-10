# Autotuning Architecture Refactor

## Executive Summary

This document proposes a comprehensive refactoring of the current autotuning implementation to create a reusable, extensible framework suitable for long-term performance optimization work. The refactoring addresses significant code duplication, improves separation of concerns, and creates a foundation for future optimizations.

---

## Current Implementation Analysis

### Identified Issues

#### 1. **Code Duplication Across Kernels**
Each kernel (grayscale, negative, gaussian_blur) has nearly identical:
- Launch wrapper structures (`*LaunchArgs`)
- Launch wrapper functions (`launch_*_kernel`)
- Autotuned application functions (`apply_*_autotuned`)
- Grid/block calculation logic
- Hardcoded warmup/timing parameters (5/10)

**Impact**: Adding new kernels requires ~80 lines of boilerplate per kernel.

#### 2. **Tight Coupling**
- Kernel execution logic tightly bound to autotuning framework
- AutoTuner directly calls kernel launch wrappers
- No abstraction between benchmarking and kernel semantics
- Launch arguments require manual `void*` casting with type-specific structs

#### 3. **Limited Extensibility**
- Only block dimensions (x, y) are tunable
- No support for:
  - Shared memory size tuning
  - Vectorization factors (e.g., `float4` vs `float`)
  - Loop unroll factors
  - Work-per-thread ratios
- Adding new tunable parameters requires modifying `KernelConfig` struct

#### 4. **Inflexible Candidate Generation**
- Fixed candidate list in `get_candidate_configs()`
- No kernel-specific constraints (e.g., shared memory limits)
- Cannot express parameter dependencies (e.g., "if block_x > 128, reduce unroll factor")

#### 5. **Cache Key Limitations**
- Cache keys only include: `gpu_arch`, `kernel_name`
- Missing context:
  - Image size categories (small/medium/large)
  - Kernel-specific parameters (e.g., `blur_amount`)
- `image_size_cat` field exists but unused

#### 6. **Minimal Runtime Overhead** (Current Strength)
- ✅ Cache lookup is O(n) but fast in practice
- ✅ Once tuned, overhead is negligible
- **Goal**: Preserve this property in refactor

---

## Proposed Architecture

### Design Principles

1. **Separation of Concerns**: Decouple kernel execution, benchmarking, decision logic, and caching
2. **Compile-Time Polymorphism**: Use templates and type traits to avoid runtime overhead
3. **Declarative Configuration**: Kernels describe their tunable parameters via traits
4. **Extensibility**: Support arbitrary tunable parameters without modifying core framework
5. **Type Safety**: Eliminate `void*` casting with type-safe parameter passing

---

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Code (Filters)                 │
│  - Declares kernel traits (parameters, constraints)     │
│  - Calls autotuner with type-safe parameters            │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              AutoTuner Framework (Core)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  TuningOrchestrator<KernelTraits>                │  │
│  │  - Coordinates tuning process                    │  │
│  │  - Dispatches to subsystems                      │  │
│  └─────┬────────────────────────────────────────────┘  │
│        │                                                │
│  ┌─────▼──────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ ParameterSpace │  │  Benchmarker │  │ CacheStore │ │
│  │ - Generate     │  │  - Warmup    │  │ - Load     │ │
│  │   candidates   │  │  - Time      │  │ - Save     │ │
│  │ - Apply        │  │  - Validate  │  │ - Lookup   │ │
│  │   constraints  │  │              │  │            │ │
│  └────────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│               Kernel Launcher (Filters)                  │
│  - Type-safe kernel invocation                          │
│  - Grid/block dimension calculation                     │
└─────────────────────────────────────────────────────────┘
```

---

## Concrete Interfaces and Types

### 1. Core Parameter Types

```cpp
namespace imgfx::core::autotune {

/**
 * @brief Represents a single tunable parameter with value and metadata
 */
template<typename T>
struct Parameter {
    std::string name;
    T value;
    T min_value;
    T max_value;
    
    Parameter(std::string n, T val, T min_v = T{}, T max_v = std::numeric_limits<T>::max())
        : name(std::move(n)), value(val), min_value(min_v), max_value(max_v) {}
    
    bool is_valid() const { return value >= min_value && value <= max_value; }
};

/**
 * @brief Configuration for a specific kernel launch
 * 
 * Extensible to support arbitrary tunable parameters.
 * Uses a variant-based approach for different parameter types.
 */
class TuningConfig {
public:
    using ParamValue = std::variant<int, float, bool>;
    
    void set(const std::string& name, ParamValue value);
    ParamValue get(const std::string& name) const;
    bool has(const std::string& name) const;
    
    // Common parameters (with type-safe accessors)
    int block_x() const;
    int block_y() const;
    int block_z() const { return get_or("block_z", 1); }
    
    // Serialization for caching
    std::string to_key_string() const;
    static TuningConfig from_key_string(const std::string& str);
    
    bool operator==(const TuningConfig& other) const;
    
private:
    std::unordered_map<std::string, ParamValue> params_;
    
    template<typename T>
    T get_or(const std::string& name, T default_val) const;
};

} // namespace imgfx::core::autotune
```

### 2. Kernel Traits (Type-Based Interface)

```cpp
namespace imgfx::filters {

/**
 * @brief Traits for grayscale kernel autotuning
 * 
 * Each kernel defines a traits class describing:
 * - Kernel-specific arguments
 * - Tunable parameter space
 * - Constraints
 * - Launch logic
 */
struct GrayscaleKernelTraits {
    // Unique kernel identifier
    static constexpr const char* name() { return "grayscale"; }
    
    // Kernel-specific arguments (type-safe)
    struct Args {
        const unsigned char* input;
        unsigned char* output;
        const imgfx::core::image_meta_t* metas;
        int num_images;
        size_t max_image_bytes;
    };
    
    // Context for cache key generation
    struct Context {
        size_t image_bytes;  // For size-based tuning
        
        std::string cache_key() const {
            // Categorize: small (<1MB), medium (<10MB), large (>=10MB)
            if (image_bytes < 1024 * 1024) return "small";
            if (image_bytes < 10 * 1024 * 1024) return "medium";
            return "large";
        }
    };
    
    // Define tunable parameter space
    static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates() {
        std::vector<imgfx::core::autotune::TuningConfig> configs;
        
        // 1D configs
        for (int bx : {64, 128, 256}) {
            imgfx::core::autotune::TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", 1);
            configs.push_back(cfg);
        }
        
        // 2D configs
        for (auto [bx, by] : std::vector<std::pair<int,int>>{{16,8}, {16,16}, {32,8}}) {
            imgfx::core::autotune::TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", by);
            configs.push_back(cfg);
        }
        
        return configs;
    }
    
    // Apply kernel-specific constraints
    static bool is_valid_config(const imgfx::core::autotune::TuningConfig& cfg, const Args& args) {
        int threads = cfg.block_x() * cfg.block_y();
        // Must be multiple of wavefront size (64)
        if (threads % 64 != 0) return false;
        // Reasonable thread count limits
        if (threads < 64 || threads > 1024) return false;
        return true;
    }
    
    // Launch kernel with given configuration
    static void launch(
        const imgfx::core::autotune::TuningConfig& cfg,
        const Args& args,
        hipStream_t stream)
    {
        int threads_per_block = cfg.block_x() * cfg.block_y();
        int blocks_x = (args.max_image_bytes + threads_per_block - 1) / threads_per_block;
        
        dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
        dim3 grid_dim(blocks_x, args.num_images, 1);
        
        hipLaunchKernelGGL(
            grayscale_kernel,
            grid_dim,
            block_dim,
            0,  // shared memory
            stream,
            args.input,
            args.output,
            args.metas,
            args.num_images);
    }
};

// Similar traits for other kernels...
struct GaussianBlurKernelTraits { /* ... */ };
struct NegativeKernelTraits { /* ... */ };

} // namespace imgfx::filters
```

### 3. Tuning Orchestrator (Core Framework)

```cpp
namespace imgfx::core::autotune {

/**
 * @brief Main autotuning orchestrator with type-safe interface
 * 
 * Template-based design eliminates void* casting and enables
 * compile-time optimization.
 */
template<typename KernelTraits>
class TuningOrchestrator {
public:
    using Args = typename KernelTraits::Args;
    using Context = typename KernelTraits::Context;
    
    /**
     * @brief Get or compute optimal configuration
     * 
     * @param args Kernel arguments (used for benchmarking)
     * @param ctx Context for cache key generation
     * @param options Tuning options (warmup runs, timing runs, etc.)
     * @return Optimal TuningConfig
     */
    TuningConfig get_or_tune(
        const Args& args,
        const Context& ctx,
        const TuningOptions& options = TuningOptions::defaults())
    {
        // Generate cache key
        CacheKey key{
            gpu_arch_,
            KernelTraits::name(),
            ctx.cache_key()
        };
        
        // Check cache
        if (auto cached = cache_.lookup(key)) {
            return *cached;
        }
        
        // Perform autotuning
        TuningConfig best_config = tune(args, options);
        
        // Store in cache
        cache_.insert(key, best_config);
        
        return best_config;
    }
    
    /**
     * @brief Execute kernel with optimal configuration
     * 
     * Convenience wrapper that combines get_or_tune + launch.
     */
    void execute(
        const Args& args,
        const Context& ctx,
        hipStream_t stream,
        const TuningOptions& options = TuningOptions::defaults())
    {
        TuningConfig config = get_or_tune(args, ctx, options);
        KernelTraits::launch(config, args, stream);
    }
    
private:
    TuningConfig tune(const Args& args, const TuningOptions& options);
    
    std::string gpu_arch_;
    CacheStore cache_;
};

/**
 * @brief Tuning options for benchmarking
 */
struct TuningOptions {
    int warmup_runs = 5;
    int timing_runs = 10;
    bool verbose = true;
    
    static TuningOptions defaults() { return TuningOptions{}; }
};

} // namespace imgfx::core::autotune
```

### 4. Benchmarking Subsystem

```cpp
namespace imgfx::core::autotune {

/**
 * @brief Result of benchmarking a single configuration
 */
struct BenchmarkResult {
    TuningConfig config;
    float avg_time_ms;
    float stddev_ms;
    bool valid;
    
    BenchmarkResult() : avg_time_ms(0.0f), stddev_ms(0.0f), valid(false) {}
    
    bool operator<(const BenchmarkResult& other) const {
        return avg_time_ms < other.avg_time_ms;
    }
};

/**
 * @brief Benchmarking engine for kernel configurations
 * 
 * Separated from orchestrator for testability and reuse.
 */
template<typename KernelTraits>
class Benchmarker {
public:
    using Args = typename KernelTraits::Args;
    
    /**
     * @brief Benchmark a single configuration
     */
    BenchmarkResult benchmark(
        const TuningConfig& config,
        const Args& args,
        hipStream_t stream,
        const TuningOptions& options)
    {
        // Validate configuration
        if (!KernelTraits::is_valid_config(config, args)) {
            return BenchmarkResult{};  // Invalid
        }
        
        // Warmup phase
        for (int i = 0; i < options.warmup_runs; ++i) {
            KernelTraits::launch(config, args, stream);
        }
        HIP_ERRCHK(hipStreamSynchronize(stream));
        
        // Timing phase
        std::vector<float> times;
        times.reserve(options.timing_runs);
        
        for (int i = 0; i < options.timing_runs; ++i) {
            HIPEvent start, end;
            start.record(stream);
            KernelTraits::launch(config, args, stream);
            end.record(stream);
            end.synchronize();
            
            times.push_back(HIPEvent::elapsed_time(start, end));
        }
        
        // Compute statistics
        BenchmarkResult result;
        result.config = config;
        result.avg_time_ms = compute_mean(times);
        result.stddev_ms = compute_stddev(times, result.avg_time_ms);
        result.valid = true;
        
        return result;
    }
    
    /**
     * @brief Benchmark all candidate configurations
     */
    std::vector<BenchmarkResult> benchmark_all(
        const std::vector<TuningConfig>& candidates,
        const Args& args,
        hipStream_t stream,
        const TuningOptions& options)
    {
        std::vector<BenchmarkResult> results;
        results.reserve(candidates.size());
        
        for (const auto& config : candidates) {
            auto result = benchmark(config, args, stream, options);
            if (result.valid) {
                results.push_back(result);
                
                if (options.verbose) {
                    printf("  [%dx%d] = %.4f ± %.4f ms\n",
                           config.block_x(), config.block_y(),
                           result.avg_time_ms, result.stddev_ms);
                }
            }
        }
        
        return results;
    }
    
private:
    static float compute_mean(const std::vector<float>& values);
    static float compute_stddev(const std::vector<float>& values, float mean);
};

} // namespace imgfx::core::autotune
```

### 5. Cache Store

```cpp
namespace imgfx::core::autotune {

/**
 * @brief Cache key for looking up tuned configurations
 */
struct CacheKey {
    std::string gpu_arch;
    std::string kernel_name;
    std::string context;  // Size category, parameter hash, etc.
    
    bool operator==(const CacheKey& other) const {
        return gpu_arch == other.gpu_arch &&
               kernel_name == other.kernel_name &&
               context == other.context;
    }
    
    std::string to_string() const {
        return gpu_arch + ":" + kernel_name + ":" + context;
    }
};

} // namespace std {
    template<>
    struct hash<imgfx::core::autotune::CacheKey> {
        size_t operator()(const imgfx::core::autotune::CacheKey& k) const {
            return std::hash<std::string>{}(k.to_string());
        }
    };
}

namespace imgfx::core::autotune {

/**
 * @brief Cache entry for serialization
 */
struct CacheEntry {
    CacheKey key;
    TuningConfig config;
    float benchmark_time_ms;  // For reference
    std::string timestamp;
};

/**
 * @brief On-disk cache management
 * 
 * Separated from orchestrator for independent testing and
 * potential multi-process scenarios.
 */
class CacheStore {
public:
    CacheStore();
    
    /**
     * @brief Load cache from disk
     */
    bool load(const std::string& path = ".autotune_cache.json");
    
    /**
     * @brief Save cache to disk
     */
    bool save(const std::string& path = ".autotune_cache.json") const;
    
    /**
     * @brief Lookup configuration in cache
     */
    std::optional<TuningConfig> lookup(const CacheKey& key) const;
    
    /**
     * @brief Insert or update configuration in cache
     */
    void insert(const CacheKey& key, const TuningConfig& config, float time_ms = 0.0f);
    
    /**
     * @brief Check if cache contains key
     */
    bool contains(const CacheKey& key) const;
    
    /**
     * @brief Get number of cached entries
     */
    size_t size() const { return cache_.size(); }
    
    /**
     * @brief Clear all cached entries
     */
    void clear() { cache_.clear(); }
    
private:
    std::unordered_map<CacheKey, CacheEntry> cache_;
    
    // JSON serialization helpers
    std::string serialize() const;
    void deserialize(const std::string& json_content);
};

} // namespace imgfx::core::autotune
```

---

## Migration Guide

### Step 1: Implement Core Framework

**Files to create:**
- `src/core/autotune/tuning_config.h` - `TuningConfig` class
- `src/core/autotune/tuning_config.cpp` - Implementation
- `src/core/autotune/cache_store.h` - `CacheStore` class
- `src/core/autotune/cache_store.cpp` - Implementation
- `src/core/autotune/benchmarker.h` - `Benchmarker` template
- `src/core/autotune/orchestrator.h` - `TuningOrchestrator` template
- `src/core/autotune/types.h` - Common types and utilities

**Estimated effort:** 2-3 days

### Step 2: Migrate Grayscale Kernel (Reference Implementation)

**Files to modify:**
- `src/filters/grayscale_autotune.hip.cpp` → Remove boilerplate
- `src/filters/filters.h` → Add `GrayscaleKernelTraits`

**New approach:**
```cpp
// In filters.h
struct GrayscaleKernelTraits {
    static constexpr const char* name() { return "grayscale"; }
    struct Args { /* ... */ };
    struct Context { /* ... */ };
    static std::vector<TuningConfig> generate_candidates();
    static bool is_valid_config(const TuningConfig& cfg, const Args& args);
    static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream);
};

// In grayscale_autotune.hip.cpp
void apply_grayscale_autotuned(
    const unsigned char* input,
    unsigned char* output,
    const imgfx::core::image_meta_t* metas,
    int num_images,
    size_t max_image_bytes,
    imgfx::core::AutoTuner& /* DEPRECATED */,
    hipStream_t stream)
{
    using namespace imgfx::core::autotune;
    
    // Create orchestrator (could be cached at higher level)
    static TuningOrchestrator<GrayscaleKernelTraits> orchestrator;
    
    // Prepare arguments
    GrayscaleKernelTraits::Args args{input, output, metas, num_images, max_image_bytes};
    GrayscaleKernelTraits::Context ctx{max_image_bytes};
    
    // Execute with autotuning
    orchestrator.execute(args, ctx, stream);
}
```

**Estimated effort:** 1 day

### Step 3: Migrate Remaining Kernels

Apply same pattern to:
- `negative_autotune.hip.cpp` → Define `NegativeKernelTraits`
- `gaussian_blur_autotune.hip.cpp` → Define `GaussianBlurKernelTraits`

Each kernel requires:
1. Define traits struct (~30 lines)
2. Update `apply_*_autotuned` function (~10 lines)
3. Remove old boilerplate (~70 lines)

**Estimated effort:** 0.5 days per kernel

### Step 4: Deprecate Old AutoTuner

**Files to modify:**
- `src/core/autotuning.h` → Mark deprecated
- `src/core/autotuning.cpp` → Keep for backward compatibility temporarily

**Migration path:**
1. Keep old `AutoTuner` class for 1 release cycle
2. Add deprecation warnings
3. Update all call sites to new framework
4. Remove in next major version

**Estimated effort:** 0.5 days

### Step 5: Extend for Advanced Parameters

**Example: Add vectorization tuning to grayscale**

```cpp
struct GrayscaleKernelTraits {
    // Add to candidate generation
    static std::vector<TuningConfig> generate_candidates() {
        std::vector<TuningConfig> configs;
        
        for (int bx : {64, 128, 256}) {
            for (int vec_width : {1, 2, 4}) {  // NEW: vectorization
                TuningConfig cfg;
                cfg.set("block_x", bx);
                cfg.set("block_y", 1);
                cfg.set("vec_width", vec_width);  // NEW
                configs.push_back(cfg);
            }
        }
        
        return configs;
    }
    
    // Update launch to use vec_width
    static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream) {
        int vec_width = cfg.get("vec_width");  // NEW
        
        // Launch specialized kernel based on vec_width
        if (vec_width == 4) {
            hipLaunchKernelGGL(grayscale_kernel_vec4, ...);
        } else if (vec_width == 2) {
            hipLaunchKernelGGL(grayscale_kernel_vec2, ...);
        } else {
            hipLaunchKernelGGL(grayscale_kernel, ...);
        }
    }
};
```

**No changes to core framework needed!**

---

## Runtime Overhead Analysis

### Current Implementation
- Cache lookup: O(n) linear search, typically n < 10
- Once cached: ~10ns overhead (config struct copy)
- **Total overhead:** Negligible after warmup

### Proposed Implementation
- Cache lookup: O(1) hash table lookup
- Once cached: ~15ns overhead (variant access + config copy)
- **Total overhead:** Still negligible, potentially faster

### Optimization: Static Caching

For maximum performance, add compile-time caching:

```cpp
template<typename KernelTraits>
class TuningOrchestrator {
public:
    TuningConfig get_or_tune(const Args& args, const Context& ctx, const TuningOptions& opts) {
        // Check thread-local static cache first (fastest path)
        thread_local static std::unordered_map<std::string, TuningConfig> fast_cache;
        
        std::string fast_key = KernelTraits::name() + ctx.cache_key();
        auto it = fast_cache.find(fast_key);
        if (it != fast_cache.end()) {
            return it->second;  // ~5ns overhead
        }
        
        // Fall back to persistent cache / tuning
        CacheKey key{gpu_arch_, KernelTraits::name(), ctx.cache_key()};
        TuningConfig config = /* ... cache lookup or tune ... */;
        
        fast_cache[fast_key] = config;  // Populate fast cache
        return config;
    }
};
```

**Result:** After first lookup, overhead drops to ~5ns (comparable to function call).

---

## Future Extensions

### 1. Multi-Dimensional Parameter Tuning

```cpp
struct AdvancedKernelTraits {
    static std::vector<TuningConfig> generate_candidates() {
        std::vector<TuningConfig> configs;
        
        for (auto [bx, by] : block_sizes) {
            for (int unroll : {1, 2, 4, 8}) {
                for (int smem_tiles : {0, 1, 2}) {
                    for (int vec_width : {1, 2, 4}) {
                        TuningConfig cfg;
                        cfg.set("block_x", bx);
                        cfg.set("block_y", by);
                        cfg.set("unroll_factor", unroll);
                        cfg.set("smem_tiles", smem_tiles);
                        cfg.set("vec_width", vec_width);
                        configs.push_back(cfg);
                    }
                }
            }
        }
        
        return configs;
    }
};
```

### 2. Adaptive Candidate Pruning

```cpp
class SmartCandidateGenerator {
public:
    std::vector<TuningConfig> generate(const KernelTraits::Args& args) {
        // Start with coarse search
        auto coarse = generate_coarse_grid();
        auto coarse_results = benchmark(coarse);
        
        // Identify promising regions
        auto top_3 = select_top_n(coarse_results, 3);
        
        // Refine around best candidates
        auto refined = generate_refined_grid(top_3);
        
        return refined;
    }
};
```

### 3. Device-Specific Optimizations

```cpp
struct CacheKey {
    std::string gpu_arch;
    std::string kernel_name;
    std::string context;
    
    // NEW: Add device-specific fields
    int compute_units;
    size_t shared_mem_per_block;
    size_t global_mem_bandwidth;
    
    std::string to_string() const {
        // Include device characteristics in cache key
        // Allows sharing cache across similar devices
    }
};
```

### 4. Compile-Time Kernel Generation

```cpp
// Generate specialized kernel versions at compile time
template<int BlockX, int BlockY, int VecWidth>
__global__ void grayscale_kernel_specialized(...) {
    // Fully unrolled, no runtime branching
}

// Dispatch table generated at compile time
template<typename Traits>
struct KernelDispatcher {
    using LaunchFunc = void(*)(const Traits::Args&, hipStream_t);
    
    static constexpr LaunchFunc get_launcher(const TuningConfig& cfg) {
        // Compile-time lookup table
        if (cfg.block_x() == 64 && cfg.vec_width() == 4)
            return &launch_specialized<64, 1, 4>;
        // ...
    }
};
```

---

## Testing Strategy

### Unit Tests

1. **TuningConfig**
   - Serialization/deserialization
   - Type-safe get/set operations
   - Equality comparison

2. **CacheStore**
   - Insert/lookup operations
   - Load/save to disk
   - Collision handling

3. **Benchmarker**
   - Timing accuracy
   - Error handling (invalid configs)
   - Statistics computation

### Integration Tests

1. **End-to-End Tuning**
   - Verify best config selected
   - Cache persistence across runs
   - Performance regression tests

2. **Kernel Migration**
   - Ensure results identical to old implementation
   - Performance parity (no slowdown)
   - Cache compatibility

---

## Performance Goals

### Tuning Time (First Run)
- **Current:** ~500ms per kernel (6 candidates × 10 runs × 8ms)
- **Target:** Same (500ms) - tuning thoroughness unchanged
- **Stretch:** 300ms with adaptive pruning (future work)

### Cached Lookup Time
- **Current:** ~10ns (linear search, small n)
- **Target:** ~5ns (hash table + thread-local cache)

### Code Maintenance
- **Current:** ~80 lines boilerplate per kernel
- **Target:** ~30 lines traits definition per kernel
- **Benefit:** 60% reduction in kernel-specific code

---

## Risk Mitigation

### Risk 1: Template Bloat
**Mitigation:** 
- Use explicit instantiation for common kernel types
- Keep template logic in headers, implementation in .cpp files
- Profile binary size increase (target: <5% growth)

### Risk 2: Compile Time Increase
**Mitigation:**
- Measure before/after (baseline: ~15s)
- Use precompiled headers for framework types
- Consider splitting large template implementations

### Risk 3: Cache Invalidation
**Mitigation:**
- Version cache file format
- Add cache validation on load
- Graceful fallback to re-tuning

### Risk 4: Backward Compatibility
**Mitigation:**
- Keep old `AutoTuner` class for 1 release
- Add deprecation warnings
- Provide automated migration script

---

## Conclusion

The proposed refactoring creates a clean, extensible foundation for long-term autotuning work while maintaining (or improving) runtime performance. The key improvements are:

1. **Eliminates 60% of boilerplate** through traits-based design
2. **Enables arbitrary tunable parameters** without framework changes
3. **Improves type safety** by removing `void*` casts
4. **Preserves performance** with O(1) cache lookups and thread-local optimization
5. **Supports future extensions** (vectorization, shared memory, unrolling)

The migration is incremental and low-risk, with clear validation criteria at each step.

---

## Appendix: File Structure

### Proposed Directory Layout

```
src/
├── core/
│   ├── autotune/                    # NEW: Autotuning framework
│   │   ├── types.h                  # Common types, CacheKey, etc.
│   │   ├── tuning_config.h/.cpp     # TuningConfig class
│   │   ├── cache_store.h/.cpp       # CacheStore class
│   │   ├── benchmarker.h            # Benchmarker template
│   │   ├── orchestrator.h           # TuningOrchestrator template
│   │   └── json_utils.h/.cpp        # JSON serialization helpers
│   ├── autotuning.h/.cpp            # DEPRECATED: Old AutoTuner
│   ├── gpu_utils.h/.cpp
│   └── image.h/.cpp
└── filters/
    ├── filters.h                    # Kernel declarations + traits
    ├── grayscale.hip.cpp
    ├── grayscale_autotune.hip.cpp   # Uses new framework
    ├── negative.hip.cpp
    ├── negative_autotune.hip.cpp    # Uses new framework
    ├── gaussian_blur.hip.cpp
    └── gaussian_blur_autotune.hip.cpp  # Uses new framework
```

### Estimated Line Counts

| Component | Current LOC | Proposed LOC | Delta |
|-----------|-------------|--------------|-------|
| Core framework | 400 | 800 | +400 |
| Grayscale autotune | 90 | 40 | -50 |
| Negative autotune | 80 | 40 | -40 |
| Gaussian autotune | 95 | 45 | -50 |
| **Total** | **665** | **925** | **+260** |

**Analysis:** 40% increase in total lines, but:
- Framework is reusable (one-time cost)
- Per-kernel code reduced by 55%
- Adding 4th kernel: Old = +80 lines, New = +40 lines
- Break-even at 7 kernels, net savings beyond

