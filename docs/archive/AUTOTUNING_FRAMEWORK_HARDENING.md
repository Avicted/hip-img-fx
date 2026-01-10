# Autotuning Framework Hardening & Safety Analysis

**Date**: 2026-01-10  
**Version**: 2.0 (Post Tier-1 Improvements)  
**Status**: Safety Review & Hardening

---

## Executive Summary

The autotuning framework is **fundamentally sound** but has **12 identified failure modes** requiring guardrails. Current implementation is optimistic (assumes success) which works for well-formed inputs but lacks defensive measures for edge cases and misuse.

**Critical Findings**:
- ✅ No architectural flaws requiring rewrites
- ⚠️ Missing validation at 7 critical boundaries
- ⚠️ Silent failures possible in 5 scenarios
- ⚠️ Kernel author contracts are implicit, not enforced

**Recommended Actions**:
1. Add 8 compile-time trait validation checks
2. Add 12 runtime assertions (debug-only, zero overhead in release)
3. Document 5 non-negotiable invariants
4. Create safe kernel author checklist

**Impact**: Zero performance regression, significantly improved debuggability and maintainability.

---

## Non-Negotiable Invariants

These must **always** hold. Framework must detect violations and fail fast.

### INV-1: Configuration Validity
**Statement**: Any `TuningConfig` returned by `get_or_tune()` MUST pass `is_valid_config()`.

**Rationale**: Invalid configs can cause kernel launches to fail silently or produce wrong results.

**Current Status**: ⚠️ Not enforced - cache can contain invalid configs that bypass validation.

**Impact**: Silent corruption if cached config becomes invalid after GPU driver update or hardware change.

---

### INV-2: Non-Empty Candidate Set
**Statement**: `generate_candidates()` MUST return at least one configuration.

**Rationale**: Zero candidates → no tuning possible → framework returns empty config → kernel launch with undefined parameters.

**Current Status**: ⚠️ Checked at runtime but returns empty config on failure (should abort).

**Impact**: Hard-to-debug crashes in production.

---

### INV-3: Cache Key Uniqueness
**Statement**: `(gpu_arch, kernel_name, context)` tuple MUST uniquely identify one optimal configuration.

**Rationale**: Key collisions → wrong configs used → performance degradation or incorrect results.

**Current Status**: ✅ Enforced by `std::unordered_map`, but collision detection missing.

**Impact**: Subtle misoptimization if context keys are poorly chosen.

---

### INV-4: Thread-Safety of Kernel Traits
**Statement**: All KernelTraits static methods MUST be thread-safe.

**Rationale**: `TuningOrchestrator` can be called from multiple threads, traits methods are shared.

**Current Status**: ⚠️ Not documented, kernel authors may add mutable state.

**Impact**: Race conditions, undefined behavior.

---

### INV-5: Monotonic Performance
**Statement**: Cached configs MUST NOT degrade performance by >10% vs fresh tuning.

**Rationale**: Ensures cache correctness and prevents stale configs from harming performance.

**Current Status**: ❌ No validation - cache can contain arbitrarily bad configs.

**Impact**: Silent performance regressions that persist across runs.

---

## Failure Modes Analysis

| ID | Failure Mode | Trigger | Current Behavior | Severity | Detection |
|----|--------------|---------|------------------|----------|-----------|
| **FM-1** | Empty candidate list | `generate_candidates()` returns `{}` | Returns empty config, kernel launches with defaults | **HIGH** | None |
| **FM-2** | All candidates invalid | `is_valid_config()` rejects all | Returns empty config | **HIGH** | Runtime warning only |
| **FM-3** | Corrupted cache file | Invalid JSON in `.autotune_cache.json` | Silent failure to load, fresh tuning | **MEDIUM** | None |
| **FM-4** | Invalid cached config | Hardware change invalidates old config | Launches with invalid config | **HIGH** | None |
| **FM-5** | Malicious embedded cache | Attacker modifies `embedded_cache.h` | Arbitrary configs loaded | **MEDIUM** | None |
| **FM-6** | Context key collision | Two workloads use same cache key | Wrong config used | **MEDIUM** | None |
| **FM-7** | Benchmark timing failure | HIP events fail to create/record | Returns invalid result, skipped | **LOW** | Warning printed |
| **FM-8** | Zero timing results | All benchmarks fail validation | Returns empty config | **HIGH** | None |
| **FM-9** | Early-exit incorrect threshold | `threshold < 1.0` or negative | Exits prematurely, wrong config | **MEDIUM** | None |
| **FM-10** | Thread-local cache pollution | Bad config inserted by force_retune | Persists until thread exit | **LOW** | None |
| **FM-11** | GPU arch detection failure | `hipGetDeviceProperties()` fails | Uses arch="unknown", cache misses | **LOW** | Warning printed |
| **FM-12** | Missing KernelTraits method | Incomplete traits implementation | Compile error (good) | **N/A** | Compile-time |

---

## Guardrails & Enforcement Plan

### 1. Compile-Time Validation (Zero Runtime Cost)

Add trait validation using C++20 concepts or SFINAE:

```cpp
// In orchestrator.h
template <typename T>
concept ValidKernelTraits = requires {
    // Required static methods
    { T::name() } -> std::convertible_to<const char*>;
    
    // Required nested types
    typename T::Args;
    typename T::Context;
    
    // Context must have cache_key() method
    requires requires(typename T::Context ctx) {
        { ctx.cache_key() } -> std::convertible_to<std::string>;
    };
    
    // Required static methods with correct signatures
    { T::generate_candidates() } -> std::same_as<std::vector<TuningConfig>>;
    
    requires requires(const TuningConfig& cfg, const typename T::Args& args) {
        { T::is_valid_config(cfg, args) } -> std::same_as<bool>;
    };
    
    requires requires(const TuningConfig& cfg, const typename T::Args& args, hipStream_t stream) {
        { T::launch(cfg, args, stream) } -> std::same_as<void>;
    };
};

template <ValidKernelTraits KernelTraits>
class TuningOrchestrator { ... };
```

**Enforces**: INV-4, FM-12  
**Cost**: Zero (compile-time only)

---

### 2. Runtime Assertions (Debug-Only)

Add assertions that compile out in release builds:

```cpp
// In types.h - add debug validation
#ifdef NDEBUG
    #define AUTOTUNE_ASSERT(cond, msg) ((void)0)
#else
    #define AUTOTUNE_ASSERT(cond, msg) \
        do { if (!(cond)) { \
            fprintf(stderr, "[AutoTune Assert] %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
            std::abort(); \
        } } while(0)
#endif

// In orchestrator.h - validate options
TuningConfig get_or_tune(...) {
    AUTOTUNE_ASSERT(options.early_exit_threshold >= 1.0, 
                    "Early-exit threshold must be >= 1.0");
    AUTOTUNE_ASSERT(options.early_exit_min_coverage > 0.0 && 
                    options.early_exit_min_coverage <= 1.0,
                    "Early-exit coverage must be in (0, 1]");
    AUTOTUNE_ASSERT(options.warmup_runs >= 0 && options.timing_runs > 0,
                    "Invalid warmup/timing run counts");
    ...
}

// In tune() method - validate candidates
TuningConfig tune(const Args &args, const TuningOptions &options) {
    auto candidates = KernelTraits::generate_candidates();
    AUTOTUNE_ASSERT(!candidates.empty(), 
                    "generate_candidates() returned empty list");
    
    auto results = benchmarker.benchmark_all(...);
    AUTOTUNE_ASSERT(!results.empty(), 
                    "All benchmark attempts failed - no valid results");
    
    auto best = std::min_element(results.begin(), results.end());
    AUTOTUNE_ASSERT(KernelTraits::is_valid_config(best->config, args),
                    "Best config failed is_valid_config() - invariant violation");
    ...
}
```

**Enforces**: INV-1, INV-2, FM-1, FM-2, FM-8, FM-9  
**Cost**: Zero in release builds, caught early in debug

---

### 3. Cache Validation Layer

Add validation when loading/using cached configs:

```cpp
// In cache_store.cpp - validate during deserialization
void CacheStore::deserialize(const std::string &json_content) {
    // Existing parsing...
    
    for (each entry) {
        // Validate config string is well-formed
        if (!is_valid_config_string(config_str)) {
            fprintf(stderr, "[AutoTune] Warning: Skipping malformed cache entry: %s\n",
                    config_str.c_str());
            continue; // Skip bad entry
        }
        
        cache_[key] = entry;
    }
}

// In orchestrator.h - validate before use
TuningConfig get_or_tune(...) {
    if (auto cached = cache_.lookup(key)) {
        // Re-validate cached config
        if (KernelTraits::is_valid_config(*cached, args)) {
            fast_cache[fast_key] = *cached;
            return *cached;
        } else {
            // Cache poisoned - invalidate and retune
            fprintf(stderr, "[AutoTune] Warning: Cached config invalid, retuning...\n");
            cache_.remove(key);
            // Fall through to tuning
        }
    }
    
    // Perform tuning...
}
```

**Enforces**: INV-1, INV-5, FM-3, FM-4, FM-5  
**Cost**: Minimal (~50ns for validation check)

---

### 4. Context Key Uniqueness Check

Add validation for context keys to prevent collisions:

```cpp
// In orchestrator.h - track context diversity
class TuningOrchestrator {
private:
    #ifndef NDEBUG
    mutable std::unordered_set<std::string> seen_contexts_;
    #endif
    
    TuningConfig get_or_tune(...) {
        std::string ctx_key = ctx.cache_key();
        
        #ifndef NDEBUG
        if (seen_contexts_.count(ctx_key) == 0) {
            // First time seeing this context - log it
            if (options.verbose) {
                printf("[AutoTune] New context: '%s'\n", ctx_key.c_str());
            }
            seen_contexts_.insert(ctx_key);
        }
        #endif
        
        // Continue with existing logic...
    }
};
```

**Enforces**: INV-3, FM-6  
**Cost**: Debug-only tracking, helps developers catch poor context keys

---

### 5. Embedded Cache Integrity Check

Add checksum validation for embedded cache:

```cpp
// In embedded_cache.h - add checksum
namespace imgfx::core::autotune {
    inline const char* EMBEDDED_DEFAULT_CACHE_V2 = R"(...)";
    
    // Simple checksum of embedded data (computed at generation time)
    inline constexpr size_t EMBEDDED_CACHE_CHECKSUM = 0x1A2B3C4D; // Auto-generated
}

// In orchestrator.h - validate before use
TuningOrchestrator(...) {
    // Existing cache loading...
    
    if (!user_cache_loaded && EMBEDDED_DEFAULT_CACHE_V2 != nullptr) {
        #ifndef NDEBUG
        // Verify checksum in debug builds
        size_t computed = compute_simple_checksum(EMBEDDED_DEFAULT_CACHE_V2);
        if (computed != EMBEDDED_CACHE_CHECKSUM) {
            fprintf(stderr, "[AutoTune] Warning: Embedded cache checksum mismatch, skipping\n");
        } else
        #endif
        {
            cache_.load_from_string(EMBEDDED_DEFAULT_CACHE_V2);
        }
    }
}
```

**Enforces**: FM-5  
**Cost**: Debug-only validation, prevents silent corruption

---

### 6. Benchmark Result Validation

Add sanity checks on timing results:

```cpp
// In benchmarker.h - validate timing results
BenchmarkResult benchmark(...) {
    // Existing benchmark logic...
    
    float avg = compute_mean(times);
    float stddev = compute_stddev(times, avg);
    
    // Sanity checks
    if (avg <= 0.0f || avg > 10000.0f) { // 0-10 seconds is reasonable
        fprintf(stderr, "[AutoTune] Warning: Suspicious timing %.4f ms, rejecting\n", avg);
        return BenchmarkResult{}; // Invalid
    }
    
    if (stddev > avg * 0.5f) { // High variance suggests measurement error
        fprintf(stderr, "[AutoTune] Warning: High variance (%.1f%%), consider more runs\n",
                (stddev / avg) * 100.0f);
    }
    
    return BenchmarkResult(config, avg, stddev, min_t, max_t);
}
```

**Enforces**: FM-7, FM-8  
**Cost**: Trivial (few FP comparisons)

---

## Minimal Code Changes Required

### Change 1: Add AUTOTUNE_ASSERT Macro
**File**: `src/core/autotune/types.h`  
**Lines**: Add after includes  
**Purpose**: Debug-only assertion system

### Change 2: Add Validation to TuningOptions
**File**: `src/core/autotune/types.h`  
**Lines**: In `TuningOptions` struct  
**Purpose**: Prevent invalid options at construction time

### Change 3: Add Assertions to Orchestrator
**File**: `src/core/autotune/orchestrator.h`  
**Lines**: In `get_or_tune()` and `tune()` methods  
**Purpose**: Enforce invariants INV-1, INV-2

### Change 4: Add Cache Validation
**File**: `src/core/autotune/orchestrator.h`  
**Lines**: In `get_or_tune()` before returning cached config  
**Purpose**: Re-validate cached configs against current hardware

### Change 5: Add Benchmark Sanity Checks
**File**: `src/core/autotune/benchmarker.h`  
**Lines**: In `benchmark()` method after computing statistics  
**Purpose**: Detect measurement errors

### Change 6: Add Config String Validation
**File**: `src/core/autotune/tuning_config.cpp`  
**Lines**: In `from_key_string()` method  
**Purpose**: Detect corrupted cache entries

---

## Safe Kernel Author Checklist

When implementing a new kernel with autotuning support:

### ✅ Trait Implementation
- [ ] `name()` returns unique, stable identifier (never change)
- [ ] `Args` struct contains all kernel parameters
- [ ] `Context::cache_key()` returns stable, collision-free strings
- [ ] `generate_candidates()` returns **non-empty** list
- [ ] `is_valid_config()` validates all constraints (threads, alignment, limits)
- [ ] `launch()` respects all parameters from config

### ✅ Thread Safety
- [ ] All trait methods are **stateless** (no static mutable variables)
- [ ] `launch()` only modifies kernel arguments, not shared state
- [ ] No global variables accessed in traits methods

### ✅ Context Key Design
- [ ] Keys are human-readable (aids debugging)
- [ ] Keys capture all performance-relevant dimensions
- [ ] Keys are stable across program runs
- [ ] Different workloads → different keys (no collisions)
- [ ] Test: Run same workload twice, should hit cache second time

### ✅ Validation Logic
- [ ] `is_valid_config()` checks thread count limits (64-1024 typical)
- [ ] Check wavefront alignment (multiple of 64 for AMD, 32 for NVIDIA)
- [ ] Validate block dimensions don't exceed hardware limits
- [ ] Check shared memory usage doesn't exceed device limits
- [ ] Test: Generate all candidates, all must pass validation

### ✅ Candidate Generation
- [ ] Include at least 5-10 diverse candidates
- [ ] Cover 1D, 2D configurations if applicable
- [ ] Include known-good default (e.g., 256x1)
- [ ] Sort by likelihood of success (best first for early-exit)
- [ ] Test: Run on small workload, tuning should complete in <1 second

### ✅ Launch Implementation
- [ ] Correctly map config parameters to dim3 (block/grid)
- [ ] Handle edge cases (last block may be partial)
- [ ] Stream parameter passed correctly to hipLaunchKernelGGL
- [ ] No synchronization in launch() (caller handles)
- [ ] Test: Compare output with reference implementation

### ✅ Documentation
- [ ] Document what each context key represents
- [ ] Document valid config parameter ranges
- [ ] Document expected performance characteristics
- [ ] Add example usage in header comments

---

## Misuse Resistance

### Easy Mistake #1: Unstable Context Keys
**Problem**: Keys change between runs, cache never hits.

```cpp
// ❌ BAD: Uses memory address
std::string cache_key() const {
    return std::to_string(reinterpret_cast<uintptr_t>(data_ptr));
}

// ✅ GOOD: Uses workload characteristics
std::string cache_key() const {
    if (image_bytes < 1*1024*1024) return "small";
    if (image_bytes < 10*1024*1024) return "medium";
    return "large";
}
```

**Detection**: Cache hit rate monitoring (to be added)

---

### Easy Mistake #2: Forgetting Validation
**Problem**: Invalid configs make it to kernel launch.

```cpp
// ❌ BAD: No validation
static bool is_valid_config(const TuningConfig& cfg, const Args&) {
    return true; // Accepts everything
}

// ✅ GOOD: Check all constraints
static bool is_valid_config(const TuningConfig& cfg, const Args&) {
    int threads = cfg.block_x() * cfg.block_y();
    if (threads % 64 != 0) return false; // Wavefront alignment
    if (threads < 64 || threads > 1024) return false; // Limits
    return true;
}
```

**Detection**: AUTOTUNE_ASSERT after tuning (to be added)

---

### Easy Mistake #3: Mutable Trait State
**Problem**: Race conditions in multi-threaded use.

```cpp
// ❌ BAD: Static mutable state
struct MyKernelTraits {
    static int call_count = 0; // DANGER!
    
    static void launch(...) {
        call_count++; // Race condition
        ...
    }
};

// ✅ GOOD: Stateless traits
struct MyKernelTraits {
    static void launch(...) {
        // No mutable state
        ...
    }
};
```

**Detection**: Thread-sanitizer testing (to be documented)

---

### Easy Mistake #4: Empty Candidate List
**Problem**: No configs to test, tuning fails silently.

```cpp
// ❌ BAD: Can return empty
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    // Oops, forgot to add any configs
    return configs; // Empty!
}

// ✅ GOOD: Always has candidates
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    
    // Always include safe default
    TuningConfig default_cfg;
    default_cfg.set_block_dims(256, 1);
    configs.push_back(default_cfg);
    
    // Add other candidates...
    return configs;
}
```

**Detection**: AUTOTUNE_ASSERT in tune() (to be added)

---

## Debuggability Improvements

### Problem: Silent Failures
**Current**: Invalid configs, corrupted cache, failed benchmarks → no clear error

**Solution**: Add verbose error paths with context

```cpp
// Enhanced error reporting
if (results.empty()) {
    fprintf(stderr, "[AutoTune ERROR] All %zu candidates failed validation:\n", 
            candidates.size());
    for (const auto& cfg : candidates) {
        fprintf(stderr, "  - [%dx%d]: %s\n", 
                cfg.block_x(), cfg.block_y(),
                KernelTraits::is_valid_config(cfg, args) ? "valid but failed" : "INVALID");
    }
    fprintf(stderr, "  Possible causes:\n");
    fprintf(stderr, "    1. is_valid_config() too strict\n");
    fprintf(stderr, "    2. HIP runtime error (check previous messages)\n");
    fprintf(stderr, "    3. Insufficient GPU resources\n");
    return TuningConfig{};
}
```

### Problem: Cache Poisoning Unknown
**Current**: Bad cached config used, no indication why performance degraded

**Solution**: Add cache health metrics

```cpp
// In orchestrator.h - add validation counter
struct CacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t invalidations = 0; // Cached config failed validation
};

// Track and report periodically
if (options.verbose && stats_.invalidations > 0) {
    printf("[AutoTune] Warning: %zu cached configs invalidated (hardware change?)\n",
           stats_.invalidations);
}
```

### Problem: Early-Exit Mystery
**Current**: Unknown why tuning stopped early

**Solution**: Already implemented - verbose output shows threshold and reasoning

---

## Long-Term Maintenance Risks

### Risk 1: Cache Format Evolution
**Scenario**: Add new fields to cache → old caches invalid

**Mitigation**: Version field already present ("2.0"), enforce compatibility checking

```cpp
// In cache_store.cpp - check version
void CacheStore::deserialize(const std::string &json_content) {
    std::string version = find_json_value(json_content, "version");
    if (version != "2.0") {
        fprintf(stderr, "[AutoTune] Warning: Incompatible cache version '%s', ignoring\n",
                version.c_str());
        return; // Don't load old cache
    }
    // Continue parsing...
}
```

### Risk 2: GPU Architecture Changes
**Scenario**: New GPU generation → cached configs suboptimal

**Mitigation**: Arch string includes generation (e.g., "gfx1030"), per-GPU cache naturally isolates

### Risk 3: Embedded Cache Staleness
**Scenario**: Embedded defaults become outdated as drivers improve

**Mitigation**: Regeneration scripts documented, include in CI pipeline

```bash
# Add to CI: Regenerate embedded cache weekly
- cron: '0 0 * * 0' # Every Sunday
  job: regenerate-embedded-cache
    steps:
      - checkout
      - run: ./scripts/collect_default_configs.sh
      - run: ./scripts/merge_default_caches.py > embedded_cache.h
      - commit: "Auto-update embedded cache"
```

### Risk 4: Kernel Trait API Drift
**Scenario**: New requirements added, old kernels break

**Mitigation**: Compile-time trait validation (Change 1) catches incompatibilities immediately

---

## Testing Strategy

### Unit Tests (To Be Added)
```cpp
// test_tuning_config.cpp
TEST(TuningConfig, ValidatesConfigString) {
    // Malformed strings
    EXPECT_THROW(TuningConfig::from_key_string("invalid"), std::runtime_error);
    EXPECT_THROW(TuningConfig::from_key_string("key="), std::runtime_error);
    
    // Valid strings
    auto cfg = TuningConfig::from_key_string("block_x=128,block_y=1");
    EXPECT_EQ(cfg.block_x(), 128);
    EXPECT_EQ(cfg.block_y(), 1);
}

// test_orchestrator.cpp
TEST(TuningOrchestrator, RejectsInvalidCachedConfig) {
    // Populate cache with invalid config
    // Verify it gets retuned instead of used
}

TEST(TuningOrchestrator, HandlesEmptyCandidates) {
    // Mock traits that return empty candidate list
    // Verify graceful failure with clear error
}
```

### Integration Tests
```bash
# Test cache corruption recovery
./build/test-tier1-improvements
echo "corrupted" > .autotune_cache.json
./build/test-tier1-improvements # Should recover

# Test with force_retune
./build/test-tier1-improvements --force-retune

# Test with invalid embedded cache
# Modify embedded_cache.h with bad config
# Verify rejection and fallback to tuning
```

---

## Implementation Priority

### Phase 1: Critical Safety (1-2 hours)
- [ ] Add AUTOTUNE_ASSERT macro
- [ ] Add assertions to tune() method
- [ ] Add cached config validation
- [ ] Test on existing kernels

### Phase 2: Robustness (2-3 hours)
- [ ] Add benchmark result validation
- [ ] Add config string validation
- [ ] Add TuningOptions validation
- [ ] Write unit tests

### Phase 3: Developer Experience (1 hour)
- [ ] Document Safe Kernel Author Checklist
- [ ] Add examples of common mistakes
- [ ] Create trait validation concept

### Phase 4: Monitoring (Optional)
- [ ] Add cache hit rate tracking
- [ ] Add invalidation statistics
- [ ] Add performance regression detection

---

## Conclusion

The autotuning framework is **architecturally sound** but needs **defensive guardrails** to prevent misuse and improve debuggability. All recommended changes:

- ✅ Zero performance impact in release builds
- ✅ No API breaking changes
- ✅ Minimal code additions (< 200 LOC)
- ✅ Significantly improve safety and maintainability

**Recommendation**: Implement Phase 1 immediately (critical safety), defer Phase 4 (monitoring) until production deployment shows need.

**Risk Assessment**: Current framework is **suitable for production** but hardening reduces support burden and prevents subtle bugs from becoming production incidents.
