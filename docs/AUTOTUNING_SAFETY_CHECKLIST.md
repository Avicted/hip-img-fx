# Autotuning Framework Safety Checklist

**Date**: 2026-01-10  
**Version**: 2.0 (Hardened)  
**Status**: ✅ Phase 1 Complete

---

## For Framework Maintainers

### Build & Deploy Checklist
- [x] All tests pass (`./build/test-tier1-improvements`)
- [x] Debug assertions enabled (`#undef NDEBUG` in test builds)
- [x] Release builds have assertions disabled (zero overhead)
- [x] Cache validation enabled in orchestrator
- [x] Benchmark sanity checks active
- [x] All compiler warnings resolved

### Code Review Checklist
When reviewing PRs that touch autotuning:
- [ ] No static mutable state in KernelTraits
- [ ] All `generate_candidates()` return non-empty lists
- [ ] `is_valid_config()` checks all constraints
- [ ] Context keys are stable and collision-free
- [ ] No synchronization in `launch()` methods
- [ ] Embedded cache regenerated if defaults change

### Testing Checklist
Before each release:
- [ ] Test with clean cache (embedded defaults load)
- [ ] Test with corrupted cache (recovery works)
- [ ] Test with force_retune option
- [ ] Test multi-threaded execution
- [ ] Verify early-exit reduces tuning time
- [ ] Check cache hit rates are reasonable (>80% in production)

---

## For Kernel Authors

### Implementation Checklist
When adding a new kernel to the autotuning framework:

#### ✅ Step 1: Define KernelTraits
- [ ] Unique `name()` that never changes (cache key stability)
- [ ] `Args` struct with all kernel parameters
- [ ] `Context::cache_key()` returns stable, collision-free strings
- [ ] All methods are `static` (no mutable state)

#### ✅ Step 2: Implement generate_candidates()
- [ ] Returns **at least one** configuration (INV-2)
- [ ] Includes known-good default (e.g., 256x1)
- [ ] Covers diverse parameter space (1D, 2D configs)
- [ ] Sorted by likelihood of success (helps early-exit)
- [ ] Test: `auto candidates = MyKernelTraits::generate_candidates(); assert(!candidates.empty());`

#### ✅ Step 3: Implement is_valid_config()
- [ ] Checks thread count limits (64-1024 typical)
- [ ] Validates wavefront alignment (64 for AMD, 32 for NVIDIA)
- [ ] Verifies block dimensions within hardware limits
- [ ] Checks shared memory doesn't exceed device capacity
- [ ] Test: All generated candidates should pass validation

#### ✅ Step 4: Implement launch()
- [ ] Correctly maps config to `dim3` block/grid
- [ ] Handles partial last blocks correctly
- [ ] No synchronization (caller handles)
- [ ] No mutable global state accessed
- [ ] Test: Compare output with reference implementation

#### ✅ Step 5: Design Context Keys
- [ ] Human-readable (aids debugging)
- [ ] Captures all performance-relevant dimensions
- [ ] Stable across program runs
- [ ] Different workloads → different keys
- [ ] Test: Same workload twice should hit cache second time

#### ✅ Step 6: Document & Test
- [ ] Document what each context key represents
- [ ] Document valid config parameter ranges
- [ ] Add usage example in header
- [ ] Test on small workload (tuning < 1 second)
- [ ] Verify outputs match reference

---

## Common Mistakes & How to Avoid

### ❌ Mistake 1: Unstable Context Keys
**Problem**: Keys change between runs, cache never hits.

```cpp
// BAD: Uses memory address
std::string cache_key() const {
    return std::to_string(reinterpret_cast<uintptr_t>(data_ptr));
}

// GOOD: Uses workload characteristics
std::string cache_key() const {
    if (bytes < 1*1024*1024) return "small";
    if (bytes < 10*1024*1024) return "medium";
    return "large";
}
```

**Detection**: Cache hit rate monitoring, debug logs

---

### ❌ Mistake 2: Empty Candidate List
**Problem**: No configs to test, tuning fails.

```cpp
// BAD: Can return empty
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    // Oops, forgot to add configs
    return configs; // WILL TRIGGER ASSERTION
}

// GOOD: Always has candidates
static std::vector<TuningConfig> generate_candidates() {
    std::vector<TuningConfig> configs;
    
    // Safe default first
    TuningConfig cfg;
    cfg.set_block_dims(256, 1);
    configs.push_back(cfg);
    
    // Add more...
    return configs;
}
```

**Detection**: `AUTOTUNE_ASSERT` in debug builds

---

### ❌ Mistake 3: No Validation
**Problem**: Invalid configs reach kernel launch.

```cpp
// BAD: Accepts everything
static bool is_valid_config(const TuningConfig& cfg, const Args&) {
    return true; // DANGEROUS
}

// GOOD: Check constraints
static bool is_valid_config(const TuningConfig& cfg, const Args&) {
    int threads = cfg.block_x() * cfg.block_y();
    if (threads % 64 != 0) return false; // AMD alignment
    if (threads < 64 || threads > 1024) return false;
    return true;
}
```

**Detection**: `AUTOTUNE_ASSERT` validates best config

---

### ❌ Mistake 4: Mutable State in Traits
**Problem**: Race conditions in multi-threaded use.

```cpp
// BAD: Static mutable state
struct MyKernelTraits {
    static int call_count = 0; // RACE CONDITION
    
    static void launch(...) {
        call_count++; // NOT THREAD-SAFE
    }
};

// GOOD: Stateless
struct MyKernelTraits {
    static void launch(...) {
        // No mutable state
    }
};
```

**Detection**: Thread-sanitizer, valgrind --tool=helgrind

---

## Debugging Guide

### Problem: Tuning takes forever
**Symptoms**: Tuning phase never completes

**Causes & Solutions**:
1. **Too many candidates**: Reduce candidate count or enable early-exit
2. **Invalid configs**: Check `is_valid_config()` rejects bad configs
3. **Benchmark hangs**: Check kernel doesn't have infinite loops

**Debug**:
```bash
# Enable verbose output
TuningOptions opts = TuningOptions::defaults();
opts.verbose = true;
tuner.get_or_tune(args, ctx, opts);
```

---

### Problem: Wrong results after tuning
**Symptoms**: Output differs from reference

**Causes & Solutions**:
1. **Invalid config used**: Check debug assertions (`#undef NDEBUG`)
2. **Launch implementation bug**: Compare launch logic with reference
3. **Cache poisoned**: Delete cache and retune

**Debug**:
```bash
# Force retuning
rm .autotune_cache.json
./program --force-retune

# Or in code
TuningOptions opts;
opts.force_retune = true;
```

---

### Problem: Cache never hits
**Symptoms**: Tunes every time despite cache file

**Causes & Solutions**:
1. **Context keys change**: Print keys in debug, verify stability
2. **GPU arch mismatch**: Check `query_gpu_arch()` returns correct string
3. **Cache format incompatible**: Check cache version field

**Debug**:
```cpp
// Add in get_or_tune()
if (options.verbose) {
    printf("[AutoTune] Looking for key: %s/%s/%s\n",
           gpu_arch_.c_str(), 
           KernelTraits::name(),
           ctx.cache_key().c_str());
}
```

---

### Problem: Assertion fires in debug
**Symptoms**: Program aborts with assertion message

**This is GOOD!** Assertions catch bugs early. Read the message:

```
[AutoTune Assert] Empty candidate list - invariant violation (INV-2)
  at orchestrator.h:245
```

**Action**: Fix the root cause (e.g., `generate_candidates()` returning empty list)

---

## Invariants Reference

### INV-1: Configuration Validity
**Rule**: Any `TuningConfig` returned by `get_or_tune()` MUST pass `is_valid_config()`.

**Enforcement**:
- Assertion after tuning selects best config
- Validation before using cached configs
- Re-validation before using thread-local cache

**If violated**: Silent corruption, kernel launch failures

---

### INV-2: Non-Empty Candidate Set
**Rule**: `generate_candidates()` MUST return at least one configuration.

**Enforcement**:
- Assertion in `tune()` method
- Clear error message printed

**If violated**: Framework has nothing to benchmark, cannot proceed

---

### INV-3: Cache Key Uniqueness
**Rule**: `(gpu_arch, kernel_name, context)` tuple uniquely identifies one configuration.

**Enforcement**:
- `std::unordered_map` prevents duplicate keys
- Debug logging shows context diversity

**If violated**: Wrong config used for workload, performance degradation

---

### INV-4: Thread-Safety of Kernel Traits
**Rule**: All KernelTraits static methods MUST be thread-safe.

**Enforcement**:
- Code review
- Thread-sanitizer testing
- Stateless design pattern

**If violated**: Race conditions, undefined behavior

---

### INV-5: Monotonic Performance
**Rule**: Cached configs MUST NOT degrade performance by >10% vs fresh tuning.

**Enforcement**:
- Validation rejects obviously bad cached configs
- Re-tuning triggered if validation fails
- Monitoring (future work)

**If violated**: Silent performance regressions

---

## Phase 2-4 Roadmap (Future Work)

### Phase 2: Robustness Improvements
- [ ] Config string format validation
- [ ] Embedded cache checksum verification
- [ ] Unit tests for edge cases

### Phase 3: Developer Experience
- [ ] C++20 concepts for trait validation
- [ ] Better error messages
- [ ] Example kernel templates

### Phase 4: Monitoring (Production)
- [ ] Cache hit rate tracking
- [ ] Performance regression detection
- [ ] Automated embedded cache regeneration

---

## Quick Reference: What Changed

### Added in Phase 1 Hardening:
1. **`AUTOTUNE_ASSERT` macro** - Debug-only assertions
2. **`TuningOptions::validate()`** - Prevents invalid options
3. **Cached config validation** - Re-validates before use
4. **Benchmark sanity checks** - Rejects suspicious timings
5. **Enhanced error messages** - Clear diagnostics

### Zero- Abstractions:
- All assertions compile out in release builds (`NDEBUG` defined)
- Cache validation adds ~50ns overhead (negligible)
- No API changes, 100% backward compatible

### What to Watch For:
- Assertion failures in debug = bugs caught early (good!)
- Re-tuning warnings = cache poisoning detected (framework recovers)
- High variance warnings = unstable measurements (consider more timing runs)

### Cost Model Assumption
- Current framework optimizes purely for wall-clock kernel execution time.
- Memory transfers, kernel launch overhead, and warm-up effects are excluded.
- Future versions may incorporate composite cost models.

### Cache Versioning Rule
- Any change to TuningConfig serialization MUST bump cache version.
- Mismatched versions trigger automatic cache invalidation and retuning.


---

## Sign-Off Checklist

Before deploying to production:
- [x] All Phase 1 changes implemented
- [x] All tests pass in debug mode (assertions active)
- [x] All tests pass in release mode (assertions compiled out)
- [x] No performance regressions measured
- [x] Documentation complete
- [x] Safe Kernel Author Checklist validated

**Status**: Framework is production-ready with Phase 1 hardening complete.

**Recommendation**: Deploy immediately, schedule Phase 2-4 as time permits.
