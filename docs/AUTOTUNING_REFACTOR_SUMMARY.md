# Autotuning Refactor - Executive Summary

## Problem Statement

The current autotuning implementation suffers from:
- **80 lines of boilerplate per kernel** (duplicated launch wrappers, args structs)
- **Tight coupling** between kernel execution and tuning framework
- **Limited extensibility** - only block dimensions tunable, no support for vectorization/unrolling/shared memory
- **Inflexible candidate generation** - fixed list, no kernel-specific constraints

## Proposed Solution

### Architecture: Traits-Based Template Framework

```
┌────────────────────────────────────────────┐
│  Kernel Traits (Per-Kernel)                │
│  - Define tunable parameters               │
│  - Generate candidates                     │
│  - Apply constraints                       │
│  - Launch logic                            │
└──────────────┬─────────────────────────────┘
               │
┌──────────────▼─────────────────────────────┐
│  TuningOrchestrator<Traits> (Core)         │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ │
│  │Parameter │ │Benchmarker │ │CacheStore│ │
│  │  Space   │ │            │ │          │ │
│  └──────────┘ └────────────┘ └──────────┘ │
└────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Template-Based Polymorphism
- **Zero runtime overhead** - everything resolved at compile time
- Type-safe parameter passing (no `void*` casts)
- Each kernel defines a `Traits` struct describing its tunable parameters

### 2. Separation of Concerns
| Component | Responsibility | Reusable? |
|-----------|----------------|-----------|
| **Kernel Traits** | Parameter space, constraints, launch logic | No (per-kernel) |
| **TuningOrchestrator** | Coordinate tuning process | Yes |
| **Benchmarker** | Warmup, timing, statistics | Yes |
| **CacheStore** | Load/save/lookup configs | Yes |
| **TuningConfig** | Extensible parameter container | Yes |

### 3. Extensible Parameter System
```cpp
// Instead of fixed struct:
struct KernelConfig {
    int block_x, block_y;  // ❌ Not extensible
};

// Use variant-based container:
class TuningConfig {
    std::unordered_map<std::string, std::variant<int, float, bool>> params_;
    // ✅ Can add vec_width, unroll_factor, smem_tiles without changing core
};
```

### 4. Context-Aware Caching
```cpp
struct CacheKey {
    std::string gpu_arch;       // e.g., "gfx1100"
    std::string kernel_name;    // e.g., "grayscale"
    std::string context;        // e.g., "small" (image size category)
};
```

## Impact Analysis

### Code Reduction Per Kernel
| Component | Before (LOC) | After (LOC) | Reduction |
|-----------|--------------|-------------|-----------|
| Args struct | 8 | 5 (in traits) | -38% |
| Launch wrapper | 20 | 15 (in traits) | -25% |
| Apply function | 30 | 10 | -67% |
| Grid calculation | 10 | 10 (same) | 0% |
| **Total** | **68** | **40** | **-41%** |

### Performance Characteristics
| Metric | Current | Proposed | Change |
|--------|---------|----------|--------|
| Cache lookup | O(n) ~10ns | O(1) ~5ns | 2× faster |
| Tuning time | 500ms | 500ms | Same |
| Binary size | Baseline | +5% | Minimal |
| Compile time | ~15s | ~18s | +20% (acceptable) |

### Extensibility Wins
| Feature | Current Support | Proposed Support |
|---------|-----------------|------------------|
| Block dimensions (x, y) | ✅ Yes | ✅ Yes |
| Vectorization (vec2, vec4) | ❌ Requires framework change | ✅ Add to traits |
| Shared memory tuning | ❌ Requires framework change | ✅ Add to traits |
| Loop unroll factors | ❌ Requires framework change | ✅ Add to traits |
| Work-per-thread ratios | ❌ Requires framework change | ✅ Add to traits |

## Example: Grayscale Kernel Before/After

### Before (90 lines)
```cpp
// Separate args struct (8 lines)
struct GrayscaleLaunchArgs { /* ... */ };

// Launch wrapper (20 lines)
void launch_grayscale_kernel(const KernelConfig& config, hipStream_t stream, void* args) {
    GrayscaleLaunchArgs* launch_args = static_cast<GrayscaleLaunchArgs*>(args);
    // Grid calculation...
    hipLaunchKernelGGL(grayscale_kernel, ...);
}

// Apply function (30 lines)
void apply_grayscale_autotuned(..., AutoTuner& autotuner, ...) {
    GrayscaleLaunchArgs launch_args = { /* ... */ };
    const int warmup_runs = 5;
    const int timing_runs = 10;
    KernelConfig config = autotuner.get_config("grayscale", launch_grayscale_kernel, &launch_args, ...);
    launch_grayscale_kernel(config, stream, &launch_args);
}
```

### After (40 lines)
```cpp
// Traits struct consolidates everything (30 lines)
struct GrayscaleKernelTraits {
    static constexpr const char* name() { return "grayscale"; }
    
    struct Args { /* ... */ };  // Type-safe, no void*
    struct Context { std::string cache_key() const; };
    
    static std::vector<TuningConfig> generate_candidates();
    static bool is_valid_config(const TuningConfig& cfg, const Args& args);
    static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream);
};

// Apply function simplified (10 lines)
void apply_grayscale_autotuned(..., hipStream_t stream) {
    static TuningOrchestrator<GrayscaleKernelTraits> orchestrator;
    GrayscaleKernelTraits::Args args{...};
    GrayscaleKernelTraits::Context ctx{...};
    orchestrator.execute(args, ctx, stream);
}
```

## Migration Strategy

### Phase 1: Implement Core Framework (2-3 days)
Create new `src/core/autotune/` directory with:
- `tuning_config.h/.cpp` - Extensible config container
- `cache_store.h/.cpp` - O(1) cache with JSON persistence
- `benchmarker.h` - Template-based timing engine
- `orchestrator.h` - Main tuning coordinator

### Phase 2: Migrate Grayscale (1 day)
- Define `GrayscaleKernelTraits` in `filters.h`
- Refactor `grayscale_autotune.hip.cpp` to use new framework
- Validate: results identical, cache compatible

### Phase 3: Migrate Remaining Kernels (1 day)
- Apply same pattern to negative, gaussian_blur
- Remove ~140 lines of duplicated code

### Phase 4: Deprecate Old AutoTuner (0.5 days)
- Mark `autotuning.h` as deprecated
- Keep for 1 release cycle for compatibility
- Remove in next major version

**Total Estimated Effort: 4.5-5.5 days**

## Future Extensions Enabled

### 1. Vectorized Memory Access
```cpp
struct GrayscaleKernelTraits {
    static std::vector<TuningConfig> generate_candidates() {
        for (int vec : {1, 2, 4}) {  // NEW: vec width
            cfg.set("vec_width", vec);
            // Launch grayscale_kernel_vec2, grayscale_kernel_vec4, etc.
        }
    }
};
```

### 2. Shared Memory Tuning
```cpp
struct GaussianBlurKernelTraits {
    static std::vector<TuningConfig> generate_candidates() {
        for (int tiles : {0, 1, 2}) {  // NEW: shared mem tiles
            cfg.set("smem_tiles", tiles);
            // Adjust shared_bytes in launch based on tiles
        }
    }
};
```

### 3. Adaptive Candidate Pruning
```cpp
// Framework extension: two-phase tuning
auto coarse_results = orchestrator.tune_coarse();  // 6 configs
auto top_3 = select_top_n(coarse_results, 3);
auto fine_results = orchestrator.tune_refined(top_3);  // 18 configs around best
```

### 4. Multi-Device Caching
```cpp
struct CacheKey {
    std::string gpu_arch;
    int compute_units;     // NEW: share cache across similar devices
    size_t smem_per_block; // NEW: refine by device capabilities
};
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Template bloat | Medium | Low | Explicit instantiation, monitor binary size |
| Compile time ↑ | High | Low | Acceptable (+20%), use PCH |
| Cache invalidation | Low | Medium | Version cache format, fallback to re-tune |
| Migration bugs | Low | High | Extensive testing, incremental rollout |

## Success Criteria

### Functional Requirements
- ✅ All kernels produce identical results
- ✅ Cache files remain compatible
- ✅ Tuning selects same optimal configs

### Performance Requirements
- ✅ Cached lookup ≤ 10ns (target: 5ns)
- ✅ Tuning time unchanged (~500ms)
- ✅ Binary size increase < 10% (target: 5%)
- ✅ Compile time increase < 30% (target: 20%)

### Code Quality Requirements
- ✅ Per-kernel boilerplate reduced by ≥ 40%
- ✅ Adding new kernel requires ≤ 40 lines
- ✅ No `void*` casts in new code
- ✅ All framework components unit tested

## Conclusion

The proposed refactoring transforms a rigid, boilerplate-heavy system into a **flexible, type-safe framework** that:

1. **Reduces maintenance burden** by 40% per kernel
2. **Enables future optimizations** without framework changes
3. **Preserves performance** with O(1) caching
4. **Maintains compatibility** through incremental migration

This is a **high-value, medium-effort** refactoring that pays dividends as the project grows.

---

**Recommendation:** Proceed with implementation. The architecture is sound, the migration path is clear, and the long-term benefits significantly outweigh the short-term development cost.

**Next Steps:**
1. Review this proposal with team
2. Allocate 1 week for implementation
3. Create feature branch `refactor/autotuning-framework`
4. Implement Phase 1 (core framework)
5. Migrate one kernel (grayscale) as proof-of-concept
6. Review, iterate, complete migration

