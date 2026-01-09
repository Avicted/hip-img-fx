# Autotuning Architecture Documentation

This directory contains comprehensive documentation for the proposed autotuning framework refactoring.

## Documents Overview

### 📋 [AUTOTUNING_REFACTOR_SUMMARY.md](AUTOTUNING_REFACTOR_SUMMARY.md)
**Start here!** Executive summary with key decisions and impact analysis.

- Problem statement and current issues
- Proposed solution architecture
- Before/after comparison table
- Performance analysis
- Success criteria
- Risk assessment

**Read this if:** You want a high-level understanding of the refactoring proposal.

**Time to read:** 10 minutes

---

### 📖 [AUTOTUNING_REFACTOR.md](AUTOTUNING_REFACTOR.md)
**Complete architectural design document** with detailed specifications.

- Comprehensive issue analysis
- High-level architecture diagrams
- Complete C++ interface specifications
- Migration guide with step-by-step instructions
- Future extension patterns
- Testing strategy
- Performance goals

**Read this if:** You're implementing the refactoring or need detailed design specs.

**Time to read:** 45 minutes

---

### 🔄 [AUTOTUNING_BEFORE_AFTER.md](AUTOTUNING_BEFORE_AFTER.md)
**Side-by-side code comparisons** showing concrete changes.

- 10 detailed before/after examples
- Line-by-line code diffs
- Extensibility demonstrations
- Performance measurements
- Migration checklist per kernel

**Read this if:** You want to see exactly what the code changes look like.

**Time to read:** 30 minutes

---

### ⚡ [AUTOTUNING_QUICK_REFERENCE.md](AUTOTUNING_QUICK_REFERENCE.md)
**Quick-start guide and reference** for using the new framework.

- Template code for new kernels
- Common patterns (context, validation, grid calculation)
- Adding tunable parameters (vectorization, shared memory, etc.)
- Debugging tips
- Performance tips
- Common mistakes
- Testing checklist

**Read this if:** You're adding a new kernel or modifying existing autotuning code.

**Time to read:** 15 minutes (reference)

---

### 💻 [examples/](examples/)
**Concrete code examples** of the new framework.

#### Core Framework Headers
- `orchestrator.h` - Main tuning orchestrator template
- `tuning_config.h` - Extensible configuration container
- `benchmarker.h` - Benchmarking engine template
- `cache_store.h` - On-disk cache management

#### Kernel Example
- `grayscale_kernel_traits_example.h` - Complete example of migrated kernel

**Read this if:** You're implementing the framework or want to see working code.

**Time to review:** 20 minutes

---

## Quick Navigation

### 👀 "I just want to understand the proposal"
→ Read [AUTOTUNING_REFACTOR_SUMMARY.md](AUTOTUNING_REFACTOR_SUMMARY.md)

### 🛠️ "I'm implementing the refactoring"
1. [AUTOTUNING_REFACTOR.md](AUTOTUNING_REFACTOR.md) - Design specs
2. [examples/](examples/) - Reference implementation
3. [AUTOTUNING_BEFORE_AFTER.md](AUTOTUNING_BEFORE_AFTER.md) - Migration patterns

### ✨ "I'm adding a new kernel"
→ Use [AUTOTUNING_QUICK_REFERENCE.md](AUTOTUNING_QUICK_REFERENCE.md) as a guide

### 🔍 "I want to see the code changes"
→ Read [AUTOTUNING_BEFORE_AFTER.md](AUTOTUNING_BEFORE_AFTER.md)

### 🎯 "I need to migrate an existing kernel"
→ Follow migration guide in [AUTOTUNING_REFACTOR.md](AUTOTUNING_REFACTOR.md#migration-guide)

---

## Key Benefits Summary

### ✅ Type Safety
- **Before:** `void*` casting, runtime errors
- **After:** Compile-time type checking

### ✅ Extensibility  
- **Before:** Framework changes needed for new parameters
- **After:** Add parameters in kernel traits only

### ✅ Context Awareness
- **Before:** One config for all workloads
- **After:** Different configs per size/context

### ✅ Code Organization
- **Before:** Scattered across multiple files
- **After:** Grouped in kernel traits

### ✅ Performance
- **Before:** O(n) cache lookup (~10ns)
- **After:** O(1) lookup with thread-local cache (~5ns)

### ✅ Maintainability
- **Before:** 80 lines boilerplate per kernel
- **After:** 40 lines traits definition per kernel

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────┐
│                   Client Code                        │
│  Defines KernelTraits for each kernel               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│       TuningOrchestrator<KernelTraits>              │
│  ┌──────────────────────────────────────────────┐  │
│  │  - get_or_tune(args, context)                │  │
│  │  - execute(args, context, stream)            │  │
│  └────┬─────────────────────────────────────────┘  │
│       │                                             │
│  ┌────▼─────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ ParameterSpace│  │Benchmarker│  │ CacheStore  │ │
│  │ - Candidates │  │ - Warmup  │  │ - Load/Save │ │
│  │ - Validation │  │ - Timing  │  │ - O(1) Lookup│ │
│  └──────────────┘  └───────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Key Abstractions

1. **KernelTraits** (Per-kernel definition)
   - `name()` - Unique identifier
   - `Args` - Type-safe arguments
   - `Context` - Cache key generation
   - `generate_candidates()` - Parameter space
   - `is_valid_config()` - Validation
   - `launch()` - Kernel invocation

2. **TuningConfig** (Extensible parameter container)
   - `set(name, value)` - Add parameter
   - `get<T>(name)` - Type-safe access
   - `get_or(name, default)` - Safe access with fallback

3. **TuningOrchestrator** (Framework coordinator)
   - `get_or_tune()` - Get optimal config
   - `execute()` - Convenience wrapper
   - Thread-local + persistent caching

4. **CacheStore** (Persistence layer)
   - O(1) lookup via hash table
   - JSON serialization
   - Per-GPU, per-kernel, per-context caching

---

## Implementation Roadmap

### Phase 1: Core Framework (2-3 days)
- [ ] Implement `TuningConfig` class
- [ ] Implement `CacheStore` with JSON serialization
- [ ] Implement `Benchmarker` template
- [ ] Implement `TuningOrchestrator` template
- [ ] Unit tests for each component

### Phase 2: Reference Migration (1 day)
- [ ] Define `GrayscaleKernelTraits`
- [ ] Refactor `apply_grayscale_autotuned()`
- [ ] Validate results match old implementation
- [ ] Performance testing

### Phase 3: Remaining Kernels (1 day)
- [ ] Migrate `negative` kernel
- [ ] Migrate `gaussian_blur` kernel
- [ ] Update documentation
- [ ] Integration tests

### Phase 4: Deprecation (0.5 days)
- [ ] Mark old `AutoTuner` as deprecated
- [ ] Update all call sites
- [ ] Add migration notes to docs

**Total Estimated Time:** 4.5-5.5 days

---

## Testing Strategy

### Unit Tests
- `TuningConfig` serialization/deserialization
- `CacheStore` load/save operations
- `Benchmarker` timing accuracy
- Config validation logic

### Integration Tests
- End-to-end autotuning
- Cache persistence across runs
- Multiple kernels, multiple contexts
- Performance regression tests

### Migration Validation
- Output equivalence (old vs new)
- Performance parity
- Cache file compatibility

---

## Performance Expectations

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Cache lookup | ~10ns | ~5ns | Thread-local + O(1) |
| Tuning time | ~500ms | ~500ms | Unchanged (6 configs) |
| Binary size | Baseline | +5% | Template instantiation |
| Compile time | ~15s | ~18s | Acceptable increase |
| Per-kernel LOC | 81 lines | 40 lines | 50% reduction |

---

## Success Criteria

### Must Have ✅
- [ ] All kernels produce identical results
- [ ] Cache files remain compatible
- [ ] No performance regression
- [ ] Compilation succeeds without warnings
- [ ] All tests pass

### Should Have ✅
- [ ] Cache lookup ≤ 10ns
- [ ] Per-kernel LOC reduced by ≥ 40%
- [ ] Binary size increase < 10%
- [ ] Documentation complete

### Nice to Have ✨
- [ ] Cache lookup < 5ns
- [ ] Compile time increase < 20%
- [ ] Example of extended parameter (vectorization)
- [ ] Migration script for automated refactoring

---

## Questions?

For specific questions about:
- **Architecture design** → See [AUTOTUNING_REFACTOR.md](AUTOTUNING_REFACTOR.md)
- **Code changes** → See [AUTOTUNING_BEFORE_AFTER.md](AUTOTUNING_BEFORE_AFTER.md)
- **Usage patterns** → See [AUTOTUNING_QUICK_REFERENCE.md](AUTOTUNING_QUICK_REFERENCE.md)
- **Implementation** → See [examples/](examples/)

---

## Contributing

When adding new documentation:
1. Update this README's table of contents
2. Cross-reference related documents
3. Add concrete code examples
4. Include performance analysis
5. Provide migration path for existing code

---

## Related Documentation

- [AUTOTUNING.md](AUTOTUNING.md) - Original autotuning design (current implementation)
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Performance measurements
- [AUTOTUNING_QUICKSTART.md](AUTOTUNING_QUICKSTART.md) - Getting started guide
- [AUTOTUNING_SUMMARY.md](AUTOTUNING_SUMMARY.md) - Tuning results analysis

