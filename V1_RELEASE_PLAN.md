# V1.0 Release Cleanup & Stabilization Plan

**Target**: Production-ready v1.0 release with clean API, minimal surface area, and comprehensive documentation

**Status**: Analysis Complete - Ready for Implementation

---

## Executive Summary

### Current State
- **Dual Autotuning Systems**: Old `AutoTuner` (unused) + New `TuningOrchestrator` (active)
- **Documentation Overload**: 23 docs (10,183 total lines), significant overlap
- **Public API Undefined**: No clear separation between public/internal headers
- **Build System**: Clean but lacks install targets for headers
- **Test Framework**: Comprehensive but includes standalone test file not integrated

### Critical Findings
✅ **Zero runtime overhead maintained** (NDEBUG behavior correct)  
✅ **Compile-time safety working** (C++20 concepts enforced)  
⚠️ **Old autotuning system unused** (can be removed)  
⚠️ **Documentation fragmentation** (needs consolidation)  
⚠️ **Public API exposure** (all headers currently internal)

---

## 1️⃣ Codebase Cleanup

### 1.1 Dead Code Removal

#### Old Autotuning System (100% unused)
**Remove these files:**
- `src/core/autotuning.h` (197 lines) - Old AutoTuner class
- `src/core/autotuning.cpp` (305 lines) - Old implementation
- `src/filters/*_autotune.hip.cpp` (3 files) - Old v1 implementations
  - `grayscale_autotune.hip.cpp`
  - `negative_autotune.hip.cpp`
  - `gaussian_blur_autotune.hip.cpp`

**Justification**: All production code uses `_v2` variants with `TuningOrchestrator`. The old `AutoTuner` class is never instantiated.

**Impact**: -900 lines of unused code

#### Commented Code
**File**: `src/app/process.cpp`
- Line 123: `// printf("Launching GPU filter kernel...` (debug printf)
- Line 170: `// print_image_info(&image);` (unused helper)

**Action**: Remove commented debug statements

### 1.2 Rename v2 → Production

**Rationale**: The "v2" suffix indicates experimental status. For v1.0 release, the new system IS the system.

**Changes**:
```
# Functions
apply_grayscale_autotuned_v2    → apply_grayscale_autotuned
apply_negative_autotuned_v2     → apply_negative_autotuned
apply_gaussian_blur_autotuned_v2 → apply_gaussian_blur_autotuned

# Kernel names (affects cache keys)
"grayscale_v2"      → "grayscale"
"negative_v2"       → "negative"
"gaussian_blur_v2"  → "gaussian_blur"

# Files
grayscale_autotune_v2.hip.cpp    → grayscale_autotune.hip.cpp
negative_autotune_v2.hip.cpp     → negative_autotune.hip.cpp
gaussian_blur_autotune_v2.hip.cpp → gaussian_blur_autotune.hip.cpp
```

**Migration Path**: Cache invalidation acceptable for v1.0 (users retune once)

### 1.3 Consolidate Test Framework

**Issue**: `src/core/autotune/test_framework.cpp` exists but isn't built/run

**Options**:
1. **Delete** - Functionality already tested in `bench/test_tier1_improvements.cpp`
2. **Move** - Relocate to `bench/` and add to build
3. **Document** - Mark as developer-only validation tool

**Recommendation**: Delete (redundant with integration tests)

---

## 2️⃣ Public API Definition

### 2.1 API Surface Area

#### Public Headers (for kernel authors)

**Tier 1: Essential Autotuning API**
```
include/hip-img-fx/autotune/
├── orchestrator.h           # TuningOrchestrator<> (main API)
├── tuning_config.h          # TuningConfig, TuningOptions
├── types.h                  # AUTOTUNE_ASSERT, TuningOptions
└── kernel_traits_concepts.h # ValidKernelTraits concept
```

**Tier 2: Optional Framework Components**
```
include/hip-img-fx/autotune/
├── benchmarker.h            # TuningBenchmarker<> (for custom benchmarking)
├── cache_store.h            # CacheStore (for custom caching strategies)
└── embedded_cache.h         # EmbeddedCacheInitializer<> (optional optimization)
```

#### Internal Headers (framework maintainers only)

**Keep in `src/core/autotune/`:**
- (None - all headers are potentially public API)

**Application Headers (not for library users):**
```
src/app/
├── process.h                # Application-specific processing logic
└── main.cpp

src/cli/
└── cli_parser.h             # CLI parsing (not library concern)

src/core/
├── gpu_utils.h              # Internal GPU utilities
├── image.h                  # Internal image handling
└── vendor_stb.h             # Vendor header wrapper

src/filters/
└── filters.h                # Example filter implementations
```

### 2.2 Header Organization

**Proposed Structure**:
```
hip-img-fx/
├── include/                 # PUBLIC API (installed)
│   └── hip-img-fx/
│       └── autotune/        # Autotuning framework headers
│           ├── orchestrator.h
│           ├── tuning_config.h
│           ├── types.h
│           ├── kernel_traits_concepts.h
│           ├── benchmarker.h
│           ├── cache_store.h
│           └── embedded_cache.h
│
├── src/                     # INTERNAL IMPLEMENTATION (not installed)
│   ├── app/                 # Application code (not a library)
│   ├── cli/                 # CLI tools (not a library)
│   ├── core/                # Core utilities (internal)
│   │   ├── autotune/        # Implementation files (.cpp)
│   │   │   ├── tuning_config.cpp
│   │   │   └── cache_store.cpp
│   │   ├── gpu_utils.h/cpp
│   │   ├── image.h/cpp
│   │   └── vendor_stb.h
│   └── filters/             # Example implementations
│
└── examples/                # Example usage (for library users)
    └── custom_kernel_example.hip.cpp
```

### 2.3 API Guarantees (v1.0)

**Stable & Supported:**
- `TuningOrchestrator<KernelTraits>` template interface
- `ValidKernelTraits` concept requirements
- `TuningConfig` get/set/iteration API
- `TuningOptions` configuration structure
- Cache file format (JSON schema)
- Compile-time validation behavior

**Not Yet Stable:**
- Benchmarking heuristics (early-exit thresholds)
- Cache pruning strategies
- Specific candidate generation patterns
- Error message wording (can improve without breaking)

**Explicitly Not Guaranteed:**
- Performance characteristics (hardware-dependent)
- Cache file locations (may become configurable)
- Verbose output formatting

---

## 3️⃣ Build & Configuration Hygiene

### 3.1 Install Targets

**Add to meson.build:**
```meson
# Install public headers
install_headers(
  'include/hip-img-fx/autotune/orchestrator.h',
  'include/hip-img-fx/autotune/tuning_config.h',
  'include/hip-img-fx/autotune/types.h',
  'include/hip-img-fx/autotune/kernel_traits_concepts.h',
  'include/hip-img-fx/autotune/benchmarker.h',
  'include/hip-img-fx/autotune/cache_store.h',
  'include/hip-img-fx/autotune/embedded_cache.h',
  subdir: 'hip-img-fx/autotune'
)

# Install pkg-config file
pkg = import('pkgconfig')
pkg.generate(
  name: 'hip-img-fx',
  description: 'HIP autotuning framework for GPU kernels',
  version: meson.project_version(),
  subdirs: 'hip-img-fx'
)
```

### 3.2 Build Flags Review

**Current flags (meson.build):**
```meson
default_options: [
  'cpp_std=c++20',        # ✅ Required for concepts
  'b_lto=true',           # ✅ Aggressive optimization
  'buildtype=release',    # ⚠️  Should be 'release' for install, flexible for dev
  'warning_level=0',      # ❌ Should be '2' or '3' for clean builds
]
```

**Recommendations:**
1. **Change `warning_level=0` → `warning_level=2`** (catch real issues)
2. **Remove `buildtype=release` from defaults** (let user choose)
3. **Add debug option for verbose autotuning**:
   ```meson
   add_project_arguments(
     '-DAUTOTUNE_VERBOSE',  # Optional: enable by default
     language: 'cpp'
   )
   ```

### 3.3 NDEBUG Behavior

**Verification**: ✅ Correctly implemented
- `AUTOTUNE_ASSERT` compiles to `((void)0)` when `NDEBUG` defined
- No runtime overhead in release builds
- Debug diagnostics verbose and helpful

**Action**: No changes needed

### 3.4 Warnings Cleanup

**Current state**: `warning_level=0` hides potential issues

**Action**: Enable warnings and fix any issues
```bash
# Test with high warning level
meson configure build -Dwarning_level=3
ninja -C build
```

**Expected issues**:
- Unused parameters (mark with `[[maybe_unused]]`)
- Potential sign comparison warnings
- Missing virtual destructors (if any)

---

## 4️⃣ Documentation Cleanup

### 4.1 Documentation Debt Analysis

**23 documents, 10,183 total lines**

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| Autotuning Core | 13 | 6,812 | 📦 Consolidate |
| Phase Reports | 5 | 2,084 | 🗑️ Archive |
| Benchmark Reports | 2 | 503 | ✅ Keep |
| Safety/Compile | 2 | 901 | 📦 Merge |
| Bug Reports | 1 | 178 | 🗑️ Archive |

### 4.2 Consolidation Plan

#### Delete (Archive to `docs/archive/`)
**Redundant/historical documents:**
- `AUTOTUNING_BEFORE_AFTER.md` (584 lines) - Historical comparison
- `AUTOTUNING_REFACTOR.md` (971 lines) - Migration guide (completed)
- `AUTOTUNING_REFACTOR_README.md` (302 lines) - Duplicate
- `AUTOTUNING_REFACTOR_SUMMARY.md` (265 lines) - Duplicate
- `AUTOTUNING_SKELETON_SUMMARY.md` (568 lines) - Internal design notes
- `AUTOTUNING_FRAMEWORK_HARDENING.md` (719 lines) - Implementation notes
- `PHASE1_COMPLETION_REPORT.md` (436 lines) - Project management
- `PHASE2A_CODE_COMPARISON.md` (206 lines) - Historical comparison
- `PHASE2A_GRAYSCALE_MIGRATION_REPORT.md` (536 lines) - Migration notes
- `TIER1_IMPLEMENTATION_SUMMARY.md` (260 lines) - Implementation notes
- `TIER1_VERIFICATION_CHECKLIST.md` (98 lines) - Internal checklist
- `BUGFIX_2D_BLOCKS.md` (178 lines) - Bug report (fixed)

**Total**: 5,723 lines to archive

#### Consolidate

**Merge into single comprehensive guide:**

**NEW: `docs/AUTOTUNING_GUIDE.md`** (consolidates 5 files)
```markdown
# HIP Kernel Autotuning Framework Guide

## Quick Start (from AUTOTUNING_QUICKSTART.md)
- 5-minute getting started
- Example usage
- Common patterns

## Concepts & Architecture (from AUTOTUNING.md + AUTOTUNING_EXTENDED.md)
- How autotuning works
- Three-tier caching
- Benchmarking process
- Cache format

## API Reference (from AUTOTUNING_QUICK_REFERENCE.md)
- KernelTraits requirements
- TuningOrchestrator API
- TuningConfig methods
- Configuration examples

## Advanced Topics (from AUTOTUNING_OPTIMIZATION_STRATEGIES.md)
- Custom benchmarking strategies
- Cache warming
- Multi-GPU considerations
- Performance tuning tips

## Safety & Validation (from compile-time safety docs)
- Compile-time checks
- Runtime assertions
- Debug vs release behavior
- Common mistakes
```

**Merged files** (6 → 1):
- `AUTOTUNING_QUICKSTART.md` (228 lines)
- `AUTOTUNING.md` (325 lines)
- `AUTOTUNING_EXTENDED.md` (338 lines)
- `AUTOTUNING_QUICK_REFERENCE.md` (583 lines)
- `AUTOTUNING_OPTIMIZATION_STRATEGIES.md` (1,562 lines)
- `COMPILE_TIME_SAFETY_ENFORCEMENT.md` (570 lines)

**Result**: ~3,600 lines → ~2,000 lines (consolidated, deduped)

#### Keep As-Is
- `BENCHMARK_RESULTS.md` (266 lines) - Empirical data reference
- `BENCHMARK_FIX_REPORT.md` (237 lines) - Important methodology notes
- `AUTOTUNING_SUMMARY.md` (236 lines) - High-level overview (for README)
- `COMPILE_TIME_SAFETY_QUICK_REFERENCE.md` (331 lines) - Quick reference card

### 4.3 New Entry Point

**Create: `docs/README.md`** (Start Here)
```markdown
# HIP Image FX Documentation

## New Users Start Here

1. **[Quick Start](../README.md#quick-start)** - Build and run in 5 minutes
2. **[Autotuning Guide](AUTOTUNING_GUIDE.md)** - Complete framework documentation
3. **[Examples](../examples/)** - Sample kernel implementations

## Performance Analysis

- **[Benchmark Results](BENCHMARK_RESULTS.md)** - Empirical performance data
- **[Benchmark Methodology](BENCHMARK_FIX_REPORT.md)** - How we measure

## Reference

- **[Quick Reference Card](COMPILE_TIME_SAFETY_QUICK_REFERENCE.md)** - API cheat sheet
- **[Framework Overview](AUTOTUNING_SUMMARY.md)** - Architecture summary

## Archive

Historical development documents: [docs/archive/](archive/)
```

### 4.4 Code Snippet Verification

**Action**: Verify all code examples compile against current API

**Files to check**:
- `docs/examples/*.h` (4 example files)
- Inline code blocks in documentation

**Tool**: Create validation script
```bash
#!/bin/bash
# scripts/verify_doc_examples.sh
for example in docs/examples/*.h; do
  echo "Checking $example..."
  gcc -std=c++20 -fsyntax-only -I include/ -I /opt/rocm/include "$example"
done
```

---

## 5️⃣ Error Handling & Diagnostics

### 5.1 Compile-Time Error Messages

**Current state**: ✅ Excellent

**Examples**:
```cpp
static_assert(StatelessKernelTraits<KernelTraits>,
    "KernelTraits must be stateless (no mutable members)");

static_assert(HasStableCacheKey<typename KernelTraits::Context>,
    "KernelTraits::Context must provide cache_key() method");
```

**Action**: No changes needed (already actionable and clear)

### 5.2 Runtime Error Messages

**Review findings**:
- ✅ Debug assertions are verbose and helpful
- ✅ Cache errors provide file paths
- ⚠️ Some HIP error codes not checked

**Improvements needed**:

**File**: `src/core/autotune/benchmarker.h`
```cpp
// Current: Silent failure on HIP errors
hipDeviceSynchronize();

// Better: Check and report
hipError_t err = hipDeviceSynchronize();
if (err != hipSuccess) {
    fprintf(stderr, "[AutoTune] GPU sync failed: %s\n", 
            hipGetErrorString(err));
    return BenchmarkResult(); // invalid result
}
```

### 5.3 User-Facing Messages

**Standardize format**:
```
[AutoTune] Loading cache from 'path/to/cache.json'...
[AutoTune] Tuning kernel 'grayscale' (16 candidates)...
[AutoTune] ├─ Tested [64x1]: 0.0382 ms
[AutoTune] ├─ Tested [128x1]: 0.0371 ms
[AutoTune] └─ Best: [16x8] = 0.0337 ms
[AutoTune] Saved cache to 'path/to/cache.json'

[AutoTune Error] Failed to open cache file: permission denied
[AutoTune Warning] Skipping invalid config [1024x2]: exceeds thread limit
```

**Tone**: Informative, consistent prefix, actionable on errors

---

## 6️⃣ Testing & Validation

### 6.1 Current Test Coverage

**Validation Tests**:
- ✅ `bench/validate_grayscale_migration.cpp` - Correctness validation
- ✅ `bench/test_tier1_improvements.cpp` - Integration test
- ✅ `bench/test_compile_time_safety.cpp` - Concept validation (not built)

**Benchmarking**:
- ✅ `bench/run_bench.cpp` - Full benchmark suite
- ✅ `bench/benchmark_grayscale_migration.cpp` - Migration benchmark
- ✅ `bench/scripts/run_benchmark.sh` - Automated benchmarking

### 6.2 Test Organization

**Current structure**:
```
bench/
├── run_bench.cpp                      # Main benchmark
├── benchmark_grayscale_migration.cpp  # Specific benchmark
├── validate_grayscale_migration.cpp   # Correctness test
├── test_tier1_improvements.cpp        # Integration test
└── test_compile_time_safety.cpp       # ⚠️ Not built!
```

**Recommendation**: Clarify purpose
```
bench/
├── benchmarks/                        # Performance tests
│   ├── run_bench.cpp
│   └── benchmark_grayscale.cpp
│
├── tests/                             # Correctness & validation
│   ├── test_correctness.cpp           # (rename validate_grayscale_migration)
│   ├── test_integration.cpp           # (rename test_tier1_improvements)
│   └── test_compile_safety.cpp        # (rename + add to build)
│
└── scripts/                           # Automation
    ├── run_benchmark.sh
    └── run_analysis.sh
```

### 6.3 Missing Tests

**Edge Cases**:
- Empty candidate list (should static_assert)
- All candidates pruned as invalid (runtime assert)
- Cache file corruption (graceful fallback to retuning)
- Multiple threads accessing cache concurrently (thread safety)

**Recommendation**: Add `bench/tests/test_edge_cases.cpp`

### 6.4 CI Suitability

**Current blockers**:
1. **Hardware dependency**: Requires AMD GPU
2. **Build time**: Full benchmark suite ~30 seconds
3. **Non-determinism**: GPU timing has variance

**CI Strategy**:
```yaml
# .github/workflows/ci.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build (no GPU required)
        run: meson setup build && ninja -C build
      
      - name: Compile-time tests
        run: ./build/test-compile-safety  # exits without GPU
  
  benchmark:
    runs-on: [self-hosted, amd-gpu]  # Requires GPU runner
    steps:
      - name: Quick benchmark
        run: ./bench/scripts/run_quick_bench.sh  # Subset of tests
```

---

## 7️⃣ Release Artifacts

### 7.1 README.md Updates

**Current README**: ✅ Good foundation (442 lines)

**Additions needed**:

**Section: "Installing as Library"** (new)
```markdown
## Installing as Library

### System-wide installation

```bash
meson setup build --prefix=/usr/local
ninja -C build install
```

### Using in your project

```meson
# meson.build
hip_img_fx = dependency('hip-img-fx')
executable('my_app', 'main.cpp', dependencies: hip_img_fx)
```

```cpp
// my_kernel.hip.cpp
#include <hip-img-fx/autotune/orchestrator.h>

// Define your kernel traits...
```
```

**Section: "Project Status"** (update)
```markdown
## Project Status

**Version**: 1.0.0 (January 2026)  
**Status**: Production-ready  
**License**: MIT  

### What's Stable (v1.0)
- ✅ Autotuning framework API
- ✅ Compile-time safety enforcement
- ✅ Three-tier caching system
- ✅ Example filter implementations

### Roadmap
- Multi-GPU orchestration
- Distributed cache sharing
- Additional optimization heuristics
```

### 7.2 CHANGELOG.md (new)

**Create**: Root-level changelog
```markdown
# Changelog

All notable changes to HIP Image FX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-11

### Added
- Complete autotuning framework with trait-based kernel interface
- Compile-time safety enforcement via C++20 concepts
- Three-tier caching system (thread-local, persistent, tuning)
- Comprehensive benchmark suite with statistical analysis
- Example implementations (grayscale, negative, Gaussian blur)

### Framework Features
- `TuningOrchestrator<KernelTraits>` template-based orchestration
- `ValidKernelTraits` concept for compile-time validation
- Configurable benchmarking with early-exit heuristics
- JSON-based persistent cache with GPU architecture detection
- Zero runtime overhead in release builds (NDEBUG)

### Performance
- 577× speedup on Gaussian blur (vs single-threaded CPU)
- 41.8× speedup on Gaussian blur (vs 32-thread OpenMP)
- Comprehensive benchmark results documented

### Documentation
- Complete API reference
- Quick start guide
- Performance analysis
- Compile-time safety reference

### Stability Guarantees (v1.0)
- Public API: `TuningOrchestrator`, `TuningConfig`, `ValidKernelTraits`
- Cache format: JSON schema stable
- Compile-time validation: Concept requirements stable

### Not Yet Stable
- Benchmarking heuristics (may improve)
- Cache pruning strategies (may optimize)
- Error message wording (may clarify)

## [Unreleased]
- Multi-GPU support
- Distributed cache synchronization
- Additional optimization strategies
```

### 7.3 Version Tagging

**Current**: `version: '0.2.0'` in meson.build

**Update to**: `version: '1.0.0'`

**Git tags**:
```bash
git tag -a v1.0.0 -m "Release v1.0.0: Production-ready autotuning framework"
git push origin v1.0.0
```

### 7.4 License Verification

**Current**: `LICENSE` file exists (MIT)

**Action**: Verify all source files have license headers

```bash
# Check for missing headers
grep -L "Copyright\|License" src/**/*.{h,cpp} include/**/*.h
```

---

## Commit Structure

### Recommended Commit Sequence

```
1. refactor: Remove old AutoTuner system
   - Delete src/core/autotuning.{h,cpp}
   - Delete src/filters/*_autotune.hip.cpp (v1 files)
   - Update meson.build to remove old files
   
2. refactor: Rename v2 → production
   - Rename filter functions (remove _v2 suffix)
   - Rename kernel cache keys
   - Rename files (*_v2.hip.cpp → *.hip.cpp)
   - Update all callsites
   
3. refactor: Reorganize public API headers
   - Create include/hip-img-fx/autotune/ directory
   - Move public headers from src/core/autotune/
   - Update include paths throughout codebase
   - Add install targets to meson.build
   
4. refactor: Clean up commented code
   - Remove commented debug printfs
   - Clean up commented experiments
   
5. build: Improve build configuration
   - Enable warning_level=2
   - Fix all compiler warnings
   - Add pkg-config file generation
   - Update default build options
   
6. docs: Consolidate documentation
   - Archive historical documents
   - Merge autotuning docs into single guide
   - Create docs/README.md entry point
   - Verify all code examples compile
   
7. fix: Improve error handling
   - Add HIP error checking to benchmarker
   - Standardize error message format
   - Improve user-facing diagnostics
   
8. test: Reorganize test structure
   - Move bench files to bench/tests/ and bench/benchmarks/
   - Add test_compile_safety to build
   - Create test_edge_cases.cpp
   
9. chore: Prepare v1.0 release
   - Update README.md (add installation section)
   - Create CHANGELOG.md
   - Update version to 1.0.0
   - Verify license headers
   
10. docs: Add examples for library users
    - Create examples/custom_kernel_tutorial.md
    - Add minimal working example
    - Document API usage patterns
```

---

## Blocking Issues

### Critical (Must Fix Before Release)

**None identified** - Codebase is functionally complete and correct

### High Priority (Should Fix)

1. **Public API Reorganization** - Headers need to move to `include/`
2. **Documentation Consolidation** - 23 files → ~8 files
3. **Remove Dead Code** - Old autotuner unused (900 lines)

### Medium Priority (Good to Fix)

4. **Rename v2 → Production** - Remove "experimental" naming
5. **Warning Level** - Enable and fix warnings
6. **Test Organization** - Clarify bench structure

### Low Priority (Post-v1.0)

7. **CI Integration** - Requires GPU runner setup
8. **Examples Repository** - Separate examples/tutorials
9. **Performance** - Further optimization opportunities

---

## Post-Release Follow-ups

### v1.1 Candidates
- Multi-GPU support (partition workloads)
- Remote cache synchronization (team caching)
- Python bindings (for ML frameworks)
- ROCm 6.0+ compatibility testing

### v2.0 Considerations
- Distributed autotuning (cluster-wide optimization)
- Cost models (analytical performance prediction)
- JIT kernel generation (runtime specialization)
- Breaking API changes (if needed for above features)

---

## Success Metrics

### Technical Metrics
- [ ] Build produces zero warnings at `-Wextra`
- [ ] All tests pass consistently (deterministic)
- [ ] Documentation examples compile and run
- [ ] Public API < 10 header files
- [ ] Core docs < 3000 lines total

### Usability Metrics
- [ ] New contributor can build project in < 5 minutes
- [ ] Kernel author can integrate autotuning in < 30 minutes
- [ ] Public API understandable without reading internal code
- [ ] Documentation search finds answers in < 2 clicks

### Release Readiness
- [ ] All "Critical" blocking issues resolved
- [ ] CHANGELOG.md documents all changes
- [ ] Version tagged in git
- [ ] GitHub release with artifacts
- [ ] Installation instructions verified on clean system

---

## Appendix: File-by-File Changes

### Files to DELETE
```
src/core/autotuning.h                               # 197 lines
src/core/autotuning.cpp                             # 305 lines
src/filters/grayscale_autotune.hip.cpp              # 97 lines
src/filters/negative_autotune.hip.cpp               # 92 lines
src/filters/gaussian_blur_autotune.hip.cpp          # 98 lines
src/core/autotune/test_framework.cpp                # 348 lines (optional: move to bench/)

docs/AUTOTUNING_BEFORE_AFTER.md                     # 584 lines
docs/AUTOTUNING_REFACTOR.md                         # 971 lines
docs/AUTOTUNING_REFACTOR_README.md                  # 302 lines
docs/AUTOTUNING_REFACTOR_SUMMARY.md                 # 265 lines
docs/AUTOTUNING_SKELETON_SUMMARY.md                 # 568 lines
docs/AUTOTUNING_FRAMEWORK_HARDENING.md              # 719 lines
docs/PHASE1_COMPLETION_REPORT.md                    # 436 lines
docs/PHASE2A_CODE_COMPARISON.md                     # 206 lines
docs/PHASE2A_GRAYSCALE_MIGRATION_REPORT.md          # 536 lines
docs/TIER1_IMPLEMENTATION_SUMMARY.md                # 260 lines
docs/TIER1_VERIFICATION_CHECKLIST.md                # 98 lines
docs/BUGFIX_2D_BLOCKS.md                            # 178 lines

Total: ~6,660 lines removed
```

### Files to RENAME
```
src/filters/grayscale_autotune_v2.hip.cpp    → src/filters/grayscale_autotune.hip.cpp
src/filters/negative_autotune_v2.hip.cpp     → src/filters/negative_autotune.hip.cpp
src/filters/gaussian_blur_autotune_v2.hip.cpp → src/filters/gaussian_blur_autotune.hip.cpp
```

### Files to MOVE
```
src/core/autotune/orchestrator.h            → include/hip-img-fx/autotune/orchestrator.h
src/core/autotune/tuning_config.h           → include/hip-img-fx/autotune/tuning_config.h
src/core/autotune/types.h                   → include/hip-img-fx/autotune/types.h
src/core/autotune/kernel_traits_concepts.h  → include/hip-img-fx/autotune/kernel_traits_concepts.h
src/core/autotune/benchmarker.h             → include/hip-img-fx/autotune/benchmarker.h
src/core/autotune/cache_store.h             → include/hip-img-fx/autotune/cache_store.h
src/core/autotune/embedded_cache.h          → include/hip-img-fx/autotune/embedded_cache.h

bench/validate_grayscale_migration.cpp      → bench/tests/test_correctness.cpp
bench/test_tier1_improvements.cpp           → bench/tests/test_integration.cpp
bench/test_compile_time_safety.cpp          → bench/tests/test_compile_safety.cpp
bench/run_bench.cpp                         → bench/benchmarks/run_bench.cpp
bench/benchmark_grayscale_migration.cpp     → bench/benchmarks/benchmark_grayscale.cpp
```

### Files to MERGE
```
docs/AUTOTUNING_QUICKSTART.md               ┐
docs/AUTOTUNING.md                          │
docs/AUTOTUNING_EXTENDED.md                 ├─→ docs/AUTOTUNING_GUIDE.md
docs/AUTOTUNING_QUICK_REFERENCE.md          │
docs/AUTOTUNING_OPTIMIZATION_STRATEGIES.md  │
docs/COMPILE_TIME_SAFETY_ENFORCEMENT.md     ┘
```

### Files to CREATE
```
CHANGELOG.md                                # Release notes
docs/README.md                              # Documentation index
docs/AUTOTUNING_GUIDE.md                    # Consolidated guide
include/hip-img-fx/autotune/*.h             # Public API headers
bench/tests/test_edge_cases.cpp             # Additional test coverage
scripts/verify_doc_examples.sh              # CI validation script
examples/custom_kernel_tutorial.md          # Library usage guide
```

### Files to MODIFY
```
README.md                                   # Add installation section, update status
meson.build                                 # Remove old files, add install targets, fix warnings
src/core/gpu_utils.cpp                      # Update function calls (remove _v2)
src/app/process.cpp                         # Clean commented code
src/filters/filters.h                       # Update function declarations
src/core/autotune/benchmarker.h             # Add HIP error checking
All files in src/                           # Update include paths after header move
```

---

## Execution Plan

### Phase 1: Code Cleanup (1-2 days)
- Remove old autotuner system
- Clean commented code
- Fix compiler warnings

### Phase 2: API Stabilization (1 day)
- Reorganize headers to include/
- Rename v2 → production
- Update all include paths

### Phase 3: Documentation (1-2 days)
- Archive historical docs
- Consolidate autotuning docs
- Create entry point
- Verify examples

### Phase 4: Testing & Validation (1 day)
- Reorganize test structure
- Add edge case tests
- Run full benchmark suite
- Verify clean builds

### Phase 5: Release Preparation (0.5 day)
- Update version numbers
- Create CHANGELOG
- Update README
- Tag release

**Total Estimate**: 4-6 days of focused work

---

## Risk Assessment

### Low Risk
- Removing dead code (unused autotuner)
- Documentation consolidation
- Version number update

### Medium Risk
- Renaming v2 → production (breaks existing caches)
  - **Mitigation**: Document that users must retune once
- Moving headers to include/ (breaks internal includes)
  - **Mitigation**: Update all includes in same commit
- Enabling warnings (may reveal issues)
  - **Mitigation**: Fix incrementally before merging

### High Risk
- None identified

### Rollback Plan
- Keep archive branches for each major change
- Tag intermediate milestones: `v1.0.0-rc1`, `v1.0.0-rc2`, etc.
- Maintain git history for easy revert if issues found

---

**END OF RELEASE PLAN**

**Next Steps**: 
1. Review and approve plan
2. Create GitHub issues for tracking
3. Execute phases sequentially
4. Review after each phase completion
