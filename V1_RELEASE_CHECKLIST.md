# V1.0 Release Implementation Checklist

**Use this checklist to track implementation progress.**

---

## Phase 1: Code Cleanup (Est: 1-2 days)

### Remove Old AutoTuner System
- [ ] Delete `src/core/autotuning.h`
- [ ] Delete `src/core/autotuning.cpp`
- [ ] Delete `src/filters/grayscale_autotune.hip.cpp` (old v1)
- [ ] Delete `src/filters/negative_autotune.hip.cpp` (old v1)
- [ ] Delete `src/filters/gaussian_blur_autotune.hip.cpp` (old v1)
- [ ] Remove references from `meson.build`
- [ ] Remove forward declarations from `src/filters/filters.h`
- [ ] Verify build: `ninja -C build`
- [ ] **Commit**: "refactor: Remove old AutoTuner system"

### Clean Commented Code
- [ ] Remove commented printf at `src/app/process.cpp:123`
- [ ] Remove commented helper at `src/app/process.cpp:170`
- [ ] Search for other commented experiments: `grep -r "^[[:space:]]*//" src/`
- [ ] **Commit**: "refactor: Clean up commented debug code"

### Fix Compiler Warnings
- [ ] Update `meson.build`: change `warning_level=0` → `warning_level=2`
- [ ] Run: `meson configure build -Dwarning_level=3`
- [ ] Run: `ninja -C build 2>&1 | tee warnings.log`
- [ ] Fix unused parameter warnings (use `[[maybe_unused]]`)
- [ ] Fix sign comparison warnings
- [ ] Verify: `ninja -C build` produces zero warnings
- [ ] **Commit**: "build: Enable warnings and fix issues"

---

## Phase 2: API Stabilization (Est: 1 day)

### Rename v2 → Production
- [ ] Update function names in `src/filters/filters.h`:
  - `apply_grayscale_autotuned_v2` → `apply_grayscale_autotuned`
  - `apply_negative_autotuned_v2` → `apply_negative_autotuned`
  - `apply_gaussian_blur_autotuned_v2` → `apply_gaussian_blur_autotuned`
- [ ] Update kernel names in `src/filters/filters.h`:
  - `"grayscale_v2"` → `"grayscale"`
  - `"negative_v2"` → `"negative"`
  - `"gaussian_blur_v2"` → `"gaussian_blur"`
- [ ] Rename files:
  - `grayscale_autotune_v2.hip.cpp` → `grayscale_autotune.hip.cpp`
  - `negative_autotune_v2.hip.cpp` → `negative_autotune.hip.cpp`
  - `gaussian_blur_autotune_v2.hip.cpp` → `gaussian_blur_autotune.hip.cpp`
- [ ] Update function names in implementation files (remove _v2)
- [ ] Update callsites in `src/core/gpu_utils.cpp`
- [ ] Update callsites in `bench/` directory
- [ ] Update `meson.build` with new filenames
- [ ] Verify build: `ninja -C build`
- [ ] Run tests: `./build/test-tier1-improvements`
- [ ] **Commit**: "refactor: Rename v2 → production (removes experimental suffix)"

### Reorganize Public API Headers
- [ ] Create directory: `mkdir -p include/hip-img-fx/autotune`
- [ ] Move headers:
  - `orchestrator.h` → `include/hip-img-fx/autotune/`
  - `tuning_config.h` → `include/hip-img-fx/autotune/`
  - `types.h` → `include/hip-img-fx/autotune/`
  - `kernel_traits_concepts.h` → `include/hip-img-fx/autotune/`
  - `benchmarker.h` → `include/hip-img-fx/autotune/`
  - `cache_store.h` → `include/hip-img-fx/autotune/`
  - `embedded_cache.h` → `include/hip-img-fx/autotune/`
- [ ] Update include paths in all headers (relative includes → full paths)
- [ ] Update include paths in all `.cpp` files
- [ ] Add to `meson.build`: `inc_pub = include_directories('include')`
- [ ] Update executable includes to use both `inc_src` and `inc_pub`
- [ ] Add install targets to `meson.build`:
  ```meson
  install_headers(
    'include/hip-img-fx/autotune/orchestrator.h',
    # ... (all 7 headers)
    subdir: 'hip-img-fx/autotune'
  )
  ```
- [ ] Verify build: `ninja -C build`
- [ ] Test include paths: `#include <hip-img-fx/autotune/orchestrator.h>`
- [ ] **Commit**: "refactor: Move public API headers to include/"

### Add Pkg-Config Support
- [ ] Add to `meson.build`:
  ```meson
  pkg = import('pkgconfig')
  pkg.generate(
    name: 'hip-img-fx',
    description: 'HIP autotuning framework for GPU kernels',
    version: meson.project_version(),
    subdirs: 'hip-img-fx'
  )
  ```
- [ ] Test installation: `meson setup build_test --prefix=/tmp/install`
- [ ] Test: `ninja -C build_test install`
- [ ] Verify: `PKG_CONFIG_PATH=/tmp/install/lib/pkgconfig pkg-config --cflags hip-img-fx`
- [ ] **Commit**: "build: Add pkg-config support for library users"

---

## Phase 3: Documentation (Est: 1-2 days)

### Archive Historical Documents
- [ ] Create: `mkdir docs/archive`
- [ ] Move historical docs (12 files):
  - `AUTOTUNING_BEFORE_AFTER.md`
  - `AUTOTUNING_REFACTOR*.md` (3 files)
  - `AUTOTUNING_SKELETON_SUMMARY.md`
  - `AUTOTUNING_FRAMEWORK_HARDENING.md`
  - `PHASE*.md` (3 files)
  - `TIER1_*.md` (2 files)
  - `BUGFIX_2D_BLOCKS.md`
- [ ] Update any links in remaining docs
- [ ] **Commit**: "docs: Archive historical development documents"

### Consolidate Autotuning Documentation
- [ ] Create `docs/AUTOTUNING_GUIDE.md`
- [ ] Merge content from:
  - `AUTOTUNING_QUICKSTART.md` (Quick Start section)
  - `AUTOTUNING.md` (Concepts & Architecture)
  - `AUTOTUNING_EXTENDED.md` (Advanced topics)
  - `AUTOTUNING_QUICK_REFERENCE.md` (API Reference)
  - `AUTOTUNING_OPTIMIZATION_STRATEGIES.md` (Advanced Topics)
  - `COMPILE_TIME_SAFETY_ENFORCEMENT.md` (Safety & Validation)
- [ ] Remove redundant sections
- [ ] Add clear table of contents
- [ ] Verify internal links work
- [ ] Delete source files after merge
- [ ] **Commit**: "docs: Consolidate autotuning documentation into single guide"

### Create Documentation Entry Point
- [ ] Create `docs/README.md`
- [ ] Add sections:
  - New Users Start Here
  - Performance Analysis
  - Reference
  - Archive
- [ ] Link to all primary docs
- [ ] Update root `README.md` to reference `docs/README.md`
- [ ] **Commit**: "docs: Add documentation entry point"

### Verify Code Examples
- [ ] Create `scripts/verify_doc_examples.sh`
- [ ] Test compile all examples in `docs/examples/`
- [ ] Fix any compilation errors
- [ ] Update examples to use new include paths
- [ ] Run: `./scripts/verify_doc_examples.sh`
- [ ] **Commit**: "docs: Verify and fix code examples"

---

## Phase 4: Testing & Polish (Est: 1 day)

### Reorganize Test Structure
- [ ] Create directories:
  - `mkdir -p bench/tests`
  - `mkdir -p bench/benchmarks`
- [ ] Move files:
  - `validate_grayscale_migration.cpp` → `bench/tests/test_correctness.cpp`
  - `test_tier1_improvements.cpp` → `bench/tests/test_integration.cpp`
  - `test_compile_time_safety.cpp` → `bench/tests/test_compile_safety.cpp`
  - `run_bench.cpp` → `bench/benchmarks/run_bench.cpp`
  - `benchmark_grayscale_migration.cpp` → `bench/benchmarks/benchmark_grayscale.cpp`
- [ ] Update `meson.build` with new paths
- [ ] Verify build: `ninja -C build`
- [ ] Run all tests to verify they still work
- [ ] **Commit**: "test: Reorganize test structure for clarity"

### Add test_compile_safety to Build
- [ ] Add executable to `meson.build`:
  ```meson
  executable(
    'test-compile-safety',
    'bench/tests/test_compile_time_safety.cpp',
    # ... dependencies
    install: false,
  )
  ```
- [ ] Fix any build issues
- [ ] Run: `./build/test-compile-safety`
- [ ] **Commit**: "test: Add compile-time safety tests to build"

### Improve Error Handling
- [ ] Review `src/core/autotune/benchmarker.h`
- [ ] Add HIP error checking after `hipDeviceSynchronize()`
- [ ] Add HIP error checking after `hipEventElapsedTime()`
- [ ] Standardize error message format: `[AutoTune Error]`, `[AutoTune Warning]`
- [ ] Test error paths (if possible)
- [ ] **Commit**: "fix: Improve error handling and diagnostics"

### Run Full Validation
- [ ] Clean build: `ninja -C build clean && ninja -C build`
- [ ] Run all tests:
  - [ ] `./build/test-correctness`
  - [ ] `./build/test-integration`
  - [ ] `./build/test-compile-safety`
- [ ] Run benchmark: `./build/hip-img-fx-bench`
- [ ] Test main app: `./build/hip-img-fx --help`
- [ ] Verify zero warnings: `ninja -C build 2>&1 | grep warning`
- [ ] **Commit**: "test: Verify all tests pass with clean builds"

---

## Phase 5: Release Preparation (Est: 0.5 day)

### Create CHANGELOG.md
- [ ] Create file in project root
- [ ] Add v1.0.0 section with:
  - Added (features)
  - Framework Features
  - Performance highlights
  - Documentation
  - Stability Guarantees
  - Not Yet Stable
- [ ] Add Unreleased section (placeholders)
- [ ] **Commit**: "docs: Add CHANGELOG for v1.0.0 release"

### Update README.md
- [ ] Add "Installing as Library" section
- [ ] Add "Using in your project" subsection (meson + code)
- [ ] Update "Project Status" section
- [ ] Add version badge
- [ ] Add "What's Stable" and "Roadmap" subsections
- [ ] Verify all links work
- [ ] **Commit**: "docs: Update README for v1.0.0 release"

### Update Version Number
- [ ] Change `meson.build`: `version: '0.2.0'` → `version: '1.0.0'`
- [ ] Reconfigure: `meson setup build --reconfigure`
- [ ] Verify: `meson introspect build --projectinfo | grep version`
- [ ] **Commit**: "chore: Bump version to 1.0.0"

### Verify License Headers
- [ ] Check all source files: `grep -L "Copyright\|License" src/**/*.{h,cpp}`
- [ ] Add missing headers where needed
- [ ] Verify LICENSE file is up to date
- [ ] **Commit**: "chore: Verify license headers"

### Create Release Tag
- [ ] Ensure all changes committed
- [ ] Run final validation: `ninja -C build && ./build/test-integration`
- [ ] Tag release: `git tag -a v1.0.0 -m "Release v1.0.0: Production-ready autotuning framework"`
- [ ] Push tag: `git push origin v1.0.0`

### Test Installation
- [ ] Clean install test:
  ```bash
  rm -rf /tmp/hip-img-fx-install
  meson setup build_install --prefix=/tmp/hip-img-fx-install
  ninja -C build_install install
  ```
- [ ] Verify headers installed:
  ```bash
  ls /tmp/hip-img-fx-install/include/hip-img-fx/autotune/
  ```
- [ ] Verify pkg-config works:
  ```bash
  PKG_CONFIG_PATH=/tmp/hip-img-fx-install/lib/pkgconfig pkg-config --cflags --libs hip-img-fx
  ```
- [ ] **Verify**: Installation successful

---

## Optional Enhancements

### Create Example Tutorial
- [ ] Create `examples/custom_kernel_tutorial.md`
- [ ] Add minimal working example
- [ ] Show step-by-step kernel trait implementation
- [ ] Test example compiles and runs
- [ ] **Commit**: "docs: Add custom kernel tutorial"

### CI Configuration
- [ ] Create `.github/workflows/ci.yml`
- [ ] Add build job (no GPU required)
- [ ] Add compile-time safety test job
- [ ] Add warning check job
- [ ] Test CI locally: `act` (if available)
- [ ] **Commit**: "ci: Add GitHub Actions workflow"

---

## Final Checklist

### Quality Gates
- [ ] Zero compiler warnings at `-Wextra`
- [ ] All tests passing
- [ ] Documentation builds without errors
- [ ] Examples compile successfully
- [ ] Installation verified on clean system

### Release Artifacts
- [ ] CHANGELOG.md complete
- [ ] README.md updated
- [ ] Version number updated (1.0.0)
- [ ] Git tag created (v1.0.0)
- [ ] GitHub release created (with notes)

### Success Metrics
- [ ] Build completes in < 5 minutes
- [ ] Public API < 10 header files ✓ (7 headers)
- [ ] Core docs < 3000 lines ✓ (~2000 after consolidation)
- [ ] Documentation searchable in < 2 clicks ✓ (docs/README.md)

---

## Notes

**Estimated Total Time**: 4-6 days of focused work

**Progress Tracking**: Update this checklist as you complete each item

**Rollback Points**: Each commit is a stable checkpoint

**Questions/Issues**: Document any blockers or questions for team review

---

**Current Status**: Ready to begin Phase 1

**Next Action**: Start with "Remove Old AutoTuner System"
