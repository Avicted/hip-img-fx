# Testing Documentation

## Overview

The hip-img-fx project includes comprehensive tests covering filters, image I/O, CLI parsing, autotuning, and GPU integration. Tests are built using GoogleTest and support both CPU-only and GPU environments.

**Quick Start:** See [QUICKREF.md](QUICKREF.md) for common commands.

## Test Structure

```
tests/
├── test_main.cpp                       # Main entry with GPU detection
├── test_helpers.h/cpp                  # Utility functions for test data generation
│
├── Core Filters & Processing
├── test_filters_cpu.cpp                # CPU filter correctness tests
├── test_filters_cpu_extended.cpp       # Extended CPU filter tests
├── test_filters_gpu.cpp                # GPU vs CPU validation tests
├── test_filters_traits.cpp             # Filter trait system tests
├── test_process.cpp                    # Process function tests
├── test_process_extended.cpp           # Extended process tests
├── test_process_errors.cpp             # Process error handling tests
│
├── Image I/O
├── test_image_io.cpp                   # Image loading/saving tests
├── test_image_formats.cpp              # Extended format tests
├── test_image_edge_cases.cpp           # Edge case handling tests
│
├── CLI & Application
├── test_cli_parser.cpp                 # CLI argument parsing tests
├── test_cli_parser_extended.cpp        # Extended CLI tests
├── test_cli_parser_death_tests.cpp     # CLI death/error tests
├── test_main_app.cpp                   # Main application tests
│
├── GPU Infrastructure
├── test_gpu_utils.cpp                  # GPU utility tests
├── test_gpu_utils_extended.cpp         # Extended GPU utility tests
├── test_gpu_integration.cpp            # GPU memory/batch tests
│
├── Autotuning Framework
├── test_autotune_config.cpp            # TuningConfig validation tests
├── test_autotune_extended.cpp          # Extended autotune tests
├── test_cache_store.cpp                # Cache persistence tests
├── test_cache_store_spec.cpp           # Cache store specification tests
├── test_tuning_config_spec.cpp         # TuningConfig specification tests
├── test_tuning_options_spec.cpp        # TuningOptions specification tests
├── test_orchestrator_spec.cpp          # Orchestrator specification tests
├── test_benchmarker_spec.cpp           # Benchmarker specification tests
├── test_hip_event_spec.cpp             # HIP event wrapper tests
├── test_kernel_traits_concepts_spec.cpp # Kernel traits & concepts tests
├── test_generate_candidates.cpp        # Candidate generation tests
│
└── Benchmarking
    └── test_benchmark_runner.cpp       # Benchmark utility function tests
```

**Total: ~690 tests across 33 test files**

**Coverage Achieved (as of v1.1.0):**
- **Line Coverage**: 71.0% (11213/15801 lines)
- **Function Coverage**: 92.2% (2540/2756 functions)
- **Branch Coverage**: 33.2% (15514/46698 branches)

## Building Tests

### Prerequisites

Install GoogleTest:
```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev

# Arch Linux
sudo pacman -S gtest

# From source
git clone https://github.com/google/googletest.git
cd googletest
cmake -S . -B build
sudo cmake --build build --target install
```

### Build with Tests

```bash
# Configure with tests enabled (GoogleTest must be available)
meson setup build --native-file native/hip.ini

# Build including tests
ninja -C build

# The test executable will be: build/hip-img-fx-tests
```

If GoogleTest is not found, Meson will show a warning and skip test building.

## Running Tests

### Run All Tests

```bash
# Using Meson test runner (recommended)
meson test -C build --print-errorlogs

# Or run executable directly
./build/hip-img-fx-tests

# From VSCode: Run task "Test: All" (Ctrl+Shift+P -> Tasks: Run Task)
```

### Run Specific Test Suites

```bash
# Run only CPU tests (no GPU required)
meson test -C build --suite cpu --print-errorlogs
./build/hip-img-fx-tests --gtest_filter="-*GPU*"

# Run only GPU tests (requires GPU)
./build/hip-img-fx-tests --gtest_filter="*GPU*"

# Run specific test file
./build/hip-img-fx-tests --gtest_filter="FiltersCPU.*"
./build/hip-img-fx-tests --gtest_filter="ImageIO.*"
./build/hip-img-fx-tests --gtest_filter="CLIParser.*"
./build/hip-img-fx-tests --gtest_filter="AutotuneConfig.*"
./build/hip-img-fx-tests --gtest_filter="CacheStore.*"

# Run specific test
./build/hip-img-fx-tests --gtest_filter="FiltersCPU.GrayscaleKnownValues"
```

### Verbose Output

```bash
# Show detailed test output with Meson
meson test -C build --verbose --print-errorlogs

# Show GoogleTest verbose output
./build/hip-img-fx-tests --gtest_print_time=1 --gtest_color=yes

# List all tests without running
./build/hip-img-fx-tests --gtest_list_tests
```

### VSCode Tasks

Access tests via Command Palette (Ctrl+Shift+P -> Tasks: Run Task):

- **Test: All** - Run all tests (default test task)
- **Test: CPU Only** - Run only CPU tests (no GPU required)
- **Test: Run Executable** - Run test executable directly
- **Test: Verbose** - Run with verbose output
- **Test: Coverage Report** - Generate coverage report

## Code Coverage

### Generate Coverage Report

```bash
# Using provided script
./scripts/generate_coverage.sh

# Or manually
meson setup build --native-file native/hip.ini -Db_coverage=true --reconfigure
ninja -C build
meson test -C build
ninja -C build coverage-html

# View HTML report
xdg-open build/meson-logs/coveragereport/index.html

# From VSCode: Run task "Test: Coverage Report"
```

### Coverage Summary

```bash
# Install gcovr for detailed coverage
pip install gcovr

# Generate text summary
cd build
gcovr --root .. --print-summary

# Generate detailed HTML report with gcovr
gcovr --root .. --html --html-details -o coverage.html
```

### Coverage Goals

- **Core filters**: >90% line coverage
- **Image I/O**: >80% line coverage  
- **CLI parser**: >80% line coverage
- **Autotuning framework**: >75% line coverage
- **GPU utilities**: >70% line coverage (limited by GPU availability)

## Test Behavior

### GPU Detection

Tests automatically detect GPU availability at startup:

**With GPU:**
```
=== GPU detected: AMD Radeon RX 6900 XT (Device count: 1) ===
[==========] Running N tests from M test suites.
...
```

**Without GPU:**
```
=== No GPU detected - GPU tests will be skipped ===
[==========] Running N tests from M test suites.
...
[  SKIPPED ] GPUFilterTest.GrayscaleGPUvsCPU (0 ms)
[  SKIPPED ] GPUFilterTest.NegativeGPUvsCPU (0 ms)
...
```

### Test Inventory (Current)

The test binary currently contains **690 tests across 35 GoogleTest suites**.

Notes:
- CI typically runs the Meson test `unit_tests_cpu_only` (suite `cpu`), which uses `--gtest_filter=-*GPU*`. Tests that require a HIP device but don’t match `*GPU*` should `GTEST_SKIP()` when no device is available.
- To see the authoritative list for your build: `./build/hip-img-fx-tests --gtest_list_tests`.

| GoogleTest Suite | Tests | GPU Required | Runs in CI |
|------------------|-------|--------------|------------|
| AppMainFixture | 21 | ❌ No | ✅ Yes |
| AutotuneConfig | 1 | ❌ No | ✅ Yes |
| AutotuneExtendedTest | 26 | ❌ No | ✅ Yes |
| BenchmarkerSpec | 25 | ✅ Yes (skips without HIP) | ⚠️ Skipped |
| BenchmarkResultSpec | 4 | ❌ No | ✅ Yes |
| BenchmarkRunner | 28 | ❌ No | ✅ Yes |
| CacheEntrySpec | 3 | ❌ No | ✅ Yes |
| CacheKeySpec | 13 | ❌ No | ✅ Yes |
| CacheStore | 4 | ❌ No | ✅ Yes |
| CacheStoreSpec | 42 | ❌ No | ✅ Yes |
| CLIParser | 7 | ❌ No | ✅ Yes |
| CLIParserExtended | 4 | ❌ No | ✅ Yes |
| CLIParserFixture | 21 | ❌ No | ✅ Yes |
| FiltersCPU | 7 | ❌ No | ✅ Yes |
| FiltersCPUExtended | 17 | ❌ No | ✅ Yes |
| FiltersTraits | 37 | ❌ No | ✅ Yes |
| GenerateCandidates | 29 | ❌ No | ✅ Yes |
| GPUFilterTest | 6 | ✅ Yes | ⚠️ Skipped |
| GPUIntegrationTest | 6 | ✅ Yes | ⚠️ Skipped |
| GPUUtils | 10 | ✅ Yes (skips without HIP) | ⚠️ Skipped |
| GPUUtilsExtended | 11 | ✅ Yes (skips without HIP) | ⚠️ Skipped |
| HIPEventSpec | 26 | ✅ Yes (skips without HIP) | ⚠️ Skipped |
| ImageEdgeCasesTest | 19 | ❌ No | ✅ Yes |
| ImageFormatsTest | 12 | ❌ No | ✅ Yes |
| ImageIO | 6 | ❌ No | ✅ Yes |
| KernelTraitsConceptsSpec | 35 | ❌ No | ✅ Yes |
| OrchestratorSpec | 73 | ✅ Yes (skips without HIP) | ⚠️ Skipped |
| ProcessErrorFixture | 39 | ❌ No | ✅ Yes |
| ProcessExtendedTest | 14 | ❌ No | ✅ Yes |
| ProcessGPUTest | 8 | ✅ Yes | ⚠️ Skipped |
| ProcessTest | 7 | ❌ No | ✅ Yes |
| TestHelpers | 17 | ❌ No | ✅ Yes |
| TuningConfigExtended | 32 | ❌ No | ✅ Yes |
| TuningConfigSpec | 42 | ❌ No | ✅ Yes |
| TuningOptionsSpec | 38 | ❌ No | ✅ Yes |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CI Tests

on: [push, pull_request]

jobs:
  test-cpu-only:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y meson ninja-build libgtest-dev
          # Install ROCm headers (for compilation, not runtime)
          wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
          echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
          sudo apt-get update
          sudo apt-get install -y rocm-dev
      
      - name: Configure
        run: meson setup build --native-file native/hip.ini
      
      - name: Build
        run: ninja -C build
      
      - name: Run tests
        run: meson test -C build --print-errorlogs
        # GPU tests automatically skipped via GTEST_SKIP()
```

## Expected Test Results

### All Tests Pass (with GPU)
```
[==========] Running 40 tests from 9 test suites.
[----------] Global test environment set-up.
=== GPU detected: AMD Radeon RX 6900 XT (Device count: 1) ===
[----------] 7 tests from FiltersCPU
[ RUN      ] FiltersCPU.GrayscaleKnownValues
[       OK ] FiltersCPU.GrayscaleKnownValues (0 ms)
...
[==========] 40 tests from 9 test suites ran. (2543 ms total)
[  PASSED  ] 40 tests.
```

### CPU Tests Pass, GPU Tests Skipped (no GPU)
```
[==========] Running 40 tests from 9 test suites.
[----------] Global test environment set-up.
=== No GPU detected - GPU tests will be skipped ===
[----------] 7 tests from FiltersCPU
[ RUN      ] FiltersCPU.GrayscaleKnownValues
[       OK ] FiltersCPU.GrayscaleKnownValues (1 ms)
...
[----------] 6 tests from GPUFilterTest
[ RUN      ] GPUFilterTest.GrayscaleGPUvsCPU
[  SKIPPED ] GPUFilterTest.GrayscaleGPUvsCPU (0 ms)
...
[==========] 40 tests from 9 test suites ran. (145 ms total)
[  PASSED  ] 28 tests.
[  SKIPPED ] 12 tests.
```

## Test Development

### Adding New Tests

1. **CPU-only test:**
```cpp
TEST(TestSuite, NewTest) {
    // Your test code
    EXPECT_EQ(expected, actual);
}
```

2. **GPU-dependent test:**
```cpp
TEST_F(GPUFilterTest, NewGPUTest) {
    // GPU detection happens in SetUp()
    // Test will be skipped if no GPU available
    EXPECT_EQ(err, hipSuccess);
}
```

### Test Helpers

Use provided helper functions for common tasks:

```cpp
#include "test_helpers.h"

// Generate test images
auto solid = test_helpers::generate_solid_color_image(64, 64, 3, 255, 0, 0);
auto gradient = test_helpers::generate_gradient_image(128, 128, 3);
auto checker = test_helpers::generate_checkerboard_image(256, 256, 3, 16);

// Compare images with tolerance
int diff = test_helpers::compare_images_with_tolerance(img1, img2, size, 1);

// Check GPU availability
if (test_helpers::has_gpu_available()) { /* ... */ }
```

## Troubleshooting

### Tests Don't Build

```bash
# Check if GoogleTest is installed
pkg-config --cflags --libs gtest

# If not found, install it
sudo apt-get install libgtest-dev
```

### Tests Crash on GPU Systems

```bash
# Verify HIP runtime is working
rocm-smi

# Check device availability
./build/hip-img-fx-tests --gtest_filter="GPUIntegrationTest.DeviceMemoryLifetime"
```

### All GPU Tests Skipped

This is expected behavior when no GPU is detected. Verify GPU presence:
```bash
rocm-smi
hipconfig --check
```

## Coverage Goals

- **Core filters**: >90% line coverage
- **Image I/O**: >80% line coverage  
- **CLI parser**: >80% line coverage
- **Autotuning framework**: >75% line coverage
- **GPU utilities**: >70% line coverage (limited by GPU availability)

## Performance

Tests should complete quickly:
- **CPU-only tests**: < 200ms total
- **With GPU tests**: < 3 seconds total
- **Individual test**: < 100ms (except large image tests)

## Future Enhancements

- [ ] Add performance regression tests using benchmark infrastructure
- [ ] Increase coverage for autotuning orchestrator
- [ ] Add integration tests for live autotuning
- [ ] Mock HIP API for comprehensive GPU utility testing without hardware
- [ ] Add property-based testing for filter correctness
- [ ] Continuous fuzzing for image format edge cases
