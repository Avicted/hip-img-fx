# Quick Test Reference

## Running Tests

```bash
# All tests
meson test -C build

# CPU tests only (no GPU required)
meson test -C build --suite cpu

# Verbose output
meson test -C build --verbose --print-errorlogs

# Direct execution with GoogleTest filters
./build/hip-img-fx-tests
./build/hip-img-fx-tests --gtest_filter="FiltersCPU.*"
./build/hip-img-fx-tests --gtest_filter="-*GPU*"
```

## VSCode Tasks

Press `Ctrl+Shift+P` → `Tasks: Run Task`:

- **Test: All** - Run all tests (default)
- **Test: CPU Only** - Run CPU tests only
- **Test: Run Executable** - Direct test execution
- **Test: Verbose** - Detailed output
- **Test: Coverage Report** - Generate coverage

## Coverage

```bash
# Quick coverage report
./scripts/generate_coverage.sh

# View HTML report
xdg-open build/meson-logs/coveragereport/index.html
```

## Test Organization

The test suite includes comprehensive coverage across multiple categories:

- **Core Filters & Processing**: ~150 tests covering CPU/GPU filters and processing
- **Image I/O**: ~50 tests for loading, saving, and format handling
- **CLI & Application**: ~60 tests for command-line parsing and application logic
- **GPU Infrastructure**: ~40 tests for GPU utilities and integration
- **Autotuning Framework**: ~360 tests for the autotuning system (spec tests)
- **Benchmarking**: 28 tests for benchmark utility functions

**Total: ~690 tests across 33 test files**

**Coverage Statistics (v1.1.0):**
- **Line Coverage**: 71.0% (11213/15801 lines)
- **Function Coverage**: 92.2% (2540/2756 functions)
- **Branch Coverage**: 33.2% (15514/46698 branches)

## Common Commands

```bash
# List all tests
./build/hip-img-fx-tests --gtest_list_tests

# Run specific test
./build/hip-img-fx-tests --gtest_filter="FiltersCPU.GrayscaleKnownValues"

# Repeat test multiple times
./build/hip-img-fx-tests --gtest_repeat=10

# Shuffle test order
./build/hip-img-fx-tests --gtest_shuffle

# Run tests with color output
./build/hip-img-fx-tests --gtest_color=yes
```

## CI Integration

```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    meson test -C build --suite cpu --print-errorlogs
```

## Troubleshooting

```bash
# Check GoogleTest installation
pkg-config --cflags --libs gtest

# Check for test binary
ls -lh build/hip-img-fx-tests

# View test log
cat build/meson-logs/testlog.txt

# Check coverage tools
gcov --version
gcovr --version
```
