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

| Test Suite | Tests | GPU Required |
|------------|-------|--------------|
| FiltersCPU | 7 | No |
| GPUFilterTest | 6 | Yes |
| ImageIO | 6 | No |
| ImageFormatsTest | 12 | No |
| CLIParser | 7 | No |
| AutotuneConfig | 1 | No |
| AutotuneExtendedTest | 26 | No |
| CacheStore | 4 | No |
| GPUIntegrationTest | 6 | Yes |
| GPUUtils | 11 | No |
| ProcessTest | 6 | No |
| ProcessGPUTest | 8 | Yes |
| **Total** | **100** | **19 GPU tests** |

**Coverage Statistics:**
- Line Coverage: 49.6% (4617/9308)
- Function Coverage: 73.7% (585/794)
- Branch Coverage: 30.3% (3816/12596)
| GPUIntegrationTest | 6 | Yes |
| **Total** | **37** | 12 GPU, 25 CPU |

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
