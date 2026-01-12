# HIP Image FX

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red)]()
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-green)]()
[![HIP](https://img.shields.io/badge/HIP-C%2B%2B20-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Version](https://img.shields.io/badge/version-1.1.0-green)]()

GPU-accelerated image processing with automatic kernel tuning for AMD and NVIDIA GPUs.

## Features

- Three image filters: `grayscale`, `negative`, `gaussian-blur`
- Automatic GPU configuration tuning (12-18% speedup, transparent caching)
- Batch processing for directories
- CPU fallback for comparison
- Reusable autotuning framework for custom HIP kernels

## Documentation

- **[Complete Documentation](docs/README.md)** - Full guides and references
- **[Autotuning Guide](docs/AUTOTUNING_GUIDE.md)** - Framework details and API
- **[Testing Guide](tests/README.md)** - Running tests and coverage
- **[Changelog](docs/CHANGELOG.md)** - Version history

## Quick Start

### Prerequisites

- **AMD GPUs**: ROCm 5.0+ with HIP runtime
- **NVIDIA GPUs**: CUDA 11.0+ (HIP translates to CUDA via hipcc)
- Meson build system & C++20 compiler

### Build

```bash
meson setup build --native-file native/hip.ini --reconfigure
ninja -C build
```

### Usage

```bash
# Single image
./build/hip-img-fx --input photo.jpg --output result.jpg --filter grayscale

# Batch processing (directory)
./build/hip-img-fx --input images/ --output results/ --filter gaussian-blur --batch-size 64

# CPU fallback
./build/hip-img-fx --input photo.jpg --output result.jpg --filter negative --use-cpu
```

Run `./build/hip-img-fx --help` for all options.

## Benchmarking

Run the full benchmark suite and generate analysis:

```bash
./scripts/run_benchmark.sh
./scripts/run_analysis.sh bench/results/benchmark_*.csv
```

See [bench/results/benchmark_report.md](bench/results/benchmark_report.md) for detailed performance results.

## Testing

```bash
# Run all tests
meson test -C build --print-errorlogs

# CPU tests only (no GPU required)
meson test -C build --suite cpu
```

See [tests/README.md](tests/README.md) for comprehensive testing documentation.

## Using as a Library

Install headers and use the autotuning framework in your own HIP projects:

```bash
meson install -C build
pkg-config --cflags hip-img-fx
```

Refer to [docs/AUTOTUNING_GUIDE.md](docs/AUTOTUNING_GUIDE.md) for integration examples.

## License

MIT - See [LICENSE](LICENSE)

