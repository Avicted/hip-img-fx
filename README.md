# HIP Image FX

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red)]()
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-green)]()
[![HIP](https://img.shields.io/badge/HIP-C%2B%2B20-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Version](https://img.shields.io/badge/version-1.1.0-green)]()

**HIP Image FX** is a high-performance, GPU-accelerated image processing toolkit built with **HIP**, supporting both **AMD (ROCm)** and **NVIDIA (CUDA via hipcc)** GPUs.  
It features **automatic kernel configuration tuning** to maximize performance across hardware, with a transparent CPU fallback for comparison and validation.

The project is designed as both a practical image-processing tool and a **reusable framework for GPU kernel autotuning** in real-world workloads.

---

## About the Project

This project was built to explore and demonstrate:

- Cross-vendor GPU compute using HIP
- Where GPU acceleration meaningfully outperforms CPU execution
- Automatic kernel tuning strategies for real workloads
- Writing portable, testable, and benchmark-driven C++20 systems

Rather than focusing on a single optimized kernel, HIP Image FX emphasizes **reusability, measurement, and adaptability across different GPU architectures**.

---

## Features

- GPU-accelerated image filters:
  - `grayscale`
  - `negative`
  - `gaussian-blur`
- Automatic GPU configuration tuning  
  *(~12–18% speedup with transparent caching)*
- Batch processing for image directories
- CPU fallback path for comparison and testing
- Reusable autotuning framework for custom HIP kernels
- CLI-driven workflow suitable for scripting and automation

---

## Documentation

- **[Complete Documentation](docs/README.md)** – Full guides and references  
- **[Autotuning Guide](docs/AUTOTUNING_GUIDE.md)** – Framework design, API, and usage  
- **[Testing Guide](tests/README.md)** – Test structure and execution  

---

## Quick Start

### Prerequisites

- **AMD GPUs**: ROCm 5.0+ with HIP runtime  
- **NVIDIA GPUs**: CUDA 11.0+ (HIP compiled via `hipcc`)  
- Meson build system  
- C++20-compatible compiler  

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

Run `./build/hip-img-fx --help` for the full list of options.

---

## Benchmarking

Run the full benchmark suite and generate performance analysis:

```bash
./scripts/run_benchmark.sh
./scripts/run_analysis.sh bench/results/benchmark_*.csv
```

Detailed benchmark results and analysis are available in:  
**[`bench/results/benchmark_report.md`](bench/results/benchmark_report.md)**

The benchmarks compare:
- GPU vs CPU execution
- Tuned vs untuned kernels
- Performance scaling across batch sizes

---

## Testing

```bash
# Run all tests
meson test -C build --print-errorlogs

# CPU-only tests (no GPU required)
meson test -C build --suite cpu
```

See **[tests/README.md](tests/README.md)** for detailed testing documentation.

---

## Using HIP Image FX as a Library

HIP Image FX can be installed and reused as a library, including its autotuning framework:

```bash
meson install -C build
pkg-config --cflags hip-img-fx
```

This allows you to integrate GPU autotuning and image-processing kernels into existing HIP-based pipelines **without building custom tuning infrastructure from scratch**.

Refer to **[docs/AUTOTUNING_GUIDE.md](docs/AUTOTUNING_GUIDE.md)** for integration examples.

---

## Roadmap

- Additional image filters and convolution kernels  
- Expanded autotuning strategies and heuristics  
- CI validation across multiple GPU architectures  
- Improved profiling and visualization tooling  

---

## About the Author

This project is developed and maintained by **Victor Anderssén**.

HIP Image FX was built to demonstrate experience in:

- GPU programming with HIP (AMD ROCm & NVIDIA CUDA backends)
- Performance analysis, benchmarking, and kernel autotuning
- Modern C++ (C++20), Meson-based builds, and testable system design
- Designing reusable frameworks rather than one-off demos

If you are evaluating this project for professional use, or are interested in **collaboration, consulting, or full-time work related to GPU compute or high-performance systems**, feel free to reach out.

### Contact

- Portfolio: https://victoranderssen.com  
- GitHub: https://github.com/Avicted  
- LinkedIn: https://www.linkedin.com/in/victoranderssen/

---

## License

Apache License 2.0 - see [LICENSE](LICENSE)

