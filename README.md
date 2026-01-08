# HIP Image FX - GPU-Accelerated Image Processing

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red)]()
[![HIP](https://img.shields.io/badge/HIP-C%2B%2B20-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

GPU-accelerated image processing framework demonstrating real-world HIP performance engineering principles.

This project showcases production-grade GPU optimization through empirical measurement and data-driven architectural decisions. Through extensive benchmarking, we determined that **single-image synchronous processing provides optimal performance** for these workloads. The framework demonstrates fine-grained profiling, honest performance analysis, and the value of measuring rather than assuming.

**Performance Highlight**: Achieves **735× speedup** on Gaussian blur vs single-threaded CPU and **40× vs OpenMP** (32 threads) on AMD Radeon RX 6900 XT

**Architecture**: Simple, empirically-validated single-image synchronous pipeline

**Full Results**: See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed performance analysis

## Performance Engineering Focus

This repository demonstrates:

- **GPU Profiling**: Fine-grained timing with HIP events (H2D, kernel, D2H breakdown)
- **Memory Analysis**: Bandwidth utilization and transfer overhead measurement
- **Data-Driven Decisions**: Extensive benchmarking to validate architectural choices
- **Reproducible Benchmarking**: Production-quality harness with statistical analysis
- **Honest Performance Reporting**: Documenting both successes (735× speedup) and failures (GPU slower than CPU for simple filters)

## Performance Characteristics

### Current Architecture

```
Single-Image Synchronous Pipeline
┌────────────────────────────────────────┐
│         For each image:                │
│  ┌──────────────────────────────────┐  │
│  │ 1. Load image from disk          │  │
│  │ 2. H2D: Copy to GPU (hipMemcpy)  │  │
│  │ 3. Launch kernel                 │  │
│  │ 4. D2H: Copy from GPU            │  │
│  │ 5. Write result to disk          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

Simple, maintainable, and empirically optimal
```

### Key Findings

The current architecture was validated through extensive benchmarking. Key insights from our performance testing:

- **Compute-Bound Wins**: Gaussian blur achieves consistent 39-40× speedup vs OpenMP
- **Memory-Bound Struggles**: Simple filters (grayscale, negative) show only 0.6-4× speedup
- **Transfer Overhead**: Averages 68% of total GPU time across all tests
- **Resolution Scaling**: Larger images (4096×4096) benefit more from GPU acceleration


## Quick Start

### Prerequisites

- AMD ROCm 5.0+ ([installation guide](https://rocmdocs.amd.com/))
- HIP-enabled AMD GPU (tested on RDNA/GCN architectures)
- Meson build system
- OpenMP for CPU fallback
- C++20 compiler

### Build

```bash
# Configure with HIP native file
meson setup build --native-file native/hip.ini --reconfigure

# Build all targets (main app + benchmark suite)
ninja -C build

# Verify build
./build/hip-img-fx --help
```

### Basic Usage

```bash
# Single image (GPU)
./build/hip-img-fx \
    --input examples/example_01.jpg \
    --filter grayscale \
    --output output.jpg

# Directory processing (processes all images in directory)
./build/hip-img-fx \
    --input examples/ \
    --filter gaussian-blur \
    --output examples/output/

# CPU fallback (for comparison)
./build/hip-img-fx \
    --input examples/example_01.jpg \
    --filter negative \
    --output output.jpg \
    --use-cpu
```


## Benchmark Suite

### Running Benchmarks

```bash
# Full benchmark sweep (512² → 4096², 3 filters)
./bench/scripts/run_benchmark.sh

# Custom benchmark configuration
./build/hip-img-fx-bench \
    --warmup 5 \
    --iterations 20 \
    --output my_results.csv \
    --verbose

# Analyze results (uses matplotlib & pandas in isolated venv)
# Automatically creates venv, installs dependencies, generates visualizations
./bench/scripts/run_analysis.sh bench/results/benchmark_*.csv
```

### What Gets Benchmarked

- **CPU Performance**:
  - Single-threaded baseline
  - OpenMP parallelized (32 threads)
- **GPU Performance**:
  - H2D transfer time (host to device)
  - Kernel execution time
  - D2H transfer time (device to host)
  - Total pipeline latency
- **Sweep Parameters**:
  - Image resolutions: 512², 1024², 2048², 4096²
  - Filter types: grayscale, negative, gaussian_blur

### Sample Results (AMD Radeon RX 6900 XT - January 2026)

**Gaussian Blur - 4096×4096:**
```
CPU (single):   9815.00 ms
CPU (OpenMP):    534.00 ms
GPU H2D:           1.95 ms
GPU Kernel:        9.59 ms
GPU D2H:           1.80 ms
GPU Total:        13.35 ms
Speedup vs Single: 735×
Speedup vs OpenMP:  40×
Bandwidth:          7.53 GB/s
Transfer Overhead: 28% of GPU time (compute-bound)
```

**Grayscale - 2048×2048:**
```
CPU (single):     13.09 ms
CPU (OpenMP):      1.91 ms
GPU H2D:           0.52 ms
GPU Kernel:        0.11 ms
GPU D2H:           0.51 ms
GPU Total:         1.14 ms
Speedup vs Single: 11.5×
Speedup vs OpenMP:  0.6× (CPU OpenMP is faster)
Bandwidth:         22.13 GB/s
Transfer Overhead: 91% of GPU time (memory-bound)
```

**See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for complete analysis (12 test configurations)**

### Visualizations

The analysis script generates publication-quality visualizations:
- **speedup_vs_resolution.png** - GPU vs CPU performance comparison
- **gpu_time_breakdown.png** - H2D/Kernel/D2H time distribution
- **transfer_overhead.png** - Transfer overhead percentage by filter
- **bandwidth.png** - Memory bandwidth utilization
- **absolute_times.png** - Absolute time comparison across implementations
- **benchmark_report.html** - HTML report with embedded graphs


## Performance Optimizations

### 1. Fine-Grained GPU Profiling

**Implementation**: HIP event-based timing infrastructure

```cpp
GPUTimings timings;
apply_filter_gpu(FILTER_GRAYSCALE, input, output, true, &timings);

printf("H2D:    %.2f ms\n", timings.h2d_ms);
printf("Kernel: %.2f ms\n", timings.kernel_ms);
printf("D2H:    %.2f ms\n", timings.d2h_ms);
```

**Impact**: Identifies bottlenecks (memory-bound vs compute-bound)

### 2. Single-Image Synchronous Architecture

**Implementation**: Simple H2D to Kernel to D2H pipeline

```cpp
// Simple, empirically-proven optimal architecture
apply_filter_gpu(filter, input, output, profile, &timings);
```

**Key Finding**: For these fast-executing kernels (<10ms), a simple synchronous pipeline proved optimal through empirical testing.

### 3. Memory Access Optimization

**Kernel Characteristics** (from measured benchmark data at 2048×2048):

| Filter | GPU Total | Kernel Time | Bandwidth | Speedup (vs OpenMP) |
|--------|-----------|-------------|-----------|---------------------|
| Grayscale | 1.14ms | 0.11ms | 22.13 GB/s | 0.6× |
| Negative | 1.16ms | 0.09ms | 21.73 GB/s | 0.8× |
| Gaussian Blur | 3.37ms | 2.33ms | 7.49 GB/s | 38.6× |

**Key Finding**: Transfer overhead (88-93% of time for simple filters) dominates performance for memory-bound operations. Only compute-intensive filters (Gaussian blur) achieve significant GPU speedup.

All kernels use:
- **512 threads per block** (optimal for AMD RDNA/GCN)
- Coalesced memory access patterns
- Minimal register pressure (< 32 regs/thread)


## Kernel Deep Dive

### Grayscale Kernel

```cpp
__global__ void grayscale_kernel(const unsigned char *input, ...)
{
    // Convert RGB to luminance-preserving grayscale
    // Formula: Y = 0.21R + 0.72G + 0.07B (Rec. 709)
    
    unsigned char gray = 0.21f * r + 0.72f * g + 0.07f * b;
}
```

**Performance Analysis**:
- **Memory-bound**: 3 FLOPs per 7 bytes transferred
- **Coalesced access**: Sequential RGB reads
- **Occupancy**: Near-maximum (low register usage)
- **Transfer overhead**: 88-93% of execution time

### Gaussian Blur Kernel

```cpp
__global__ void gaussian_blur_kernel(const unsigned char *input, ...)
{
    // Shared memory for Gaussian kernel coefficients
    extern __shared__ float kernel[];
    
    // Convolution with 11x11 kernel (121 neighbor reads)
    for (int ky = -5; ky <= 5; ky++) {
        for (int kx = -5; kx <= 5; kx++) {
            pixel_value += input[neighbor] * kernel[k];
        }
    }
}
```

**Performance Analysis**:
- **Compute-intensive**: 121 reads and 121 multiply-adds per pixel
- **Shared memory**: Kernel coefficients (484 bytes for 11x11)
- **Optimization opportunity**: Separable convolution (potential 3-5× speedup)
- **Current limitation**: Naive 2D approach (not tile-based)

**Recommended Improvements**:
1. Implement separable Gaussian (2× 1D passes): **3-5× faster**
2. Tile-based shared memory for image data: **additional 2-3× speedup**
3. Combined improvement: **6-15× faster** than current implementation


## Testing & Validation

### Build & Run Tests

```bash
# Build with debugging enabled
meson setup build --native-file native/hip.ini --reconfigure
ninja -C build

# Run benchmark suite (includes correctness validation)
./build/hip-img-fx-bench --iterations 5

# Compare CPU vs GPU outputs (should be identical)
./build/hip-img-fx --input test.jpg --output gpu_out.jpg
./build/hip-img-fx --input test.jpg --output cpu_out.jpg --use-cpu
diff <(xxd gpu_out.jpg) <(xxd cpu_out.jpg)
```

## Project Structure

```
hip-img-fx/
├── src/
│   ├── app/                   # Application entry point & batch processing
│   ├── cli/                   # Command-line argument parsing
│   ├── core/                  # GPU utilities, timing, image I/O
│   │   ├── gpu_utils.cpp/h    # HIP pipeline, events, streams
│   │   └── image.cpp/h        # STB-based image loading
│   └── filters/               # HIP kernels & CPU implementations
│       ├── grayscale.hip.cpp
│       ├── negative.hip.cpp
│       └── gaussian_blur.hip.cpp
├── bench/
│   ├── run_bench.cpp          # Benchmark harness
│   ├── scripts/
│   │   ├── run_benchmark.sh   # Automated benchmark runner
│   │   └── analyze_results.py # Performance analysis tool
│   └── results/               # CSV output directory
├── examples/                  # Sample images
├── native/
│   └── hip.ini                # Meson HIP configuration
└── meson.build                # Build system
```


## Advanced Configuration

### GPU Architecture Targeting

Edit [`native/hip.ini`](native/hip.ini) to target your GPU:

```ini
[properties]
# RDNA 2 (RX 6000 series)
offload_arch = 'gfx1030'

# RDNA 3 (RX 7000 series)
# offload_arch = 'gfx1100'

# GCN 5 (Vega)
# offload_arch = 'gfx900'
```

### Tuning Parameters

In [`src/core/gpu_utils.h`](src/core/gpu_utils.h):

```cpp
// Gaussian blur kernel size (must be odd)
constexpr int GAUSSIAN_BLUR_AMOUNT = 11;  // Tune: 3, 5, 7, 11, 15

// Thread block size
int threads = 512;  // Tune: 256, 512, 1024
```


## Performance Analysis Tools

### Python Analysis Script

```bash
./bench/scripts/run_analysis.sh results.csv
```

**Output**:
- Summary by filter type
- Transfer overhead analysis

## Learning Outcomes

This project demonstrates:

1. **HIP Fundamentals**
   - Kernel launch configuration (`hipLaunchKernelGGL`)
   - Memory management (`hipMalloc`, `hipMemcpy`)
   - Event-based profiling (`hipEventRecord`, `hipEventElapsedTime`)
   - Synchronous execution pipeline

2. **GPU Performance Engineering**
   - Identifying memory-bound vs compute-bound kernels
   - Analyzing bandwidth utilization
   - Optimizing memory access patterns (coalescing)
   - Understanding when GPU acceleration is beneficial vs CPU

3. **Production Engineering Practices**
   - Reproducible benchmark infrastructure
   - Statistical analysis (mean, std dev)
   - Performance regression detection
   - Clear documentation of optimization tradeoffs


## Future Work

### High-Priority Optimizations

1. **Separable Gaussian Blur** (3-5× speedup expected)
   - Split 11×11 2D convolution into 2× 11×1 passes
   - Reduces memory reads from 121 to 22 per pixel

2. **Tile-Based Processing with Shared Memory**
   - Load image tiles into shared memory
   - Reuse data across threads
   - Reduces global memory pressure

3. **Multi-GPU Support**
   - Distribute workload across multiple GPUs
   - Peer-to-peer transfers between GPUs

4. **Half-Precision (FP16) Kernels**
   - Leverage RDNA matrix acceleration
   - 2× throughput for blur operations

### Research Directions

- **Auto-tuning**: Grid search for optimal block sizes per GPU
- **Warp-level primitives**: Use shuffle intrinsics for reduction ops
- **Texture memory**: Test performance with texture cache for blur


## License

MIT License - See [LICENSE](LICENSE)


## References

- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP)
- [GPU Performance Optimization Techniques](https://developer.amd.com/resources/rocm-learning-center/)
- [Separable Gaussian Blur](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-40-incremental-computation-gaussian)


## Credits

Images (unsplash.com):
- [example_01](https://unsplash.com/photos/pagoda-surrounded-by-trees-E_eWwM29wfU)
- [example_02](https://unsplash.com/photos/blue-and-brown-bird-on-brown-tree-trunk-DPXytK8Z59Y)
- [example_03](https://unsplash.com/photos/selective-focus-photo-of-giraffe-D6TqIa-tWRY)
- [example_04](https://unsplash.com/photos/black-white-and-yellow-bird-on-brown-tree-branch-during-daytime-vjFC9OjrOtA)
- [example_05](https://unsplash.com/photos/two-white-ferrets-zQTw2g6JY6U)

---

**Built with HIP, engineered for performance.**
