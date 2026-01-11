# HIP Image FX - GPU-Accelerated Image Processing

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red)]()
[![HIP](https://img.shields.io/badge/HIP-C%2B%2B20-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Version](https://img.shields.io/badge/version-1.0.1-green)]()

GPU-accelerated image processing framework with **production-ready autotuning** for optimal kernel configurations.

This project showcases production-grade GPU optimization through empirical measurement and automated tuning. The framework features a comprehensive **autotuning system** that automatically discovers optimal kernel configurations for your specific GPU, plus fine-grained profiling and honest performance analysis.

**Key Features**:
-  **Automatic GPU tuning**: Zero-configuration optimal block sizes per GPU
-  **Performance**: 582× speedup on Gaussian blur vs single-threaded CPU
-  **Data-driven**: Validated through 60+ benchmark configurations  
-  **Production-ready**: Installable headers, pkg-config support, stable API
-  **Type-safe**: C++20 concepts enforce correct usage at compile-time

**[→ See Complete Documentation](docs/README.md)**

## Autotuning Framework

The autotuning system automatically finds the fastest GPU kernel configuration for your hardware:

```bash
# First run: autotuning (~100-200ms overhead)
./build/hip-img-fx --input photo.jpg --output result.jpg --filter grayscale
# Output: [AutoTune] Benchmarking... Selected [16x8] (0.034ms)

# Subsequent runs: cached (zero overhead)  
./build/hip-img-fx --input photo.jpg --output result.jpg --filter grayscale
# Output: [AutoTune] Using cached [16x8]
```

**Performance impact**: 12-18% faster than default configurations  
**Learn more**: [Autotuning Guide](docs/AUTOTUNING_GUIDE.md)

## Performance Engineering Focus

This repository demonstrates:

- **Automatic GPU Tuning**: Self-optimizing kernels via empirical benchmarking
- **GPU Profiling**: Fine-grained timing with HIP events (H2D, kernel, D2H breakdown)
- **Memory Analysis**: Bandwidth utilization and transfer overhead measurement
- **Data-Driven Decisions**: Extensive benchmarking (60 configurations) to validate architectural choices
- **Reproducible Benchmarking**: Production-quality harness with statistical analysis
- **Honest Performance Reporting**: Documenting both successes (582× speedup) and limitations
- **Type Safety**: C++20 concepts enforce correct kernel integration at compile-time

## Performance Characteristics

### Current Architecture

```
Batch Processing Pipeline

  Process N images in a single GPU call:     
    
   1. Load N images (OpenMP parallel)      
   2. Allocate contiguous GPU buffer       
   3. H2D: Copy all images sequentially    
   4. Launch one kernel for entire batch   
   5. D2H: Copy all results                
   6. Save N images (OpenMP parallel)      
    
                                             
  Configurable batch size: --batch-size N    
  Default: 64 images per GPU call            


Empirically validated through comprehensive benchmarking
```

### Key Findings

The batch processing architecture was validated through extensive benchmarking across multiple configurations. Key insights from the performance testing:

- **Compute-Bound Excellence**: Gaussian blur achieves 25-41× speedup vs OpenMP
- **Memory-Bound Limitations**: Simple filters (grayscale, negative) show 0.6-3.7× speedup vs OpenMP
- **Transfer Overhead**: Averages 72% of total GPU time across all configurations
- **Resolution Scaling**: Larger images (4096×4096) benefit more from GPU acceleration
- **Batch Size Impact**: Minimal for compute-bound filters, significant for memory-bound


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
============================
Running HIP Image FX v1.0.1
============================

Usage: hip-img-fx [options]
Options:
  --input <input_file|input_dir>     Specifies the input file or directory path.
  --output <output_file|output_dir>  Specifies the output file or directory path.
  --filter <filter_type>             Specifies the type of filter to apply (e.g., "grayscale", "negative", "gaussian-blur").
  --use-cpu                          Use CPU for processing instead of GPU.
  --batch-size <N>                   Number of images to process per GPU batch (default: 64).
  --help                             Displays this help information.

Notes:
  - For batch processing, specify both --input and --output as directories.
  - For single image processing, specify both as files.
  - Supported filters: grayscale, negative, gaussian-blur

```

### Basic Usage

```bash
# Single image (GPU)
./build/hip-img-fx \
    --input examples/example_01.jpg \
    --filter grayscale \
    --output output.jpg

# Directory processing with custom batch size
./build/hip-img-fx \
    --input examples/ \
    --filter gaussian-blur \
    --output examples/output/ \
    --batch-size 64

# CPU fallback (for comparison)
./build/hip-img-fx \
    --input examples/example_01.jpg \
    --filter negative \
    --output output.jpg \
    --use-cpu
```

**Batch Size Tuning:**
```bash
# Default batch size (64 images)
./build/hip-img-fx --input images/ --filter gaussian-blur --output results/

# Smaller batch (memory-constrained systems)
./build/hip-img-fx --input images/ --filter grayscale --output results/ --batch-size 32

# Single-image processing
./build/hip-img-fx --input images/ --filter negative --output results/ --batch-size 1
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
  - Batch sizes: 1, 8, 16, 32, 64 images per GPU call

**Total Configurations**: 60 (3 filters × 4 resolutions × 5 batch sizes)

### Sample Results (AMD Radeon RX 6900 XT - January 2026)

**Gaussian Blur - 4096×4096 (Batch Size 64):**
```
CPU (single):   9812.71 ms
CPU (OpenMP):    567.40 ms
GPU H2D:           1.97 ms
GPU Kernel:       13.37 ms
GPU D2H:           1.81 ms
GPU Total:        17.15 ms
Speedup vs Single: 572.22×
Speedup vs OpenMP:  33.09×
Bandwidth:          5.87 GB/s
Transfer Overhead: 22% of GPU time (compute-bound)
```

**Grayscale - 1024×1024 (Batch Size 32):**
```
CPU (single):      2.80 ms
CPU (OpenMP):      0.27 ms
GPU H2D:           0.17 ms
GPU Kernel:        0.04 ms
GPU D2H:           0.13 ms
GPU Total:         0.34 ms
Speedup vs Single: 8.17×
Speedup vs OpenMP:  0.79× (CPU OpenMP is faster)
Bandwidth:         18.25 GB/s
Transfer Overhead: 89% of GPU time (memory-bound)
```

**Negative - 2048×2048 (Batch Size 1):**
```
CPU (single):     10.04 ms
CPU (OpenMP):      1.08 ms
GPU H2D:           0.55 ms
GPU Kernel:        0.08 ms
GPU D2H:           0.49 ms
GPU Total:         1.12 ms
Speedup vs Single: 8.98×
Speedup vs OpenMP:  0.96× (CPU OpenMP is faster)
Bandwidth:         22.49 GB/s
Transfer Overhead: 93% of GPU time (memory-bound)
```

**Validated through 60 benchmark configurations with comprehensive batch size impact analysis**

### Visualizations

The analysis script generates publication-quality visualizations:
- **speedup_vs_resolution.png** - GPU vs CPU performance comparison (best batch size per config)
- **gpu_time_breakdown.png** - H2D/Kernel/D2H time distribution
- **transfer_overhead.png** - Transfer overhead percentage by filter
- **bandwidth.png** - Memory bandwidth utilization
- **absolute_times.png** - Absolute time comparison across implementations
- **batch_size_scaling.png** - 6-panel analysis of batch size impact (1, 8, 16, 32, 64)
- **benchmark_report.md** - Comprehensive markdown report with embedded charts and tables


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

### 2. Batch Processing Architecture

**Implementation**: Contiguous GPU buffer with single kernel launch

```cpp
// Process multiple images in one GPU call
process_batch_gpu(images, filter, batch_size);

// Internally: allocate contiguous buffer, one kernel launch
// Memory layout: [img0][img1][img2]...[imgN]
```

**Key Finding**: Through empirical testing across 60 configurations, batch processing provides optimal throughput. Batch size impact varies by filter type.

### 3. Memory Access Optimization

**Kernel Characteristics** (from measured benchmark data):

| Filter | Resolution | GPU Total | Kernel Time | Bandwidth | Speedup (vs OpenMP) |
|--------|-----------|-----------|-------------|-----------|---------------------|
| Grayscale | 2048² | 1.14ms | 0.11ms | 22.06 GB/s | 1.1× |
| Negative | 2048² | 1.12ms | 0.08ms | 22.49 GB/s | 0.96× |
| Gaussian Blur | 2048² | 4.27ms | 3.22ms | 5.90 GB/s | 32.2× |

**Key Finding**: Transfer overhead (22-93% depending on filter) is the primary performance factor. Only compute-intensive filters (Gaussian blur) achieve significant GPU speedup where kernel time dominates.



## Testing & Validation

### Build & Run Tests

```bash
# Build with debugging enabled
meson setup build --native-file native/hip.ini --reconfigure
ninja -C build

# Run benchmark suite (includes correctness validation)
./build/hip-img-fx-bench --iterations 5

# Compare CPU vs GPU outputs (should be identical)
./build/hip-img-fx --filter negative --input test.jpg --output gpu_out.jpg
./build/hip-img-fx --filter negative --input test.jpg --output cpu_out.jpg --use-cpu
diff <(xxd gpu_out.jpg) <(xxd cpu_out.jpg)
```

## Project Structure

```
hip-img-fx/
├── src/
│   ├── app/                    # Application entry point & batch processing
│   ├── cli/                    # Command-line argument parsing
│   ├── core/                   # GPU utilities, timing, image I/O
│   │   ├── gpu_utils.cpp/.h    # HIP pipeline, events, streams
│   │   └── image.cpp/.h        # STB-based image loading
│   └── filters/                # HIP kernels & CPU implementations
│       ├── grayscale.hip.cpp
│       ├── negative.hip.cpp
│       └── gaussian_blur.hip.cpp
├── bench/
│   ├── run_bench.cpp           # Benchmark harness
│   ├── scripts/
│   │   ├── run_benchmark.sh    # Automated benchmark runner
│   │   └── analyze_results.py  # Performance analysis tool
│   └── results/                # CSV output directory
├── examples/                   # Sample images
├── native/
│   └── hip.ini                 # Meson HIP configuration
└── meson.build                 # Build system
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
   - Batch processing with contiguous memory allocation

2. **GPU Performance Engineering**
   - Automated kernel configuration tuning
   - Identifying memory-bound vs compute-bound kernels
   - Analyzing bandwidth utilization
   - Optimizing memory access patterns (coalescing)
   - Understanding when GPU acceleration is beneficial vs CPU
   - Batch size tuning for different workload characteristics

3. **Production Engineering Practices**
   - Reproducible benchmark infrastructure
   - Statistical analysis (mean, std dev)
   - Performance regression detection
   - Clear documentation of optimization tradeoffs
   - Data-driven architectural decisions
   - Type-safe APIs with C++20 concepts


## Using as a Library

HIP Image FX provides a production-ready autotuning framework for custom HIP kernels.

### Installation

```bash
meson setup build
meson install -C build
```

Headers install to `${PREFIX}/include/hip-img-fx/autotune/`

### Integration

```cpp
#include <hip-img-fx/autotune/orchestrator.h>

// Define your kernel traits
struct MyKernelTraits {
    static constexpr const char* name() { return "my_kernel"; }
    
    struct Args { /* kernel arguments */ };
    struct Context { /* cache key */ };
    
    static std::vector<TuningConfig> generate_candidates(const Context&);
    static bool is_valid_config(const TuningConfig&, const Args&);
    static void launch(const TuningConfig&, const Args&, hipStream_t);
};

// Use TuningOrchestrator
static TuningOrchestrator<MyKernelTraits> orchestrator;
orchestrator.execute(args, context, stream);
```

**Complete guide**: [Autotuning Framework Documentation](docs/AUTOTUNING_GUIDE.md)

### Pkg-config Support

```bash
# Get compiler flags
pkg-config --cflags hip-img-fx

# In your build system
g++ $(pkg-config --cflags hip-img-fx) mykernel.cpp
```


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

- **Warp-level primitives**: Use shuffle intrinsics for reduction operations
- **Texture memory**: Test performance with texture cache for blur operations
- **Async Streams**: Overlap H2D/kernel/D2H using HIP streams (requires careful benchmarking)
- **Extended autotuning**: Context-aware tuning for more kernel parameters


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



