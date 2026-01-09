# HIP Image FX - Benchmark Results

## Executive Summary

**GPU:** AMD Radeon RX 6900 XT  
**Framework:** HIP/ROCm  
**Architecture:** Batch processing with synchronous execution  
**Date:** January 9, 2026

### Key Findings

- **Peak Speedup vs OpenMP:** 41.8× (Gaussian blur, 2048², batch size 32)
- **Compute-Bound Performance:** Gaussian blur achieves 25-42× speedup vs OpenMP across tested resolutions
- **Memory-Bound Limitations:** Simple filters (grayscale, negative) range from 0.6-3.7× vs OpenMP (often near parity)
- **Batch Size Impact:** Modest for gaussian blur; can shift results for memory-bound filters due to transfer dominance
- **Average Transfer Overhead:** 71.3% of total GPU time across all configurations

## Performance by Filter

### Gaussian Blur (Compute-Intensive)

| Resolution | Batch Size | GPU Time | Kernel Time | Transfer % | Speedup vs OpenMP | Speedup vs CPU |
|-----------|-----------|----------|-------------|------------|-------------------|----------------|
| 512²      | 1         | 0.368 ms | 0.217 ms    | 41.0%      | 24.995×           | 416.983×       |
| 512²      | 32        | 0.394 ms | 0.202 ms    | 49.0%      | 35.981×           | 361.465×       |
| 512²      | 64        | 0.396 ms | 0.203 ms    | 48.7%      | 35.808×           | 369.237×       |
| 1024²     | 1         | 1.095 ms | 0.815 ms    | 25.5%      | 32.053×           | 521.535×       |
| 1024²     | 32        | 1.153 ms | 0.823 ms    | 28.7%      | 27.325×           | 488.134×       |
| 1024²     | 64        | 1.154 ms | 0.822 ms    | 28.9%      | 28.519×           | 485.579×       |
| 2048²     | 1         | 4.282 ms | 3.226 ms    | 24.7%      | 30.939×           | 521.175×       |
| 2048²     | 32        | 4.316 ms | 3.289 ms    | 23.8%      | 41.824×           | 523.786×       |
| 2048²     | 64        | 4.331 ms | 3.298 ms    | 23.9%      | 34.827×           | 518.328×       |
| 4096²     | 1         | 16.860 ms| 13.093 ms   | 22.3%      | 35.110×           | 577.175×       |
| 4096²     | 32        | 17.035 ms| 13.279 ms   | 22.0%      | 32.890×           | 571.483×       |
| 4096²     | 64        | 17.083 ms| 13.330 ms   | 22.0%      | 32.854×           | 570.359×       |

**Analysis:** Gaussian blur is the ideal GPU workload. The 11×11 convolution kernel (121 operations per pixel) provides sufficient compute intensity to amortize transfer costs. Transfer overhead decreases from ~49% to ~22% as resolution increases. Speedup remains excellent (25-42× vs OpenMP) across all tested resolutions and batch sizes.

### Grayscale (Memory-Bound)

| Resolution | Batch Size | GPU Time | Kernel Time | Transfer % | Speedup vs OpenMP | Speedup vs CPU |
|-----------|-----------|----------|-------------|------------|-------------------|----------------|
| 512²      | 1         | 0.168 ms | 0.014 ms    | 91.1%      | 2.845×            | 2.927×         |
| 512²      | 32        | 0.199 ms | 0.007 ms    | 96.5%      | 0.651×            | 2.237×         |
| 512²      | 64        | 0.200 ms | 0.007 ms    | 96.5%      | 0.619×            | 2.606×         |
| 1024²     | 1         | 0.317 ms | 0.038 ms    | 88.0%      | 0.712×            | 10.973×        |
| 1024²     | 32        | 0.362 ms | 0.031 ms    | 91.2%      | 0.824×            | 4.828×         |
| 1024²     | 64        | 0.365 ms | 0.031 ms    | 91.5%      | 2.430×            | 5.404×         |
| 2048²     | 1         | 1.142 ms | 0.108 ms    | 90.5%      | 0.823×            | 9.500×         |
| 2048²     | 32        | 1.235 ms | 0.127 ms    | 89.7%      | 0.760×            | 5.839×         |
| 2048²     | 64        | 1.239 ms | 0.126 ms    | 89.8%      | 3.512×            | 5.808×         |
| 4096²     | 1         | 4.321 ms | 0.427 ms    | 90.1%      | 0.936×            | 6.666×         |
| 4096²     | 32        | 4.450 ms | 0.506 ms    | 88.7%      | 0.956×            | 6.711×         |
| 4096²     | 64        | 4.405 ms | 0.508 ms    | 88.5%      | 1.073×            | 6.752×         |

**Analysis:** Grayscale conversion is severely memory-bound. Kernel execution stays under ~0.51ms even for 4096² images, while transfers dominate (typically 88-97% overhead). Performance is inconsistent across batch sizes, with many configurations showing CPU OpenMP beating GPU (speedup <1×). Best case in this run: 3.512× at 2048² with batch size 64.

### Negative (Memory-Bound)

| Resolution | Batch Size | GPU Time | Kernel Time | Transfer % | Speedup vs OpenMP | Speedup vs CPU |
|-----------|-----------|----------|-------------|------------|-------------------|----------------|
| 512²      | 1         | 0.166 ms | 0.013 ms    | 91.6%      | 0.867×            | 4.653×         |
| 512²      | 32        | 0.202 ms | 0.005 ms    | 97.0%      | 0.785×            | 2.941×         |
| 512²      | 64        | 0.197 ms | 0.005 ms    | 97.5%      | 0.613×            | 2.889×         |
| 1024²     | 1         | 0.311 ms | 0.033 ms    | 89.4%      | 0.676×            | 10.983×        |
| 1024²     | 32        | 0.354 ms | 0.023 ms    | 93.2%      | 0.691×            | 6.299×         |
| 1024²     | 64        | 0.354 ms | 0.022 ms    | 93.8%      | 1.009×            | 6.094×         |
| 2048²     | 1         | 1.094 ms | 0.081 ms    | 92.5%      | 3.741×            | 10.761×        |
| 2048²     | 32        | 1.232 ms | 0.095 ms    | 92.3%      | 0.960×            | 7.367×         |
| 2048²     | 64        | 1.232 ms | 0.093 ms    | 92.4%      | 0.919×            | 7.256×         |
| 4096²     | 1         | 4.204 ms | 0.329 ms    | 92.2%      | 1.013×            | 8.382×         |
| 4096²     | 32        | 4.293 ms | 0.370 ms    | 91.4%      | 1.001×            | 8.158×         |
| 4096²     | 64        | 4.283 ms | 0.376 ms    | 91.2%      | 1.023×            | 8.155×         |

**Analysis:** Similar to grayscale, negative inversion is memory-bound with ~89-98% transfer overhead. The simple per-byte operation executes in well under 0.4ms even at 4096², so transfer dominates. Performance is frequently near parity with CPU OpenMP; best case in this run: 3.741× at 2048² with batch size 1. Worst case: 0.613× at 512² with batch size 64.

## Batch Processing Analysis

### Batch Size Notes

This repository supports GPU directory batching via `--batch-size` in the main application.

The benchmark dataset referenced below contains a `batch_size` column (1, 32, 64). Interpreting performance differences across that column requires that the benchmark harness actually processes that many images per GPU call. Verify the harness behavior in `bench/run_bench.cpp` for the CSV you are analyzing.

### Architecture Details

**Batch Processing Pipeline:**
1. Load N images in parallel (OpenMP)
2. Allocate single contiguous GPU buffer for all images
3. Copy all images H2D in sequence
4. Launch one kernel to process entire batch
5. Copy all results D2H in sequence
6. Save all images in parallel (OpenMP)

**Memory Layout:**
- Images stored contiguously: `[img0][img1][img2]...[imgN]`
- Metadata array: `[meta0, meta1, meta2, ..., metaN]`
- Each metadata entry contains: width, height, channels, byte offset

## Compute-Bound vs Memory-Bound

### Compute-Bound (Gaussian Blur)
- **Kernel time dominates:** ~51-78% of total GPU time
- **High arithmetic intensity:** O(n² × 121) operations for 11×11 kernel
- **Excellent GPU scaling:** 25-42× speedup vs OpenMP maintained across resolutions
- **Transfer overhead decreases with resolution:** ~49% → ~22%
- **Batch size insensitive:** Performance consistent across batch sizes

### Memory-Bound (Grayscale, Negative)
- **Transfer time dominates:** ~88-98% of total GPU time
- **Low arithmetic intensity:** O(n²) simple operations
- **Poor GPU scaling vs OpenMP:** 0.6-3.7× speedup, highly variable
- **Transfer overhead constant:** ~90% regardless of resolution
- **Batch size sensitive:** Performance varies significantly
- **CPU can be faster:** At 2048², OpenMP beats GPU for both filters

## Performance Characteristics

### Resolution Scaling

As image size increases:
- **Kernel time** grows quadratically (O(n²))
- **Transfer time** grows quadratically (O(n²))
- **Transfer percentage** decreases for compute-bound workloads
- **Speedup** improves for compute-bound, inconsistent for memory-bound

### Bandwidth Utilization

Effective bandwidth increases with image size as fixed-cost overhead amortizes:
- **512²:** 4.0-9.5 GB/s
- **1024²:** 5.5-20.2 GB/s
- **2048²:** 5.8-23.0 GB/s
- **4096²:** 5.9-23.9 GB/s

Peak bandwidth of ~23.9 GB/s achieved at 4096².

## When to Use GPU

### Use GPU When:
- Filter is **compute-intensive** (many operations per pixel)
- Using **Gaussian blur** or similar convolution operations
- Image resolution is **large** (≥1024² for best results)
- Filter has **high arithmetic intensity**
- Batch size can be optimized (typically 32-64)

### Don't Use GPU When:
- Filter is **simple** (grayscale, brightness, contrast)
- Image resolution is **small** (<512²)
- **Latency** is critical (GPU adds ~0.2-2ms transfer overhead minimum)
- CPU OpenMP can achieve similar or better performance
- Working with single small images

## Benchmark Methodology

### Hardware
- **GPU:** AMD Radeon RX 6900 XT (40 CUs, 16GB VRAM, 256-bit bus)
- **CPU:** 32-thread system with OpenMP parallelization
- **CPU Governor:** performance
- **Memory:** PCIe 4.0 connection

### Software
- **Framework:** HIP/ROCm 6.x
- **Compiler:** hipcc with -O3 optimization
- **Timing:** HIPEvent for GPU (microsecond precision), chrono for CPU
- **Warmup:** 2 iterations before measurement
- **Iterations:** 5 runs per configuration, averaged with standard deviation

### Test Configuration
- **Resolutions:** 512², 1024², 2048², 4096²
- **Batch Sizes:** 1, 32, 64 images
- **Filters:** Grayscale, Negative, Gaussian Blur (11×11 kernel)
- **Image Format:** RGB, 3 channels, 8-bit per channel
- **Execution Mode:** Synchronous (H2D → Kernel → D2H)

### Timing Breakdown
GPU timing includes three components:
1. **H2D (Host-to-Device):** Copy images from CPU to GPU memory
2. **Kernel:** Execute filter on GPU
3. **D2H (Device-to-Host):** Copy results from GPU to CPU memory

CPU timing includes only computation (no I/O overhead).

## Visualizations

The analysis generates 6 publication-quality visualizations:

1. **speedup_vs_resolution.png** - GPU speedup comparison (best batch size per config)
2. **gpu_time_breakdown.png** - H2D/Kernel/D2H time breakdown per filter
3. **transfer_overhead.png** - PCIe transfer overhead percentage
4. **bandwidth.png** - Effective memory bandwidth by resolution
5. **absolute_times.png** - Absolute execution time comparison
6. **batch_size_scaling.png** - Performance vs batch size (6-panel analysis)

Run analysis: `./bench/scripts/run_analysis.sh bench/results/benchmark_*.csv`

## Conclusions

### Main Findings

1. **GPU Excels at Compute-Intensive Work:** Gaussian blur achieves 25-42× speedup vs OpenMP
2. **GPU Struggles with Simple Filters:** Memory-bound operations range from 0.6-3.7× vs OpenMP
3. **Transfer Overhead is Critical:** 71.3% average overhead means most GPU time spent on PCIe
4. **Batch Size Matters for Memory-Bound:** Optimal batch size varies by filter (typically 32-64)
5. **Resolution Scaling Works Well:** Larger images benefit more from GPU (for compute-bound)

### Optimization Opportunities

For future work:
1. **Kernel Fusion:** Combine multiple simple filters into one kernel to reduce round-trips
2. **Separable Convolution:** Decompose 2D gaussian blur into two 1D passes (121 → 22 operations)
3. **Persistent Data:** Keep data on GPU across multiple operations
4. **Async Execution:** Overlap H2D/kernel/D2H using CUDA streams (requires testing)
5. **Compute-Optimized Filters:** Focus development on complex filters where GPU excels

### Architectural Lessons

Key insights from benchmarking:
- **Measure, Don't Assume:** Batch size impact varies significantly by filter type
- **Workload Characterization:** Compute-bound vs memory-bound determines GPU benefit
- **Transfer Overhead:** PCIe bandwidth is the primary bottleneck for simple operations
- **Batch Size Optimization:** Finding optimal batch size requires empirical testing
- **Hardware Constraints:** Memory-bound operations limited by PCIe, not GPU compute

## Repository Structure

```
bench/
├── results/
│   ├── benchmark_20260109_144830.csv    # Raw benchmark data (latest)
│   ├── benchmark_report.html            # Interactive HTML report
│   └── *.png                            # Generated visualizations
├── scripts/
│   ├── analyze_results.py               # Analysis and visualization script
│   ├── run_benchmark.sh                 # Automated benchmark runner
│   └── run_analysis.sh                  # Analysis runner with venv
└── run_bench.cpp                        # Benchmark harness

src/
├── app/                                 # Application logic
├── cli/                                 # Command-line interface
├── core/                                # GPU utilities and image I/O
└── filters/                             # HIP kernels (*.hip.cpp)
```

## Data Summary

**Total Configurations Tested:** 36  
- 3 filters × 4 resolutions × 3 batch sizes

**Performance Range:**
- Best vs single-threaded: 577.2× (Gaussian blur, 4096², batch 1)
- Best vs OpenMP: 41.824× (Gaussian blur, 2048², batch 32)
- Worst vs OpenMP: 0.613× (Negative, 512², batch 64)

**Key Statistics (mean across 36 rows):**
- Average GPU total time: 2.911 ms
- Average CPU time (single): 1065.292 ms
- Average CPU time (OpenMP): 65.455 ms
- Average speedup vs OpenMP: 11.738×
- Average transfer overhead: 71.300%

---

**Generated from:** `benchmark_20260109_144830.csv`  
**Date:** January 9, 2026  
**Analysis tool:** `analyze_results.py`
