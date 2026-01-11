#!/usr/bin/env python3
"""Analysis of HIP image-processing benchmarks.

This script analyzes CSV output produced by the benchmark harness in
`bench/run_bench.cpp`.

Current benchmark sweep (as configured in the harness):
- Resolutions: 512, 1024, 2048, 4096
- Filters: grayscale, negative, gaussian_blur
- Batch sizes: 1, 8, 16, 32, 64

Notes:
- The main application supports directory batching via `--batch-size`.
- The benchmark harness records a `batch_size` column; interpretation depends on
    the harness implementation used to generate the CSV.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif"
})


def _fmt_int_list(values) -> str:
    return ", ".join(str(int(v)) for v in values)


def _save_figure(fig, outdir: Path, stem: str):
    """Save figure as PNG for embedding in markdown."""
    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Generated: {png_path.name}")


# -------------------------
# Loading
# -------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "filter", "resolution", "batch_size",
        "gpu_total_ms", "gpu_kernel_ms",
        "gpu_h2d_ms", "gpu_d2h_ms",
        "speedup_vs_single", "speedup_vs_omp",
        "bandwidth_gb_s"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    return df


# -------------------------
# Text summary
# -------------------------
def print_summary(df: pd.DataFrame):
    print("=" * 80)
    print("HIP Image FX – Batch Processing Performance Summary")
    print("=" * 80)

    # Define fixed column widths for perfect alignment
    w_res = 12
    w_batch = 8
    w_total = 12
    w_kernel = 12
    w_transfer = 12
    w_omp = 18
    w_single = 18

    header = (f"  {'Resolution':<{w_res}} {'Batch':<{w_batch}} {'GPU Total':>{w_total}} {'Kernel':>{w_kernel}} "
              f"{'Transfer %':>{w_transfer}} {'Speedup vs OMP':>{w_omp}} {'Speedup vs Single':>{w_single}}")
    sep = '-' * len(header)

    for flt in sorted(df["filter"].unique()):
        print(f"\nFILTER: {flt}")
        fdf = df[df["filter"] == flt]

        print(header)
        print(sep)

        for res in sorted(fdf["resolution"].unique()):
            for batch_size in sorted(fdf["batch_size"].unique()):
                rdf = fdf[(fdf["resolution"] == res) & (fdf["batch_size"] == batch_size)]
                if rdf.empty:
                    continue
                row = rdf.iloc[0]

                transfer_pct = (
                    (row["gpu_h2d_ms"] + row["gpu_d2h_ms"]) / row["gpu_total_ms"] * 100
                )

                res_str = f"{res}×{res}"
                batch_str = f"{batch_size}"
                gpu_total_str = f"{row['gpu_total_ms']:.2f} ms"
                kernel_str = f"{row['gpu_kernel_ms']:.2f} ms"
                transfer_str = f"{transfer_pct:.1f}%"
                speedup_omp_str = f"{row['speedup_vs_omp']:.2f}×"
                speedup_single_str = f"{row['speedup_vs_single']:.2f}×"

                print(
                    f"  {res_str:<{w_res}} {batch_str:<{w_batch}} {gpu_total_str:>{w_total}} {kernel_str:>{w_kernel}} "
                    f"{transfer_str:>{w_transfer}} {speedup_omp_str:>{w_omp}} {speedup_single_str:>{w_single}}"
                )


# -------------------------
# Plot 1: Speedup vs Resolution (Both Single and OpenMP)
# -------------------------
def plot_speedup_vs_resolution(df, outdir):
    """Plot GPU speedup vs both single-threaded and OpenMP CPU for each filter (best batch size only)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {"grayscale": "#1e88e5", "negative": "#fb8c00", "gaussian_blur": "#43a047"}
    
    # For each filter and resolution, select the best batch size (highest speedup)
    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    
    # Speedup vs Single-threaded (Linear)
    resolutions = sorted(df_best["resolution"].unique())
    x_positions = list(range(len(resolutions)))
    
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax1.plot(
            x_positions,
            fdf["speedup_vs_single"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="CPU Parity")
    ax1.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax1.set_ylabel("Speedup vs Single-threaded CPU", fontsize=12)
    ax1.set_title("GPU Speedup vs Single-threaded CPU (Linear, Best Batch Size)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{r}²" for r in resolutions])
    
    # Speedup vs Single-threaded (Log)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax2.plot(
            x_positions,
            fdf["speedup_vs_single"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="CPU Parity")
    ax2.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax2.set_ylabel("Speedup vs Single-threaded CPU (log)", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_title("GPU Speedup vs Single-threaded CPU (Log, Best Batch Size)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"{r}²" for r in resolutions])
    
    # Speedup vs OpenMP (Linear)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax3.plot(
            x_positions,
            fdf["speedup_vs_omp"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    ax3.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="CPU Parity")
    ax3.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax3.set_ylabel("Speedup vs OpenMP CPU (32 threads)", fontsize=12)
    ax3.set_title("GPU Speedup vs OpenMP CPU (Linear, Best Batch Size)", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([f"{r}²" for r in resolutions])
    
    # Speedup vs OpenMP (Log)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax4.plot(
            x_positions,
            fdf["speedup_vs_omp"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    ax4.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="CPU Parity")
    ax4.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax4.set_ylabel("Speedup vs OpenMP CPU (32 threads, log)", fontsize=12)
    ax4.set_yscale("log")
    ax4.set_title("GPU Speedup vs OpenMP CPU (Log, Best Batch Size)", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels([f"{r}²" for r in resolutions])
    
    plt.tight_layout()
    _save_figure(fig, outdir, "speedup_vs_resolution")
    plt.close(fig)


# -------------------------
# Plot 2: GPU Time Breakdown
# -------------------------
def plot_gpu_time_breakdown(df, outdir):
    """Stacked bar chart showing H2D, kernel, and D2H breakdown for all configurations (best batch size)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # For each filter and resolution, select the best batch size
    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    
    filters = sorted(df_best["filter"].unique())
    
    for idx, flt in enumerate(filters):
        ax = axes[idx]
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        
        resolutions = [f"{r}²" for r in fdf["resolution"]]
        h2d_times = fdf["gpu_h2d_ms"].values
        kernel_times = fdf["gpu_kernel_ms"].values
        d2h_times = fdf["gpu_d2h_ms"].values
        
        x = range(len(resolutions))
        ax.bar(x, h2d_times, label="H2D Transfer", color="#1e88e5")
        ax.bar(x, kernel_times, bottom=h2d_times, label="Kernel", color="#43a047")
        ax.bar(x, d2h_times, bottom=h2d_times + kernel_times, label="D2H Transfer", color="#e53935")
        
        ax.set_xticks(x)
        ax.set_xticklabels(resolutions, fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(flt.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("GPU Time Breakdown by Filter", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, outdir, "gpu_time_breakdown")
    plt.close(fig)


# -------------------------
# Plot 3: Transfer Overhead Percentage
# -------------------------
def plot_transfer_overhead(df, outdir):
    """Plot transfer overhead percentage (best batch size for each config)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For each filter and resolution, select the best batch size
    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    
    colors = {"grayscale": "#1e88e5", "negative": "#fb8c00", "gaussian_blur": "#43a047"}
    
    resolutions = sorted(df_best["resolution"].unique())
    x_positions = list(range(len(resolutions)))
    
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        
        transfer_pct = (fdf["gpu_h2d_ms"] + fdf["gpu_d2h_ms"]) / fdf["gpu_total_ms"] * 100
        
        ax.plot(
            x_positions,
            transfer_pct,
            marker="o",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    
    ax.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax.set_ylabel("Transfer Overhead (%)", fontsize=12)
    ax.set_title("PCIe Transfer Overhead as % of Total GPU Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add reference line at 50%
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(0, 52, "50% (memory-bound)", fontsize=9, color="gray")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{r}²" for r in resolutions])
    
    plt.tight_layout()
    _save_figure(fig, outdir, "transfer_overhead")
    plt.close(fig)


# -------------------------
# Plot 4: Effective Bandwidth
# -------------------------
def plot_bandwidth(df, outdir):
    """Plot effective memory bandwidth across resolutions (best batch size)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For each filter and resolution, select the best batch size
    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    
    colors = {"grayscale": "#1e88e5", "negative": "#fb8c00", "gaussian_blur": "#43a047"}
    
    resolutions = sorted(df_best["resolution"].unique())
    x_positions = list(range(len(resolutions)))
    
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        
        ax.plot(
            x_positions,
            fdf["bandwidth_gb_s"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=flt.replace("_", " ").title(),
            color=colors.get(flt, "#333")
        )
    
    ax.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_title("Effective Memory Bandwidth", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{r}²" for r in resolutions])
    
    plt.tight_layout()
    _save_figure(fig, outdir, "bandwidth")
    plt.close(fig)


# -------------------------
# Plot 5: Absolute Performance Comparison
# -------------------------
def plot_absolute_times(df, outdir):
    """Compare absolute execution times: CPU single, CPU OpenMP, and GPU (best batch size)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For each resolution, select the best batch size based on GPU performance
    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    
    # Focus on one filter for clarity (gaussian_blur shows most dramatic difference)
    flt = "gaussian_blur"
    fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
    
    resolutions = fdf["resolution"].values
    x_positions = list(range(len(resolutions)))
    cpu_single = fdf["cpu_single_ms"].values
    cpu_omp = fdf["cpu_omp_ms"].values
    gpu_total = fdf["gpu_total_ms"].values
    
    ax.plot(x_positions, cpu_single, marker="s", linewidth=2, markersize=8, 
            label="CPU Single-threaded", color="#e53935")
    ax.plot(x_positions, cpu_omp, marker="^", linewidth=2, markersize=8,
            label="CPU OpenMP (32 threads)", color="#fb8c00")
    ax.plot(x_positions, gpu_total, marker="o", linewidth=2, markersize=8,
            label="GPU (AMD RX 6900 XT)", color="#43a047")
    
    ax.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax.set_ylabel("Execution Time (ms, log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Absolute Performance: {flt.replace('_', ' ').title()}", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(r)}²" for r in resolutions])
    
    plt.tight_layout()
    _save_figure(fig, outdir, "absolute_times")
    plt.close(fig)


# -------------------------
# Plot 6: Batch Size Scaling
# -------------------------
def plot_batch_size_scaling(df, outdir):
    """Plot GPU performance vs batch size for each filter and resolution."""
    if "batch_size" not in df.columns or df["batch_size"].nunique() <= 1:
        print("Skipping batch size scaling plot (only one batch size in data)")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    markers = {512: "o", 1024: "s", 2048: "^", 4096: "D"}
    batch_sizes = sorted(df["batch_size"].unique().tolist())
    batch_positions = list(range(len(batch_sizes)))
    
    filters = sorted(df["filter"].unique())
    
    # Plot 1-3: GPU Total Time vs Batch Size (one per filter)
    for idx, flt in enumerate(filters):
        ax = axes[idx]
        fdf = df[df["filter"] == flt]
        
        for res in sorted(fdf["resolution"].unique()):
            rdf = fdf[fdf["resolution"] == res].sort_values("batch_size")
            
            # Map batch_size values to positions
            x_vals = [batch_positions[batch_sizes.index(bs)] for bs in rdf["batch_size"]]
            
            ax.plot(
                x_vals,
                rdf["gpu_total_ms"],
                marker=markers.get(res, "o"),
                linewidth=2,
                markersize=10,
                label=f"{res}×{res}",
                alpha=0.8
            )
        
        ax.set_xlabel("Batch Size (images per GPU call)", fontsize=12)
        ax.set_ylabel("GPU Total Time (ms)", fontsize=12)
        ax.set_title(f"{flt.replace('_', ' ').title()} - Total Time", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, title="Resolution", loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(batch_positions)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
    
    # Plot 4-6: Speedup vs OpenMP vs Batch Size (one per filter)
    for idx, flt in enumerate(filters):
        ax = axes[idx + 3]
        fdf = df[df["filter"] == flt]
        
        for res in sorted(fdf["resolution"].unique()):
            rdf = fdf[fdf["resolution"] == res].sort_values("batch_size")
            
            # Map batch_size values to positions
            x_vals = [batch_positions[batch_sizes.index(bs)] for bs in rdf["batch_size"]]
            
            ax.plot(
                x_vals,
                rdf["speedup_vs_omp"],
                marker=markers.get(res, "o"),
                linewidth=2,
                markersize=10,
                label=f"{res}×{res}",
                alpha=0.8
            )
        
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="CPU Parity")
        ax.set_xlabel("Batch Size (images per GPU call)", fontsize=12)
        ax.set_ylabel("Speedup vs OpenMP (32 threads)", fontsize=12)
        ax.set_title(f"{flt.replace('_', ' ').title()} - Speedup", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, title="Resolution", loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(batch_positions)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
    
    plt.suptitle("Batch Size Scaling Analysis: Performance vs Batch Size", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    _save_figure(fig, outdir, "batch_size_scaling")
    plt.close(fig)


def generate_markdown_report(df: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive Markdown report with embedded charts and tables."""
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # Calculate summary statistics
    avg_transfer = ((df["gpu_h2d_ms"] + df["gpu_d2h_ms"]) / df["gpu_total_ms"] * 100).mean()
    max_speedup_single = df["speedup_vs_single"].max()
    max_speedup_single_filter = df.loc[df["speedup_vs_single"].idxmax(), "filter"]
    max_speedup_single_res = df.loc[df["speedup_vs_single"].idxmax(), "resolution"]
    max_speedup_omp = df["speedup_vs_omp"].max()
    max_speedup_omp_filter = df.loc[df["speedup_vs_omp"].idxmax(), "filter"]
    max_speedup_omp_res = df.loc[df["speedup_vs_omp"].idxmax(), "resolution"]
    
    # Get best and worst cases
    best_single_row = df.loc[df["speedup_vs_single"].idxmax()]
    best_omp_row = df.loc[df["speedup_vs_omp"].idxmax()]
    worst_omp_row = df.loc[df["speedup_vs_omp"].idxmin()]

    df_transfer = (df["gpu_h2d_ms"] + df["gpu_d2h_ms"]) / df["gpu_total_ms"] * 100
    df_blur = df[df["filter"] == "gaussian_blur"].copy()
    blur_transfer = (df_blur["gpu_h2d_ms"] + df_blur["gpu_d2h_ms"]) / df_blur["gpu_total_ms"] * 100
    df_simple = df[df["filter"].isin(["grayscale", "negative"])].copy()
    simple_transfer = (df_simple["gpu_h2d_ms"] + df_simple["gpu_d2h_ms"]) / df_simple["gpu_total_ms"] * 100

    blur_speedup_single_min = float(df_blur["speedup_vs_single"].min()) if not df_blur.empty else float("nan")
    blur_speedup_single_max = float(df_blur["speedup_vs_single"].max()) if not df_blur.empty else float("nan")
    simple_speedup_omp_min = float(df_simple["speedup_vs_omp"].min()) if not df_simple.empty else float("nan")
    simple_speedup_omp_max = float(df_simple["speedup_vs_omp"].max()) if not df_simple.empty else float("nan")

    blur_transfer_min = float(blur_transfer.min()) if not blur_transfer.empty else float("nan")
    simple_transfer_min = float(simple_transfer.min()) if not simple_transfer.empty else float("nan")
    simple_transfer_max = float(simple_transfer.max()) if not simple_transfer.empty else float("nan")

    max_bw = float(df["bandwidth_gb_s"].max())

    df_best = df.loc[df.groupby(['filter', 'resolution'])['speedup_vs_omp'].idxmax()]
    blur_4096_best = df_best[(df_best["filter"] == "gaussian_blur") & (df_best["resolution"] == 4096)]
    blur_4096_best_ms = float(blur_4096_best.iloc[0]["gpu_total_ms"]) if not blur_4096_best.empty else float("nan")

    batch_sizes = sorted(df["batch_size"].unique().tolist())
    batch_sizes_str = _fmt_int_list(batch_sizes)
    
    # Build markdown content
    md_content = f"""# HIP Image FX – Benchmark Report

**GPU-Accelerated Image Processing Performance Analysis**

*Generated: {date_str}*

---

## Executive Summary

| Metric | Value | Details |
|--------|-------|---------|
| **Peak Speedup (vs Single-CPU)** | **{max_speedup_single:.0f}×** | {max_speedup_single_filter.replace('_', ' ').title()} @ {max_speedup_single_res}² |
| **Peak Speedup (vs OpenMP)** | **{max_speedup_omp:.0f}×** | {max_speedup_omp_filter.replace('_', ' ').title()} @ {max_speedup_omp_res}² |
| **Avg Transfer Overhead** | **{avg_transfer:.0f}%** | of total GPU time |
| **Test Configurations** | **{len(df)}** | {len(df['filter'].unique())} filters × {len(df['resolution'].unique())} resolutions |

---

## Performance Analysis

### GPU Speedup vs CPU

![Speedup vs Resolution](speedup_vs_resolution.png)

**GPU Speedup vs CPU (Both Single-threaded and OpenMP)** – Shows performance relative to single-threaded CPU (top row, up to {max_speedup_single:.0f}×) and OpenMP CPU with 32 threads (bottom row, up to {max_speedup_omp:.0f}×). Gaussian blur achieves massive speedups due to compute-intensive nature, while simple filters are memory-bound.

---

### GPU Time Breakdown

![GPU Time Breakdown](gpu_time_breakdown.png)

**GPU Time Breakdown** – Shows proportion of time spent in H2D transfer (blue), kernel execution (green), and D2H transfer (red). Note how transfer overhead dominates for simple filters but drops substantially for gaussian blur.

---

### PCIe Transfer Overhead

![Transfer Overhead](transfer_overhead.png)

**PCIe Transfer Overhead** – Percentage of total GPU time spent on memory transfers. Simple filters are {simple_transfer_min:.0f}-{simple_transfer_max:.0f}% transfer-dominated, while gaussian blur drops to ~{blur_transfer_min:.0f}% at large resolutions.

---

### Effective Memory Bandwidth

![Bandwidth](bandwidth.png)

**Effective Memory Bandwidth** – Achieved bandwidth increases with image size as transfer overhead amortizes. Peak bandwidth reaches ~{max_bw:.1f} GB/s for large images.

---

### Absolute Performance Comparison

![Absolute Times](absolute_times.png)

**Absolute Performance Comparison** – Log-scale plot comparing single-threaded CPU, OpenMP CPU (32 threads), and GPU execution times for Gaussian blur. GPU maintains consistent ~{blur_4096_best_ms:.1f}ms performance even for 4096² images (best batch for that resolution).

---

### Batch Size Scaling

![Batch Size Scaling](batch_size_scaling.png)

**Batch Size Scaling** – How per-image GPU time and OpenMP speedup vary with batch size.

---

## Detailed Results

### Best and Worst Case Performance

| Case | Filter | Resolution | GPU Time | Speedup vs Single | Speedup vs OpenMP | Transfer Overhead |
|------|--------|------------|----------|-------------------|-------------------|-------------------|
| **Best (vs Single)** | {best_single_row['filter'].replace('_', ' ').title()} | {best_single_row['resolution']}² | {best_single_row['gpu_total_ms']:.2f} ms | {best_single_row['speedup_vs_single']:.1f}× | {best_single_row['speedup_vs_omp']:.1f}× | {(best_single_row['gpu_h2d_ms'] + best_single_row['gpu_d2h_ms']) / best_single_row['gpu_total_ms'] * 100:.1f}% |
| **Best (vs OpenMP)** | {best_omp_row['filter'].replace('_', ' ').title()} | {best_omp_row['resolution']}² | {best_omp_row['gpu_total_ms']:.2f} ms | {best_omp_row['speedup_vs_single']:.1f}× | {best_omp_row['speedup_vs_omp']:.1f}× | {(best_omp_row['gpu_h2d_ms'] + best_omp_row['gpu_d2h_ms']) / best_omp_row['gpu_total_ms'] * 100:.1f}% |
| **Worst** | {worst_omp_row['filter'].replace('_', ' ').title()} | {worst_omp_row['resolution']}² | {worst_omp_row['gpu_total_ms']:.2f} ms | {worst_omp_row['speedup_vs_single']:.1f}× | {worst_omp_row['speedup_vs_omp']:.1f}× | {(worst_omp_row['gpu_h2d_ms'] + worst_omp_row['gpu_d2h_ms']) / worst_omp_row['gpu_total_ms'] * 100:.1f}% |

---

### Complete Results Table

| Filter | Resolution | Batch Size | GPU Total (ms) | Kernel (ms) | Speedup vs Single | Speedup vs OpenMP | Bandwidth (GB/s) |
|--------|------------|------------|----------------|-------------|-------------------|-------------------|------------------|
"""

    # Add all results to table
    for _, row in df.sort_values(["filter", "resolution", "batch_size"]).iterrows():
        md_content += f"| {row['filter'].replace('_', ' ').title()} | {row['resolution']}² | {row['batch_size']} | {row['gpu_total_ms']:.3f} | {row['gpu_kernel_ms']:.3f} | {row['speedup_vs_single']:.2f}× | {row['speedup_vs_omp']:.2f}× | {row['bandwidth_gb_s']:.2f} |\n"

    md_content += f"""
---

## Key Insights

### Compute-Bound vs Memory-Bound

**Gaussian blur** is compute-intensive (O(n² × kernel_size²)) and shows excellent GPU scaling, achieving {blur_speedup_single_min:.0f}-{blur_speedup_single_max:.0f}× speedup vs single-threaded CPU. **Grayscale and negative** are memory-bound (simple per-pixel operations) and range from {simple_speedup_omp_min:.2f}-{simple_speedup_omp_max:.2f}× vs OpenMP, limited by PCIe transfer overhead.

### Batch Processing Architecture

The system uses configurable batch processing:

- **Batch sizes tested:** {batch_sizes_str} images per GPU call
- **Memory allocation:** Single large contiguous buffer for entire batch
- **Kernel launch:** One kernel processes all images in batch simultaneously
- **Performance impact:** Batch size can shift results for memory-bound filters because transfers dominate
- **Optimal batch size:** Varies by filter and resolution

### Resolution Scaling

As image resolution increases, kernel execution time grows quadratically while transfer overhead (as a percentage) decreases. This makes GPU acceleration more effective for larger images, especially for compute-bound operations like gaussian blur.

---

## Technical Details

| Parameter | Value |
|-----------|-------|
| **GPU** | AMD Radeon RX 6900 XT |
| **Framework** | HIP/ROCm |
| **Architecture** | Batch processing with synchronous execution |
| **Batch Sizes Tested** | {batch_sizes_str} images |
| **Image Format** | RGB (3 channels), 8-bit per channel |
| **Test Resolutions** | 512², 1024², 2048², 4096² |
| **CPU Baseline (Single)** | Single-threaded |
| **CPU Baseline (OpenMP)** | 32 threads |
| **Filters Tested** | Grayscale, Negative, Gaussian Blur (11×11 kernel) |

---

*Generated by `analyze_results.py`*
"""
    
    # Write to file
    output_path = output_dir / "benchmark_report.md"
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown report generated: {output_path}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze HIP image processing benchmark results"
    )
    parser.add_argument("csv", type=Path, help="Path to benchmark CSV file")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: File not found: {args.csv}")
        return 1

    print(f"Loading benchmark data from: {args.csv}")
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} benchmark results\n")
    
    outdir = args.csv.parent

    print_summary(df)
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80 + "\n")

    plot_speedup_vs_resolution(df, outdir)
    plot_gpu_time_breakdown(df, outdir)
    plot_transfer_overhead(df, outdir)
    plot_bandwidth(df, outdir)
    plot_absolute_times(df, outdir)
    plot_batch_size_scaling(df, outdir)

    print("\n" + "=" * 80)
    print("Generating Markdown report...")
    print("=" * 80 + "\n")

    generate_markdown_report(df, outdir)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
