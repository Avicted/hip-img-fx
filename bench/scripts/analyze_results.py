#!/usr/bin/env python3
"""Analysis of HIP image-processing benchmarks.

This script analyzes CSV output produced by the benchmark harness in
`bench/run_bench.cpp`.

Current benchmark sweep (as configured in the harness):
- Resolutions: 512, 1024, 2048, 4096
- Filters: grayscale, negative, gaussian_blur
- Batch sizes: 1, 32, 64

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
    """Save each figure as both PNG (for quick viewing) and SVG (for inspection/zoom)."""
    png_path = outdir / f"{stem}.png"
    svg_path = outdir / f"{stem}.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Generated: {png_path.name}")
    print(f"Generated: {svg_path.name}")


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
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax1.plot(
            fdf["resolution"],
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
    
    # Speedup vs Single-threaded (Log)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax2.plot(
            fdf["resolution"],
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
    
    # Speedup vs OpenMP (Linear)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax3.plot(
            fdf["resolution"],
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
    
    # Speedup vs OpenMP (Log)
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        ax4.plot(
            fdf["resolution"],
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

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticks(sorted(df_best["resolution"].unique()))
        ax.set_xticklabels([f"{r}²" for r in sorted(df_best["resolution"].unique())])
    
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
    
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        
        transfer_pct = (fdf["gpu_h2d_ms"] + fdf["gpu_d2h_ms"]) / fdf["gpu_total_ms"] * 100
        
        ax.plot(
            fdf["resolution"],
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
    ax.text(df["resolution"].min(), 52, "50% (memory-bound)", fontsize=9, color="gray")

    ax.set_xticks(sorted(df_best["resolution"].unique()))
    ax.set_xticklabels([f"{r}²" for r in sorted(df_best["resolution"].unique())])
    
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
    
    for flt in sorted(df_best["filter"].unique()):
        fdf = df_best[df_best["filter"] == flt].sort_values("resolution")
        
        ax.plot(
            fdf["resolution"],
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

    ax.set_xticks(sorted(df_best["resolution"].unique()))
    ax.set_xticklabels([f"{r}²" for r in sorted(df_best["resolution"].unique())])
    
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
    cpu_single = fdf["cpu_single_ms"].values
    cpu_omp = fdf["cpu_omp_ms"].values
    gpu_total = fdf["gpu_total_ms"].values
    
    ax.plot(resolutions, cpu_single, marker="s", linewidth=2, markersize=8, 
            label="CPU Single-threaded", color="#e53935")
    ax.plot(resolutions, cpu_omp, marker="^", linewidth=2, markersize=8,
            label="CPU OpenMP (32 threads)", color="#fb8c00")
    ax.plot(resolutions, gpu_total, marker="o", linewidth=2, markersize=8,
            label="GPU (AMD RX 6900 XT)", color="#43a047")
    
    ax.set_xlabel("Resolution (pixels per side)", fontsize=12)
    ax.set_ylabel("Execution Time (ms, log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Absolute Performance: {flt.replace('_', ' ').title()}", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    ax.set_xticks(resolutions)
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
    batch_ticks = sorted(df["batch_size"].unique().tolist())
    
    filters = sorted(df["filter"].unique())
    
    # Plot 1-3: GPU Total Time vs Batch Size (one per filter)
    for idx, flt in enumerate(filters):
        ax = axes[idx]
        fdf = df[df["filter"] == flt]
        
        for res in sorted(fdf["resolution"].unique()):
            rdf = fdf[fdf["resolution"] == res].sort_values("batch_size")
            
            ax.plot(
                rdf["batch_size"],
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
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch_ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Plot 4-6: Speedup vs OpenMP vs Batch Size (one per filter)
    for idx, flt in enumerate(filters):
        ax = axes[idx + 3]
        fdf = df[df["filter"] == flt]
        
        for res in sorted(fdf["resolution"].unique()):
            rdf = fdf[fdf["resolution"] == res].sort_values("batch_size")
            
            ax.plot(
                rdf["batch_size"],
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
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch_ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.suptitle("Batch Size Scaling Analysis: Performance vs Batch Size", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    _save_figure(fig, outdir, "batch_size_scaling")
    plt.close(fig)


def generate_html_report(df: pd.DataFrame, output_dir: Path):
    """Generate a modern, self-contained HTML report with embedded visualizations."""
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
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HIP Image FX – Benchmark Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
    color: #333;
    line-height: 1.6;
    padding: 20px;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    overflow: hidden;
}}

header {{
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
    color: white;
    padding: 60px 40px;
    text-align: center;
}}

header h1 {{
    font-size: 3em;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}}

header .subtitle {{
    font-size: 1.2em;
    opacity: 0.95;
    font-weight: 300;
}}

header .meta {{
    margin-top: 20px;
    font-size: 0.95em;
    opacity: 0.9;
}}

.content {{
    padding: 40px;
}}

h2 {{
    font-size: 2em;
    margin: 50px 0 25px 0;
    padding-bottom: 15px;
    border-bottom: 3px solid #1e88e5;
    color: #2d3748;
}}

h3 {{
    font-size: 1.4em;
    margin: 30px 0 15px 0;
    color: #4a5568;
}}

.summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 30px 0;
}}

.card {{
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
    color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
    transition: transform 0.3s ease;
}}

.card:hover {{
    transform: translateY(-5px);
}}

.card .value {{
    font-size: 2.5em;
    font-weight: 700;
    margin: 10px 0;
}}

.card .label {{
    font-size: 0.95em;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.figure {{
    background: #f7fafc;
    padding: 30px;
    margin: 30px 0;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}}

.figure img {{
    width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}}

.figure .caption {{
    margin-top: 15px;
    color: #4a5568;
    font-size: 0.95em;
    line-height: 1.5;
}}

.insight {{
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #1e88e5;
    padding: 20px 25px;
    margin: 25px 0;
    border-radius: 8px;
}}

.insight h4 {{
    color: #1565c0;
    margin-bottom: 10px;
    font-size: 1.1em;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.95em;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}}

.data-table thead {{
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
    color: white;
}}

.data-table th {{
    padding: 15px;
    text-align: left;
    font-weight: 600;
}}

.data-table td {{
    padding: 12px 15px;
    border-bottom: 1px solid #e2e8f0;
}}

.data-table tbody tr:hover {{
    background: #f7fafc;
}}

.data-table tbody tr:last-child td {{
    border-bottom: none;
}}

footer {{
    background: #2d3748;
    color: #a0aec0;
    text-align: center;
    padding: 30px;
    font-size: 0.9em;
}}

footer a {{
    color: #1e88e5;
    text-decoration: none;
}}

footer a:hover {{
    text-decoration: underline;
}}

.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 0 5px;
}}

.badge-success {{
    background: #c6f6d5;
    color: #22543d;
}}

.badge-warning {{
    background: #feebc8;
    color: #744210;
}}

.badge-info {{
    background: #bee3f8;
    color: #2c5282;
}}
</style>
</head>

<body>

<div class="container">

<header>
    <h1>HIP Image FX</h1>
    <div class="subtitle">GPU-Accelerated Image Processing Benchmark Report</div>
    <div class="meta">
        <span class="badge badge-info">AMD ROCm/HIP</span>
        <span class="badge badge-info">Generated: {date_str}</span>
    </div>
</header>

<div class="content">

<h2>Executive Summary</h2>

<div class="summary-cards">
    <div class="card">
        <div class="label">Peak Speedup (vs Single-CPU)</div>
        <div class="value">{max_speedup_single:.0f}×</div>
        <div class="label">{max_speedup_single_filter.replace('_', ' ').title()} @ {max_speedup_single_res}²</div>
    </div>
    <div class="card">
        <div class="label">Peak Speedup (vs OpenMP)</div>
        <div class="value">{max_speedup_omp:.0f}×</div>
        <div class="label">{max_speedup_omp_filter.replace('_', ' ').title()} @ {max_speedup_omp_res}²</div>
    </div>
    <div class="card">
        <div class="label">Avg Transfer Overhead</div>
        <div class="value">{avg_transfer:.0f}%</div>
        <div class="label">of total GPU time</div>
    </div>
    <div class="card">
        <div class="label">Test Configurations</div>
        <div class="value">{len(df)}</div>
        <div class="label">{len(df['filter'].unique())} filters × {len(df['resolution'].unique())} resolutions</div>
    </div>
</div>

<h2>Performance Analysis</h2>

<div class="figure">
    <img src="speedup_vs_resolution.png" alt="Speedup vs Resolution">
    <div class="caption">
        <strong>GPU Speedup vs CPU (Both Single-threaded and OpenMP)</strong> – Shows performance relative to single-threaded CPU 
        (top row, up to {max_speedup_single:.0f}×) and OpenMP CPU with 32 threads (bottom row, up to {max_speedup_omp:.0f}×). 
        Gaussian blur achieves massive speedups due to compute-intensive nature, while simple filters are memory-bound.
        <div style="margin-top: 8px;"><a href="speedup_vs_resolution.svg">Download SVG</a></div>
    </div>
</div>

<div class="figure">
    <img src="gpu_time_breakdown.png" alt="GPU Time Breakdown">
    <div class="caption">
        <strong>GPU Time Breakdown</strong> – Shows proportion of time spent in H2D transfer (blue), 
        kernel execution (green), and D2H transfer (red). Note how transfer overhead dominates for 
        simple filters but drops substantially for gaussian blur.
        <div style="margin-top: 8px;"><a href="gpu_time_breakdown.svg">Download SVG</a></div>
    </div>
</div>

<div class="figure">
    <img src="transfer_overhead.png" alt="Transfer Overhead">
    <div class="caption">
        <strong>PCIe Transfer Overhead</strong> – Percentage of total GPU time spent on memory transfers. 
        Simple filters are {simple_transfer_min:.0f}-{simple_transfer_max:.0f}% transfer-dominated, while gaussian blur drops to ~{blur_transfer_min:.0f}% at large resolutions.
        <div style="margin-top: 8px;"><a href="transfer_overhead.svg">Download SVG</a></div>
    </div>
</div>

<div class="figure">
    <img src="bandwidth.png" alt="Bandwidth">
    <div class="caption">
        <strong>Effective Memory Bandwidth</strong> – Achieved bandwidth increases with image size as 
        transfer overhead amortizes. Peak bandwidth reaches ~{max_bw:.1f} GB/s for large images.
        <div style="margin-top: 8px;"><a href="bandwidth.svg">Download SVG</a></div>
    </div>
</div>

<div class="figure">
    <img src="absolute_times.png" alt="Absolute Times">
    <div class="caption">
        <strong>Absolute Performance Comparison</strong> – Log-scale plot comparing single-threaded CPU, 
        OpenMP CPU (32 threads), and GPU execution times for Gaussian blur. GPU maintains consistent 
        ~{blur_4096_best_ms:.1f}ms performance even for 4096² images (best batch for that resolution).
        <div style="margin-top: 8px;"><a href="absolute_times.svg">Download SVG</a></div>
    </div>
</div>

<div class="figure">
    <img src="batch_size_scaling.png" alt="Batch Size Scaling">
    <div class="caption">
        <strong>Batch Size Scaling</strong> – How per-image GPU time and OpenMP speedup vary with batch size.
        <div style="margin-top: 8px;"><a href="batch_size_scaling.svg">Download SVG</a></div>
    </div>
</div>

<h2>Detailed Results</h2>

<h3>Best and Worst Case Performance</h3>

<table class="data-table">
<thead>
    <tr>
        <th>Case</th>
        <th>Filter</th>
        <th>Resolution</th>
        <th>GPU Time</th>
        <th>Speedup vs Single</th>
        <th>Speedup vs OpenMP</th>
        <th>Transfer Overhead</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td><span class="badge badge-success">Best (vs Single)</span></td>
        <td>{best_single_row['filter'].replace('_', ' ').title()}</td>
        <td>{best_single_row['resolution']}²</td>
        <td>{best_single_row['gpu_total_ms']:.2f} ms</td>
        <td>{best_single_row['speedup_vs_single']:.1f}×</td>
        <td>{best_single_row['speedup_vs_omp']:.1f}×</td>
        <td>{(best_single_row['gpu_h2d_ms'] + best_single_row['gpu_d2h_ms']) / best_single_row['gpu_total_ms'] * 100:.1f}%</td>
    </tr>
    <tr>
        <td><span class="badge badge-success">Best (vs OpenMP)</span></td>
        <td>{best_omp_row['filter'].replace('_', ' ').title()}</td>
        <td>{best_omp_row['resolution']}²</td>
        <td>{best_omp_row['gpu_total_ms']:.2f} ms</td>
        <td>{best_omp_row['speedup_vs_single']:.1f}×</td>
        <td>{best_omp_row['speedup_vs_omp']:.1f}×</td>
        <td>{(best_omp_row['gpu_h2d_ms'] + best_omp_row['gpu_d2h_ms']) / best_omp_row['gpu_total_ms'] * 100:.1f}%</td>
    </tr>
    <tr>
        <td><span class="badge badge-warning">Worst</span></td>
        <td>{worst_omp_row['filter'].replace('_', ' ').title()}</td>
        <td>{worst_omp_row['resolution']}²</td>
        <td>{worst_omp_row['gpu_total_ms']:.2f} ms</td>
        <td>{worst_omp_row['speedup_vs_single']:.1f}×</td>
        <td>{worst_omp_row['speedup_vs_omp']:.1f}×</td>
        <td>{(worst_omp_row['gpu_h2d_ms'] + worst_omp_row['gpu_d2h_ms']) / worst_omp_row['gpu_total_ms'] * 100:.1f}%</td>
    </tr>
</tbody>
</table>

<h3>Complete Results Table</h3>

<table class="data-table">
<thead>
    <tr>
        <th>Filter</th>
        <th>Resolution</th>
        <th>Batch Size</th>
        <th>GPU Total (ms)</th>
        <th>Kernel (ms)</th>
        <th>Speedup vs Single</th>
        <th>Speedup vs OpenMP</th>
        <th>Bandwidth (GB/s)</th>
    </tr>
</thead>
<tbody>
"""

    # Add all results to table
    for _, row in df.sort_values(["filter", "resolution", "batch_size"]).iterrows():
        html_content += f"""    <tr>
        <td>{row['filter'].replace('_', ' ').title()}</td>
        <td>{row['resolution']}²</td>
        <td>{row['batch_size']}</td>
        <td>{row['gpu_total_ms']:.3f}</td>
        <td>{row['gpu_kernel_ms']:.3f}</td>
        <td>{row['speedup_vs_single']:.2f}×</td>
        <td>{row['speedup_vs_omp']:.2f}×</td>
        <td>{row['bandwidth_gb_s']:.2f}</td>
    </tr>
"""

    html_content += f"""</tbody>
</table>

<h2>Key Insights</h2>

<div class="insight">
    <h4>Compute-Bound vs Memory-Bound</h4>
    <p><strong>Gaussian blur</strong> is compute-intensive (O(n² × kernel_size²)) and shows excellent GPU scaling, 
    achieving {blur_speedup_single_min:.0f}-{blur_speedup_single_max:.0f}× speedup vs single-threaded CPU. <strong>Grayscale and negative</strong> are memory-bound 
    (simple per-pixel operations) and range from {simple_speedup_omp_min:.2f}-{simple_speedup_omp_max:.2f}× vs OpenMP, limited by PCIe transfer overhead.</p>
</div>

<div class="insight">
    <h4>Batch Processing Architecture</h4>
    <p>The system uses configurable batch processing:</p>
    <ul style="margin-left: 25px; margin-top: 10px;">
        <li><strong>Batch sizes tested:</strong> {batch_sizes_str} images per GPU call</li>
        <li><strong>Memory allocation:</strong> Single large contiguous buffer for entire batch</li>
        <li><strong>Kernel launch:</strong> One kernel processes all images in batch simultaneously</li>
        <li><strong>Performance impact:</strong> Batch size can shift results for memory-bound filters because transfers dominate</li>
        <li><strong>Optimal batch size:</strong> Varies by filter and resolution</li>
    </ul>
</div>

<div class="insight">
    <h4>Resolution Scaling</h4>
    <p>As image resolution increases, kernel execution time grows quadratically while transfer overhead 
    (as a percentage) decreases. This makes GPU acceleration more effective for larger images, especially 
    for compute-bound operations like gaussian blur.</p>
</div>

<h2>Technical Details</h2>

<table class="data-table">
<thead>
    <tr><th>Parameter</th><th>Value</th></tr>
</thead>
<tbody>
    <tr><td>GPU</td><td>AMD Radeon RX 6900 XT</td></tr>
    <tr><td>Framework</td><td>HIP/ROCm</td></tr>
    <tr><td>Architecture</td><td>Batch processing with synchronous execution</td></tr>
    <tr><td>Batch Sizes Tested</td><td>{batch_sizes_str} images</td></tr>
    <tr><td>Image Format</td><td>RGB (3 channels), 8-bit per channel</td></tr>
    <tr><td>Test Resolutions</td><td>512², 1024², 2048², 4096²</td></tr>
    <tr><td>CPU Baseline (Single)</td><td>Single-threaded</td></tr>
    <tr><td>CPU Baseline (OpenMP)</td><td>32 threads</td></tr>
    <tr><td>Filters Tested</td><td>Grayscale, Negative, Gaussian Blur (11×11 kernel)</td></tr>
</tbody>
</table>

</div>

<footer>
    <p>Generated by <code>analyze_results.py</code></p>
    <p>View source: <a href="https://github.com/Avicted/hip-img-fx">github.com/Avicted/hip-img-fx</a></p>
</footer>

</div>

</body>
</html>
"""
    
    # Write to file
    output_path = output_dir / "benchmark_report.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"HTML report generated: {output_path}")
    print(f"    Open in browser: file://{output_path.absolute()}")


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
    print("Generating HTML report...")
    print("=" * 80 + "\n")

    generate_html_report(df, outdir)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
