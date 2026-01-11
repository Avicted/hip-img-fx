#!/bin/bash
# Automated benchmark runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_ROOT/bench/results"

echo "======================================================================"
echo "  HIP Image FX - Automated Benchmark Runner"
echo "======================================================================"
echo ""

mkdir -p "$RESULTS_DIR"

BENCH_BIN="$PROJECT_ROOT/build/hip-img-fx-bench"
if [ ! -f "$BENCH_BIN" ]; then
    echo "ERROR: Benchmark binary not found: $BENCH_BIN"
    echo "Please build the project first with: ninja -C build"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"

echo "Running benchmark suite..."
echo "Output will be saved to: $OUTPUT_CSV"
echo ""

# Disable cache saving during benchmark to avoid interference between static orchestrators
export HIP_IMG_FX_NO_CACHE_SAVE=1

"$BENCH_BIN" --warmup 2 --iterations 5 --output "$OUTPUT_CSV" --verbose

echo ""
echo "======================================================================"
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "To analyze results, you can:"
echo "  - Open the CSV in a spreadsheet application"
echo "  - Use the analysis script: ./bench/scripts/run_analysis.sh $OUTPUT_CSV"
echo "======================================================================"
