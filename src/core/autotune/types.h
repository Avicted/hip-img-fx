#pragma once

/**
 * @file types.h
 * @brief Common types and utilities for autotuning framework
 */

#include <hip/hip_runtime.h>
#include <string>

namespace imgfx::core::autotune
{
    /**
     * @brief Options for autotuning process
     */
    struct TuningOptions
    {
        int warmup_runs = 5;       ///< Number of warmup iterations
        int timing_runs = 10;      ///< Number of timed iterations
        bool verbose = true;       ///< Print tuning progress
        bool force_retune = false; ///< Ignore cache and retune

        // Early-exit benchmarking options
        bool enable_early_exit = true;        ///< Stop testing if best is clearly found
        double early_exit_threshold = 1.15;   ///< Tolerance: 15% slower than best
        double early_exit_min_coverage = 0.4; ///< Test at least 40% of candidates

        static TuningOptions defaults() { return TuningOptions{}; }

        static TuningOptions quiet()
        {
            TuningOptions opts;
            opts.verbose = false;
            return opts;
        }

        /// Conservative preset: Test all candidates (safer, slower)
        static TuningOptions conservative()
        {
            TuningOptions opts;
            opts.enable_early_exit = false; // Disable early exit
            return opts;
        }

        /// Aggressive preset: Exit earlier (faster, slight risk)
        static TuningOptions aggressive()
        {
            TuningOptions opts;
            opts.early_exit_threshold = 1.10;   // 10% tolerance
            opts.early_exit_min_coverage = 0.3; // Test only 30%
            return opts;
        }
    };

} // namespace imgfx::core::autotune
