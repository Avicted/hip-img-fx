// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#pragma once

/**
 * @file types.h
 * @brief Common types and utilities for autotuning framework
 */

#include <hip/hip_runtime.h>
#include <string>
#include <cstdio>
#include <cstdlib>

/**
 * @brief Debug-only assertion for autotuning framework
 *
 * Compiles to nothing in release builds (NDEBUG defined).
 * In debug builds, prints diagnostic message and aborts on failure.
 */
#ifdef NDEBUG
#define AUTOTUNE_ASSERT(cond, msg) ((void)0)
#else
#define AUTOTUNE_ASSERT(cond, msg)                                                          \
    do                                                                                      \
    {                                                                                       \
        if (!(cond))                                                                        \
        {                                                                                   \
            fprintf(stderr, "[AutoTune Assert] %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
            std::abort();                                                                   \
        }                                                                                   \
    } while (0)
#endif

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

        /**
         * @brief Validate options are sane
         * @return true if all options are valid
         */
        bool validate() const
        {
            if (warmup_runs < 0 || timing_runs <= 0)
            {
                return false;
            }
            if (early_exit_threshold < 1.0)
            {
                return false;
            }
            if (early_exit_min_coverage <= 0.0 || early_exit_min_coverage > 1.0)
            {
                return false;
            }

            return true;
        }

        static TuningOptions defaults()
        {
            TuningOptions opts;
            AUTOTUNE_ASSERT(opts.validate(), "Default options validation failed");
            return opts;
        }

        static TuningOptions quiet()
        {
            TuningOptions opts;
            opts.verbose = false;
            AUTOTUNE_ASSERT(opts.validate(), "Quiet options validation failed");
            return opts;
        }

        /// Conservative preset: Test all candidates (safer, slower)
        static TuningOptions conservative()
        {
            TuningOptions opts;
            opts.enable_early_exit = false; // Disable early exit
            AUTOTUNE_ASSERT(opts.validate(), "Conservative options validation failed");
            return opts;
        }

        /// Aggressive preset: Exit earlier (faster, slight risk)
        static TuningOptions aggressive()
        {
            TuningOptions opts;
            opts.early_exit_threshold = 1.10;   // 10% tolerance
            opts.early_exit_min_coverage = 0.3; // Test only 30%
            AUTOTUNE_ASSERT(opts.validate(), "Aggressive options validation failed");
            return opts;
        }
    };

} // namespace imgfx::core::autotune
