#pragma once

/**
 * @file benchmarker.h
 * @brief Benchmarking engine for kernel configurations
 */

#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "tuning_config.h"
#include "types.h"
#include "../gpu_utils.h"

namespace imgfx::core::autotune
{
    /**
     * @brief Result of benchmarking a single configuration
     */
    struct BenchmarkResult
    {
        TuningConfig config;
        float avg_time_ms;
        float stddev_ms;
        float min_time_ms;
        float max_time_ms;
        bool valid;

        BenchmarkResult()
            : avg_time_ms(0.0f), stddev_ms(0.0f),
              min_time_ms(0.0f), max_time_ms(0.0f), valid(false) {}

        BenchmarkResult(const TuningConfig &cfg, float avg, float stddev, float min_t, float max_t)
            : config(cfg), avg_time_ms(avg), stddev_ms(stddev),
              min_time_ms(min_t), max_time_ms(max_t), valid(true) {}

        /**
         * @brief Comparison operator for finding best config
         */
        bool operator<(const BenchmarkResult &other) const
        {
            return avg_time_ms < other.avg_time_ms;
        }
    };

    /**
     * @brief Benchmarking engine for kernel configurations
     *
     * Template-based design allows compile-time specialization
     * per kernel type.
     *
     * @tparam KernelTraits Traits struct defining kernel characteristics
     */
    template <typename KernelTraits>
    class Benchmarker
    {
    public:
        using Args = typename KernelTraits::Args;

        /**
         * @brief Benchmark a single configuration
         *
         * Performs warmup, then times multiple iterations and
         * computes statistics.
         *
         * @param config Configuration to benchmark
         * @param args Kernel arguments
         * @param stream HIP stream for execution
         * @param options Tuning options (warmup/timing runs)
         * @return BenchmarkResult with timing statistics
         */
        BenchmarkResult benchmark(
            const TuningConfig &config,
            const Args &args,
            hipStream_t stream,
            const TuningOptions &options)
        {
            // Validate configuration
            if (!KernelTraits::is_valid_config(config, args))
            {
                return BenchmarkResult{}; // Invalid
            }

            // Warmup phase
            for (int i = 0; i < options.warmup_runs; ++i)
            {
                KernelTraits::launch(config, args, stream);
            }
            HIP_ERRCHK(hipStreamSynchronize(stream));

            // Timing phase
            std::vector<float> times;
            times.reserve(options.timing_runs);

            for (int i = 0; i < options.timing_runs; ++i)
            {
                HIPEvent start, end;
                if (!start.is_valid() || !end.is_valid())
                {
                    fprintf(stderr, "Warning: Failed to create HIP events for benchmarking\n");
                    return BenchmarkResult{};
                }

                start.record(stream);
                KernelTraits::launch(config, args, stream);
                end.record(stream);
                end.synchronize();

                times.push_back(HIPEvent::elapsed_time(start, end));
            }

            // Compute statistics
            float avg = compute_mean(times);
            float stddev = compute_stddev(times, avg);
            float min_t = *std::min_element(times.begin(), times.end());
            float max_t = *std::max_element(times.begin(), times.end());

            // Sanity checks on timing results
            constexpr float MAX_REASONABLE_TIME_MS = 10000.0f; // 10 seconds
            if (avg <= 0.0f || avg > MAX_REASONABLE_TIME_MS)
            {
                fprintf(stderr, "[AutoTune] Warning: Suspicious timing %.4f ms (rejected)\n", avg);
                return BenchmarkResult{}; // Invalid
            }

            // Warn on high variance (suggests measurement instability)
            if (stddev > avg * 0.5f)
            {
                fprintf(stderr, "[AutoTune] Warning: High timing variance (%.1f%%), results may be unreliable\n",
                        (stddev / avg) * 100.0f);
            }

            return BenchmarkResult(config, avg, stddev, min_t, max_t);
        }

        /**
         * @brief Benchmark all candidate configurations
         *
         * Implements early-exit optimization: stops testing if current best
         * is clearly superior to remaining candidates.
         *
         * @param candidates List of configurations to test
         * @param args Kernel arguments
         * @param stream HIP stream for execution
         * @param options Tuning options (including early-exit settings)
         * @return Vector of valid benchmark results
         */
        std::vector<BenchmarkResult> benchmark_all(
            const std::vector<TuningConfig> &candidates,
            const Args &args,
            hipStream_t stream,
            const TuningOptions &options)
        {
            if (candidates.empty())
            {
                return {};
            }

            std::vector<BenchmarkResult> results;
            results.reserve(candidates.size());

            double best_time_ms = std::numeric_limits<double>::max();
            int candidates_tested = 0;
            int candidates_skipped = 0;

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                const auto &config = candidates[i];

                // Benchmark this candidate
                auto result = benchmark(config, args, stream, options);

                if (!result.valid)
                {
                    if (options.verbose)
                    {
                        printf("  [%dx%d] = INVALID\n",
                               config.block_x(), config.block_y());
                    }
                    continue; // Skip invalid configurations
                }

                results.push_back(result);
                candidates_tested++;

                // Track best time
                if (result.avg_time_ms < best_time_ms)
                {
                    best_time_ms = result.avg_time_ms;
                }

                // Verbose output for this candidate
                if (options.verbose)
                {
                    printf("  [%dx%d] = %.4f ± %.4f ms (min: %.4f, max: %.4f)\n",
                           config.block_x(), config.block_y(),
                           result.avg_time_ms, result.stddev_ms,
                           result.min_time_ms, result.max_time_ms);
                }

                // Early-exit logic (after testing minimum number)
                if (options.enable_early_exit && candidates_tested >= 3)
                {
                    double threshold = best_time_ms * options.early_exit_threshold;
                    double coverage = static_cast<double>(candidates_tested) / candidates.size();

                    // Exit if:
                    // 1. This candidate is significantly slower than best
                    // 2. We've tested at least minimum coverage
                    if (result.avg_time_ms > threshold &&
                        coverage >= options.early_exit_min_coverage)
                    {
                        candidates_skipped = candidates.size() - i - 1;

                        if (options.verbose && candidates_skipped > 0)
                        {
                            double percent_slower = (result.avg_time_ms / best_time_ms - 1.0) * 100.0;
                            printf("  [Early exit: last candidate %.1f%% slower, "
                                   "skipping %d remaining]\n",
                                   percent_slower, candidates_skipped);
                        }
                        break;
                    }
                }
            }

            // Report savings if early exit occurred
            if (options.verbose && candidates_skipped > 0)
            {
                // Estimate time saved (warmup + timing runs per candidate)
                double approx_time_per_candidate = best_time_ms *
                                                   (options.warmup_runs + options.timing_runs);
                double saved_ms = candidates_skipped * approx_time_per_candidate;

                printf("  [Saved ~%.1f ms by skipping %d candidates]\n",
                       saved_ms, candidates_skipped);
            }

            return results;
        }

    private:
        /**
         * @brief Compute mean of timing samples
         */
        static float compute_mean(const std::vector<float> &values)
        {
            if (values.empty())
                return 0.0f;

            float sum = 0.0f;
            for (float v : values)
            {
                sum += v;
            }
            return sum / values.size();
        }

        /**
         * @brief Compute standard deviation of timing samples
         */
        static float compute_stddev(const std::vector<float> &values, float mean)
        {
            if (values.size() <= 1)
                return 0.0f;

            float variance = 0.0f;
            for (float v : values)
            {
                float diff = v - mean;
                variance += diff * diff;
            }
            variance /= (values.size() - 1); // Sample variance

            return std::sqrt(variance);
        }
    };

} // namespace imgfx::core::autotune
