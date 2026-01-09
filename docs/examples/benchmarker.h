#pragma once

/**
 * @file benchmarker.h
 * @brief Benchmarking engine for kernel configurations
 */

#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "tuning_config.h"
#include "../gpu_utils.h"

namespace imgfx::core::autotune
{
    /**
     * @brief Forward declaration of tuning options
     */
    struct TuningOptions;

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

            return BenchmarkResult(config, avg, stddev, min_t, max_t);
        }

        /**
         * @brief Benchmark all candidate configurations
         *
         * @param candidates List of configurations to test
         * @param args Kernel arguments
         * @param stream HIP stream for execution
         * @param options Tuning options
         * @return Vector of valid benchmark results
         */
        std::vector<BenchmarkResult> benchmark_all(
            const std::vector<TuningConfig> &candidates,
            const Args &args,
            hipStream_t stream,
            const TuningOptions &options)
        {
            std::vector<BenchmarkResult> results;
            results.reserve(candidates.size());

            for (const auto &config : candidates)
            {
                auto result = benchmark(config, args, stream, options);

                if (result.valid)
                {
                    results.push_back(result);

                    if (options.verbose)
                    {
                        printf("  [%dx%d] = %.4f ± %.4f ms (min: %.4f, max: %.4f)\n",
                               config.block_x(), config.block_y(),
                               result.avg_time_ms, result.stddev_ms,
                               result.min_time_ms, result.max_time_ms);
                    }
                }
                else if (options.verbose)
                {
                    printf("  [%dx%d] = INVALID\n",
                           config.block_x(), config.block_y());
                }
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
