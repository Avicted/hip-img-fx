#pragma once

/**
 * @file orchestrator.h
 * @brief Main autotuning orchestrator using traits-based template design
 *
 * Example usage:
 *
 *     // Define kernel traits
 *     struct MyKernelTraits {
 *         static constexpr const char* name() { return "my_kernel"; }
 *         struct Args { ... };
 *         struct Context { std::string cache_key() const { ... } };
 *         static std::vector<TuningConfig> generate_candidates();
 *         static bool is_valid_config(const TuningConfig& cfg, const Args& args);
 *         static void launch(const TuningConfig& cfg, const Args& args, hipStream_t stream);
 *     };
 *
 *     // Use orchestrator
 *     TuningOrchestrator<MyKernelTraits> tuner;
 *     MyKernelTraits::Args args = {...};
 *     MyKernelTraits::Context ctx = {...};
 *     tuner.execute(args, ctx, stream);
 */

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

#include "tuning_config.h"
#include "cache_store.h"
#include "benchmarker.h"
#include "../gpu_utils.h"

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

        static TuningOptions defaults() { return TuningOptions{}; }

        static TuningOptions quiet()
        {
            TuningOptions opts;
            opts.verbose = false;
            return opts;
        }
    };

    /**
     * @brief Main autotuning orchestrator
     *
     * Coordinates the autotuning process using kernel-specific traits.
     * Template-based design ensures zero runtime overhead and type safety.
     *
     * @tparam KernelTraits Traits struct defining kernel characteristics
     *
     * KernelTraits Requirements:
     * - static constexpr const char* name()
     * - struct Args { ... } - Kernel launch arguments
     * - struct Context { std::string cache_key() const; } - Cache context
     * - static std::vector<TuningConfig> generate_candidates()
     * - static bool is_valid_config(const TuningConfig&, const Args&)
     * - static void launch(const TuningConfig&, const Args&, hipStream_t)
     */
    template <typename KernelTraits>
    class TuningOrchestrator
    {
    public:
        using Args = typename KernelTraits::Args;
        using Context = typename KernelTraits::Context;

        /**
         * @brief Construct orchestrator and load cache
         *
         * @param cache_path Path to persistent cache file
         */
        explicit TuningOrchestrator(const std::string &cache_path = ".autotune_cache.json")
            : cache_path_(cache_path)
        {
            gpu_arch_ = query_gpu_arch();
            cache_.load(cache_path_);
        }

        /**
         * @brief Destructor - saves cache to disk
         */
        ~TuningOrchestrator()
        {
            cache_.save(cache_path_);
        }

        // Disable copy, allow move
        TuningOrchestrator(const TuningOrchestrator &) = delete;
        TuningOrchestrator &operator=(const TuningOrchestrator &) = delete;
        TuningOrchestrator(TuningOrchestrator &&) noexcept = default;
        TuningOrchestrator &operator=(TuningOrchestrator &&) noexcept = default;

        /**
         * @brief Get or compute optimal configuration
         *
         * Checks thread-local cache first for maximum performance,
         * then persistent cache, then performs autotuning.
         *
         * @param args Kernel arguments (used for benchmarking)
         * @param ctx Context for cache key generation
         * @param options Tuning options
         * @return Optimal TuningConfig
         */
        TuningConfig get_or_tune(
            const Args &args,
            const Context &ctx,
            const TuningOptions &options = TuningOptions::defaults())
        {
            // Fast path: thread-local cache (~5ns overhead)
            thread_local static std::unordered_map<std::string, TuningConfig> fast_cache;
            std::string fast_key = std::string(KernelTraits::name()) + ":" + ctx.cache_key();

            if (!options.force_retune)
            {
                auto fast_it = fast_cache.find(fast_key);
                if (fast_it != fast_cache.end())
                {
                    return fast_it->second;
                }
            }

            // Medium path: persistent cache
            CacheKey key{gpu_arch_, KernelTraits::name(), ctx.cache_key()};

            if (!options.force_retune)
            {
                if (auto cached = cache_.lookup(key))
                {
                    fast_cache[fast_key] = *cached;
                    return *cached;
                }
            }

            // Slow path: perform autotuning
            TuningConfig best_config = tune(args, options);

            // Populate caches
            cache_.insert(key, best_config);
            fast_cache[fast_key] = best_config;

            return best_config;
        }

        /**
         * @brief Execute kernel with optimal configuration
         *
         * Convenience wrapper that combines get_or_tune + launch.
         *
         * @param args Kernel arguments
         * @param ctx Context for cache key generation
         * @param stream HIP stream for kernel execution
         * @param options Tuning options
         */
        void execute(
            const Args &args,
            const Context &ctx,
            hipStream_t stream,
            const TuningOptions &options = TuningOptions::defaults())
        {
            TuningConfig config = get_or_tune(args, ctx, options);
            KernelTraits::launch(config, args, stream);
        }

        /**
         * @brief Check if configuration is cached
         *
         * @param ctx Context for cache key generation
         * @return true if cached config exists
         */
        bool has_cached_config(const Context &ctx) const
        {
            CacheKey key{gpu_arch_, KernelTraits::name(), ctx.cache_key()};
            return cache_.contains(key);
        }

        /**
         * @brief Get GPU architecture string
         *
         * @return Architecture identifier (e.g., "gfx1100")
         */
        const std::string &get_gpu_arch() const { return gpu_arch_; }

        /**
         * @brief Get cache store for inspection/manipulation
         *
         * @return Reference to cache store
         */
        const CacheStore &cache() const { return cache_; }

        /**
         * @brief Clear cache and force retuning
         */
        void clear_cache()
        {
            cache_.clear();
        }

    private:
        /**
         * @brief Perform autotuning for kernel
         *
         * Generates candidate configurations, benchmarks each,
         * and selects the fastest.
         *
         * @param args Kernel arguments for benchmarking
         * @param options Tuning options
         * @return Optimal TuningConfig
         */
        TuningConfig tune(const Args &args, const TuningOptions &options)
        {
            if (options.verbose)
            {
                printf("[AutoTuner] Tuning kernel '%s' for GPU arch '%s'...\n",
                       KernelTraits::name(), gpu_arch_.c_str());
            }

            // Create dedicated stream for benchmarking
            hipStream_t stream;
            HIP_ERRCHK(hipStreamCreate(&stream));

            // Generate candidate configurations
            std::vector<TuningConfig> candidates = KernelTraits::generate_candidates();

            // Filter invalid candidates
            std::vector<TuningConfig> valid_candidates;
            for (const auto &config : candidates)
            {
                if (KernelTraits::is_valid_config(config, args))
                {
                    valid_candidates.push_back(config);
                }
            }

            if (valid_candidates.empty())
            {
                fprintf(stderr, "[AutoTuner] Warning: No valid candidate configurations\n");
                HIP_ERRCHK(hipStreamDestroy(stream));
                return TuningConfig{}; // Return default config
            }

            // Benchmark all valid candidates
            Benchmarker<KernelTraits> benchmarker;
            std::vector<BenchmarkResult> results = benchmarker.benchmark_all(
                valid_candidates, args, stream, options);

            HIP_ERRCHK(hipStreamDestroy(stream));

            if (results.empty())
            {
                fprintf(stderr, "[AutoTuner] Warning: No valid benchmark results\n");
                return TuningConfig{};
            }

            // Select fastest configuration
            auto best = std::min_element(results.begin(), results.end());

            if (options.verbose)
            {
                printf("[AutoTuner] Selected config [%dx%d] with avg time %.4f ms (±%.4f ms)\n",
                       best->config.block_x(), best->config.block_y(),
                       best->avg_time_ms, best->stddev_ms);
            }

            return best->config;
        }

        /**
         * @brief Query GPU architecture from device
         *
         * @return Architecture string (e.g., "gfx1100")
         */
        static std::string query_gpu_arch()
        {
            hipDeviceProp_t prop;
            hipError_t err = hipGetDeviceProperties(&prop, 0);
            if (err != hipSuccess)
            {
                fprintf(stderr, "Warning: Failed to query GPU properties for autotuning\n");
                return "unknown";
            }

            // AMD GPUs use gcnArchName
            std::string arch_name = prop.gcnArchName;
            if (arch_name.empty())
            {
                arch_name = prop.name;
            }

            return arch_name;
        }

        std::string gpu_arch_;
        std::string cache_path_;
        CacheStore cache_;
    };

} // namespace imgfx::core::autotune
