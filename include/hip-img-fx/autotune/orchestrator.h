// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

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
#include <cstdio>
#include <cstdlib>

#include "tuning_config.h"
#include "cache_store.h"
#include "benchmarker.h"
#include "types.h"
#include "embedded_cache.h"
#include "kernel_traits_concepts.h"
#include "hip_event.h"

namespace imgfx::core::autotune
{
    /**
     * @brief Main autotuning orchestrator with compile-time safety enforcement
     *
     * Coordinates the autotuning process using kernel-specific traits.
     * Template-based design ensures zero runtime overhead and type safety.
     *
     * NEW (January 2026): Compile-time validation using C++20 concepts
     * - Enforces stateless KernelTraits (no mutable state)
     * - Validates stable cache keys at compile time
     * - Checks for non-empty candidate sets
     * - Supports candidate pruning (skip autotuning for simple kernels)
     *
     * @tparam KernelTraits Traits struct defining kernel characteristics
     *
     * KernelTraits Requirements (enforced at compile time):
     * - static constexpr const char* name()
     * - struct Args { ... } - Kernel launch arguments
     * - struct Context { std::string cache_key() const; } - Cache context
     * - static std::vector<TuningConfig> generate_candidates()
     * - static bool is_valid_config(const TuningConfig&, const Args&)
     * - static void launch(const TuningConfig&, const Args&, hipStream_t)
     *
     * Optional Requirements:
     * - static constexpr bool autotune_needed - Set to false to skip autotuning
     * - static TuningConfig default_config() - Provide default when skipping
     */
    template <typename KernelTraits>
    class TuningOrchestrator
    {
        // ====================================================================
        // COMPILE-TIME VALIDATION (C++20 Concepts)
        // ====================================================================

        // INV-0: Traits must be stateless (no mutable state, all methods static)
        static_assert(concepts::StatelessKernelTraits<KernelTraits>,
                      "KernelTraits must be stateless - remove all non-static data members");

        // INV-1: Must have unique kernel name for caching
        static_assert(concepts::HasKernelName<KernelTraits>,
                      "KernelTraits must define static name() method returning const char*");

        // INV-2: Must define Args and Context types
        static_assert(concepts::HasArgsType<KernelTraits>,
                      "KernelTraits must define nested Args type");
        static_assert(concepts::HasContextType<KernelTraits>,
                      "KernelTraits must define nested Context type");

        // INV-3: Context must have stable cache_key() method
        static_assert(concepts::StableCacheKey<typename KernelTraits::Context>,
                      "Context type must have cache_key() const method returning std::string");

        // INV-4: Must have generate_candidates() method
        static_assert(concepts::NonEmptyCandidates<KernelTraits, TuningConfig>,
                      "KernelTraits must define generate_candidates() returning vector<TuningConfig>");

        // INV-5: Must have is_valid_config() method
        static_assert(concepts::ValidConfigurations<KernelTraits, TuningConfig, typename KernelTraits::Args>,
                      "KernelTraits must define is_valid_config(config, args) returning bool");

        // INV-6: Must have launch() method
        static_assert(concepts::HasLaunchMethod<KernelTraits, TuningConfig, typename KernelTraits::Args>,
                      "KernelTraits must define launch(config, args, stream) method");

    public:
        using Args = typename KernelTraits::Args;
        using Context = typename KernelTraits::Context;

        /**
         * @brief Construct orchestrator and load cache
         *
         * @param cache_path Path to persistent cache file
         * @param enable_save Whether to save cache on destruction (default: true, or controlled by HIP_IMG_FX_NO_CACHE_SAVE env var)
         */
        explicit TuningOrchestrator(const std::string &cache_path = ".autotune_cache.json", bool enable_save = true)
            : cache_path_(cache_path)
        {
            // Check environment variable to disable cache saving (useful for benchmarks)
            const char *no_save_env = std::getenv("HIP_IMG_FX_NO_CACHE_SAVE");
            enable_save_ = enable_save && (no_save_env == nullptr || std::string(no_save_env) != "1");

            gpu_arch_ = query_gpu_arch();

            // Load user cache first (highest priority)
            bool user_cache_loaded = cache_.load(cache_path_);

            // If user cache missing or empty, load embedded defaults as fallback
            if (!user_cache_loaded && EMBEDDED_DEFAULT_CACHE_V2 != nullptr)
            {
                cache_.load_from_string(EMBEDDED_DEFAULT_CACHE_V2);
            }
        }

        /**
         * @brief Destructor - saves cache to disk (if enabled and modified)
         */
        ~TuningOrchestrator()
        {
            if (enable_save_ && cache_.is_modified())
            {
                cache_.save(cache_path_);
            }
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
            // Validate options
            AUTOTUNE_ASSERT(options.validate(), "Invalid TuningOptions provided");

            // ================================================================
            // CANDIDATE PRUNING: Skip autotuning for simple kernels
            // ================================================================

            // Check 1: Explicit autotune_needed flag
            if constexpr (concepts::HasAutotuneFlag<KernelTraits>)
            {
                if (!KernelTraits::autotune_needed && !options.force_retune)
                {
                    // Kernel explicitly opts out of autotuning
                    return concepts::get_default_config<KernelTraits, TuningConfig>();
                }
            }

            // Check 2: Runtime heuristic for workload size
            // (Only if no explicit flag is set)
            if constexpr (!concepts::HasAutotuneFlag<KernelTraits>)
            {
                // Estimate workload size if possible
                size_t workload_bytes = 0;
                if constexpr (requires { args.num_images; args.max_image_bytes; })
                {
                    workload_bytes = args.num_images * args.max_image_bytes;
                }

                if (concepts::should_skip_autotuning<KernelTraits>(
                        concepts::PruningHeuristics{}, workload_bytes))
                {
                    // Workload too small to benefit from tuning
                    if (options.verbose)
                    {
                        printf("[AutoTune] Skipping tuning for small workload (%zu bytes)\n",
                               workload_bytes);
                    }
                    return concepts::get_default_config<KernelTraits, TuningConfig>();
                }
            }

            // ================================================================
            // NORMAL PATH: Check caches and perform autotuning if needed
            // ================================================================

            // Fast path: thread-local cache (~5ns overhead)
            thread_local static std::unordered_map<std::string, TuningConfig> fast_cache;
            std::string fast_key = std::string(KernelTraits::name()) + ":" + ctx.cache_key();

            if (!options.force_retune)
            {
                auto fast_it = fast_cache.find(fast_key);
                if (fast_it != fast_cache.end())
                {
                    // Validate cached config is still valid
                    if (KernelTraits::is_valid_config(fast_it->second, args))
                    {
                        return fast_it->second;
                    }
                    else
                    {
                        // Cached config became invalid - remove and retune
                        if (options.verbose)
                        {
                            fprintf(stderr, "[AutoTune] Warning: Thread-local cached config invalid, retuning...\n");
                        }
                        fast_cache.erase(fast_it);
                    }
                }
            }

            // Medium path: persistent cache
            CacheKey key{gpu_arch_, KernelTraits::name(), ctx.cache_key()};

            if (!options.force_retune)
            {
                if (auto cached = cache_.lookup(key))
                {
                    // Validate cached config before use (INV-1: cached configs must be valid)
                    if (KernelTraits::is_valid_config(*cached, args))
                    {
                        fast_cache[fast_key] = *cached;
                        return *cached;
                    }
                    else
                    {
                        // Cache poisoned - invalidate and retune
                        if (options.verbose)
                        {
                            fprintf(stderr, "[AutoTune] Warning: Cached config [%dx%d] invalid, retuning...\n",
                                    cached->block_x(), cached->block_y());
                        }
                        cache_.remove_if([&key](const CacheEntry &e)
                                         { return e.key == key; });
                    }
                }
            }

            // Slow path: perform autotuning
            float best_time_ms = 0.0f;
            TuningConfig best_config = tune(args, options, best_time_ms);

            // Populate caches
            cache_.insert(key, best_config, best_time_ms);
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
         * @param out_best_time_ms Output parameter for best configuration's time
         * @return Optimal TuningConfig
         */
        TuningConfig tune(const Args &args, const TuningOptions &options, float &out_best_time_ms)
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

            // ================================================================
            // RUNTIME VALIDATION: Non-empty candidate set (INV-2)
            // ================================================================
            ASSERT_NON_EMPTY_CANDIDATES(candidates, KernelTraits);

            // Additional runtime check with detailed error message
            if (!concepts::validate_candidates(candidates))
            {
                fprintf(stderr, "[AutoTune ERROR] generate_candidates() returned empty list\n");
                fprintf(stderr, "  Kernel: %s\n", KernelTraits::name());
                fprintf(stderr, "  This is a bug in the kernel traits implementation.\n");
                fprintf(stderr, "  Invariant violated: INV-2 (Non-Empty Candidate Set)\n");
                HIP_ERRCHK(hipStreamDestroy(stream));
                AUTOTUNE_ASSERT(false, "Empty candidate list - invariant violation (INV-2)");
                return TuningConfig{}; // Unreachable in debug, fallback in release
            }

            if (options.verbose)
            {
                printf("[AutoTuner] Generated %zu candidate configurations\n", candidates.size());
            }

            // ================================================================
            // RUNTIME VALIDATION: Valid configurations (INV-1)
            // ================================================================

            // Filter invalid candidates
            std::vector<TuningConfig> valid_candidates;
            for (const auto &config : candidates)
            {
                if (KernelTraits::is_valid_config(config, args))
                {
                    valid_candidates.push_back(config);
                }
            }

            // Count and report validation results
            size_t valid_count = concepts::count_valid_candidates<KernelTraits>(candidates, args);

            if (options.verbose && valid_count < candidates.size())
            {
                printf("[AutoTuner] Filtered to %zu valid candidates (%zu invalid)\n",
                       valid_count, candidates.size() - valid_count);
            }

            if (valid_candidates.empty())
            {
                fprintf(stderr, "[AutoTune ERROR] All %zu candidates failed validation\n",
                        candidates.size());
                fprintf(stderr, "  Kernel: %s\n", KernelTraits::name());
                fprintf(stderr, "  Possible causes:\n");
                fprintf(stderr, "    1. is_valid_config() too strict\n");
                fprintf(stderr, "    2. generate_candidates() produces invalid configs\n");
                fprintf(stderr, "    3. Hardware constraints not met\n");
                HIP_ERRCHK(hipStreamDestroy(stream));
                AUTOTUNE_ASSERT(false, "No valid candidates - check trait implementation");
                return TuningConfig{}; // Unreachable in debug, fallback in release
            }

            // Benchmark all valid candidates
            Benchmarker<KernelTraits> benchmarker;
            std::vector<BenchmarkResult> results = benchmarker.benchmark_all(
                valid_candidates, args, stream, options);

            HIP_ERRCHK(hipStreamDestroy(stream));

            if (results.empty())
            {
                fprintf(stderr, "[AutoTune ERROR] All benchmark attempts failed\n");
                fprintf(stderr, "  Kernel: %s\n", KernelTraits::name());
                fprintf(stderr, "  Tested %zu candidates, all failed\n", valid_candidates.size());
                fprintf(stderr, "  Check HIP runtime errors above.\n");
                AUTOTUNE_ASSERT(false, "All benchmarks failed - runtime error likely");
                return TuningConfig{}; // Unreachable in debug, fallback in release
            }

            // Select fastest configuration
            auto best = std::min_element(results.begin(), results.end());

            // INV-1: Best config must pass validation
            AUTOTUNE_ASSERT(KernelTraits::is_valid_config(best->config, args),
                            "Best config failed validation - invariant violation (INV-1)");

            if (options.verbose)
            {
                printf("[AutoTuner] Selected config [%dx%d] with avg time %.4f ms (±%.4f ms)\n",
                       best->config.block_x(), best->config.block_y(),
                       best->avg_time_ms, best->stddev_ms);
            }

            out_best_time_ms = best->avg_time_ms;
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
        bool enable_save_;
        CacheStore cache_;
    };

} // namespace imgfx::core::autotune
