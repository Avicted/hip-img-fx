/**
 * @file kernel_traits_concepts_example.h
 * @brief Complete examples demonstrating C++20 concept-based KernelTraits validation
 *
 * This file shows:
 * 1. ✅ Correct KernelTraits implementation with compile-time validation
 * 2. ❌ Common mistakes that trigger compile errors
 * 3. 🚀 Kernels that skip autotuning using autotune_needed flag
 * 4. 🔍 Integration with TuningOrchestrator
 *
 * @author Autotuning Framework Team
 * @date January 2026
 */

#pragma once

#include "../src/core/autotune/kernel_traits_concepts.h"
#include "../src/core/autotune/tuning_config.h"
#include <hip/hip_runtime.h>
#include <vector>

// ============================================================================
// EXAMPLE 1: FULLY COMPLIANT KERNEL TRAITS (BEST PRACTICE)
// ============================================================================

/**
 * @brief Example: Grayscale kernel with full compile-time validation
 *
 * This demonstrates all required traits and best practices:
 * - Stateless design (all methods static)
 * - Stable cache keys (Context::cache_key())
 * - Non-empty candidate set
 * - Valid configuration checking
 * - Compile-time validation using concepts
 */
struct CompliantGrayscaleKernelTraits
{
    // ========================================================================
    // REQUIRED TRAITS
    // ========================================================================

    /// Unique kernel identifier (never change this after deployment!)
    static constexpr const char *name() { return "grayscale_compliant_v1"; }

    /// Type-safe kernel arguments
    struct Args
    {
        const unsigned char *input;
        unsigned char *output;
        size_t num_pixels;
        int num_images;
    };

    /// Cache context for workload-specific tuning
    struct Context
    {
        size_t image_bytes;

        /// Stable cache key: same input → same key (INV-3)
        std::string cache_key() const
        {
            if (image_bytes < 1024 * 1024)
                return "small"; // < 1MB
            if (image_bytes < 10 * 1024 * 1024)
                return "medium"; // 1-10MB
            return "large";      // > 10MB
        }
    };

    /// Generate diverse candidate configurations (INV-2: non-empty)
    static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
    {
        using imgfx::core::autotune::TuningConfig;
        std::vector<TuningConfig> configs;

        // 1D configurations (memory-bound workloads)
        for (int bx : {64, 128, 256, 512, 1024})
        {
            TuningConfig cfg;
            cfg.set("block_x", bx);
            cfg.set("block_y", 1);
            configs.push_back(cfg);
        }

        // 2D configurations (spatial locality)
        for (int bx : {16, 32})
        {
            for (int by : {4, 8, 16})
            {
                if (bx * by >= 64 && bx * by <= 1024)
                {
                    TuningConfig cfg;
                    cfg.set("block_x", bx);
                    cfg.set("block_y", by);
                    configs.push_back(cfg);
                }
            }
        }

        // INV-2: Must return at least one config
        return configs; // Non-empty guaranteed
    }

    /// Validate configuration (INV-1: all configs must be valid)
    static bool is_valid_config(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args & /*args*/)
    {
        int threads = cfg.block_x() * cfg.block_y();

        // AMD wavefront alignment
        if (threads % 64 != 0)
            return false;

        // Hardware limits
        if (threads < 64 || threads > 1024)
            return false;

        return true;
    }

    /// Launch kernel with validated configuration
    static void launch(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args &args,
        hipStream_t stream)
    {
        int threads_per_block = cfg.block_x() * cfg.block_y();
        int blocks = (args.num_pixels + threads_per_block - 1) / threads_per_block;

        dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
        dim3 grid_dim(blocks, args.num_images, 1);

        // Launch kernel (implementation not shown)
        // hipLaunchKernelGGL(grayscale_kernel, grid_dim, block_dim, 0, stream, ...);
    }
};

// Compile-time validation (fails if trait implementation is incorrect)
VALIDATE_KERNEL_TRAITS(CompliantGrayscaleKernelTraits);

// ============================================================================
// EXAMPLE 2: KERNEL THAT SKIPS AUTOTUNING (SIMPLE KERNEL)
// ============================================================================

/**
 * @brief Example: Simple memory copy kernel that doesn't need autotuning
 *
 * Rationale:
 * - Trivial operation (memory copy)
 * - Deterministic optimal config (1024 threads, 1D)
 * - Runtime too short to benefit from tuning overhead
 *
 * Sets autotune_needed = false to skip benchmarking.
 */
struct SimpleMemcpyKernelTraits
{
    /// Compile-time flag: Skip autotuning for this kernel
    static constexpr bool autotune_needed = false;

    static constexpr const char *name() { return "simple_memcpy_v1"; }

    struct Args
    {
        const void *src;
        void *dst;
        size_t num_bytes;
    };

    struct Context
    {
        std::string cache_key() const { return "default"; }
    };

    /// Provide optimal default configuration
    static imgfx::core::autotune::TuningConfig default_config()
    {
        imgfx::core::autotune::TuningConfig cfg;
        cfg.set("block_x", 1024); // Maximum throughput for memory ops
        cfg.set("block_y", 1);
        return cfg;
    }

    /// Still required, but only validates default_config()
    static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
    {
        return {default_config()}; // Return only default (INV-2: non-empty)
    }

    static bool is_valid_config(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args & /*args*/)
    {
        int threads = cfg.block_x() * cfg.block_y();
        return threads == 1024; // Only default config is valid
    }

    static void launch(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args &args,
        hipStream_t stream)
    {
        int threads_per_block = cfg.block_x();
        int blocks = (args.num_bytes + threads_per_block - 1) / threads_per_block;

        dim3 block_dim(threads_per_block, 1, 1);
        dim3 grid_dim(blocks, 1, 1);

        // Launch memcpy kernel (implementation not shown)
    }
};

VALIDATE_KERNEL_TRAITS(SimpleMemcpyKernelTraits);

// ============================================================================
// EXAMPLE 3: KERNEL WITH RUNTIME HEURISTIC (CONDITIONAL AUTOTUNING)
// ============================================================================

/**
 * @brief Example: Blur kernel that autotunes only for large images
 *
 * This kernel uses runtime heuristics:
 * - Small images (< 64KB): Use default config, skip tuning
 * - Large images: Perform full autotuning
 *
 * The autotune_needed flag is not set, so the framework will apply
 * runtime heuristics in should_skip_autotuning().
 */
struct AdaptiveBlurKernelTraits
{
    // No autotune_needed flag → runtime heuristics apply

    static constexpr const char *name() { return "adaptive_blur_v1"; }

    struct Args
    {
        const unsigned char *input;
        unsigned char *output;
        int width;
        int height;
        int radius; // Blur radius
    };

    struct Context
    {
        size_t image_bytes;

        std::string cache_key() const
        {
            // Differentiate by image size and blur complexity
            if (image_bytes < 512 * 512)
                return "tiny";
            if (image_bytes < 1920 * 1080)
                return "hd";
            return "4k";
        }
    };

    /// Provide reasonable default for small workloads
    static imgfx::core::autotune::TuningConfig default_config()
    {
        imgfx::core::autotune::TuningConfig cfg;
        cfg.set("block_x", 16);
        cfg.set("block_y", 16); // 2D layout for spatial locality
        return cfg;
    }

    static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
    {
        using imgfx::core::autotune::TuningConfig;
        std::vector<TuningConfig> configs;

        // 2D block configurations for image processing
        for (int bx : {8, 16, 32})
        {
            for (int by : {8, 16, 32})
            {
                if (bx * by >= 64 && bx * by <= 1024)
                {
                    TuningConfig cfg;
                    cfg.set("block_x", bx);
                    cfg.set("block_y", by);
                    configs.push_back(cfg);
                }
            }
        }

        return configs;
    }

    static bool is_valid_config(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args & /*args*/)
    {
        int threads = cfg.block_x() * cfg.block_y();
        return threads >= 64 && threads <= 1024 && threads % 64 == 0;
    }

    static void launch(
        const imgfx::core::autotune::TuningConfig &cfg,
        const Args &args,
        hipStream_t stream)
    {
        dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
        dim3 grid_dim(
            (args.width + cfg.block_x() - 1) / cfg.block_x(),
            (args.height + cfg.block_y() - 1) / cfg.block_y(),
            1);

        // Launch blur kernel (implementation not shown)
    }
};

VALIDATE_KERNEL_TRAITS(AdaptiveBlurKernelTraits);

// ============================================================================
// EXAMPLE 4: COMMON MISTAKES (COMPILE ERRORS)
// ============================================================================

#if 0 // Disabled - these are intentionally broken examples

/**
 * ❌ MISTAKE 1: Non-static member (violates StatelessKernelTraits)
 */
struct BrokenStatefulTraits {
    int mutable_state = 0;  // ❌ ERROR: Traits must be stateless
    
    static constexpr const char* name() { return "broken_stateful"; }
    // ... rest of implementation
};
// VALIDATE_KERNEL_TRAITS(BrokenStatefulTraits);  // ❌ Compile error

/**
 * ❌ MISTAKE 2: Missing cache_key() method
 */
struct BrokenCacheKeyTraits {
    static constexpr const char* name() { return "broken_cache"; }
    
    struct Context {
        size_t data;
        // ❌ ERROR: Missing cache_key() method
    };
    // ... rest of implementation
};
// VALIDATE_KERNEL_TRAITS(BrokenCacheKeyTraits);  // ❌ Compile error

/**
 * ❌ MISTAKE 3: Empty candidate list (violates INV-2)
 */
struct BrokenEmptyCandidatesTraits {
    static constexpr const char* name() { return "broken_empty"; }
    
    // This compiles but fails at runtime
    static std::vector<TuningConfig> generate_candidates() {
        return {};  // ❌ RUNTIME ERROR: INV-2 violation
    }
    // ... rest of implementation
};

/**
 * ❌ MISTAKE 4: Invalid cached config (violates INV-1)
 */
struct BrokenValidationTraits {
    static std::vector<TuningConfig> generate_candidates() {
        TuningConfig cfg;
        cfg.set("block_x", 999);  // Invalid config
        return {cfg};
    }
    
    static bool is_valid_config(const TuningConfig& cfg, const Args&) {
        return cfg.block_x() % 64 == 0;  // 999 % 64 != 0
    }
    // ❌ RUNTIME ERROR: All candidates fail validation
};

#endif // End of broken examples

// ============================================================================
// USAGE IN TUNING ORCHESTRATOR
// ============================================================================

namespace usage_example
{
    using namespace imgfx::core::autotune;

    /**
     * @brief Example: Using compile-time validation in TuningOrchestrator
     *
     * This shows how the orchestrator integrates concept checks.
     */
    template <typename KernelTraits>
    class SafeTuningOrchestrator
    {
        // Compile-time validation: KernelTraits must satisfy all requirements
        static_assert(concepts::ValidKernelTraits<KernelTraits>,
                      "KernelTraits does not satisfy framework requirements");

        // Additional specific checks
        static_assert(concepts::StableCacheKey<typename KernelTraits::Context>,
                      "Context::cache_key() must be const and return std::string");

        static_assert(concepts::has_candidate_generator<KernelTraits>(),
                      "KernelTraits must implement generate_candidates()");

    public:
        using Args = typename KernelTraits::Args;
        using Context = typename KernelTraits::Context;

        /**
         * @brief Get or tune with compile-time checks and candidate pruning
         */
        TuningConfig get_or_tune(
            const Args &args,
            const Context &ctx,
            size_t workload_size_bytes = 0)
        {
            // Check 1: Should we skip autotuning?
            if (SHOULD_SKIP_AUTOTUNE(KernelTraits, workload_size_bytes))
            {
                // Return default config without benchmarking
                return concepts::get_default_config<KernelTraits, TuningConfig>();
            }

            // Check 2: Generate and validate candidates
            auto candidates = KernelTraits::generate_candidates();

            // Runtime validation (debug builds only)
            ASSERT_NON_EMPTY_CANDIDATES(candidates, KernelTraits);

            // Check 3: Count valid candidates
            size_t valid_count = concepts::count_valid_candidates<KernelTraits>(candidates, args);

            if (valid_count == 0)
            {
                fprintf(stderr, "[AutoTune ERROR] No valid candidates for kernel '%s'\n",
                        KernelTraits::name());
                // Fallback to default config
                return concepts::get_default_config<KernelTraits, TuningConfig>();
            }

            // Proceed with normal autotuning...
            // (benchmark candidates, select best, cache result)

            return TuningConfig{}; // Placeholder
        }
    };

    // Instantiate with compliant traits (compiles successfully)
    using GrayscaleTuner = SafeTuningOrchestrator<CompliantGrayscaleKernelTraits>;

    // Instantiate with skip-tuning traits (compiles, skips benchmarking)
    using MemcpyTuner = SafeTuningOrchestrator<SimpleMemcpyKernelTraits>;

    // Instantiate with adaptive traits (compiles, uses heuristics)
    using BlurTuner = SafeTuningOrchestrator<AdaptiveBlurKernelTraits>;

} // namespace usage_example

// ============================================================================
// MAINTAINER GUIDELINES
// ============================================================================

/**
 * @section Guidelines for Kernel Authors
 *
 * ✅ DO:
 * - Use VALIDATE_KERNEL_TRAITS() macro after trait definition
 * - Ensure generate_candidates() returns at least one config
 * - Make cache_key() deterministic (same input → same output)
 * - Set autotune_needed = false for trivial kernels
 * - Provide default_config() when skipping autotuning
 * - Test with static_assert before deploying
 *
 * ❌ DON'T:
 * - Add mutable state to traits (breaks StatelessKernelTraits)
 * - Return empty candidate list (violates INV-2)
 * - Change kernel name() after deployment (breaks cache)
 * - Make cache_key() non-deterministic (breaks INV-3)
 * - Generate invalid candidates (breaks INV-1)
 *
 * @section Performance Impact
 *
 * Compile-time checks: Zero runtime overhead
 * Runtime validation: Only in debug builds (NDEBUG guards)
 * Candidate pruning: Saves ~10-50ms per skipped autotuning session
 *
 * @section Backward Compatibility
 *
 * Existing kernels without concepts still work (soft migration).
 * New kernels MUST use concepts for safety guarantees.
 * Orchestrator validates both old and new trait styles.
 */
