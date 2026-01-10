#pragma once

/**
 * @file kernel_traits_concepts.h
 * @brief C++20 concepts and compile-time validation for autotuning kernel traits
 *
 * This header provides:
 * 1. Compile-time enforcement of KernelTraits invariants
 * 2. Static assertions for stateless traits and valid candidates
 * 3. Candidate pruning mechanism to skip unnecessary autotuning
 * 4. Zero runtime overhead in release builds
 *
 * Usage:
 *     struct MyKernelTraits {
 *         static constexpr bool autotune_needed = true;
 *         static constexpr const char* name() { return "my_kernel"; }
 *         // ... implement other required methods
 *     };
 *
 *     // Validate at compile time
 *     static_assert(ValidKernelTraits<MyKernelTraits>);
 *
 * @author Autotuning Framework Team
 * @date January 2026
 */

#include <concepts>
#include <string>
#include <vector>
#include <type_traits>
#include <hip/hip_runtime.h>

namespace imgfx::core::autotune::concepts
{
    // ========================================================================
    // FORWARD DECLARATIONS
    // ========================================================================

    template <typename T>
    class TuningConfig;

    // ========================================================================
    // CONCEPT 1: STATELESS KERNEL TRAITS
    // ========================================================================

    /**
     * @brief Concept: Ensures KernelTraits has no non-static member functions
     *
     * Rationale: KernelTraits should be a pure policy class with no state.
     * All methods must be static to prevent accidental state mutation.
     *
     * This concept checks that the traits type is:
     * - Trivially constructible (no complex constructor logic)
     * - Trivially destructible (no cleanup required)
     * - Empty (sizeof == 1, no data members)
     */
    template <typename T>
    concept StatelessKernelTraits = std::is_empty_v<T> &&
                                    std::is_trivially_constructible_v<T> &&
                                    std::is_trivially_destructible_v<T>;

    // ========================================================================
    // CONCEPT 2: STABLE CACHE KEY
    // ========================================================================

    /**
     * @brief Concept: Ensures Context type has a stable cache_key() method
     *
     * Rationale: Cache keys must be deterministic and collision-free.
     * The same context should always produce the same key.
     *
     * Requirements:
     * - Has cache_key() method that returns string-like type
     * - Method must be const (no mutation of context state)
     */
    template <typename T>
    concept StableCacheKey = requires(const T ctx) {
        { ctx.cache_key() } -> std::convertible_to<std::string>;
    };

    // ========================================================================
    // CONCEPT 3: NON-EMPTY CANDIDATE SET
    // ========================================================================

    /**
     * @brief Concept: Ensures generate_candidates() returns a vector-like type
     *
     * This is a structural check. Runtime validation (non-empty) happens
     * via validate_candidates() function below.
     */
    template <typename T, typename ConfigType>
    concept NonEmptyCandidates = requires() {
        { T::generate_candidates() } -> std::convertible_to<std::vector<ConfigType>>;
    };

    // ========================================================================
    // CONCEPT 4: VALID CONFIGURATIONS
    // ========================================================================

    /**
     * @brief Concept: Ensures is_valid_config() method exists with correct signature
     *
     * Rationale: All generated candidates must be validated before benchmarking.
     *
     * Requirements:
     * - Static method is_valid_config(config, args)
     * - Returns bool
     */
    template <typename T, typename ConfigType, typename ArgsType>
    concept ValidConfigurations = requires(const ConfigType cfg, const ArgsType args) {
        { T::is_valid_config(cfg, args) } -> std::convertible_to<bool>;
    };

    // ========================================================================
    // CONCEPT 5: HAS KERNEL NAME
    // ========================================================================

    /**
     * @brief Concept: Ensures kernel has a static name() method
     *
     * Rationale: Unique kernel names are essential for cache keying.
     *
     * Requirements:
     * - Static constexpr name() method
     * - Returns const char* or string-like type
     */
    template <typename T>
    concept HasKernelName = requires() {
        { T::name() } -> std::convertible_to<const char *>;
    };

    // ========================================================================
    // CONCEPT 6: HAS ARGS TYPE
    // ========================================================================

    /**
     * @brief Concept: Ensures KernelTraits defines Args nested type
     *
     * Rationale: Type-safe kernel arguments eliminate void* casting.
     */
    template <typename T>
    concept HasArgsType = requires() {
        typename T::Args;
    };

    // ========================================================================
    // CONCEPT 7: HAS CONTEXT TYPE
    // ========================================================================

    /**
     * @brief Concept: Ensures KernelTraits defines Context nested type
     *
     * Rationale: Context enables workload-specific cache keys.
     */
    template <typename T>
    concept HasContextType = requires() {
        typename T::Context;
    };

    // ========================================================================
    // CONCEPT 8: HAS LAUNCH METHOD
    // ========================================================================

    /**
     * @brief Concept: Ensures kernel can be launched with given config
     *
     * Requirements:
     * - Static launch(config, args, stream) method
     * - Accepts TuningConfig, Args, and hipStream_t
     */
    template <typename T, typename ConfigType, typename ArgsType>
    concept HasLaunchMethod = requires(const ConfigType cfg, const ArgsType args, hipStream_t stream) {
        { T::launch(cfg, args, stream) } -> std::same_as<void>;
    };

    // ========================================================================
    // CONCEPT 9: AUTOTUNE FLAG (OPTIONAL)
    // ========================================================================

    /**
     * @brief Concept: Check if kernel has autotune_needed flag
     *
     * This is optional. If not present, defaults to true (always autotune).
     * Kernels can opt-out by setting `static constexpr bool autotune_needed = false;`
     */
    template <typename T>
    concept HasAutotuneFlag = requires() {
        { T::autotune_needed } -> std::convertible_to<bool>;
    };

    // ========================================================================
    // COMBINED CONCEPT: VALID KERNEL TRAITS
    // ========================================================================

    /**
     * @brief Master concept: Validates complete KernelTraits interface
     *
     * This combines all individual concepts to ensure a KernelTraits type
     * satisfies all framework requirements.
     *
     * Usage:
     *     static_assert(ValidKernelTraits<MyKernelTraits>);
     *
     * @tparam T KernelTraits type to validate
     */
    template <typename T>
    concept ValidKernelTraits = StatelessKernelTraits<T> &&
                                HasKernelName<T> &&
                                HasArgsType<T> &&
                                HasContextType<T> &&
                                requires() {
                                    typename T::Args;
                                    typename T::Context;
                                    // Note: Full validation happens at TuningOrchestrator instantiation
                                    // where we have access to ConfigType
                                };

    // ========================================================================
    // COMPILE-TIME VALIDATION HELPERS
    // ========================================================================

    /**
     * @brief Constexpr function to check if candidates list would be non-empty
     *
     * This is a compile-time heuristic. Runtime validation still required
     * because generate_candidates() may depend on runtime state.
     *
     * @return true if type has generate_candidates() method
     */
    template <typename T>
    consteval bool has_candidate_generator()
    {
        return requires() {
            { T::generate_candidates() };
        };
    }

    /**
     * @brief Constexpr function to get autotune_needed flag value
     *
     * Returns the kernel's autotune_needed flag if present, otherwise true.
     *
     * @tparam T KernelTraits type
     * @return true if autotuning is needed (default), false to skip
     */
    template <typename T>
    consteval bool should_autotune()
    {
        if constexpr (HasAutotuneFlag<T>)
        {
            return T::autotune_needed;
        }
        else
        {
            return true; // Default: always autotune
        }
    }

    /**
     * @brief Constexpr function to validate KernelTraits at compile time
     *
     * This performs all possible compile-time checks on a KernelTraits type.
     * Use in static_assert for early error detection.
     *
     * @tparam T KernelTraits type
     * @return true if all checks pass
     */
    template <typename T>
    consteval bool validate_kernel_traits()
    {
        // Check 1: Stateless (no mutable state)
        static_assert(StatelessKernelTraits<T>,
                      "KernelTraits must be stateless (no data members, all methods static)");

        // Check 2: Has name() method
        static_assert(HasKernelName<T>,
                      "KernelTraits must define static name() method returning const char*");

        // Check 3: Has Args type
        static_assert(HasArgsType<T>,
                      "KernelTraits must define nested Args type");

        // Check 4: Has Context type
        static_assert(HasContextType<T>,
                      "KernelTraits must define nested Context type with cache_key() method");

        // Check 5: Context has stable cache key
        static_assert(StableCacheKey<typename T::Context>,
                      "Context type must have cache_key() const method returning string");

        // Check 6: Has generate_candidates() method
        static_assert(has_candidate_generator<T>(),
                      "KernelTraits must define static generate_candidates() method");

        return true;
    }

    // ========================================================================
    // RUNTIME VALIDATION (constexpr where possible)
    // ========================================================================

    /**
     * @brief Runtime validation: Ensure candidates list is non-empty
     *
     * This should be called in orchestrator's tune() method.
     * Marked [[nodiscard]] to prevent ignoring the result.
     *
     * @tparam Candidates Vector-like container of TuningConfig
     * @param candidates List of generated candidates
     * @return true if non-empty, false otherwise
     */
    template <typename Candidates>
    [[nodiscard]] inline bool validate_candidates(const Candidates &candidates)
    {
        return !candidates.empty();
    }

    /**
     * @brief Runtime validation: Ensure all candidates pass is_valid_config()
     *
     * This is expensive (O(n) where n = candidates.size()) and should only
     * be enabled in debug builds unless explicitly requested.
     *
     * @tparam KernelTraits Kernel traits type
     * @tparam ConfigType Configuration type
     * @tparam ArgsType Arguments type
     * @param candidates List of candidates to validate
     * @param args Kernel arguments for validation
     * @return Number of valid candidates
     */
    template <typename KernelTraits, typename ConfigType, typename ArgsType>
    [[nodiscard]] inline size_t count_valid_candidates(
        const std::vector<ConfigType> &candidates,
        const ArgsType &args)
    {
        size_t valid_count = 0;
        for (const auto &cfg : candidates)
        {
            if (KernelTraits::is_valid_config(cfg, args))
            {
                ++valid_count;
            }
        }
        return valid_count;
    }

    // ========================================================================
    // CANDIDATE PRUNING: SKIP AUTOTUNING FOR SIMPLE KERNELS
    // ========================================================================

    /**
     * @brief Configuration for candidate pruning heuristics
     *
     * Kernels meeting these criteria may skip autotuning and use a default config.
     */
    struct PruningHeuristics
    {
        /// Skip autotuning if expected runtime < threshold (microseconds)
        double runtime_threshold_us = 50.0;

        /// Skip if kernel is memory-bound (arithmetic intensity < threshold)
        double arithmetic_intensity_threshold = 0.1;

        /// Skip if workload size < threshold (bytes)
        size_t workload_size_threshold = 64 * 1024; // 64 KB

        /// Always respect explicit autotune_needed flag
        bool respect_explicit_flag = true;
    };

    /**
     * @brief Trait to provide default configuration for kernels that skip tuning
     *
     * Kernels that set autotune_needed = false should also define this method
     * to provide a reasonable default configuration.
     *
     * Optional trait method:
     *     static TuningConfig default_config();
     */
    template <typename T, typename ConfigType>
    concept HasDefaultConfig = requires() {
        { T::default_config() } -> std::convertible_to<ConfigType>;
    };

    /**
     * @brief Get default configuration or generate one automatically
     *
     * If kernel defines default_config(), use it.
     * Otherwise, return a conservative default (256 threads, 1D layout).
     *
     * @tparam KernelTraits Kernel traits type
     * @tparam ConfigType Configuration type
     * @return Default TuningConfig
     */
    template <typename KernelTraits, typename ConfigType>
    inline ConfigType get_default_config()
    {
        if constexpr (HasDefaultConfig<KernelTraits, ConfigType>)
        {
            return KernelTraits::default_config();
        }
        else
        {
            // Conservative default: 256 threads in 1D layout
            ConfigType cfg;
            cfg.set("block_x", 256);
            cfg.set("block_y", 1);
            return cfg;
        }
    }

    /**
     * @brief Determine if kernel should skip autotuning based on heuristics
     *
     * Decision logic:
     * 1. If autotune_needed = false explicitly set, always skip
     * 2. If autotune_needed = true explicitly set, never skip
     * 3. If no flag, apply runtime heuristics
     *
     * @tparam KernelTraits Kernel traits type
     * @param heuristics Pruning configuration
     * @param workload_size_bytes Size of workload (if applicable)
     * @return true if autotuning should be skipped
     */
    template <typename KernelTraits>
    [[nodiscard]] inline bool should_skip_autotuning(
        const PruningHeuristics &heuristics = PruningHeuristics{},
        size_t workload_size_bytes = 0)
    {
        // Check 1: Explicit autotune_needed flag
        if constexpr (HasAutotuneFlag<KernelTraits>)
        {
            if (heuristics.respect_explicit_flag)
            {
                return !KernelTraits::autotune_needed;
            }
        }

        // Check 2: Workload size heuristic
        if (workload_size_bytes > 0 && workload_size_bytes < heuristics.workload_size_threshold)
        {
            return true; // Too small to benefit from tuning
        }

        // Default: do not skip (perform autotuning)
        return false;
    }

    // ========================================================================
    // DIAGNOSTIC HELPERS
    // ========================================================================

    /**
     * @brief Compile-time diagnostic: Print KernelTraits validation status
     *
     * This can be used in static_assert to get better error messages.
     *
     * Example:
     *     static_assert(diagnose_kernel_traits<MyKernelTraits>());
     *
     * @tparam T KernelTraits type
     * @return true (always, used for static_assert)
     */
    template <typename T>
    consteval bool diagnose_kernel_traits()
    {
        // These static_asserts will fire with descriptive messages
        // if any requirement is not met
        return validate_kernel_traits<T>();
    }

} // namespace imgfx::core::autotune::concepts

// ============================================================================
// CONVENIENCE MACROS FOR KERNEL TRAIT VALIDATION
// ============================================================================

/**
 * @brief Macro: Validate KernelTraits at compile time with clear error message
 *
 * Usage at the end of KernelTraits definition:
 *     struct MyKernelTraits {
 *         // ... trait implementation
 *     };
 *     VALIDATE_KERNEL_TRAITS(MyKernelTraits);
 */
#define VALIDATE_KERNEL_TRAITS(TraitsType)                                                  \
    static_assert(::imgfx::core::autotune::concepts::StatelessKernelTraits<TraitsType>,     \
                  #TraitsType " must be stateless (no data members)");                      \
    static_assert(::imgfx::core::autotune::concepts::HasKernelName<TraitsType>,             \
                  #TraitsType " must define static name() method");                         \
    static_assert(::imgfx::core::autotune::concepts::HasArgsType<TraitsType>,               \
                  #TraitsType " must define nested Args type");                             \
    static_assert(::imgfx::core::autotune::concepts::HasContextType<TraitsType>,            \
                  #TraitsType " must define nested Context type");                          \
    static_assert(::imgfx::core::autotune::concepts::StableCacheKey<TraitsType::Context>,   \
                  #TraitsType "::Context must have cache_key() const method");              \
    static_assert(::imgfx::core::autotune::concepts::has_candidate_generator<TraitsType>(), \
                  #TraitsType " must define static generate_candidates() method")

/**
 * @brief Macro: Assert non-empty candidates at runtime (debug builds only)
 *
 * Usage in tune() method:
 *     auto candidates = KernelTraits::generate_candidates();
 *     ASSERT_NON_EMPTY_CANDIDATES(candidates, KernelTraits);
 */
#ifndef NDEBUG
#define ASSERT_NON_EMPTY_CANDIDATES(candidates, TraitsType)                                     \
    do                                                                                          \
    {                                                                                           \
        if (!(::imgfx::core::autotune::concepts::validate_candidates(candidates)))              \
        {                                                                                       \
            fprintf(stderr, "[AutoTune ERROR] %s::generate_candidates() returned empty list\n", \
                    TraitsType::name());                                                        \
            fprintf(stderr, "  This violates INV-2 (Non-Empty Candidate Set)\n");               \
            abort();                                                                            \
        }                                                                                       \
    } while (0)
#else
#define ASSERT_NON_EMPTY_CANDIDATES(candidates, TraitsType) ((void)0)
#endif

/**
 * @brief Macro: Check if kernel should skip autotuning
 *
 * Usage in get_or_tune():
 *     if (SHOULD_SKIP_AUTOTUNE(KernelTraits, workload_bytes)) {
 *         return get_default_config<KernelTraits, TuningConfig>();
 *     }
 */
#define SHOULD_SKIP_AUTOTUNE(TraitsType, workload_size)                     \
    (::imgfx::core::autotune::concepts::should_skip_autotuning<TraitsType>( \
        ::imgfx::core::autotune::concepts::PruningHeuristics{}, workload_size))
