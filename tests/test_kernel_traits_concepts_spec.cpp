/**
 * @file test_kernel_traits_concepts_spec.cpp
 * @brief Specification tests for kernel traits concepts (kernel_traits_concepts.h)
 *
 * Following SKILL.md guidelines:
 * - Encode contracts not implementation
 * - Negative tests required for every positive case
 * - Break-the-code validation
 *
 * Coverage targets:
 * - Compile-time concepts (StatelessKernelTraits, StableCacheKey, etc.)
 * - Runtime validation functions (validate_candidates, count_valid_candidates)
 * - Candidate pruning (should_skip_autotuning, get_default_config)
 * - Diagnostic helpers
 */

#include <gtest/gtest.h>
#include "hip-img-fx/autotune/tuning_config.h"
#include "hip-img-fx/autotune/types.h"
#include "hip-img-fx/autotune/kernel_traits_concepts.h"
#include <hip/hip_runtime.h>

using namespace imgfx::core::autotune;

// ============================================================================
// MOCK KERNEL TRAITS FOR TESTING CONCEPTS
// ============================================================================

/**
 * @brief Valid kernel traits that satisfies all concepts
 */
struct ValidKernelTraits
{
    static constexpr const char *name() { return "valid_kernel"; }

    struct Args
    {
        int width{0};
        int height{0};
    };

    struct Context
    {
        int size{0};
        std::string cache_key() const
        {
            return std::to_string(size);
        }
    };

    static std::vector<TuningConfig> generate_candidates()
    {
        std::vector<TuningConfig> configs;
        TuningConfig cfg;
        cfg.set("block_x", 256);
        cfg.set("block_y", 1);
        configs.push_back(cfg);
        return configs;
    }

    static bool is_valid_config(const TuningConfig &cfg, const Args &)
    {
        int block_x = cfg.get_or<int>("block_x", 0);
        return block_x > 0 && block_x <= 1024;
    }

    static void launch(const TuningConfig &, const Args &, hipStream_t)
    {
    }
};

/**
 * @brief Kernel traits with autotune_needed = false
 */
struct NoAutotuneTraits
{
    static constexpr bool autotune_needed = false;
    static constexpr const char *name() { return "no_autotune"; }

    struct Args
    {
        int dummy{0};
    };

    struct Context
    {
        std::string cache_key() const { return "fixed"; }
    };

    static TuningConfig default_config()
    {
        TuningConfig cfg;
        cfg.set("block_x", 256);
        cfg.set("block_y", 1);
        return cfg;
    }

    static std::vector<TuningConfig> generate_candidates()
    {
        return {default_config()};
    }

    static bool is_valid_config(const TuningConfig &, const Args &)
    {
        return true;
    }

    static void launch(const TuningConfig &, const Args &, hipStream_t)
    {
    }
};

/**
 * @brief Kernel traits with default_config() method
 */
struct HasDefaultConfigTraits
{
    static constexpr const char *name() { return "has_default"; }

    struct Args
    {
        int dummy{0};
    };

    struct Context
    {
        std::string cache_key() const { return "test"; }
    };

    static TuningConfig default_config()
    {
        TuningConfig cfg;
        cfg.set("block_x", 512);
        cfg.set("block_y", 2);
        return cfg;
    }

    static std::vector<TuningConfig> generate_candidates()
    {
        return {default_config()};
    }

    static bool is_valid_config(const TuningConfig &, const Args &)
    {
        return true;
    }

    static void launch(const TuningConfig &, const Args &, hipStream_t)
    {
    }
};

/**
 * @brief Kernel traits without default_config() method
 */
struct NoDefaultConfigTraits
{
    static constexpr const char *name() { return "no_default"; }

    struct Args
    {
        int dummy{0};
    };

    struct Context
    {
        std::string cache_key() const { return "test"; }
    };

    static std::vector<TuningConfig> generate_candidates()
    {
        std::vector<TuningConfig> configs;
        TuningConfig cfg;
        cfg.set("block_x", 128);
        configs.push_back(cfg);
        return configs;
    }

    static bool is_valid_config(const TuningConfig &, const Args &)
    {
        return true;
    }

    static void launch(const TuningConfig &, const Args &, hipStream_t)
    {
    }
};

// ============================================================================
// TEST SUITE: Compile-Time Concepts
// ============================================================================

TEST(KernelTraitsConceptsSpec, StatelessKernelTraitsAcceptsEmptyStruct)
{
    EXPECT_TRUE(concepts::StatelessKernelTraits<ValidKernelTraits>);
    EXPECT_TRUE(concepts::StatelessKernelTraits<NoAutotuneTraits>);
}

TEST(KernelTraitsConceptsSpec, HasKernelNameDetectsNameMethod)
{
    EXPECT_TRUE(concepts::HasKernelName<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasKernelName<NoAutotuneTraits>);
}

TEST(KernelTraitsConceptsSpec, HasArgsTypeDetectsNestedType)
{
    EXPECT_TRUE(concepts::HasArgsType<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasArgsType<NoAutotuneTraits>);
}

TEST(KernelTraitsConceptsSpec, HasContextTypeDetectsNestedType)
{
    EXPECT_TRUE(concepts::HasContextType<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasContextType<NoAutotuneTraits>);
}

TEST(KernelTraitsConceptsSpec, StableCacheKeyDetectsCacheKeyMethod)
{
    EXPECT_TRUE(concepts::StableCacheKey<ValidKernelTraits::Context>);
    EXPECT_TRUE(concepts::StableCacheKey<NoAutotuneTraits::Context>);
}

TEST(KernelTraitsConceptsSpec, NonEmptyCandidatesDetectsGenerateCandidatesMethod)
{
    EXPECT_TRUE((concepts::NonEmptyCandidates<ValidKernelTraits, TuningConfig>));
    EXPECT_TRUE((concepts::NonEmptyCandidates<NoAutotuneTraits, TuningConfig>));
}

TEST(KernelTraitsConceptsSpec, ValidConfigurationsDetectsIsValidConfigMethod)
{
    EXPECT_TRUE((concepts::ValidConfigurations<ValidKernelTraits, TuningConfig, ValidKernelTraits::Args>));
    EXPECT_TRUE((concepts::ValidConfigurations<NoAutotuneTraits, TuningConfig, NoAutotuneTraits::Args>));
}

TEST(KernelTraitsConceptsSpec, HasLaunchMethodDetectsLaunchMethod)
{
    EXPECT_TRUE((concepts::HasLaunchMethod<ValidKernelTraits, TuningConfig, ValidKernelTraits::Args>));
    EXPECT_TRUE((concepts::HasLaunchMethod<NoAutotuneTraits, TuningConfig, NoAutotuneTraits::Args>));
}

TEST(KernelTraitsConceptsSpec, HasAutotuneFlagDetectsOptionalFlag)
{
    EXPECT_FALSE(concepts::HasAutotuneFlag<ValidKernelTraits>); // No flag defined
    EXPECT_TRUE(concepts::HasAutotuneFlag<NoAutotuneTraits>);   // Flag defined
}

TEST(KernelTraitsConceptsSpec, HasDefaultConfigDetectsOptionalMethod)
{
    EXPECT_TRUE((concepts::HasDefaultConfig<HasDefaultConfigTraits, TuningConfig>));
    EXPECT_TRUE((concepts::HasDefaultConfig<NoAutotuneTraits, TuningConfig>));
    EXPECT_FALSE((concepts::HasDefaultConfig<NoDefaultConfigTraits, TuningConfig>));
}

// ============================================================================
// TEST SUITE: Runtime Validation Functions
// ============================================================================

TEST(KernelTraitsConceptsSpec, ValidateCandidatesReturnsTrueForNonEmpty)
{
    std::vector<TuningConfig> candidates;
    TuningConfig cfg;
    cfg.set("block_x", 256);
    candidates.push_back(cfg);

    EXPECT_TRUE(concepts::validate_candidates(candidates));
}

TEST(KernelTraitsConceptsSpec, ValidateCandidatesReturnsFalseForEmpty)
{
    std::vector<TuningConfig> empty_candidates;

    EXPECT_FALSE(concepts::validate_candidates(empty_candidates));
}

TEST(KernelTraitsConceptsSpec, CountValidCandidatesCountsCorrectly)
{
    std::vector<TuningConfig> candidates;

    // Add some valid configs
    for (int block_x : {128, 256, 512})
    {
        TuningConfig cfg;
        cfg.set("block_x", block_x);
        cfg.set("block_y", 1);
        candidates.push_back(cfg);
    }

    // Add an invalid config (block_x > 1024)
    TuningConfig invalid;
    invalid.set("block_x", 2048);
    invalid.set("block_y", 1);
    candidates.push_back(invalid);

    ValidKernelTraits::Args args{};

    size_t valid_count = concepts::count_valid_candidates<ValidKernelTraits>(candidates, args);

    EXPECT_EQ(valid_count, 3u); // 3 valid, 1 invalid
}

TEST(KernelTraitsConceptsSpec, CountValidCandidatesReturnsZeroForAllInvalid)
{
    std::vector<TuningConfig> candidates;

    // Add only invalid configs
    for (int block_x : {2048, 4096, 8192})
    {
        TuningConfig cfg;
        cfg.set("block_x", block_x); // All > 1024
        cfg.set("block_y", 1);
        candidates.push_back(cfg);
    }

    ValidKernelTraits::Args args{};

    size_t valid_count = concepts::count_valid_candidates<ValidKernelTraits>(candidates, args);

    EXPECT_EQ(valid_count, 0u);
}

TEST(KernelTraitsConceptsSpec, CountValidCandidatesHandlesEmptyList)
{
    std::vector<TuningConfig> empty_candidates;
    ValidKernelTraits::Args args{};

    size_t valid_count = concepts::count_valid_candidates<ValidKernelTraits>(empty_candidates, args);

    EXPECT_EQ(valid_count, 0u);
}

// ============================================================================
// TEST SUITE: Candidate Pruning - should_skip_autotuning()
// ============================================================================

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningReturnsTrueForExplicitFalseFlag)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.respect_explicit_flag = true;

    bool should_skip = concepts::should_skip_autotuning<NoAutotuneTraits>(heuristics);

    EXPECT_TRUE(should_skip);
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningReturnsFalseForNoFlag)
{
    concepts::PruningHeuristics heuristics{};

    bool should_skip = concepts::should_skip_autotuning<ValidKernelTraits>(heuristics);

    EXPECT_FALSE(should_skip); // Default: do not skip
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningReturnsTrueForSmallWorkload)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.workload_size_threshold = 100 * 1024; // 100 KB

    size_t small_workload = 10 * 1024; // 10 KB

    bool should_skip = concepts::should_skip_autotuning<ValidKernelTraits>(heuristics, small_workload);

    EXPECT_TRUE(should_skip);
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningReturnsFalseForLargeWorkload)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.workload_size_threshold = 100 * 1024; // 100 KB

    size_t large_workload = 1024 * 1024; // 1 MB

    bool should_skip = concepts::should_skip_autotuning<ValidKernelTraits>(heuristics, large_workload);

    EXPECT_FALSE(should_skip);
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningRespectsExplicitFlagOverHeuristics)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.respect_explicit_flag = true;
    heuristics.workload_size_threshold = 100 * 1024;

    size_t large_workload = 1024 * 1024; // 1 MB (large, but flag says no autotune)

    bool should_skip = concepts::should_skip_autotuning<NoAutotuneTraits>(heuristics, large_workload);

    EXPECT_TRUE(should_skip); // Flag takes precedence
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningWithZeroWorkloadUsesFlagOnly)
{
    concepts::PruningHeuristics heuristics{};

    bool should_skip_no_flag = concepts::should_skip_autotuning<ValidKernelTraits>(heuristics, 0);
    bool should_skip_with_flag = concepts::should_skip_autotuning<NoAutotuneTraits>(heuristics, 0);

    EXPECT_FALSE(should_skip_no_flag);
    EXPECT_TRUE(should_skip_with_flag);
}

// ============================================================================
// TEST SUITE: get_default_config()
// ============================================================================

TEST(KernelTraitsConceptsSpec, GetDefaultConfigUsesTraitsDefaultIfAvailable)
{
    TuningConfig config = concepts::get_default_config<HasDefaultConfigTraits, TuningConfig>();

    EXPECT_EQ(config.block_x(), 512);
    EXPECT_EQ(config.block_y(), 2);
}

TEST(KernelTraitsConceptsSpec, GetDefaultConfigUsesConservativeDefaultIfNoMethod)
{
    TuningConfig config = concepts::get_default_config<NoDefaultConfigTraits, TuningConfig>();

    // Should use conservative default: 256 threads, 1D layout
    EXPECT_EQ(config.block_x(), 256);
    EXPECT_EQ(config.block_y(), 1);
}

TEST(KernelTraitsConceptsSpec, GetDefaultConfigForNoAutotuneTraits)
{
    TuningConfig config = concepts::get_default_config<NoAutotuneTraits, TuningConfig>();

    EXPECT_EQ(config.block_x(), 256);
    EXPECT_EQ(config.block_y(), 1);
}

// ============================================================================
// TEST SUITE: Compile-Time Validation Helpers
// ============================================================================

TEST(KernelTraitsConceptsSpec, HasCandidateGeneratorReturnsTrueForValidTraits)
{
    constexpr bool has_generator = concepts::has_candidate_generator<ValidKernelTraits>();

    EXPECT_TRUE(has_generator);
}

TEST(KernelTraitsConceptsSpec, ShouldAutotuneReturnsTrueByDefault)
{
    constexpr bool should = concepts::should_autotune<ValidKernelTraits>();

    EXPECT_TRUE(should);
}

TEST(KernelTraitsConceptsSpec, ShouldAutotuneReturnsFalseForExplicitFlag)
{
    constexpr bool should = concepts::should_autotune<NoAutotuneTraits>();

    EXPECT_FALSE(should);
}

TEST(KernelTraitsConceptsSpec, ValidateKernelTraitsReturnsTrueForValidTraits)
{
    constexpr bool is_valid = concepts::validate_kernel_traits<ValidKernelTraits>();

    EXPECT_TRUE(is_valid);
}

// ============================================================================
// TEST SUITE: concepts::PruningHeuristics Configuration
// ============================================================================

TEST(KernelTraitsConceptsSpec, PruningHeuristicsHasReasonableDefaults)
{
    concepts::PruningHeuristics heuristics{};

    EXPECT_GT(heuristics.runtime_threshold_us, 0.0);
    EXPECT_GT(heuristics.arithmetic_intensity_threshold, 0.0);
    EXPECT_GT(heuristics.workload_size_threshold, 0u);
    EXPECT_TRUE(heuristics.respect_explicit_flag);
}

TEST(KernelTraitsConceptsSpec, PruningHeuristicsCanBeCustomized)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.runtime_threshold_us = 100.0;
    heuristics.arithmetic_intensity_threshold = 0.5;
    heuristics.workload_size_threshold = 1024 * 1024; // 1 MB
    heuristics.respect_explicit_flag = false;

    EXPECT_EQ(heuristics.runtime_threshold_us, 100.0);
    EXPECT_EQ(heuristics.arithmetic_intensity_threshold, 0.5);
    EXPECT_EQ(heuristics.workload_size_threshold, 1024u * 1024u);
    EXPECT_FALSE(heuristics.respect_explicit_flag);
}

// ============================================================================
// TEST SUITE: Edge Cases and Boundary Conditions
// ============================================================================

TEST(KernelTraitsConceptsSpec, ValidateCandidatesWithSingleCandidate)
{
    std::vector<TuningConfig> candidates;
    TuningConfig cfg;
    cfg.set("block_x", 256);
    candidates.push_back(cfg);

    EXPECT_TRUE(concepts::validate_candidates(candidates));
    EXPECT_EQ(candidates.size(), 1u);
}

TEST(KernelTraitsConceptsSpec, CountValidCandidatesWithAllValid)
{
    std::vector<TuningConfig> candidates;

    for (int block_x : {64, 128, 256, 512, 1024})
    {
        TuningConfig cfg;
        cfg.set("block_x", block_x);
        cfg.set("block_y", 1);
        candidates.push_back(cfg);
    }

    ValidKernelTraits::Args args{};

    size_t valid_count = concepts::count_valid_candidates<ValidKernelTraits>(candidates, args);

    EXPECT_EQ(valid_count, candidates.size());
}

TEST(KernelTraitsConceptsSpec, ShouldSkipAutotuningWithExactThresholdWorkload)
{
    concepts::PruningHeuristics heuristics{};
    heuristics.workload_size_threshold = 100 * 1024; // 100 KB

    size_t exact_threshold = 100 * 1024;

    bool should_skip = concepts::should_skip_autotuning<ValidKernelTraits>(heuristics, exact_threshold);

    // Exact threshold should NOT skip (< threshold skips, >= does not)
    EXPECT_FALSE(should_skip);
}

TEST(KernelTraitsConceptsSpec, GetDefaultConfigProducesValidConfig)
{
    TuningConfig config = concepts::get_default_config<ValidKernelTraits, TuningConfig>();

    ValidKernelTraits::Args args{};

    // Default config should be valid
    EXPECT_TRUE(ValidKernelTraits::is_valid_config(config, args));
}

// ============================================================================
// TEST SUITE: Integration with Real Kernel Traits
// ============================================================================

TEST(KernelTraitsConceptsSpec, ConceptsWorkWithActualKernelTraits)
{
    // This tests that our concepts work with real kernel traits from the codebase
    // We use ValidKernelTraits as a stand-in, but in real usage it would be
    // GrayscaleKernelTraits, GaussianBlurKernelTraits, etc.

    EXPECT_TRUE(concepts::StatelessKernelTraits<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasKernelName<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasArgsType<ValidKernelTraits>);
    EXPECT_TRUE(concepts::HasContextType<ValidKernelTraits>);
    EXPECT_TRUE((concepts::NonEmptyCandidates<ValidKernelTraits, TuningConfig>));

    std::vector<TuningConfig> candidates = ValidKernelTraits::generate_candidates();
    EXPECT_TRUE(concepts::validate_candidates(candidates));

    ValidKernelTraits::Args args{};
    size_t valid = concepts::count_valid_candidates<ValidKernelTraits>(candidates, args);
    EXPECT_GT(valid, 0u);
}
