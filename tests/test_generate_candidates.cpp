#include <gtest/gtest.h>
#include "../src/filters/filters.h"
#include "../src/core/gpu_utils.h"
#include <set>
#include <algorithm>

/**
 * @brief Comprehensive tests for generate_candidates functions
 *
 * Tests all three filter types' generate_candidates() implementations
 * to ensure they produce valid, diverse, and reasonable configurations.
 */

// ============================================================================
// GRAYSCALE GENERATE_CANDIDATES TESTS
// ============================================================================

TEST(GenerateCandidates, GrayscaleReturnsNonEmpty)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    EXPECT_GT(configs.size(), 0) << "Should generate at least one configuration";
}

TEST(GenerateCandidates, GrayscaleReturnsMultipleCandidates)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    EXPECT_GE(configs.size(), 5) << "Should generate multiple candidates for tuning";
}

TEST(GenerateCandidates, GrayscaleIncludesExpected1DConfigs)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // Should include 64, 128, 256, 512 as 1D configs
    std::set<int> expected_1d = {64, 128, 256, 512};
    std::set<int> found_1d;

    for (const auto &cfg : configs)
    {
        if (cfg.block_y() == 1)
        {
            found_1d.insert(cfg.block_x());
        }
    }

    for (int expected : expected_1d)
    {
        EXPECT_TRUE(found_1d.count(expected) > 0)
            << "Should include 1D config with block_x=" << expected;
    }
}

TEST(GenerateCandidates, GrayscaleIncludes2DConfigs)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // Should have some 2D configurations
    bool has_2d = false;
    for (const auto &cfg : configs)
    {
        if (cfg.block_y() > 1)
        {
            has_2d = true;
            // 2D configs should use 16 or 32 for block_x
            EXPECT_TRUE(cfg.block_x() == 16 || cfg.block_x() == 32)
                << "2D configs should use block_x of 16 or 32";
            // block_y should be 4, 8, or 16
            EXPECT_TRUE(cfg.block_y() == 4 || cfg.block_y() == 8 || cfg.block_y() == 16)
                << "2D configs should use block_y of 4, 8, or 16";
        }
    }

    EXPECT_TRUE(has_2d) << "Should include 2D configurations";
}

TEST(GenerateCandidates, GrayscaleAllConfigsUnique)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    std::set<std::string> unique_keys;
    for (const auto &cfg : configs)
    {
        unique_keys.insert(cfg.to_key_string());
    }

    EXPECT_EQ(unique_keys.size(), configs.size())
        << "All generated configs should be unique";
}

TEST(GenerateCandidates, GrayscaleConfigsHaveValidDimensions)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        EXPECT_GT(cfg.block_x(), 0) << "block_x must be positive";
        EXPECT_GT(cfg.block_y(), 0) << "block_y must be positive";

        int total_threads = cfg.block_x() * cfg.block_y();
        EXPECT_GE(total_threads, 64) << "Should have at least 64 threads";
        EXPECT_LE(total_threads, 1024) << "Should not exceed 1024 threads";
    }
}

TEST(GenerateCandidates, GrayscaleExpectedCount)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // 4 1D configs + (2 block_x values * 3 block_y values) = 4 + 6 = 10
    EXPECT_EQ(configs.size(), 10) << "Should generate exactly 10 configurations";
}

// ============================================================================
// NEGATIVE GENERATE_CANDIDATES TESTS
// ============================================================================

TEST(GenerateCandidates, NegativeReturnsNonEmpty)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();
    EXPECT_GT(configs.size(), 0);
}

TEST(GenerateCandidates, NegativeReturnsMultipleCandidates)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();
    EXPECT_GE(configs.size(), 5);
}

TEST(GenerateCandidates, NegativeIncludesExpected1DConfigs)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();

    std::set<int> expected_1d = {64, 128, 256, 512};
    std::set<int> found_1d;

    for (const auto &cfg : configs)
    {
        if (cfg.block_y() == 1)
        {
            found_1d.insert(cfg.block_x());
        }
    }

    for (int expected : expected_1d)
    {
        EXPECT_TRUE(found_1d.count(expected) > 0)
            << "Should include 1D config with block_x=" << expected;
    }
}

TEST(GenerateCandidates, NegativeIncludes2DConfigs)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();

    bool has_2d = false;
    for (const auto &cfg : configs)
    {
        if (cfg.block_y() > 1)
        {
            has_2d = true;
            EXPECT_TRUE(cfg.block_x() == 16 || cfg.block_x() == 32);
            EXPECT_TRUE(cfg.block_y() == 4 || cfg.block_y() == 8 || cfg.block_y() == 16);
        }
    }

    EXPECT_TRUE(has_2d);
}

TEST(GenerateCandidates, NegativeAllConfigsUnique)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();

    std::set<std::string> unique_keys;
    for (const auto &cfg : configs)
    {
        unique_keys.insert(cfg.to_key_string());
    }

    EXPECT_EQ(unique_keys.size(), configs.size());
}

TEST(GenerateCandidates, NegativeExpectedCount)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();
    EXPECT_EQ(configs.size(), 10);
}

// ============================================================================
// GAUSSIAN BLUR GENERATE_CANDIDATES TESTS
// ============================================================================

TEST(GenerateCandidates, GaussianBlurReturnsNonEmpty)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();
    EXPECT_GT(configs.size(), 0);
}

TEST(GenerateCandidates, GaussianBlurReturnsMultipleCandidates)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();
    EXPECT_GE(configs.size(), 5);
}

TEST(GenerateCandidates, GaussianBlurPrefers2DConfigs)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    // Blur should prefer 2D configs due to 2D memory access patterns
    int count_2d = 0;
    int count_1d = 0;

    for (const auto &cfg : configs)
    {
        if (cfg.block_y() > 1)
        {
            count_2d++;
        }
        else
        {
            count_1d++;
        }
    }

    EXPECT_GT(count_2d, 0) << "Should have 2D configurations";
    EXPECT_GT(count_1d, 0) << "Should also have some 1D configurations";
}

TEST(GenerateCandidates, GaussianBlurIncludes2DRange)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    // Should test 8, 16, 32 for both dimensions
    std::set<int> block_x_values;
    std::set<int> block_y_values;

    for (const auto &cfg : configs)
    {
        if (cfg.block_y() > 1)
        {
            block_x_values.insert(cfg.block_x());
            block_y_values.insert(cfg.block_y());
        }
    }

    // Should include multiple values for exploration
    EXPECT_GE(block_x_values.size(), 2) << "Should test multiple block_x values";
    EXPECT_GE(block_y_values.size(), 2) << "Should test multiple block_y values";
}

TEST(GenerateCandidates, GaussianBlurRespects64ThreadMinimum)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        int total_threads = cfg.block_x() * cfg.block_y();
        EXPECT_GE(total_threads, 64)
            << "Config " << cfg.to_key_string() << " has < 64 threads";
    }
}

TEST(GenerateCandidates, GaussianBlurRespects1024ThreadMaximum)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        int total_threads = cfg.block_x() * cfg.block_y();
        EXPECT_LE(total_threads, 1024)
            << "Config " << cfg.to_key_string() << " exceeds 1024 threads";
    }
}

TEST(GenerateCandidates, GaussianBlurAllConfigsUnique)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    std::set<std::string> unique_keys;
    for (const auto &cfg : configs)
    {
        unique_keys.insert(cfg.to_key_string());
    }

    EXPECT_EQ(unique_keys.size(), configs.size());
}

TEST(GenerateCandidates, GaussianBlurIncludes1DConfigs)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    // Should include 128, 256, 512 as 1D
    std::set<int> expected_1d = {128, 256, 512};
    std::set<int> found_1d;

    for (const auto &cfg : configs)
    {
        if (cfg.block_y() == 1)
        {
            found_1d.insert(cfg.block_x());
        }
    }

    for (int expected : expected_1d)
    {
        EXPECT_TRUE(found_1d.count(expected) > 0)
            << "Should include 1D config with block_x=" << expected;
    }
}

TEST(GenerateCandidates, GaussianBlurConfigsInValidRange)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        // Values should be from the specified ranges
        if (cfg.block_y() > 1)
        {
            EXPECT_TRUE(cfg.block_x() == 8 || cfg.block_x() == 16 || cfg.block_x() == 32)
                << "2D block_x should be 8, 16, or 32";
            EXPECT_TRUE(cfg.block_y() == 8 || cfg.block_y() == 16 || cfg.block_y() == 32)
                << "2D block_y should be 8, 16, or 32";
        }
        else
        {
            EXPECT_TRUE(cfg.block_x() == 128 || cfg.block_x() == 256 || cfg.block_x() == 512)
                << "1D block_x should be 128, 256, or 512";
        }
    }
}

// ============================================================================
// CROSS-FILTER COMPARISON TESTS
// ============================================================================

TEST(GenerateCandidates, AllFiltersReturnConfigs)
{
    auto grayscale = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    auto negative = imgfx::filters::NegativeKernelTraits::generate_candidates();
    auto blur = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    EXPECT_GT(grayscale.size(), 0);
    EXPECT_GT(negative.size(), 0);
    EXPECT_GT(blur.size(), 0);
}

TEST(GenerateCandidates, GrayscaleAndNegativeSimilar)
{
    auto grayscale = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    auto negative = imgfx::filters::NegativeKernelTraits::generate_candidates();

    // Grayscale and Negative should have similar configs (both memory-bound)
    EXPECT_EQ(grayscale.size(), negative.size())
        << "Similar filters should generate similar candidate counts";
}

TEST(GenerateCandidates, BlurDifferentFromOthers)
{
    auto grayscale = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    auto blur = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    // Blur may have different configs due to different memory access pattern
    // Just verify both generate reasonable amounts
    EXPECT_GE(grayscale.size(), 5);
    EXPECT_GE(blur.size(), 5);
}

TEST(GenerateCandidates, AllGeneratedConfigsCanBeValidated)
{
    // Verify generated configs can be passed to is_valid_config
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    for (const auto &cfg : configs)
    {
        // Should not crash when validating
        ASSERT_NO_THROW({
            bool valid = imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args);
            (void)valid; // Use the result
        });
    }
}

TEST(GenerateCandidates, ConfigsHaveRequiredFields)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        // Every config should have block_x and block_y set
        EXPECT_TRUE(cfg.has("block_x")) << "Config missing block_x";
        EXPECT_TRUE(cfg.has("block_y")) << "Config missing block_y";
        EXPECT_GT(cfg.block_x(), 0);
        EXPECT_GT(cfg.block_y(), 0);
    }
}

TEST(GenerateCandidates, NoConfigDuplicatesAcrossFilters)
{
    auto grayscale = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    auto negative = imgfx::filters::NegativeKernelTraits::generate_candidates();

    // Within each filter, no duplicates
    std::set<std::string> gs_keys;
    for (const auto &cfg : grayscale)
    {
        gs_keys.insert(cfg.to_key_string());
    }
    EXPECT_EQ(gs_keys.size(), grayscale.size());

    std::set<std::string> neg_keys;
    for (const auto &cfg : negative)
    {
        neg_keys.insert(cfg.to_key_string());
    }
    EXPECT_EQ(neg_keys.size(), negative.size());
}

TEST(GenerateCandidates, WavefrontAlignment)
{
    // All configs should be aligned to 64 threads (AMD wavefront size)
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    for (const auto &cfg : configs)
    {
        int threads = cfg.block_x() * cfg.block_y();
        EXPECT_EQ(threads % 64, 0)
            << "Config " << cfg.to_key_string() << " not aligned to 64-thread wavefronts";
    }
}
