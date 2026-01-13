// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/filters/filters.h"
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"

/**
 * @brief Tests for filters.h - focusing on kernel traits and configuration
 */

TEST(FiltersTraits, GrayscaleKernelTraitsName)
{
    EXPECT_STREQ(imgfx::filters::GrayscaleKernelTraits::name(), "grayscale");
}

TEST(FiltersTraits, NegativeKernelTraitsName)
{
    EXPECT_STREQ(imgfx::filters::NegativeKernelTraits::name(), "negative");
}

TEST(FiltersTraits, GaussianBlurKernelTraitsName)
{
    EXPECT_STREQ(imgfx::filters::GaussianBlurKernelTraits::name(), "gaussian_blur");
}

TEST(FiltersTraits, GrayscaleContextCacheKeySmall)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;
    ctx.image_bytes = 500 * 1024; // 500 KB

    EXPECT_EQ(ctx.cache_key(), "small");
}

TEST(FiltersTraits, GrayscaleContextCacheKeyMedium)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;
    ctx.image_bytes = 5 * 1024 * 1024; // 5 MB

    EXPECT_EQ(ctx.cache_key(), "medium");
}

TEST(FiltersTraits, GrayscaleContextCacheKeyLarge)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;
    ctx.image_bytes = 20 * 1024 * 1024; // 20 MB

    EXPECT_EQ(ctx.cache_key(), "large");
}

TEST(FiltersTraits, GrayscaleContextCacheKeyBoundary1MB)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;

    ctx.image_bytes = 1024 * 1024 - 1; // Just under 1MB
    EXPECT_EQ(ctx.cache_key(), "small");

    ctx.image_bytes = 1024 * 1024; // Exactly 1MB
    EXPECT_EQ(ctx.cache_key(), "medium");
}

TEST(FiltersTraits, GrayscaleContextCacheKeyBoundary10MB)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;

    ctx.image_bytes = 10 * 1024 * 1024 - 1; // Just under 10MB
    EXPECT_EQ(ctx.cache_key(), "medium");

    ctx.image_bytes = 10 * 1024 * 1024; // Exactly 10MB
    EXPECT_EQ(ctx.cache_key(), "large");
}

TEST(FiltersTraits, NegativeContextCacheKeySmall)
{
    imgfx::filters::NegativeKernelTraits::Context ctx;
    ctx.image_bytes = 100 * 1024; // 100 KB

    EXPECT_EQ(ctx.cache_key(), "small");
}

TEST(FiltersTraits, NegativeContextCacheKeyMedium)
{
    imgfx::filters::NegativeKernelTraits::Context ctx;
    ctx.image_bytes = 2 * 1024 * 1024; // 2 MB

    EXPECT_EQ(ctx.cache_key(), "medium");
}

TEST(FiltersTraits, NegativeContextCacheKeyLarge)
{
    imgfx::filters::NegativeKernelTraits::Context ctx;
    ctx.image_bytes = 50 * 1024 * 1024; // 50 MB

    EXPECT_EQ(ctx.cache_key(), "large");
}

TEST(FiltersTraits, GaussianBlurContextCacheKeySmall)
{
    imgfx::filters::GaussianBlurKernelTraits::Context ctx;
    ctx.image_bytes = 256 * 1024; // 256 KB
    ctx.blur_amount = 3;          // Small blur

    EXPECT_EQ(ctx.cache_key(), "small_blur_small");
}

TEST(FiltersTraits, GaussianBlurContextCacheKeyMedium)
{
    imgfx::filters::GaussianBlurKernelTraits::Context ctx;
    ctx.image_bytes = 7 * 1024 * 1024; // 7 MB
    ctx.blur_amount = 7;               // Large blur

    EXPECT_EQ(ctx.cache_key(), "medium_blur_large");
}

TEST(FiltersTraits, GaussianBlurContextCacheKeyLarge)
{
    imgfx::filters::GaussianBlurKernelTraits::Context ctx;
    ctx.image_bytes = 100 * 1024 * 1024; // 100 MB
    ctx.blur_amount = 10;                // Large blur

    EXPECT_EQ(ctx.cache_key(), "large_blur_large");
}

TEST(FiltersTraits, GrayscaleGenerateCandidatesNotEmpty)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    EXPECT_GT(configs.size(), 0) << "Should generate at least one candidate";
}

TEST(FiltersTraits, GrayscaleGenerateCandidatesHas1DConfigs)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // Should have configurations with block_y = 1
    bool has_1d = false;
    for (const auto &cfg : configs)
    {
        if (cfg.block_y() == 1)
        {
            has_1d = true;
            break;
        }
    }

    EXPECT_TRUE(has_1d) << "Should have 1D configurations";
}

TEST(FiltersTraits, GrayscaleGenerateCandidatesHas2DConfigs)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // Should have configurations with block_y > 1
    bool has_2d = false;
    for (const auto &cfg : configs)
    {
        if (cfg.block_y() > 1)
        {
            has_2d = true;
            break;
        }
    }

    EXPECT_TRUE(has_2d) << "Should have 2D configurations";
}

TEST(FiltersTraits, GrayscaleIsValidConfigWavefrontAlignment)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // 128 threads = 2 wavefronts, should be valid
    imgfx::core::autotune::TuningConfig cfg1;
    cfg1.set("block_x", 128);
    cfg1.set("block_y", 1);
    EXPECT_TRUE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg1, args));

    // 100 threads, not aligned to 64
    imgfx::core::autotune::TuningConfig cfg2;
    cfg2.set("block_x", 100);
    cfg2.set("block_y", 1);
    EXPECT_FALSE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg2, args));
}

TEST(FiltersTraits, GrayscaleIsValidConfigThreadLimits)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // Too few threads
    imgfx::core::autotune::TuningConfig cfg_small;
    cfg_small.set("block_x", 32);
    cfg_small.set("block_y", 1);
    EXPECT_FALSE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg_small, args));

    // Too many threads
    imgfx::core::autotune::TuningConfig cfg_large;
    cfg_large.set("block_x", 2048);
    cfg_large.set("block_y", 1);
    EXPECT_FALSE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg_large, args));
}

TEST(FiltersTraits, GrayscaleIsValidConfig2D)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // 16x16 = 256 threads, aligned to 64
    imgfx::core::autotune::TuningConfig cfg;
    cfg.set("block_x", 16);
    cfg.set("block_y", 16);
    EXPECT_TRUE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args));
}

TEST(FiltersTraits, NegativeGenerateCandidatesNotEmpty)
{
    auto configs = imgfx::filters::NegativeKernelTraits::generate_candidates();

    EXPECT_GT(configs.size(), 0);
}

TEST(FiltersTraits, NegativeIsValidConfigWavefront)
{
    imgfx::filters::NegativeKernelTraits::Args args{};

    imgfx::core::autotune::TuningConfig cfg_valid;
    cfg_valid.set("block_x", 256);
    cfg_valid.set("block_y", 1);
    EXPECT_TRUE(imgfx::filters::NegativeKernelTraits::is_valid_config(cfg_valid, args));

    imgfx::core::autotune::TuningConfig cfg_invalid;
    cfg_invalid.set("block_x", 100);
    cfg_invalid.set("block_y", 1);
    EXPECT_FALSE(imgfx::filters::NegativeKernelTraits::is_valid_config(cfg_invalid, args));
}

TEST(FiltersTraits, GaussianBlurGenerateCandidatesNotEmpty)
{
    auto configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    EXPECT_GT(configs.size(), 0);
}

TEST(FiltersTraits, GaussianBlurIsValidConfigWavefront)
{
    imgfx::filters::GaussianBlurKernelTraits::Args args{};

    imgfx::core::autotune::TuningConfig cfg_valid;
    cfg_valid.set("block_x", 128);
    cfg_valid.set("block_y", 1);
    EXPECT_TRUE(imgfx::filters::GaussianBlurKernelTraits::is_valid_config(cfg_valid, args));

    imgfx::core::autotune::TuningConfig cfg_invalid;
    cfg_invalid.set("block_x", 50);
    cfg_invalid.set("block_y", 1);
    EXPECT_FALSE(imgfx::filters::GaussianBlurKernelTraits::is_valid_config(cfg_invalid, args));
}

TEST(FiltersTraits, ArgsStructSizeGrayscale)
{
    imgfx::filters::GrayscaleKernelTraits::Args args;

    // Just verify we can construct and use the args struct
    args.input = nullptr;
    args.output = nullptr;
    args.metas = nullptr;
    args.num_images = 0;
    args.max_image_bytes = 0;

    EXPECT_EQ(args.num_images, 0);
}

TEST(FiltersTraits, ArgsStructSizeNegative)
{
    imgfx::filters::NegativeKernelTraits::Args args;

    args.input = nullptr;
    args.output = nullptr;
    args.metas = nullptr;
    args.num_images = 5;
    args.max_image_bytes = 1024;

    EXPECT_EQ(args.num_images, 5);
    EXPECT_EQ(args.max_image_bytes, 1024);
}

TEST(FiltersTraits, ArgsStructSizeGaussianBlur)
{
    imgfx::filters::GaussianBlurKernelTraits::Args args;

    args.input = nullptr;
    args.output = nullptr;
    args.metas = nullptr;
    args.num_images = 10;
    args.max_image_bytes = 2048;

    EXPECT_EQ(args.num_images, 10);
    EXPECT_EQ(args.max_image_bytes, 2048);
}

TEST(FiltersTraits, ContextSizeComparison)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx1;
    ctx1.image_bytes = 100;

    imgfx::filters::GrayscaleKernelTraits::Context ctx2;
    ctx2.image_bytes = 200;

    EXPECT_LT(ctx1.image_bytes, ctx2.image_bytes);
}

TEST(FiltersTraits, GeneratedConfigsAreDistinct)
{
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // Check that we have distinct configurations
    std::set<std::string> unique_configs;
    for (const auto &cfg : configs)
    {
        unique_configs.insert(cfg.to_key_string());
    }

    EXPECT_EQ(unique_configs.size(), configs.size())
        << "All generated configs should be unique";
}

TEST(FiltersTraits, ValidConfigsPassValidation)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};
    auto configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();

    // At least some generated configs should be valid
    int valid_count = 0;
    for (const auto &cfg : configs)
    {
        if (imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args))
        {
            valid_count++;
        }
    }

    EXPECT_GT(valid_count, 0) << "Should have at least some valid configs";
}

TEST(FiltersTraits, ConsistentCacheKeysForSameSize)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx1;
    ctx1.image_bytes = 500 * 1024;

    imgfx::filters::GrayscaleKernelTraits::Context ctx2;
    ctx2.image_bytes = 500 * 1024;

    EXPECT_EQ(ctx1.cache_key(), ctx2.cache_key());
}

TEST(FiltersTraits, DifferentCacheKeysForDifferentSizes)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx_small;
    ctx_small.image_bytes = 500 * 1024;

    imgfx::filters::GrayscaleKernelTraits::Context ctx_large;
    ctx_large.image_bytes = 50 * 1024 * 1024;

    EXPECT_NE(ctx_small.cache_key(), ctx_large.cache_key());
}

TEST(FiltersTraits, ZeroByteImageContext)
{
    imgfx::filters::GrayscaleKernelTraits::Context ctx;
    ctx.image_bytes = 0;

    // Should categorize as small
    EXPECT_EQ(ctx.cache_key(), "small");
}

TEST(FiltersTraits, MinimalValidConfig)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // Minimal valid: 64 threads (1 wavefront)
    imgfx::core::autotune::TuningConfig cfg;
    cfg.set("block_x", 64);
    cfg.set("block_y", 1);

    EXPECT_TRUE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args));
}

TEST(FiltersTraits, MaximalValidConfig)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // Maximal valid: 1024 threads
    imgfx::core::autotune::TuningConfig cfg;
    cfg.set("block_x", 1024);
    cfg.set("block_y", 1);

    EXPECT_TRUE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args));
}

TEST(FiltersTraits, BoundaryValid2DConfig)
{
    imgfx::filters::GrayscaleKernelTraits::Args args{};

    // 32x8 = 256, aligned
    imgfx::core::autotune::TuningConfig cfg;
    cfg.set("block_x", 32);
    cfg.set("block_y", 8);

    EXPECT_TRUE(imgfx::filters::GrayscaleKernelTraits::is_valid_config(cfg, args));
}

TEST(FiltersTraits, AllKernelTraitsHaveConsistentInterface)
{
    // All three kernel traits should have the same methods
    // Just verify they can be called

    auto grayscale_name = imgfx::filters::GrayscaleKernelTraits::name();
    auto negative_name = imgfx::filters::NegativeKernelTraits::name();
    auto blur_name = imgfx::filters::GaussianBlurKernelTraits::name();

    EXPECT_NE(strlen(grayscale_name), 0);
    EXPECT_NE(strlen(negative_name), 0);
    EXPECT_NE(strlen(blur_name), 0);

    auto grayscale_configs = imgfx::filters::GrayscaleKernelTraits::generate_candidates();
    auto negative_configs = imgfx::filters::NegativeKernelTraits::generate_candidates();
    auto blur_configs = imgfx::filters::GaussianBlurKernelTraits::generate_candidates();

    EXPECT_GT(grayscale_configs.size(), 0);
    EXPECT_GT(negative_configs.size(), 0);
    EXPECT_GT(blur_configs.size(), 0);
}
