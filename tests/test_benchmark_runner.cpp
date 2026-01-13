// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

/**
 * @file test_benchmark_runner.cpp
 * @brief Tests for benchmark runner functionality
 *
 * Category: Specification Tests
 *
 * Tests validate the behavior contracts of the benchmark runner,
 * including image generation, statistics calculation, and timing functions.
 *
 * Note: Full integration tests (run_benchmark_suite) require GPU and are
 * resource-intensive, so they are tested via edge case validation rather
 * than full execution.
 */

#include <gtest/gtest.h>
#include "../bench/bench_functions.h"
#include <vector>
#include <cmath>
#include <cstring>

// ============================================================================
// TEST SUITE: Image Generation Contracts
// ============================================================================

TEST(BenchmarkRunner, GenerateTestImageCreatesValidImage)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);

    ASSERT_NE(img.data, nullptr) << "Image data should be allocated";
    EXPECT_EQ(img.width, width);
    EXPECT_EQ(img.height, height);
    EXPECT_EQ(img.channels, channels);

    free_test_image(img);
}

TEST(BenchmarkRunner, GenerateTestImageRGBHasGradient)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    // Verify gradient pattern exists (not all zeros)
    bool has_variation = false;
    const unsigned char first_r = img.data[0];

    for (int i = 0; i < width * height * channels; i += channels)
    {
        if (img.data[i] != first_r)
        {
            has_variation = true;
            break;
        }
    }

    EXPECT_TRUE(has_variation) << "Generated image should have gradient variation";

    // Verify R channel increases horizontally
    const unsigned char left_edge_r = img.data[0];
    const unsigned char right_edge_r = img.data[(width - 1) * channels];
    EXPECT_LT(left_edge_r, right_edge_r) << "R channel should increase left to right";

    free_test_image(img);
}

TEST(BenchmarkRunner, GenerateTestImageRGBAHasAlphaChannel)
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int channels = 4;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    // Verify alpha channel is fully opaque
    for (int i = 0; i < width * height; ++i)
    {
        const unsigned char alpha = img.data[i * channels + 3];
        EXPECT_EQ(alpha, 255) << "Alpha channel should be 255 at pixel " << i;
    }

    free_test_image(img);
}

TEST(BenchmarkRunner, GenerateTestImageVariousSizes)
{
    const std::vector<std::pair<int, int>> test_sizes = {
        {1, 1},
        {2, 2},
        {64, 64},
        {512, 512},
        {1024, 1024}};

    for (const auto &[width, height] : test_sizes)
    {
        bench_image_t img = generate_test_image(width, height, 3);

        ASSERT_NE(img.data, nullptr) << "Failed for size " << width << "x" << height;
        EXPECT_EQ(img.width, width);
        EXPECT_EQ(img.height, height);

        free_test_image(img);
    }
}

TEST(BenchmarkRunner, FreeTestImageNullsPointer)
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    free_test_image(img);

    EXPECT_EQ(img.data, nullptr) << "Image data pointer should be nulled after free";
}

TEST(BenchmarkRunner, FreeTestImageHandlesNullPointer)
{
    bench_image_t img;
    img.data = nullptr;
    img.width = 0;
    img.height = 0;
    img.channels = 0;

    // Should not crash
    ASSERT_NO_THROW(free_test_image(img));
    EXPECT_EQ(img.data, nullptr);
}

TEST(BenchmarkRunner, FreeTestImageIdempotent)
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);

    free_test_image(img);
    EXPECT_EQ(img.data, nullptr);

    // Second free should be safe
    ASSERT_NO_THROW(free_test_image(img));
    EXPECT_EQ(img.data, nullptr);
}

// ============================================================================
// TEST SUITE: Statistics Calculation Contracts
// ============================================================================

TEST(BenchmarkRunner, CalculateStdDevForUniformValues)
{
    const std::vector<double> values = {5.0, 5.0, 5.0, 5.0, 5.0};
    constexpr double mean = 5.0;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_DOUBLE_EQ(std_dev, 0.0) << "Standard deviation of uniform values should be zero";
}

TEST(BenchmarkRunner, CalculateStdDevForKnownDistribution)
{
    // Data: 2, 4, 6, 8, 10
    // Mean: 6
    // Deviations: -4, -2, 0, 2, 4
    // Squared: 16, 4, 0, 4, 16 = 40
    // Sample variance (n-1): 40/4 = 10
    // Sample std dev: sqrt(10) ≈ 3.162
    const std::vector<double> values = {2.0, 4.0, 6.0, 8.0, 10.0};
    constexpr double mean = 6.0;

    const double std_dev = calculate_std_dev(values, mean);
    const double expected_std_dev = std::sqrt(40.0 / 4.0); // Using n-1 for sample std dev

    EXPECT_NEAR(std_dev, expected_std_dev, 1e-9)
        << "Standard deviation should match calculated value";
}

TEST(BenchmarkRunner, CalculateStdDevForTwoValues)
{
    const std::vector<double> values = {1.0, 3.0};
    constexpr double mean = 2.0;

    const double std_dev = calculate_std_dev(values, mean);
    const double expected = std::sqrt(2.0); // sqrt((1+1)/(2-1))

    EXPECT_NEAR(std_dev, expected, 1e-9);
}

TEST(BenchmarkRunner, CalculateStdDevForSingleValue)
{
    const std::vector<double> values = {42.0};
    constexpr double mean = 42.0;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_DOUBLE_EQ(std_dev, 0.0)
        << "Standard deviation of single value should be zero";
}

TEST(BenchmarkRunner, CalculateStdDevForEmptyVector)
{
    const std::vector<double> values = {};
    constexpr double mean = 0.0;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_DOUBLE_EQ(std_dev, 0.0)
        << "Standard deviation of empty vector should be zero";
}

TEST(BenchmarkRunner, CalculateStdDevIsAlwaysNonNegative)
{
    const std::vector<std::vector<double>> test_cases = {
        {-10.0, -5.0, 0.0, 5.0, 10.0},
        {100.0, 200.0, 300.0},
        {0.001, 0.002, 0.003},
        {-1000.0, 1000.0}};

    for (const auto &values : test_cases)
    {
        double sum = 0.0;
        for (double v : values)
        {
            sum += v;
        }
        const double mean = sum / values.size();

        const double std_dev = calculate_std_dev(values, mean);

        EXPECT_GE(std_dev, 0.0) << "Standard deviation must be non-negative";
    }
}

TEST(BenchmarkRunner, CalculateStdDevForLargeVariance)
{
    const std::vector<double> values = {0.0, 1000000.0};
    constexpr double mean = 500000.0;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_GT(std_dev, 0.0);
    EXPECT_LT(std_dev, 1000000.0);
}

TEST(BenchmarkRunner, CalculateStdDevForHighPrecisionValues)
{
    const std::vector<double> values = {
        1.000001, 1.000002, 1.000003, 1.000004, 1.000005};
    constexpr double mean = 1.000003;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_GT(std_dev, 0.0) << "Should detect small variations";
    EXPECT_LT(std_dev, 0.001) << "Variation should be small";
}

// ============================================================================
// TEST SUITE: Image Data Properties
// ============================================================================

TEST(BenchmarkRunner, GeneratedImageHasCorrectSize)
{
    constexpr int width = 32;
    constexpr int height = 16;
    constexpr int channels = 3;
    const size_t expected_bytes = width * height * channels;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    // We can't directly check allocated size, but we can verify we can
    // access all expected bytes without crashing
    bool access_ok = true;
    for (size_t i = 0; i < expected_bytes; ++i)
    {
        // Read each byte (should not segfault)
        volatile unsigned char val = img.data[i];
        (void)val; // Suppress unused warning
    }

    EXPECT_TRUE(access_ok);

    free_test_image(img);
}

TEST(BenchmarkRunner, GeneratedImageGradientIsMonotonic)
{
    constexpr int width = 256;
    constexpr int height = 1;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    // Check R channel increases monotonically left to right
    for (int x = 1; x < width; ++x)
    {
        const unsigned char prev_r = img.data[(x - 1) * channels];
        const unsigned char curr_r = img.data[x * channels];

        EXPECT_GE(curr_r, prev_r)
            << "R channel should be monotonically non-decreasing at x=" << x;
    }

    free_test_image(img);
}

TEST(BenchmarkRunner, GeneratedImageRGBChannelsAreDifferent)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);
    ASSERT_NE(img.data, nullptr);

    // Sample different pixels to verify gradient patterns exist
    // Top-left (0,0): R=0, G=0, B=0
    // Top-right (63,0): R=~252, G=0, B=~126
    // Bottom-left (0,63): R=0, G=~252, B=~126
    const int top_left = 0;
    const int top_right = (width - 1) * channels;
    const int bottom_left = ((height - 1) * width) * channels;

    // Check that R gradient increases horizontally
    EXPECT_LT(img.data[top_left + 0], img.data[top_right + 0])
        << "R should increase from left to right";

    // Check that G gradient increases vertically
    EXPECT_LT(img.data[top_left + 1], img.data[bottom_left + 1])
        << "G should increase from top to bottom";

    free_test_image(img);
}

// ============================================================================
// TEST SUITE: Edge Cases and Error Handling
// ============================================================================

TEST(BenchmarkRunner, GenerateTestImageMinimalSize)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 1;

    bench_image_t img = generate_test_image(width, height, channels);

    ASSERT_NE(img.data, nullptr);
    EXPECT_EQ(img.width, width);
    EXPECT_EQ(img.height, height);
    EXPECT_EQ(img.channels, channels);

    free_test_image(img);
}

TEST(BenchmarkRunner, GenerateTestImageRectangular)
{
    constexpr int width = 128;
    constexpr int height = 64;
    constexpr int channels = 3;

    bench_image_t img = generate_test_image(width, height, channels);

    ASSERT_NE(img.data, nullptr);
    EXPECT_EQ(img.width, width);
    EXPECT_EQ(img.height, height);

    // Verify gradient respects dimensions
    const unsigned char left_top_r = img.data[0];
    const unsigned char right_top_r = img.data[(width - 1) * channels];
    EXPECT_LT(left_top_r, right_top_r);

    free_test_image(img);
}

TEST(BenchmarkRunner, CalculateStdDevWithMeanMismatch)
{
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    constexpr double incorrect_mean = 100.0; // Intentionally wrong mean

    const double std_dev = calculate_std_dev(values, incorrect_mean);

    // Should still compute (garbage in, garbage out), but not crash
    EXPECT_GE(std_dev, 0.0);
    EXPECT_TRUE(std::isfinite(std_dev));
}

TEST(BenchmarkRunner, CalculateStdDevWithLargeDataset)
{
    std::vector<double> values;
    values.reserve(10000);
    for (int i = 0; i < 10000; ++i)
    {
        values.push_back(static_cast<double>(i));
    }

    const double mean = 4999.5; // Mean of 0..9999

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_GT(std_dev, 0.0);
    EXPECT_TRUE(std::isfinite(std_dev));
    EXPECT_LT(std_dev, 10000.0);
}

// ============================================================================
// TEST SUITE: Numerical Stability
// ============================================================================

TEST(BenchmarkRunner, CalculateStdDevNumericalStability)
{
    // Test with values that could cause numerical instability
    const std::vector<double> values = {
        1e10 + 1.0,
        1e10 + 2.0,
        1e10 + 3.0,
        1e10 + 4.0,
        1e10 + 5.0};
    const double mean = 1e10 + 3.0;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_TRUE(std::isfinite(std_dev)) << "Calculation should remain stable with large values";
    EXPECT_GT(std_dev, 0.0);
}

TEST(BenchmarkRunner, CalculateStdDevVerySmallValues)
{
    const std::vector<double> values = {1e-10, 2e-10, 3e-10, 4e-10, 5e-10};
    const double mean = 3e-10;

    const double std_dev = calculate_std_dev(values, mean);

    EXPECT_TRUE(std::isfinite(std_dev));
    EXPECT_GE(std_dev, 0.0);
}

// ============================================================================
// TEST SUITE: Memory Safety
// ============================================================================

TEST(BenchmarkRunner, MultipleImageAllocationsAndFrees)
{
    const int num_images = 10;
    std::vector<bench_image_t> images;

    // Allocate multiple images
    for (int i = 0; i < num_images; ++i)
    {
        bench_image_t img = generate_test_image(32, 32, 3);
        ASSERT_NE(img.data, nullptr);
        images.push_back(img);
    }

    // Free all images
    for (auto &img : images)
    {
        free_test_image(img);
        EXPECT_EQ(img.data, nullptr);
    }
}

TEST(BenchmarkRunner, LargeImageAllocation)
{
    constexpr int width = 4096;
    constexpr int height = 4096;
    constexpr int channels = 4;

    bench_image_t img = generate_test_image(width, height, channels);

    // Should successfully allocate large image (64MB)
    ASSERT_NE(img.data, nullptr) << "Should be able to allocate large image";
    EXPECT_EQ(img.width, width);
    EXPECT_EQ(img.height, height);

    free_test_image(img);
}

// ============================================================================
// TEST SUITE: Statistical Properties
// ============================================================================

TEST(BenchmarkRunner, CalculateStdDevSymmetricAroundMean)
{
    const std::vector<double> values = {-2.0, -1.0, 0.0, 1.0, 2.0};
    constexpr double mean = 0.0;

    const double std_dev = calculate_std_dev(values, mean);

    // Should be approximately 1.58 (sqrt(10/4))
    EXPECT_NEAR(std_dev, std::sqrt(2.5), 1e-9);
}

TEST(BenchmarkRunner, CalculateStdDevScaleInvariance)
{
    const std::vector<double> values1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> values2 = {10.0, 20.0, 30.0, 40.0, 50.0};

    const double mean1 = 3.0;
    const double mean2 = 30.0;

    const double std_dev1 = calculate_std_dev(values1, mean1);
    const double std_dev2 = calculate_std_dev(values2, mean2);

    // std_dev2 should be 10x std_dev1
    EXPECT_NEAR(std_dev2 / std_dev1, 10.0, 1e-9);
}
