// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/core/gpu_utils.h"
#include "../src/core/image.h"
#include "test_helpers.h"
#include <vector>

/**
 * @brief Extended CPU filter tests - edge cases and variations
 *
 * These tests verify CPU filter implementations with various edge cases,
 * different image sizes, and channel counts to improve coverage.
 */

TEST(FiltersCPUExtended, GrayscaleSmallImage)
{
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 255, 128, 64, 0);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);

    // Verify grayscale conversion (all channels should be equal)
    for (int i = 0; i < width * height; ++i)
    {
        unsigned char gray = output[i * channels];
        EXPECT_EQ(output[i * channels + 0], gray);
        EXPECT_EQ(output[i * channels + 1], gray);
        EXPECT_EQ(output[i * channels + 2], gray);
    }
}

TEST(FiltersCPUExtended, Grayscale1x1Image)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 200, 100, 50, 0);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, GrayscaleLargeImage)
{
    constexpr int width = 512;
    constexpr int height = 512;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, Grayscale4Channel)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 4;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 255);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);

    // Alpha channel should be preserved
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(output[i * channels + 3], 255);
    }
}

TEST(FiltersCPUExtended, GrayscalePureColors)
{
    constexpr int width = 3;
    constexpr int height = 1;
    constexpr int channels = 3;

    // Create image with pure red, green, blue
    std::vector<unsigned char> input(width * height * channels);
    // Red pixel
    input[0] = 255;
    input[1] = 0;
    input[2] = 0;
    // Green pixel
    input[3] = 0;
    input[4] = 255;
    input[5] = 0;
    // Blue pixel
    input[6] = 0;
    input[7] = 0;
    input[8] = 255;

    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);

    // Each pixel should be grayscale (all channels equal)
    for (int i = 0; i < width; ++i)
    {
        EXPECT_EQ(output[i * channels + 0], output[i * channels + 1]);
        EXPECT_EQ(output[i * channels + 1], output[i * channels + 2]);
    }
}

TEST(FiltersCPUExtended, NegativeSmallImage)
{
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 2);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);

    // Verify negative inversion
    for (size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_EQ(output[i], 255 - input[i]);
    }
}

TEST(FiltersCPUExtended, Negative1x1Image)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;

    std::vector<unsigned char> input = {100, 150, 200};
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
    EXPECT_EQ(output[0], 155);
    EXPECT_EQ(output[1], 105);
    EXPECT_EQ(output[2], 55);
}

TEST(FiltersCPUExtended, NegativeLargeImage)
{
    constexpr int width = 256;
    constexpr int height = 256;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, Negative4Channel)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 4;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 50, 100, 150, 200);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, NegativeBlackAndWhite)
{
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr int channels = 3;

    std::vector<unsigned char> input = {
        0, 0, 0,       // Black
        255, 255, 255, // White
        0, 0, 0,       // Black
        255, 255, 255  // White
    };
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);

    // Black should become white
    EXPECT_EQ(output[0], 255);
    EXPECT_EQ(output[1], 255);
    EXPECT_EQ(output[2], 255);

    // White should become black
    EXPECT_EQ(output[3], 0);
    EXPECT_EQ(output[4], 0);
    EXPECT_EQ(output[5], 0);
}

TEST(FiltersCPUExtended, GaussianBlurSmallImage)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 8);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, GaussianBlur1x1Image)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 128, 128, 128, 0);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, GaussianBlurLargeImage)
{
    constexpr int width = 256;
    constexpr int height = 256;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, GaussianBlur4Channel)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 4;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 255);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, GaussianBlurNonSquareImage)
{
    constexpr int width = 100;
    constexpr int height = 50;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(result, hipSuccess);
}

TEST(FiltersCPUExtended, MultipleFiltersSequential)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 16);
    std::vector<unsigned char> temp1(input.size());
    std::vector<unsigned char> temp2(input.size());
    std::vector<unsigned char> output(input.size());

    // Apply grayscale
    hipError_t result1 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        temp1.data(),
        width,
        height,
        channels);
    EXPECT_EQ(result1, hipSuccess);

    // Apply negative
    hipError_t result2 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        temp1.data(),
        temp2.data(),
        width,
        height,
        channels);
    EXPECT_EQ(result2, hipSuccess);

    // Apply gaussian blur
    hipError_t result3 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        temp2.data(),
        output.data(),
        width,
        height,
        channels);
    EXPECT_EQ(result3, hipSuccess);
}

TEST(FiltersCPUExtended, FilterWithRandomData)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;
    constexpr size_t size = width * height * channels;

    std::vector<unsigned char> input(size);
    test_helpers::fill_random_image(input.data(), size, 12345);

    std::vector<unsigned char> output(size);

    // Test all filters with random data
    hipError_t r1 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(), output.data(), width, height, channels);
    EXPECT_EQ(r1, hipSuccess);

    hipError_t r2 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(), output.data(), width, height, channels);
    EXPECT_EQ(r2, hipSuccess);

    hipError_t r3 = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(), output.data(), width, height, channels);
    EXPECT_EQ(r3, hipSuccess);
}
