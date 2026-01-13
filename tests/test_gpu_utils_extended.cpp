// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <string>

/**
 * @brief Additional tests for GPU utilities - edge cases and utility functions
 *
 * These tests cover utility functions and edge cases that may not be covered
 * by the main GPU integration tests. Note: FilterTypeToString is tested in
 * test_gpu_utils.cpp and is not duplicated here.
 */

TEST(GPUUtilsExtended, GaussianBlurAmountConstant)
{
    // Verify the gaussian blur amount is odd as required
    EXPECT_TRUE(imgfx::core::GAUSSIAN_BLUR_AMOUNT % 2 == 1)
        << "GAUSSIAN_BLUR_AMOUNT must be odd";
    EXPECT_GT(imgfx::core::GAUSSIAN_BLUR_AMOUNT, 0)
        << "GAUSSIAN_BLUR_AMOUNT must be positive";
}

TEST(GPUUtilsExtended, GPUTimingsStructure)
{
    // Test GPUTimings structure initialization and methods
    imgfx::core::GPUTimings timings;

    // Default values should be 0
    EXPECT_FLOAT_EQ(timings.h2d_ms, 0.0f);
    EXPECT_FLOAT_EQ(timings.kernel_ms, 0.0f);
    EXPECT_FLOAT_EQ(timings.d2h_ms, 0.0f);
    EXPECT_FLOAT_EQ(timings.total_ms, 0.0f);

    // Set some values
    timings.h2d_ms = 1.5f;
    timings.kernel_ms = 10.2f;
    timings.d2h_ms = 2.3f;
    timings.total_ms = 14.0f;

    EXPECT_FLOAT_EQ(timings.h2d_ms, 1.5f);
    EXPECT_FLOAT_EQ(timings.kernel_ms, 10.2f);
    EXPECT_FLOAT_EQ(timings.d2h_ms, 2.3f);
    EXPECT_FLOAT_EQ(timings.total_ms, 14.0f);

    // Test print method doesn't crash
    ASSERT_NO_THROW(timings.print());
}

TEST(GPUUtilsExtended, ApplyFilterCPUUndefinedFilter)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 0);
    std::vector<unsigned char> output(input.size());

    hipError_t result = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::UNDEFINED,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_NE(result, hipSuccess) << "Undefined filter should fail";
}

TEST(GPUUtilsExtended, ApplyFilterCPUAllFilters)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 8);
    std::vector<unsigned char> output(input.size());

    // Test grayscale
    {
        hipError_t result = imgfx::core::apply_filter_cpu(
            imgfx::core::FILTER_TYPE::GRAYSCALE,
            input.data(),
            output.data(),
            width,
            height,
            channels);
        EXPECT_EQ(result, hipSuccess) << "Grayscale CPU filter should succeed";
    }

    // Test negative
    {
        hipError_t result = imgfx::core::apply_filter_cpu(
            imgfx::core::FILTER_TYPE::NEGATIVE,
            input.data(),
            output.data(),
            width,
            height,
            channels);
        EXPECT_EQ(result, hipSuccess) << "Negative CPU filter should succeed";
    }

    // Test gaussian blur
    {
        hipError_t result = imgfx::core::apply_filter_cpu(
            imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
            input.data(),
            output.data(),
            width,
            height,
            channels);
        EXPECT_EQ(result, hipSuccess) << "Gaussian blur CPU filter should succeed";
    }
}

TEST(GPUUtilsExtended, ApplyFilterCPUDifferentChannels)
{
    constexpr int width = 16;
    constexpr int height = 16;

    // Test with 3 channels
    {
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

    // Test with 4 channels
    {
        constexpr int channels = 4;
        auto input = test_helpers::generate_solid_color_image(width, height, channels, 128, 64, 192, 255);
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
}

TEST(GPUUtilsExtended, ApplyFilterCPUSmallImage)
{
    // Test with very small image
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 255, 255, 255, 0);
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

TEST(GPUUtilsExtended, ApplyFilterCPULargeImage)
{
    // Test with larger image
    constexpr int width = 512;
    constexpr int height = 512;
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

TEST(GPUUtilsExtended, DeviceBufferConstruction)
{
    // Test DeviceBuffer can be constructed (just compilation test)
    imgfx::core::DeviceBuffer buffer;
    // Default construction should be safe
    EXPECT_TRUE(true);
}

TEST(GPUUtilsExtended, GetHIPDevicesExecution)
{
    // Test that get_hip_devices runs without crashing
    // It may return -1 if no GPU is available, which is fine
    int device_count = imgfx::core::get_hip_devices();
    EXPECT_TRUE(device_count >= -1) << "get_hip_devices should return -1 or positive count";

    if (device_count > 0)
    {
        EXPECT_GT(device_count, 0) << "Should have at least one device if successful";
    }
}

TEST(GPUUtilsExtended, ImageMetaStructFields)
{
    // Test image_meta_t structure fields
    imgfx::core::image_meta_t meta;
    meta.offset = 0;
    meta.width = 100;
    meta.height = 200;
    meta.channels = 3;

    EXPECT_EQ(meta.offset, 0);
    EXPECT_EQ(meta.width, 100);
    EXPECT_EQ(meta.height, 200);
    EXPECT_EQ(meta.channels, 3);

    // Test with different values
    meta.offset = 12345;
    meta.width = 640;
    meta.height = 480;
    meta.channels = 4;

    EXPECT_EQ(meta.offset, 12345);
    EXPECT_EQ(meta.width, 640);
    EXPECT_EQ(meta.height, 480);
    EXPECT_EQ(meta.channels, 4);
}

TEST(GPUUtilsExtended, FilterTypeEnumValues)
{
    // Test that filter type enum values are distinct
    EXPECT_NE(static_cast<int>(imgfx::core::FILTER_TYPE::UNDEFINED),
              static_cast<int>(imgfx::core::FILTER_TYPE::GRAYSCALE));
    EXPECT_NE(static_cast<int>(imgfx::core::FILTER_TYPE::GRAYSCALE),
              static_cast<int>(imgfx::core::FILTER_TYPE::NEGATIVE));
    EXPECT_NE(static_cast<int>(imgfx::core::FILTER_TYPE::NEGATIVE),
              static_cast<int>(imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR));
}
