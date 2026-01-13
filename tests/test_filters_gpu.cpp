// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/core/gpu_utils.h"
#include "../src/filters/filters.h"
#include "test_helpers.h"
#include <vector>

/**
 * @brief GPU filter tests - these skip if no GPU is available
 *
 * These tests validate GPU filter correctness by comparing against
 * CPU reference implementations.
 */

class GPUFilterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!test_helpers::has_gpu_available())
        {
            GTEST_SKIP() << "GPU not available - skipping GPU filter test";
        }
    }
};

TEST_F(GPUFilterTest, GrayscaleGPUvsCPU)
{
    constexpr int width = 256;
    constexpr int height = 256;
    constexpr int channels = 3;
    constexpr size_t total_bytes = width * height * channels;

    // Generate test image
    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> cpu_output(total_bytes);
    std::vector<unsigned char> gpu_output(total_bytes);

    // CPU reference
    imgfx::filters::grayscale_cpu(input.data(), cpu_output.data(),
                                  width, height, channels);

    // GPU execution
    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output_img{gpu_output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        gpu_input, gpu_output_img,
        false, nullptr);

    ASSERT_EQ(err, hipSuccess) << "GPU filter execution failed";

    // Compare outputs (allow ±1 for floating-point rounding)
    int mismatches = test_helpers::compare_images_with_tolerance(
        cpu_output.data(), gpu_output.data(), total_bytes, 1);

    EXPECT_EQ(mismatches, 0)
        << "GPU and CPU grayscale outputs differ in " << mismatches << " bytes";
}

TEST_F(GPUFilterTest, NegativeGPUvsCPU)
{
    constexpr int width = 256;
    constexpr int height = 256;
    constexpr int channels = 3;
    constexpr size_t total_bytes = width * height * channels;

    // Generate checkerboard pattern
    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 16);
    std::vector<unsigned char> cpu_output(total_bytes);
    std::vector<unsigned char> gpu_output(total_bytes);

    // CPU reference
    imgfx::filters::negative_cpu(input.data(), cpu_output.data(),
                                 width, height, channels);

    // GPU execution
    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output_img{gpu_output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        gpu_input, gpu_output_img,
        false, nullptr);

    ASSERT_EQ(err, hipSuccess) << "GPU filter execution failed";

    // Negative should be exact (no floating-point operations)
    int mismatches = test_helpers::compare_images_with_tolerance(
        cpu_output.data(), gpu_output.data(), total_bytes, 0);

    EXPECT_EQ(mismatches, 0)
        << "GPU and CPU negative outputs differ in " << mismatches << " bytes";
}

TEST_F(GPUFilterTest, GaussianBlurGPUvsCPU)
{
    constexpr int width = 128;
    constexpr int height = 128;
    constexpr int channels = 3;
    constexpr size_t total_bytes = width * height * channels;

    // Generate solid color with some variation
    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 8);
    std::vector<unsigned char> cpu_output(total_bytes);
    std::vector<unsigned char> gpu_output(total_bytes);

    // CPU reference
    imgfx::filters::gaussian_blur_cpu(input.data(), cpu_output.data(),
                                      width, height, channels,
                                      imgfx::core::GAUSSIAN_BLUR_AMOUNT);

    // GPU execution
    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output_img{gpu_output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        gpu_input, gpu_output_img,
        false, nullptr);

    ASSERT_EQ(err, hipSuccess) << "GPU filter execution failed";

    // Blur involves floating-point, allow ±1 tolerance
    int mismatches = test_helpers::compare_images_with_tolerance(
        cpu_output.data(), gpu_output.data(), total_bytes, 1);

    EXPECT_EQ(mismatches, 0)
        << "GPU and CPU blur outputs differ in " << mismatches << " bytes";
}

TEST_F(GPUFilterTest, GPU_EdgeCase_1x1)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels,
                                                          100, 150, 200, 0);
    std::vector<unsigned char> output(input.size());

    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output{output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        gpu_input, gpu_output,
        false, nullptr);

    EXPECT_EQ(err, hipSuccess) << "1x1 image should process successfully";
    EXPECT_GT(output[0], 0) << "Output should be non-zero";
}

TEST_F(GPUFilterTest, GPU_EdgeCase_4096x4096)
{
    constexpr int width = 4096;
    constexpr int height = 4096;
    constexpr int channels = 3;
    constexpr size_t total_bytes = width * height * channels;

    std::vector<unsigned char> input(total_bytes);
    std::vector<unsigned char> output(total_bytes);

    // Fill with simple pattern
    for (size_t i = 0; i < total_bytes; ++i)
    {
        input[i] = static_cast<unsigned char>(i % 256);
    }

    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output{output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        gpu_input, gpu_output,
        false, nullptr);

    EXPECT_EQ(err, hipSuccess) << "Large 4096x4096 image should process successfully";
}

TEST_F(GPUFilterTest, GPU_AlphaChannelPreservation)
{
    constexpr int width = 128;
    constexpr int height = 128;
    constexpr int channels = 4;
    constexpr size_t total_bytes = width * height * channels;

    auto input = test_helpers::generate_solid_color_image(width, height, channels,
                                                          100, 150, 200, 128);
    std::vector<unsigned char> output(total_bytes);

    imgfx::core::image_t gpu_input{input.data(), width, height, channels};
    imgfx::core::image_t gpu_output{output.data(), width, height, channels};

    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        gpu_input, gpu_output,
        false, nullptr);

    ASSERT_EQ(err, hipSuccess);

    // Check alpha channel is preserved
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(output[i * channels + 3], 128)
            << "Alpha channel should be preserved at pixel " << i;
    }
}
