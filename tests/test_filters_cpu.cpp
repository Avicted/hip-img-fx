#include <gtest/gtest.h>
#include "../src/filters/filters.h"
#include "test_helpers.h"
#include <vector>
#include <cmath>

/**
 * @brief CPU filter tests - these always run regardless of GPU availability
 *
 * These tests validate the correctness of CPU filter implementations
 * using known input/output values and edge cases.
 */

TEST(FiltersCPU, GrayscaleKnownValues)
{
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr int channels = 3;
    constexpr size_t total_size = width * height * channels;

    unsigned char input[total_size];
    unsigned char output[total_size];

    // Test with known RGB values
    // Expected grayscale formula: 0.21*R + 0.72*G + 0.07*B
    for (int i = 0; i < width * height; ++i)
    {
        input[i * channels + 0] = 100; // R
        input[i * channels + 1] = 150; // G
        input[i * channels + 2] = 200; // B
    }

    imgfx::filters::grayscale_cpu(input, output, width, height, channels);

    // Expected: 0.21*100 + 0.72*150 + 0.07*200 = 21 + 108 + 14 = 143
    unsigned char expected = 143;

    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_NEAR(output[i * channels + 0], expected, 1) << "R channel mismatch at pixel " << i;
        EXPECT_NEAR(output[i * channels + 1], expected, 1) << "G channel mismatch at pixel " << i;
        EXPECT_NEAR(output[i * channels + 2], expected, 1) << "B channel mismatch at pixel " << i;
    }
}

TEST(FiltersCPU, NegativeInversion)
{
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr int channels = 3;
    constexpr size_t total_size = width * height * channels;

    unsigned char input[total_size];
    unsigned char output[total_size];

    // Fill with known values
    for (int i = 0; i < width * height; ++i)
    {
        input[i * channels + 0] = 50;
        input[i * channels + 1] = 100;
        input[i * channels + 2] = 200;
    }

    imgfx::filters::negative_cpu(input, output, width, height, channels);

    // Expected: 255 - value for each channel
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(output[i * channels + 0], 255 - 50) << "R channel at pixel " << i;
        EXPECT_EQ(output[i * channels + 1], 255 - 100) << "G channel at pixel " << i;
        EXPECT_EQ(output[i * channels + 2], 255 - 200) << "B channel at pixel " << i;
    }
}

TEST(FiltersCPU, GaussianBlurSmoothness)
{
    // Create a checkerboard pattern - blur should smooth it
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 8);
    std::vector<unsigned char> output(input.size());

    imgfx::filters::gaussian_blur_cpu(input.data(), output.data(),
                                      width, height, channels, 11);

    // Calculate variance before and after blur
    auto calc_variance = [](const std::vector<unsigned char> &img, int channels) -> double
    {
        double sum = 0.0;
        int count = img.size() / channels;

        for (size_t i = 0; i < img.size(); i += channels)
        {
            sum += img[i]; // Use R channel
        }

        double mean = sum / count;
        double variance = 0.0;

        for (size_t i = 0; i < img.size(); i += channels)
        {
            double diff = img[i] - mean;
            variance += diff * diff;
        }

        return variance / count;
    };

    double variance_before = calc_variance(input, channels);
    double variance_after = calc_variance(output, channels);

    // Blur should reduce variance (smoothing effect)
    EXPECT_LT(variance_after, variance_before)
        << "Blur should reduce image variance (smoothing)";
}

TEST(FiltersCPU, EdgeCases_1x1Image)
{
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;
    constexpr size_t total_size = width * height * channels;

    unsigned char input[total_size] = {100, 150, 200};
    unsigned char output[total_size];

    // Test grayscale
    imgfx::filters::grayscale_cpu(input, output, width, height, channels);
    EXPECT_GT(output[0], 0) << "Grayscale should produce non-zero output";

    // Test negative
    imgfx::filters::negative_cpu(input, output, width, height, channels);
    EXPECT_EQ(output[0], 155) << "Negative of 100 should be 155";
    EXPECT_EQ(output[1], 105) << "Negative of 150 should be 105";
    EXPECT_EQ(output[2], 55) << "Negative of 200 should be 55";
}

TEST(FiltersCPU, EdgeCases_LargeImage)
{
    constexpr int width = 2048;
    constexpr int height = 2048;
    constexpr int channels = 3;
    constexpr size_t total_size = width * height * channels;

    std::vector<unsigned char> input(total_size);
    std::vector<unsigned char> output(total_size);

    // Fill with gradient - use non-zero values
    for (int i = 0; i < width * height; ++i)
    {
        unsigned char val = static_cast<unsigned char>((i % 255) + 1); // 1-255, never 0
        input[i * channels + 0] = val;
        input[i * channels + 1] = val;
        input[i * channels + 2] = val;
    }

    // Should not crash or produce errors
    ASSERT_NO_THROW(
        imgfx::filters::grayscale_cpu(input.data(), output.data(), width, height, channels));

    // Verify output (since input is all same gray value, output should be same)
    EXPECT_GT(output[0], 0) << "First pixel should be non-zero";
    EXPECT_GT(output[total_size - channels], 0) << "Last pixel should be non-zero";
}

TEST(FiltersCPU, ThreeChannelRGB)
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels,
                                                          255, 0, 0, 0);
    std::vector<unsigned char> output(input.size());

    // Pure red should produce grayscale with R coefficient
    imgfx::filters::grayscale_cpu(input.data(), output.data(), width, height, channels);

    // 0.21 * 255 ≈ 53
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_NEAR(output[i * channels + 0], 53, 2);
    }
}

TEST(FiltersCPU, FourChannelRGBA)
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int channels = 4;

    auto input = test_helpers::generate_solid_color_image(width, height, channels,
                                                          100, 150, 200, 128);
    std::vector<unsigned char> output(input.size());

    // Grayscale should preserve alpha channel
    imgfx::filters::grayscale_cpu(input.data(), output.data(), width, height, channels);

    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(output[i * channels + 3], 128)
            << "Alpha channel should be preserved at pixel " << i;
    }
}
