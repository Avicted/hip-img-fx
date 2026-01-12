#include <gtest/gtest.h>
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"

/**
 * @brief Extended GPU utilities tests
 *
 * Tests for GPU device queries, filter type conversions, and error handling.
 */

TEST(GPUUtils, FilterTypeToString)
{
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::UNDEFINED), "UNDEFINED");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::GRAYSCALE), "GRAYSCALE");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::NEGATIVE), "NEGATIVE");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR), "GAUSSIAN_BLUR");

    // Test unknown filter type (cast to enum)
    auto unknown_filter = static_cast<imgfx::core::FILTER_TYPE>(999);
    EXPECT_EQ(imgfx::core::filter_type_to_string(unknown_filter), "UNKNOWN");
}

TEST(GPUUtils, GetHIPDevices)
{
    // This function prints device info, should not crash
    int device_count = imgfx::core::get_hip_devices();

    // Should return valid count (0 or positive if GPU available, -1 on error)
    EXPECT_GE(device_count, -1);
}

TEST(GPUUtils, CPUFilterGrayscale)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(width * height * channels);

    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipSuccess);

    // Verify output is modified
    bool modified = false;
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (input[i] != output[i])
        {
            modified = true;
            break;
        }
    }
    EXPECT_TRUE(modified) << "Filter should modify the output";
}

TEST(GPUUtils, CPUFilterNegative)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 0);
    std::vector<unsigned char> output(width * height * channels);

    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipSuccess);

    // Verify inversion (check a few pixels)
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(output[i * 3 + 0], 255 - input[i * 3 + 0]);
        EXPECT_EQ(output[i * 3 + 1], 255 - input[i * 3 + 1]);
        EXPECT_EQ(output[i * 3 + 2], 255 - input[i * 3 + 2]);
    }
}

TEST(GPUUtils, CPUFilterGaussianBlur)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 8);
    std::vector<unsigned char> output(width * height * channels);

    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipSuccess);

    // Output should be different from input (blurred)
    bool modified = false;
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (input[i] != output[i])
        {
            modified = true;
            break;
        }
    }
    EXPECT_TRUE(modified);
}

TEST(GPUUtils, CPUFilterUnsupportedType)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 128, 128, 128, 0);
    std::vector<unsigned char> output(width * height * channels);

    // Test with undefined filter type
    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::UNDEFINED,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipErrorInvalidValue);
}

TEST(GPUUtils, CPUFilterDifferentChannels)
{
    constexpr int width = 32;
    constexpr int height = 32;

    // Test with 1-channel image
    {
        std::vector<unsigned char> input(width * height * 1, 128);
        std::vector<unsigned char> output(width * height * 1);

        hipError_t err = imgfx::core::apply_filter_cpu(
            imgfx::core::FILTER_TYPE::NEGATIVE,
            input.data(),
            output.data(),
            width,
            height,
            1);

        EXPECT_EQ(err, hipSuccess);
        EXPECT_EQ(output[0], 255 - 128);
    }

    // Test with 4-channel image
    {
        auto input = test_helpers::generate_solid_color_image(width, height, 4, 100, 150, 200, 255);
        std::vector<unsigned char> output(width * height * 4);

        hipError_t err = imgfx::core::apply_filter_cpu(
            imgfx::core::FILTER_TYPE::GRAYSCALE,
            input.data(),
            output.data(),
            width,
            height,
            4);

        EXPECT_EQ(err, hipSuccess);
    }
}

TEST(GPUUtils, SingleImageSizeCalculations)
{
    constexpr int width = 100;
    constexpr int height = 200;
    constexpr int channels = 3;

    size_t expected_size = size_t(width) * height * channels;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    EXPECT_EQ(data.size(), expected_size);
}

TEST(GPUUtils, SmallImages)
{
    // Test with very small images
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr int channels = 3;

    auto input = test_helpers::generate_solid_color_image(width, height, channels, 50, 100, 150, 0);
    std::vector<unsigned char> output(width * height * channels);

    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::NEGATIVE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipSuccess);
}

TEST(GPUUtils, OddDimensions)
{
    // Test with odd dimensions
    constexpr int width = 33;
    constexpr int height = 47;
    constexpr int channels = 3;

    auto input = test_helpers::generate_checkerboard_image(width, height, channels, 5);
    std::vector<unsigned char> output(width * height * channels);

    hipError_t err = imgfx::core::apply_filter_cpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input.data(),
        output.data(),
        width,
        height,
        channels);

    EXPECT_EQ(err, hipSuccess);
}
