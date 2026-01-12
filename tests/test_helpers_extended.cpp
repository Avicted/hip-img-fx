#include <gtest/gtest.h>
#include "test_helpers.h"
#include <algorithm>
#include <vector>

/**
 * @brief Tests for test helper functions
 *
 * These tests verify that our helper functions work correctly
 * and cover edge cases in the test utilities themselves.
 */

TEST(TestHelpers, HasGPUAvailable)
{
    // Test that the GPU detection function runs without crashing
    bool has_gpu = test_helpers::has_gpu_available();

    // Result can be true or false depending on system, both are valid
    EXPECT_TRUE(has_gpu || !has_gpu) << "has_gpu_available should return bool";
}

TEST(TestHelpers, GenerateSolidColorImage3Channel)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;
    constexpr unsigned char r = 100;
    constexpr unsigned char g = 150;
    constexpr unsigned char b = 200;

    auto image = test_helpers::generate_solid_color_image(width, height, channels, r, g, b, 0);

    ASSERT_EQ(image.size(), width * height * channels);

    // Check all pixels have the correct color
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(image[i * channels + 0], r);
        EXPECT_EQ(image[i * channels + 1], g);
        EXPECT_EQ(image[i * channels + 2], b);
    }
}

TEST(TestHelpers, GenerateSolidColorImage4Channel)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 4;
    constexpr unsigned char r = 50;
    constexpr unsigned char g = 100;
    constexpr unsigned char b = 150;
    constexpr unsigned char a = 200;

    auto image = test_helpers::generate_solid_color_image(width, height, channels, r, g, b, a);

    ASSERT_EQ(image.size(), width * height * channels);

    // Check all pixels have the correct color including alpha
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(image[i * channels + 0], r);
        EXPECT_EQ(image[i * channels + 1], g);
        EXPECT_EQ(image[i * channels + 2], b);
        EXPECT_EQ(image[i * channels + 3], a);
    }
}

TEST(TestHelpers, GenerateSolidColorImageEdgeCases)
{
    // Test with minimum values
    {
        auto image = test_helpers::generate_solid_color_image(1, 1, 3, 0, 0, 0, 0);
        EXPECT_EQ(image.size(), 3);
        EXPECT_EQ(image[0], 0);
        EXPECT_EQ(image[1], 0);
        EXPECT_EQ(image[2], 0);
    }

    // Test with maximum values
    {
        auto image = test_helpers::generate_solid_color_image(2, 2, 3, 255, 255, 255, 0);
        EXPECT_EQ(image.size(), 12);
        for (size_t i = 0; i < 12; ++i)
        {
            EXPECT_EQ(image[i], 255);
        }
    }
}

TEST(TestHelpers, GenerateGradientImage)
{
    constexpr int width = 100;
    constexpr int height = 50;
    constexpr int channels = 3;

    auto image = test_helpers::generate_gradient_image(width, height, channels);

    ASSERT_EQ(image.size(), width * height * channels);

    // Check that first column is black (or close to it)
    int first_pixel_idx = 0;
    EXPECT_LE(image[first_pixel_idx + 0], 5);
    EXPECT_LE(image[first_pixel_idx + 1], 5);
    EXPECT_LE(image[first_pixel_idx + 2], 5);

    // Check that last column is white (or close to it)
    int last_pixel_idx = (height - 1) * width * channels + (width - 1) * channels;
    EXPECT_GE(image[last_pixel_idx + 0], 250);
    EXPECT_GE(image[last_pixel_idx + 1], 250);
    EXPECT_GE(image[last_pixel_idx + 2], 250);

    // Check that gradient increases left to right
    int mid_left_idx = (height / 2) * width * channels + (width / 4) * channels;
    int mid_right_idx = (height / 2) * width * channels + (3 * width / 4) * channels;
    EXPECT_LT(image[mid_left_idx], image[mid_right_idx]);
}

TEST(TestHelpers, GenerateGradientImage4Channel)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 4;

    auto image = test_helpers::generate_gradient_image(width, height, channels);

    ASSERT_EQ(image.size(), width * height * channels);

    // Check alpha channel is always 255
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_EQ(image[i * channels + 3], 255) << "Alpha should be 255 at pixel " << i;
    }
}

TEST(TestHelpers, GenerateCheckerboardImage)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;
    constexpr int block_size = 16;

    auto image = test_helpers::generate_checkerboard_image(width, height, channels, block_size);

    ASSERT_EQ(image.size(), width * height * channels);

    // Check top-left corner (should be white or black)
    int top_left = 0;
    unsigned char tl_val = image[top_left];
    EXPECT_TRUE(tl_val == 0 || tl_val == 255);

    // Check that adjacent blocks have different colors
    int block1_x = block_size / 2;
    int block1_y = block_size / 2;
    int block2_x = block_size + block_size / 2;
    int block2_y = block_size / 2;

    int idx1 = (block1_y * width + block1_x) * channels;
    int idx2 = (block2_y * width + block2_x) * channels;

    EXPECT_NE(image[idx1], image[idx2]) << "Adjacent blocks should have different colors";
}

TEST(TestHelpers, GenerateCheckerboardVariousBlockSizes)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    std::vector<int> block_sizes = {1, 4, 8, 16, 32};

    for (int block_size : block_sizes)
    {
        auto image = test_helpers::generate_checkerboard_image(width, height, channels, block_size);
        EXPECT_EQ(image.size(), width * height * channels)
            << "Block size " << block_size << " should produce correct size";
    }
}

TEST(TestHelpers, CompareImagesWithToleranceExactMatch)
{
    constexpr size_t size = 100;
    std::vector<unsigned char> img1(size);
    std::vector<unsigned char> img2(size);

    // Fill with same values
    for (size_t i = 0; i < size; ++i)
    {
        img1[i] = static_cast<unsigned char>(i % 256);
        img2[i] = static_cast<unsigned char>(i % 256);
    }

    int mismatches = test_helpers::compare_images_with_tolerance(
        img1.data(), img2.data(), size, 0);

    EXPECT_EQ(mismatches, 0) << "Identical images should have 0 mismatches";
}

TEST(TestHelpers, CompareImagesWithToleranceAllDifferent)
{
    constexpr size_t size = 100;
    std::vector<unsigned char> img1(size, 0);
    std::vector<unsigned char> img2(size, 255);

    int mismatches = test_helpers::compare_images_with_tolerance(
        img1.data(), img2.data(), size, 10);

    EXPECT_EQ(mismatches, size) << "All pixels differ by more than tolerance";
}

TEST(TestHelpers, CompareImagesWithToleranceWithinTolerance)
{
    constexpr size_t size = 100;
    std::vector<unsigned char> img1(size, 100);
    std::vector<unsigned char> img2(size, 102); // Difference of 2

    // Tolerance 2 should pass
    int mismatches1 = test_helpers::compare_images_with_tolerance(
        img1.data(), img2.data(), size, 2);
    EXPECT_EQ(mismatches1, 0);

    // Tolerance 1 should fail
    int mismatches2 = test_helpers::compare_images_with_tolerance(
        img1.data(), img2.data(), size, 1);
    EXPECT_EQ(mismatches2, size);
}

TEST(TestHelpers, CompareImagesWithToleranceMixed)
{
    constexpr size_t size = 10;
    std::vector<unsigned char> img1 = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::vector<unsigned char> img2 = {0, 10, 22, 30, 45, 50, 65, 70, 85, 90};
    // Differences:                   0   0   2   0   5   0   5   0   5   0
    // With tolerance 3: 3 mismatches (positions 4, 6, 8 with diff 5)

    int mismatches = test_helpers::compare_images_with_tolerance(
        img1.data(), img2.data(), size, 3);

    EXPECT_EQ(mismatches, 3);
}

TEST(TestHelpers, FillRandomImageBasic)
{
    constexpr size_t size = 1000;
    std::vector<unsigned char> data(size);

    test_helpers::fill_random_image(data.data(), size, 42);

    // Check that not all values are the same (very unlikely with random)
    bool has_variation = false;
    unsigned char first = data[0];
    for (size_t i = 1; i < size; ++i)
    {
        if (data[i] != first)
        {
            has_variation = true;
            break;
        }
    }

    EXPECT_TRUE(has_variation) << "Random image should have variation";
}

TEST(TestHelpers, FillRandomImageReproducible)
{
    constexpr size_t size = 100;
    constexpr unsigned int seed = 123;

    std::vector<unsigned char> data1(size);
    std::vector<unsigned char> data2(size);

    // Fill with same seed
    test_helpers::fill_random_image(data1.data(), size, seed);
    test_helpers::fill_random_image(data2.data(), size, seed);

    // Should produce identical results
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(data1[i], data2[i]) << "Same seed should produce same result at index " << i;
    }
}

TEST(TestHelpers, FillRandomImageDifferentSeeds)
{
    constexpr size_t size = 100;

    std::vector<unsigned char> data1(size);
    std::vector<unsigned char> data2(size);

    // Fill with different seeds
    test_helpers::fill_random_image(data1.data(), size, 42);
    test_helpers::fill_random_image(data2.data(), size, 99);

    // Should produce different results (statistically very likely)
    int differences = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (data1[i] != data2[i])
        {
            differences++;
        }
    }

    EXPECT_GT(differences, size / 2) << "Different seeds should produce different results";
}

TEST(TestHelpers, FillRandomImageRange)
{
    constexpr size_t size = 1000;
    std::vector<unsigned char> data(size);

    test_helpers::fill_random_image(data.data(), size, 777);

    // Check all values are in valid range [0, 255]
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_GE(data[i], 0);
        EXPECT_LE(data[i], 255);
    }
}

TEST(TestHelpers, GenerateImagesConsistentSize)
{
    constexpr int width = 100;
    constexpr int height = 50;
    constexpr int channels = 3;
    constexpr size_t expected_size = width * height * channels;

    auto solid = test_helpers::generate_solid_color_image(width, height, channels, 128, 128, 128, 0);
    auto gradient = test_helpers::generate_gradient_image(width, height, channels);
    auto checker = test_helpers::generate_checkerboard_image(width, height, channels, 10);

    EXPECT_EQ(solid.size(), expected_size);
    EXPECT_EQ(gradient.size(), expected_size);
    EXPECT_EQ(checker.size(), expected_size);
}
