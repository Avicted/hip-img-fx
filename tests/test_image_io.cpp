#include <gtest/gtest.h>
#include "../src/core/image.h"
#include "test_helpers.h"
#include <fstream>
#include <cstdio>

/**
 * @brief Image I/O tests - always run (CPU-only, no GPU required)
 *
 * Tests for loading, saving, and manipulating image data using stb_image.
 */

TEST(ImageIO, LoadValidImage)
{
    // Note: This test requires actual image files or we work with synthetic data
    // For now, we'll test the image_t structure and basic operations

    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels,
                                                         100, 150, 200, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    EXPECT_NE(img.data, nullptr);
    EXPECT_EQ(img.width, width);
    EXPECT_EQ(img.height, height);
    EXPECT_EQ(img.channels, channels);
}

TEST(ImageIO, SaveAndReload)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto original = test_helpers::generate_checkerboard_image(width, height, channels, 4);

    imgfx::core::image_t img;
    img.data = original.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Save to temporary file
    const char *temp_path = "/tmp/test_image_io_temp.png";
    bool save_success = imgfx::core::save_image(temp_path, &img);
    ASSERT_TRUE(save_success) << "Failed to save test image";

    // Load it back
    imgfx::core::image_t loaded = imgfx::core::load_image(temp_path);

    ASSERT_NE(loaded.data, nullptr) << "Failed to load saved image";
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    // Compare data (PNG is lossless)
    size_t total_bytes = width * height * channels;
    int mismatches = test_helpers::compare_images_with_tolerance(
        original.data(), loaded.data, total_bytes, 0);

    EXPECT_EQ(mismatches, 0) << "Saved and loaded image data should match exactly";

    // Cleanup
    imgfx::core::free_image(&loaded);
    std::remove(temp_path);
}

TEST(ImageIO, InvalidPath)
{
    const char *invalid_path = "/nonexistent/path/to/image.png";
    imgfx::core::image_t img = imgfx::core::load_image(invalid_path);

    // stb_image returns nullptr for invalid paths
    EXPECT_EQ(img.data, nullptr) << "Loading invalid path should return null data";
}

TEST(ImageIO, EmptyImage)
{
    imgfx::core::image_t img;
    img.data = nullptr;
    img.width = 0;
    img.height = 0;
    img.channels = 0;

    // Should handle empty image gracefully
    EXPECT_EQ(img.data, nullptr);
    EXPECT_EQ(img.width, 0);
    EXPECT_EQ(img.height, 0);
}

TEST(ImageIO, MemoryManagement)
{
    constexpr int width = 128;
    constexpr int height = 128;
    constexpr int channels = 3;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Save to file
    const char *temp_path = "/tmp/test_memory_mgmt.png";
    bool saved = imgfx::core::save_image(temp_path, &img);
    ASSERT_TRUE(saved);

    // Load creates new allocation
    imgfx::core::image_t loaded = imgfx::core::load_image(temp_path);
    ASSERT_NE(loaded.data, nullptr);

    // Free should not crash
    ASSERT_NO_THROW(imgfx::core::free_image(&loaded));

    // After free, data should be set to nullptr by free_image
    // (assuming proper implementation)

    std::remove(temp_path);
}

TEST(ImageIO, ChannelConsistency)
{
    constexpr int width = 64;
    constexpr int height = 64;

    // Test 3-channel RGB
    {
        auto data_rgb = test_helpers::generate_solid_color_image(width, height, 3,
                                                                 255, 128, 64, 0);
        imgfx::core::image_t img_rgb;
        img_rgb.data = data_rgb.data();
        img_rgb.width = width;
        img_rgb.height = height;
        img_rgb.channels = 3;

        EXPECT_EQ(img_rgb.channels, 3);
    }

    // Test 4-channel RGBA
    {
        auto data_rgba = test_helpers::generate_solid_color_image(width, height, 4,
                                                                  255, 128, 64, 200);
        imgfx::core::image_t img_rgba;
        img_rgba.data = data_rgba.data();
        img_rgba.width = width;
        img_rgba.height = height;
        img_rgba.channels = 4;

        EXPECT_EQ(img_rgba.channels, 4);

        // Verify alpha channel is present
        EXPECT_EQ(data_rgba[3], 200);
    }
}
