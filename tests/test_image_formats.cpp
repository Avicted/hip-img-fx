#include <gtest/gtest.h>
#include "../src/core/image.h"
#include "test_helpers.h"
#include <filesystem>
#include <fstream>

/**
 * @brief Extended image format and edge case tests
 *
 * Tests various image formats (JPG, BMP, TGA) and edge cases for better coverage.
 */

namespace fs = std::filesystem;

class ImageFormatsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create temp directory for tests
        test_dir_ = fs::path("/tmp/hip_img_fx_format_tests");
        fs::create_directories(test_dir_);
    }

    void TearDown() override
    {
        // Cleanup test directory
        if (fs::exists(test_dir_))
        {
            fs::remove_all(test_dir_);
        }
    }

    fs::path test_dir_;
};

TEST_F(ImageFormatsTest, SaveLoadPNG)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "test.png";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved) << "Failed to save PNG";

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, SaveLoadJPG)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 200, 100, 50, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "test.jpg";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved) << "Failed to save JPG";

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    // JPG is lossy, so allow some tolerance
    size_t total_bytes = width * height * channels;
    int mismatches = test_helpers::compare_images_with_tolerance(
        data.data(), loaded.data, total_bytes, 30); // Higher tolerance for JPG

    EXPECT_LT(mismatches, total_bytes * 0.1) << "Too many mismatches in JPG compression";

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, SaveLoadJPEG)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto data = test_helpers::generate_checkerboard_image(width, height, channels, 4);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Test .jpeg extension (not just .jpg)
    fs::path test_path = test_dir_ / "test.jpeg";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved) << "Failed to save JPEG";

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, SaveLoadBMP)
{
    constexpr int width = 48;
    constexpr int height = 48;
    constexpr int channels = 3;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "test.bmp";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved) << "Failed to save BMP";

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, SaveLoadTGA)
{
    constexpr int width = 40;
    constexpr int height = 40;
    constexpr int channels = 4;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 150, 200, 100, 255);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "test.tga";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved) << "Failed to save TGA";

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, UnsupportedFormat)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 100, 100, 100, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Try to save with unsupported extension
    fs::path test_path = test_dir_ / "test.xyz";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    EXPECT_FALSE(saved) << "Should fail with unsupported format";
}

TEST_F(ImageFormatsTest, NoExtension)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 50, 50, 50, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // File without extension
    fs::path test_path = test_dir_ / "test_no_ext";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    EXPECT_FALSE(saved) << "Should fail without file extension";
}

TEST_F(ImageFormatsTest, PrintImageInfo)
{
    constexpr int width = 100;
    constexpr int height = 200;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 128, 128, 128, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Should not crash
    ASSERT_NO_THROW(imgfx::core::print_image_info(&img));
}

TEST_F(ImageFormatsTest, FreeNullImage)
{
    imgfx::core::image_t img;
    img.data = nullptr;
    img.width = 0;
    img.height = 0;
    img.channels = 0;

    // Should handle null gracefully
    ASSERT_NO_THROW(imgfx::core::free_image(&img));
}

TEST_F(ImageFormatsTest, HasSupportedExtension)
{
    // Test supported extensions
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.jpg")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.jpeg")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.png")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.bmp")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.tga")));

    // Test case insensitivity
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.JPG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.PNG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.BMP")));

    // Test unsupported extensions
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.xyz")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.txt")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("no_extension")));
}

TEST_F(ImageFormatsTest, LargeImage)
{
    constexpr int width = 512;
    constexpr int height = 512;
    constexpr int channels = 4;

    auto data = test_helpers::generate_checkerboard_image(width, height, channels, 32);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "large.png";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved);

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageFormatsTest, GrayscaleImage)
{
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 1;

    // Create grayscale gradient
    std::vector<unsigned char> data(width * height * channels);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            data[idx] = static_cast<unsigned char>((x * 255) / width);
        }
    }

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path test_path = test_dir_ / "grayscale.png";
    bool saved = imgfx::core::save_image(test_path.string().c_str(), &img);
    ASSERT_TRUE(saved);

    imgfx::core::image_t loaded = imgfx::core::load_image(test_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    imgfx::core::free_image(&loaded);
}
