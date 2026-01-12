#include <gtest/gtest.h>
#include "../src/core/image.h"
#include "test_helpers.h"
#include <filesystem>
#include <fstream>
#include <cstdio>

/**
 * @brief Additional tests for image operations - edge cases and error handling
 *
 * These tests focus on boundary conditions, error cases, and less common scenarios
 * to improve test coverage for the image module.
 */

namespace fs = std::filesystem;

class ImageEdgeCasesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_dir_ = fs::path("/tmp/hip_img_fx_image_edge_tests");
        fs::create_directories(test_dir_);
    }

    void TearDown() override
    {
        if (fs::exists(test_dir_))
        {
            fs::remove_all(test_dir_);
        }
    }

    fs::path test_dir_;
};

TEST_F(ImageEdgeCasesTest, HasSupportedExtUpperCase)
{
    // Test uppercase extensions
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.PNG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.JPG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.JPEG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.BMP")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.TGA")));
}

TEST_F(ImageEdgeCasesTest, HasSupportedExtMixedCase)
{
    // Test mixed case extensions
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.PnG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.JpG")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("image.JpEg")));
}

TEST_F(ImageEdgeCasesTest, HasSupportedExtUnsupported)
{
    // Test unsupported extensions
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.gif")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.webp")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.svg")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image.txt")));
    EXPECT_FALSE(imgfx::core::has_supported_ext(fs::path("image")));
}

TEST_F(ImageEdgeCasesTest, HasSupportedExtAllSupported)
{
    // Test all explicitly supported extensions
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("test.jpg")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("test.jpeg")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("test.png")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("test.bmp")));
    EXPECT_TRUE(imgfx::core::has_supported_ext(fs::path("test.tga")));
}

TEST_F(ImageEdgeCasesTest, SaveImageJPEG)
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

    // Test JPEG format
    fs::path jpeg_path = test_dir_ / "test.jpg";
    bool success = imgfx::core::save_image(jpeg_path.string().c_str(), &img);
    EXPECT_TRUE(success);
    EXPECT_TRUE(fs::exists(jpeg_path));

    // Test .jpeg extension
    fs::path jpeg2_path = test_dir_ / "test.jpeg";
    success = imgfx::core::save_image(jpeg2_path.string().c_str(), &img);
    EXPECT_TRUE(success);
    EXPECT_TRUE(fs::exists(jpeg2_path));
}

TEST_F(ImageEdgeCasesTest, SaveImageBMP)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path bmp_path = test_dir_ / "test.bmp";
    bool success = imgfx::core::save_image(bmp_path.string().c_str(), &img);
    EXPECT_TRUE(success);
    EXPECT_TRUE(fs::exists(bmp_path));
}

TEST_F(ImageEdgeCasesTest, SaveImageTGA)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;

    auto data = test_helpers::generate_checkerboard_image(width, height, channels, 8);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path tga_path = test_dir_ / "test.tga";
    bool success = imgfx::core::save_image(tga_path.string().c_str(), &img);
    EXPECT_TRUE(success);
    EXPECT_TRUE(fs::exists(tga_path));
}

TEST_F(ImageEdgeCasesTest, SaveImageUnsupportedFormat)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 128, 128, 128, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Test unsupported format
    fs::path unsupported_path = test_dir_ / "test.gif";
    bool success = imgfx::core::save_image(unsupported_path.string().c_str(), &img);
    EXPECT_FALSE(success) << "Unsupported format should fail";
}

TEST_F(ImageEdgeCasesTest, SaveImageNoExtension)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 255, 0, 0, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Test no extension
    fs::path no_ext_path = test_dir_ / "test_no_extension";
    bool success = imgfx::core::save_image(no_ext_path.string().c_str(), &img);
    EXPECT_FALSE(success) << "Filename without extension should fail";
}

TEST_F(ImageEdgeCasesTest, FreeNullImage)
{
    imgfx::core::image_t img;
    img.data = nullptr;
    img.width = 0;
    img.height = 0;
    img.channels = 0;

    // Should not crash when freeing null image
    ASSERT_NO_THROW(imgfx::core::free_image(&img));
    EXPECT_EQ(img.data, nullptr);
}

TEST_F(ImageEdgeCasesTest, FreeImageTwice)
{
    constexpr int width = 16;
    constexpr int height = 16;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 64, 64, 64, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Save and load to get a stb-allocated image
    fs::path temp_path = test_dir_ / "temp_free_test.png";
    imgfx::core::save_image(temp_path.string().c_str(), &img);

    imgfx::core::image_t loaded = imgfx::core::load_image(temp_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);

    // First free
    ASSERT_NO_THROW(imgfx::core::free_image(&loaded));
    EXPECT_EQ(loaded.data, nullptr);

    // Second free should be safe
    ASSERT_NO_THROW(imgfx::core::free_image(&loaded));
}

TEST_F(ImageEdgeCasesTest, PrintImageInfo)
{
    constexpr int width = 100;
    constexpr int height = 200;
    constexpr int channels = 3;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Just verify it doesn't crash
    ASSERT_NO_THROW(imgfx::core::print_image_info(&img));
}

TEST_F(ImageEdgeCasesTest, PrintImageInfoVariousSizes)
{
    // Test various image sizes (keeping sizes reasonable for test)
    // Note: Using channels 3 and 4 (RGB and RGBA) which are properly supported
    std::vector<std::tuple<int, int, int>> sizes = {
        {1, 1, 3},
        {10, 10, 3},
        {256, 256, 4},
        {512, 512, 3}};

    for (const auto &[w, h, c] : sizes)
    {
        auto data = test_helpers::generate_solid_color_image(w, h, c, 128, 128, 128, 255);
        imgfx::core::image_t img;
        img.data = data.data();
        img.width = w;
        img.height = h;
        img.channels = c;

        ASSERT_NO_THROW(imgfx::core::print_image_info(&img));
    }
}

TEST_F(ImageEdgeCasesTest, LoadCorruptedFile)
{
    // Create a file with invalid content
    fs::path corrupt_path = test_dir_ / "corrupt.png";
    std::ofstream file(corrupt_path, std::ios::binary);
    file << "This is not a valid PNG file";
    file.close();

    imgfx::core::image_t img = imgfx::core::load_image(corrupt_path.string().c_str());
    EXPECT_EQ(img.data, nullptr) << "Loading corrupted file should fail";
    EXPECT_EQ(img.width, 0);
    EXPECT_EQ(img.height, 0);
    EXPECT_EQ(img.channels, 0);
}

TEST_F(ImageEdgeCasesTest, LoadEmptyFile)
{
    // Create an empty file
    fs::path empty_path = test_dir_ / "empty.png";
    std::ofstream file(empty_path);
    file.close();

    imgfx::core::image_t img = imgfx::core::load_image(empty_path.string().c_str());
    EXPECT_EQ(img.data, nullptr) << "Loading empty file should fail";
}

TEST_F(ImageEdgeCasesTest, SaveAndLoad4Channel)
{
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 4; // RGBA

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 100, 150, 200, 128);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path rgba_path = test_dir_ / "rgba_test.png";
    bool saved = imgfx::core::save_image(rgba_path.string().c_str(), &img);
    ASSERT_TRUE(saved);

    imgfx::core::image_t loaded = imgfx::core::load_image(rgba_path.string().c_str());
    ASSERT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, width);
    EXPECT_EQ(loaded.height, height);
    EXPECT_EQ(loaded.channels, channels);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageEdgeCasesTest, Save1x1Image)
{
    // Test smallest possible image
    constexpr int width = 1;
    constexpr int height = 1;
    constexpr int channels = 3;

    auto data = test_helpers::generate_solid_color_image(width, height, channels, 255, 128, 64, 0);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path tiny_path = test_dir_ / "tiny.png";
    bool success = imgfx::core::save_image(tiny_path.string().c_str(), &img);
    EXPECT_TRUE(success);

    // Verify it can be loaded back
    imgfx::core::image_t loaded = imgfx::core::load_image(tiny_path.string().c_str());
    EXPECT_NE(loaded.data, nullptr);
    EXPECT_EQ(loaded.width, 1);
    EXPECT_EQ(loaded.height, 1);

    imgfx::core::free_image(&loaded);
}

TEST_F(ImageEdgeCasesTest, SaveLargeImage)
{
    // Test reasonably large image
    constexpr int width = 1024;
    constexpr int height = 768;
    constexpr int channels = 3;

    auto data = test_helpers::generate_gradient_image(width, height, channels);

    imgfx::core::image_t img;
    img.data = data.data();
    img.width = width;
    img.height = height;
    img.channels = channels;

    fs::path large_path = test_dir_ / "large.png";
    bool success = imgfx::core::save_image(large_path.string().c_str(), &img);
    EXPECT_TRUE(success);
    EXPECT_TRUE(fs::exists(large_path));
}

TEST_F(ImageEdgeCasesTest, ImageMetaStruct)
{
    // Test image_meta_t structure
    imgfx::core::image_meta_t meta;
    meta.offset = 1024;
    meta.width = 640;
    meta.height = 480;
    meta.channels = 3;

    EXPECT_EQ(meta.offset, 1024);
    EXPECT_EQ(meta.width, 640);
    EXPECT_EQ(meta.height, 480);
    EXPECT_EQ(meta.channels, 3);
}
