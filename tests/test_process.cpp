// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/app/process.h"
#include "../src/core/image.h"
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <filesystem>
#include <fstream>

/**
 * @brief Tests for process.cpp functions
 *
 * Tests batch processing, error handling, and CPU/GPU processing paths.
 */

namespace fs = std::filesystem;

class ProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_dir_ = fs::path("/tmp/hip_img_fx_process_tests");
        fs::create_directories(test_dir_);

        // Create a test input image
        createTestImage();
    }

    void TearDown() override
    {
        if (fs::exists(test_dir_))
        {
            fs::remove_all(test_dir_);
        }
    }

    void createTestImage()
    {
        constexpr int width = 64;
        constexpr int height = 64;
        constexpr int channels = 3;

        auto data = test_helpers::generate_checkerboard_image(width, height, channels, 8);

        imgfx::core::image_t img;
        img.data = data.data();
        img.width = width;
        img.height = height;
        img.channels = channels;

        test_image_path_ = test_dir_ / "test_input.png";
        imgfx::core::save_image(test_image_path_.string().c_str(), &img);
    }

    fs::path test_dir_;
    fs::path test_image_path_;
};

TEST_F(ProcessTest, ProcessOneCPUSuccess)
{
    fs::path output_path = test_dir_ / "output_cpu.png";

    int result = imgfx::app::process_one_cpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0) << "CPU processing should succeed";
    EXPECT_TRUE(fs::exists(output_path)) << "Output file should be created";
}

TEST_F(ProcessTest, ProcessOneCPUInvalidInput)
{
    fs::path invalid_input = test_dir_ / "nonexistent.png";
    fs::path output_path = test_dir_ / "output.png";

    int result = imgfx::app::process_one_cpu(
        false,
        invalid_input.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with invalid input";
}

TEST_F(ProcessTest, ProcessOneCPUNegativeFilter)
{
    fs::path output_path = test_dir_ / "output_negative.png";

    int result = imgfx::app::process_one_cpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessTest, ProcessOneCPUGaussianBlur)
{
    fs::path output_path = test_dir_ / "output_blur.png";

    int result = imgfx::app::process_one_cpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessTest, ProcessOneCPUBatchMode)
{
    fs::path output_path = test_dir_ / "output_batch.png";

    // Test running_as_batch flag (should not print success message)
    int result = imgfx::app::process_one_cpu(
        true, // running_as_batch
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessTest, ProcessBatchCPUSuccess)
{
    // Create multiple test images
    std::vector<std::string> input_files;
    for (int i = 0; i < 3; ++i)
    {
        fs::path img_path = test_dir_ / ("input_" + std::to_string(i) + ".png");

        constexpr int width = 32;
        constexpr int height = 32;
        constexpr int channels = 3;

        auto data = test_helpers::generate_solid_color_image(
            width, height, channels,
            i * 50, i * 60, i * 70, 0);

        imgfx::core::image_t img;
        img.data = data.data();
        img.width = width;
        img.height = height;
        img.channels = channels;

        imgfx::core::save_image(img_path.string().c_str(), &img);
        input_files.push_back(img_path.string());
    }

    fs::path output_dir = test_dir_ / "output_batch";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);

    // Check output files exist
    for (int i = 0; i < 3; ++i)
    {
        fs::path expected_output = output_dir / ("input_" + std::to_string(i) + ".png");
        EXPECT_TRUE(fs::exists(expected_output)) << "Output file " << i << " should exist";
    }
}

TEST_F(ProcessTest, ProcessBatchCPUEmptyList)
{
    std::vector<std::string> empty_list;
    fs::path output_dir = test_dir_ / "output_empty";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        empty_list,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0) << "Should handle empty list gracefully";
}

// GPU tests - only run if GPU is available

class ProcessGPUTest : public ProcessTest
{
protected:
    void SetUp() override
    {
        ProcessTest::SetUp();

        // Check if GPU is available
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No GPU available, skipping GPU tests";
        }
    }
};

TEST_F(ProcessGPUTest, ProcessOneGPUSuccess)
{
    fs::path output_path = test_dir_ / "output_gpu.png";

    int result = imgfx::app::process_one_gpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0) << "GPU processing should succeed";
    EXPECT_TRUE(fs::exists(output_path)) << "Output file should be created";
}

TEST_F(ProcessGPUTest, ProcessOneGPUInvalidInput)
{
    fs::path invalid_input = test_dir_ / "nonexistent.png";
    fs::path output_path = test_dir_ / "output.png";

    int result = imgfx::app::process_one_gpu(
        false,
        invalid_input.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with invalid input";
}

TEST_F(ProcessGPUTest, ProcessOneGPUNegativeFilter)
{
    fs::path output_path = test_dir_ / "output_gpu_negative.png";

    int result = imgfx::app::process_one_gpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessGPUTest, ProcessOneGPUGaussianBlur)
{
    fs::path output_path = test_dir_ / "output_gpu_blur.png";

    int result = imgfx::app::process_one_gpu(
        false,
        test_image_path_.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessGPUTest, ProcessBatchGPUSuccess)
{
    // Create multiple test images
    std::vector<std::string> input_files;
    for (int i = 0; i < 4; ++i)
    {
        fs::path img_path = test_dir_ / ("batch_input_" + std::to_string(i) + ".png");

        constexpr int width = 64;
        constexpr int height = 64;
        constexpr int channels = 3;

        auto data = test_helpers::generate_gradient_image(width, height, channels);

        imgfx::core::image_t img;
        img.data = data.data();
        img.width = width;
        img.height = height;
        img.channels = channels;

        imgfx::core::save_image(img_path.string().c_str(), &img);
        input_files.push_back(img_path.string());
    }

    fs::path output_dir = test_dir_ / "output_batch_gpu";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        2); // batch_size = 2

    EXPECT_EQ(result, 0);

    // Check output files exist
    for (int i = 0; i < 4; ++i)
    {
        fs::path expected_output = output_dir / ("batch_input_" + std::to_string(i) + ".png");
        EXPECT_TRUE(fs::exists(expected_output)) << "Output file " << i << " should exist";
    }
}

TEST_F(ProcessGPUTest, ProcessBatchGPUInvalidBatchSize)
{
    std::vector<std::string> input_files = {test_image_path_.string()};
    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        0); // Invalid batch_size

    EXPECT_EQ(result, -1) << "Should fail with invalid batch size";
}

TEST_F(ProcessGPUTest, ProcessBatchGPUNegativeBatchSize)
{
    std::vector<std::string> input_files = {test_image_path_.string()};
    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        -5); // Negative batch_size

    EXPECT_EQ(result, -1) << "Should fail with negative batch size";
}

TEST_F(ProcessGPUTest, ProcessBatchGPULargeBatchSize)
{
    // Create 2 test images
    std::vector<std::string> input_files;
    for (int i = 0; i < 2; ++i)
    {
        fs::path img_path = test_dir_ / ("large_batch_" + std::to_string(i) + ".png");

        constexpr int width = 32;
        constexpr int height = 32;
        constexpr int channels = 3;

        auto data = test_helpers::generate_checkerboard_image(width, height, channels, 4);

        imgfx::core::image_t img;
        img.data = data.data();
        img.width = width;
        img.height = height;
        img.channels = channels;

        imgfx::core::save_image(img_path.string().c_str(), &img);
        input_files.push_back(img_path.string());
    }

    fs::path output_dir = test_dir_ / "output_large_batch";
    fs::create_directories(output_dir);

    // Batch size larger than number of images
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE,
        100); // Much larger than input count

    EXPECT_EQ(result, 0) << "Should handle large batch size gracefully";
}
