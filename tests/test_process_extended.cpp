// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/app/process.h"
#include "../src/core/image.h"
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <filesystem>
#include <fstream>
#include <vector>

/**
 * @brief Extended tests for process module - edge cases and error handling
 *
 * These tests cover additional scenarios for batch processing, error conditions,
 * and edge cases to improve coverage of the process module.
 */

namespace fs = std::filesystem;

class ProcessExtendedTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_dir_ = fs::path("/tmp/hip_img_fx_process_extended");
        fs::create_directories(test_dir_);
    }

    void TearDown() override
    {
        if (fs::exists(test_dir_))
        {
            fs::remove_all(test_dir_);
        }
    }

    void createTestImage(const std::string &name, int width = 64, int height = 64)
    {
        constexpr int channels = 3;
        auto data = test_helpers::generate_checkerboard_image(width, height, channels, 8);

        imgfx::core::image_t img;
        img.data = data.data();
        img.width = width;
        img.height = height;
        img.channels = channels;

        fs::path path = test_dir_ / name;
        imgfx::core::save_image(path.string().c_str(), &img);
    }

    fs::path test_dir_;
};

TEST_F(ProcessExtendedTest, ProcessOneCPUWithBatchFlag)
{
    createTestImage("input.png");
    fs::path input_path = test_dir_ / "input.png";
    fs::path output_path = test_dir_ / "output.png";

    // Test with running_as_batch = true
    int result = imgfx::app::process_one_cpu(
        true, // running_as_batch
        input_path.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessExtendedTest, ProcessOneCPUInvalidOutputPath)
{
    createTestImage("input.png");
    fs::path input_path = test_dir_ / "input.png";

    // Use an invalid output path (directory that doesn't exist)
    std::string invalid_output = "/nonexistent/directory/output.png";

    int result = imgfx::app::process_one_cpu(
        false,
        input_path.string(),
        invalid_output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    // Should fail because output directory doesn't exist
    EXPECT_EQ(result, -1);
}

TEST_F(ProcessExtendedTest, ProcessOneCPUAllFilters)
{
    createTestImage("input.png");
    fs::path input_path = test_dir_ / "input.png";

    // Test all filter types
    std::vector<imgfx::core::FILTER_TYPE> filters = {
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR};

    for (size_t i = 0; i < filters.size(); ++i)
    {
        fs::path output_path = test_dir_ / ("output_" + std::to_string(i) + ".png");
        int result = imgfx::app::process_one_cpu(
            false,
            input_path.string(),
            output_path.string(),
            filters[i]);

        EXPECT_EQ(result, 0) << "Filter " << i << " should succeed";
        EXPECT_TRUE(fs::exists(output_path)) << "Output for filter " << i << " should exist";
    }
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUInvalidBatchSize)
{
    // Create test images
    createTestImage("img1.png");
    createTestImage("img2.png");

    std::vector<std::string> input_files = {
        (test_dir_ / "img1.png").string(),
        (test_dir_ / "img2.png").string()};

    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    // Test with invalid batch size (0)
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        0); // Invalid batch size

    EXPECT_EQ(result, -1) << "Batch size 0 should fail";

    // Test with negative batch size
    result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        -5); // Invalid batch size

    EXPECT_EQ(result, -1) << "Negative batch size should fail";
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUEmptyList)
{
    std::vector<std::string> empty_files;
    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    // Should handle empty list gracefully
    int result = imgfx::app::process_batch_cpu(
        empty_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0) << "Empty batch should succeed (no-op)";
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUSingleImage)
{
    createTestImage("single.png");

    std::vector<std::string> input_files = {
        (test_dir_ / "single.png").string()};

    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_dir / "single.png"));
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUMultipleImages)
{
    // Create multiple test images
    for (int i = 0; i < 5; ++i)
    {
        createTestImage("img" + std::to_string(i) + ".png", 32, 32);
    }

    std::vector<std::string> input_files;
    for (int i = 0; i < 5; ++i)
    {
        input_files.push_back((test_dir_ / ("img" + std::to_string(i) + ".png")).string());
    }

    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);

    // Verify all output files were created
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(fs::exists(output_dir / ("img" + std::to_string(i) + ".png")))
            << "Output file " << i << " should exist";
    }
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUSomeInvalidFiles)
{
    createTestImage("valid1.png");
    createTestImage("valid2.png");

    std::vector<std::string> input_files = {
        (test_dir_ / "valid1.png").string(),
        (test_dir_ / "nonexistent.png").string(), // Invalid file
        (test_dir_ / "valid2.png").string()};

    fs::path output_dir = test_dir_ / "output";
    fs::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0) << "Should process valid files even with some invalid";

    // Valid files should be processed
    EXPECT_TRUE(fs::exists(output_dir / "valid1.png"));
    EXPECT_TRUE(fs::exists(output_dir / "valid2.png"));
}

TEST_F(ProcessExtendedTest, ProcessOneCPUDifferentImageSizes)
{
    // Test with various image sizes
    std::vector<std::tuple<int, int>> sizes = {
        {16, 16},
        {64, 32},
        {100, 200},
        {256, 256}};

    for (size_t i = 0; i < sizes.size(); ++i)
    {
        auto [w, h] = sizes[i];
        std::string name = "img_" + std::to_string(w) + "x" + std::to_string(h) + ".png";
        createTestImage(name, w, h);

        fs::path input_path = test_dir_ / name;
        fs::path output_path = test_dir_ / ("out_" + name);

        int result = imgfx::app::process_one_cpu(
            false,
            input_path.string(),
            output_path.string(),
            imgfx::core::FILTER_TYPE::GRAYSCALE);

        EXPECT_EQ(result, 0) << "Processing " << w << "x" << h << " image should succeed";
        EXPECT_TRUE(fs::exists(output_path));
    }
}

TEST_F(ProcessExtendedTest, ProcessBatchCPUAllFilterTypes)
{
    createTestImage("img1.png");
    createTestImage("img2.png");

    std::vector<std::string> input_files = {
        (test_dir_ / "img1.png").string(),
        (test_dir_ / "img2.png").string()};

    std::vector<imgfx::core::FILTER_TYPE> filters = {
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR};

    for (size_t i = 0; i < filters.size(); ++i)
    {
        fs::path output_dir = test_dir_ / ("output_" + std::to_string(i));
        fs::create_directories(output_dir);

        int result = imgfx::app::process_batch_cpu(
            input_files,
            output_dir.string(),
            filters[i]);

        EXPECT_EQ(result, 0) << "Batch processing with filter " << i << " should succeed";
    }
}

// GPU tests - only run if GPU is available
TEST_F(ProcessExtendedTest, ProcessOneGPUSuccess)
{
    if (!test_helpers::has_gpu_available())
    {
        GTEST_SKIP() << "GPU not available, skipping GPU test";
    }

    createTestImage("input_gpu.png");
    fs::path input_path = test_dir_ / "input_gpu.png";
    fs::path output_path = test_dir_ / "output_gpu.png";

    int result = imgfx::app::process_one_gpu(
        false,
        input_path.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessExtendedTest, ProcessOneGPUInvalidInput)
{
    if (!test_helpers::has_gpu_available())
    {
        GTEST_SKIP() << "GPU not available, skipping GPU test";
    }

    fs::path invalid_input = test_dir_ / "nonexistent.png";
    fs::path output_path = test_dir_ / "output.png";

    int result = imgfx::app::process_one_gpu(
        false,
        invalid_input.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with invalid input";
}

TEST_F(ProcessExtendedTest, ProcessOneGPUWithBatchFlag)
{
    if (!test_helpers::has_gpu_available())
    {
        GTEST_SKIP() << "GPU not available, skipping GPU test";
    }

    createTestImage("input.png");
    fs::path input_path = test_dir_ / "input.png";
    fs::path output_path = test_dir_ / "output.png";

    int result = imgfx::app::process_one_gpu(
        true, // running_as_batch
        input_path.string(),
        output_path.string(),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(fs::exists(output_path));
}

TEST_F(ProcessExtendedTest, ProcessBatchGPUVariousBatchSizes)
{
    if (!test_helpers::has_gpu_available())
    {
        GTEST_SKIP() << "GPU not available, skipping GPU test";
    }

    // Create test images
    for (int i = 0; i < 10; ++i)
    {
        createTestImage("batch_img" + std::to_string(i) + ".png", 32, 32);
    }

    std::vector<std::string> input_files;
    for (int i = 0; i < 10; ++i)
    {
        input_files.push_back((test_dir_ / ("batch_img" + std::to_string(i) + ".png")).string());
    }

    // Test with different batch sizes
    std::vector<int> batch_sizes = {1, 2, 5, 10, 20};

    for (int batch_size : batch_sizes)
    {
        fs::path output_dir = test_dir_ / ("output_bs" + std::to_string(batch_size));
        fs::create_directories(output_dir);

        int result = imgfx::app::process_batch_gpu(
            input_files,
            output_dir.string(),
            imgfx::core::FILTER_TYPE::GRAYSCALE,
            batch_size);

        EXPECT_EQ(result, 0) << "Batch size " << batch_size << " should succeed";
    }
}
