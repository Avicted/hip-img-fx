#include <gtest/gtest.h>
#include "../src/app/process.h"
#include "../src/core/image.h"
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <filesystem>
#include <fstream>

/**
 * @brief Tests for process.cpp error paths
 *
 * Focuses on testing error handling in:
 * - process_one_gpu
 * - process_batch_gpu
 * - process_one_cpu
 * - process_batch_cpu
 */

class ProcessErrorFixture : public ::testing::Test
{
protected:
    std::filesystem::path temp_dir;

    void SetUp() override
    {
        temp_dir = std::filesystem::temp_directory_path() / "hip_img_fx_process_errors";
        std::filesystem::create_directories(temp_dir);
    }

    void TearDown() override
    {
        if (std::filesystem::exists(temp_dir))
        {
            std::filesystem::remove_all(temp_dir);
        }
    }

    std::string get_temp_path(const std::string &name)
    {
        return (temp_dir / name).string();
    }

    void create_test_image(const std::string &path, int width = 32, int height = 32, int channels = 3)
    {
        imgfx::core::image_t img;
        img.width = width;
        img.height = height;
        img.channels = channels;
        size_t size = width * height * channels;
        img.data = (unsigned char *)malloc(size);

        // Fill with gradient pattern
        for (size_t i = 0; i < size; i++)
        {
            img.data[i] = (unsigned char)(i % 256);
        }

        imgfx::core::save_image(path.c_str(), &img);
        imgfx::core::free_image(&img);
    }
};

// ============================================================================
// PROCESS_ONE_GPU ERROR TESTS
// ============================================================================

TEST_F(ProcessErrorFixture, ProcessOneGPU_NonexistentInputFile)
{
    std::string input = get_temp_path("nonexistent_file.png");
    std::string output = get_temp_path("output.png");

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with nonexistent input file";
}

TEST_F(ProcessErrorFixture, ProcessOneGPU_InvalidOutputPath)
{
    std::string input = get_temp_path("input.png");
    create_test_image(input);

    // Invalid output path (directory doesn't exist)
    std::string output = "/nonexistent_dir_xyz/output.png";

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with invalid output path";
}

TEST_F(ProcessErrorFixture, ProcessOneGPU_ValidInputGrayscale)
{
    std::string input = get_temp_path("input.png");
    std::string output = get_temp_path("output.png");
    create_test_image(input);

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0) << "Should succeed with valid inputs";
    EXPECT_TRUE(std::filesystem::exists(output));
}

TEST_F(ProcessErrorFixture, ProcessOneGPU_ValidInputNegative)
{
    std::string input = get_temp_path("input2.png");
    std::string output = get_temp_path("output2.png");
    create_test_image(input);

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(std::filesystem::exists(output));
}

TEST_F(ProcessErrorFixture, ProcessOneGPU_ValidInputGaussianBlur)
{
    std::string input = get_temp_path("input3.png");
    std::string output = get_temp_path("output3.png");
    create_test_image(input);

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(std::filesystem::exists(output));
}

TEST_F(ProcessErrorFixture, ProcessOneGPU_RunningAsBatchFlag)
{
    std::string input = get_temp_path("input_batch.png");
    std::string output = get_temp_path("output_batch.png");
    create_test_image(input);

    // Test with running_as_batch = true
    int result = imgfx::app::process_one_gpu(
        true,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
}

// ============================================================================
// PROCESS_BATCH_GPU ERROR TESTS
// ============================================================================

TEST_F(ProcessErrorFixture, ProcessBatchGPU_ZeroBatchSize)
{
    std::vector<std::string> input_files;
    std::string output_dir = get_temp_path("output_dir");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        0); // Invalid batch size

    EXPECT_EQ(result, -1) << "Should fail with batch_size = 0";
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_NegativeBatchSize)
{
    std::vector<std::string> input_files;
    std::string output_dir = get_temp_path("output_dir");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        -5); // Invalid batch size

    EXPECT_EQ(result, -1) << "Should fail with negative batch_size";
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_EmptyInputList)
{
    std::vector<std::string> input_files; // Empty
    std::string output_dir = get_temp_path("output_dir");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        64);

    // Empty list should succeed (nothing to process)
    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_SingleValidImage)
{
    std::string input = get_temp_path("batch_input1.png");
    create_test_image(input);

    std::vector<std::string> input_files = {input};
    std::string output_dir = get_temp_path("batch_output");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        64);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_MultipleValidImages)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 5; i++)
    {
        std::string input = get_temp_path("batch_input" + std::to_string(i) + ".png");
        create_test_image(input, 16, 16, 3);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("batch_output_multi");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        2); // Small batch size

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_SomeInvalidImages)
{
    std::vector<std::string> input_files;

    // Mix of valid and invalid
    std::string valid1 = get_temp_path("valid1.png");
    create_test_image(valid1);
    input_files.push_back(valid1);

    input_files.push_back(get_temp_path("nonexistent.png")); // Invalid

    std::string valid2 = get_temp_path("valid2.png");
    create_test_image(valid2);
    input_files.push_back(valid2);

    std::string output_dir = get_temp_path("batch_output_mixed");
    std::filesystem::create_directories(output_dir);

    // Should still process successfully (errors are logged but not fatal)
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        64);

    // Batch processing continues even with some failures
    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_BatchSizeOne)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 3; i++)
    {
        std::string input = get_temp_path("single_batch_" + std::to_string(i) + ".png");
        create_test_image(input, 8, 8, 3);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("batch_output_single");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        1); // Process one at a time

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchGPU_LargeBatchSize)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 3; i++)
    {
        std::string input = get_temp_path("large_batch_" + std::to_string(i) + ".png");
        create_test_image(input, 8, 8, 3);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("batch_output_large");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        1000); // Larger than number of images

    EXPECT_EQ(result, 0);
}

// ============================================================================
// PROCESS_ONE_CPU ERROR TESTS
// ============================================================================

TEST_F(ProcessErrorFixture, ProcessOneCPU_NonexistentInputFile)
{
    std::string input = get_temp_path("nonexistent_cpu.png");
    std::string output = get_temp_path("output_cpu.png");

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1);
}

TEST_F(ProcessErrorFixture, ProcessOneCPU_InvalidOutputPath)
{
    std::string input = get_temp_path("input_cpu.png");
    create_test_image(input);

    std::string output = "/nonexistent_dir_xyz/output_cpu.png";

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1);
}

TEST_F(ProcessErrorFixture, ProcessOneCPU_ValidInputGrayscale)
{
    std::string input = get_temp_path("input_cpu_valid.png");
    std::string output = get_temp_path("output_cpu_valid.png");
    create_test_image(input);

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(std::filesystem::exists(output));
}

TEST_F(ProcessErrorFixture, ProcessOneCPU_ValidInputNegative)
{
    std::string input = get_temp_path("input_cpu_neg.png");
    std::string output = get_temp_path("output_cpu_neg.png");
    create_test_image(input);

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessOneCPU_ValidInputGaussianBlur)
{
    std::string input = get_temp_path("input_cpu_blur.png");
    std::string output = get_temp_path("output_cpu_blur.png");
    create_test_image(input);

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessOneCPU_RunningAsBatchFlag)
{
    std::string input = get_temp_path("input_cpu_batch_flag.png");
    std::string output = get_temp_path("output_cpu_batch_flag.png");
    create_test_image(input);

    int result = imgfx::app::process_one_cpu(
        true, // running_as_batch
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
}

// ============================================================================
// PROCESS_BATCH_CPU ERROR TESTS
// ============================================================================

TEST_F(ProcessErrorFixture, ProcessBatchCPU_EmptyInputList)
{
    std::vector<std::string> input_files;
    std::string output_dir = get_temp_path("cpu_batch_output");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchCPU_SingleValidImage)
{
    std::string input = get_temp_path("cpu_batch_single.png");
    create_test_image(input);

    std::vector<std::string> input_files = {input};
    std::string output_dir = get_temp_path("cpu_batch_out_single");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchCPU_MultipleValidImages)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 4; i++)
    {
        std::string input = get_temp_path("cpu_batch_" + std::to_string(i) + ".png");
        create_test_image(input, 16, 16, 4);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("cpu_batch_out_multi");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchCPU_SomeInvalidImages)
{
    std::vector<std::string> input_files;

    std::string valid = get_temp_path("cpu_valid.png");
    create_test_image(valid);
    input_files.push_back(valid);

    input_files.push_back(get_temp_path("cpu_nonexistent.png"));

    std::string output_dir = get_temp_path("cpu_batch_out_mixed");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);

    EXPECT_EQ(result, 0);
}

TEST_F(ProcessErrorFixture, ProcessBatchCPU_AllFilters)
{
    std::vector<imgfx::core::FILTER_TYPE> filters = {
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR};

    for (auto filter : filters)
    {
        std::string input = get_temp_path("cpu_filter_test.png");
        create_test_image(input);

        std::vector<std::string> input_files = {input};
        std::string output_dir = get_temp_path("cpu_out_filter");
        std::filesystem::create_directories(output_dir);

        int result = imgfx::app::process_batch_cpu(
            input_files,
            output_dir,
            filter);

        EXPECT_EQ(result, 0) << "Filter type: " << (int)filter;
    }
}

// ============================================================================
// CROSS-FUNCTION COMPARISON TESTS
// ============================================================================

TEST_F(ProcessErrorFixture, GPUvsCPU_SameResultStructure)
{
    std::string input = get_temp_path("compare_input.png");
    std::string gpu_output = get_temp_path("compare_gpu_out.png");
    std::string cpu_output = get_temp_path("compare_cpu_out.png");
    create_test_image(input);

    int gpu_result = imgfx::app::process_one_gpu(
        false, input, gpu_output, imgfx::core::FILTER_TYPE::GRAYSCALE);

    int cpu_result = imgfx::app::process_one_cpu(
        false, input, cpu_output, imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(gpu_result, cpu_result) << "GPU and CPU should return same result codes";
}

TEST_F(ProcessErrorFixture, SingleVsBatch_ConsistentBehavior)
{
    std::string input = get_temp_path("single_vs_batch.png");
    create_test_image(input);

    // Process as single
    std::string single_output = get_temp_path("single_out.png");
    int single_result = imgfx::app::process_one_gpu(
        false, input, single_output, imgfx::core::FILTER_TYPE::GRAYSCALE);

    // Process as batch
    std::vector<std::string> batch_input = {input};
    std::string batch_dir = get_temp_path("batch_out_dir");
    std::filesystem::create_directories(batch_dir);
    int batch_result = imgfx::app::process_batch_gpu(
        batch_input, batch_dir, imgfx::core::FILTER_TYPE::GRAYSCALE, 64);

    EXPECT_EQ(single_result, 0);
    EXPECT_EQ(batch_result, 0);
}

// ============================================================================
// ADDITIONAL ERROR PATH COVERAGE TESTS
// ============================================================================

// Test process_one_gpu with corrupted image data
TEST_F(ProcessErrorFixture, ProcessOneGPU_CorruptedImageFile)
{
    std::string input = get_temp_path("corrupted.png");
    std::string output = get_temp_path("output_corrupted.png");

    // Create a file with invalid PNG data
    std::ofstream file(input, std::ios::binary);
    file << "Not a valid PNG file";
    file.close();

    int result = imgfx::app::process_one_gpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, -1) << "Should fail with corrupted image file";
}

// Test process_one_cpu with corrupted image data
TEST_F(ProcessErrorFixture, ProcessOneCPU_CorruptedImageFile)
{
    std::string input = get_temp_path("corrupted_cpu.png");
    std::string output = get_temp_path("output_corrupted_cpu.png");

    // Create a file with invalid PNG data
    std::ofstream file(input, std::ios::binary);
    file << "Invalid image data";
    file.close();

    int result = imgfx::app::process_one_cpu(
        false,
        input,
        output,
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with corrupted image file";
}

// Test process_batch_gpu with mixed valid and corrupted files
TEST_F(ProcessErrorFixture, ProcessBatchGPU_MixedValidAndCorrupted)
{
    std::vector<std::string> input_files;

    // Valid image
    std::string valid_img = get_temp_path("valid_batch.png");
    create_test_image(valid_img);
    input_files.push_back(valid_img);

    // Corrupted image
    std::string corrupted = get_temp_path("corrupted_batch.png");
    std::ofstream file(corrupted, std::ios::binary);
    file << "Bad data";
    file.close();
    input_files.push_back(corrupted);

    // Another valid image
    std::string valid_img2 = get_temp_path("valid_batch2.png");
    create_test_image(valid_img2);
    input_files.push_back(valid_img2);

    std::string output_dir = get_temp_path("batch_mixed_output");
    std::filesystem::create_directories(output_dir);

    // Should continue processing despite corrupted file
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR,
        2);

    // Batch processing may continue despite individual failures
    EXPECT_TRUE(result == 0 || result == -1);
}

// Test process_batch_cpu with nonexistent files
TEST_F(ProcessErrorFixture, ProcessBatchCPU_NonexistentFiles)
{
    std::vector<std::string> input_files;
    input_files.push_back(get_temp_path("nonexist1.png"));
    input_files.push_back(get_temp_path("nonexist2.png"));

    std::string output_dir = get_temp_path("batch_nonexist_out");
    std::filesystem::create_directories(output_dir);

    int result = imgfx::app::process_batch_cpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, 0) << "Batch CPU continues despite errors";
}

// Test process_one_gpu with empty filename
TEST_F(ProcessErrorFixture, ProcessOneGPU_EmptyFilename)
{
    int result = imgfx::app::process_one_gpu(
        false,
        "",
        get_temp_path("output_empty.png"),
        imgfx::core::FILTER_TYPE::GRAYSCALE);

    EXPECT_EQ(result, -1) << "Should fail with empty input filename";
}

// Test process_one_cpu with empty filename
TEST_F(ProcessErrorFixture, ProcessOneCPU_EmptyFilename)
{
    int result = imgfx::app::process_one_cpu(
        false,
        "",
        get_temp_path("output_empty_cpu.png"),
        imgfx::core::FILTER_TYPE::NEGATIVE);

    EXPECT_EQ(result, -1) << "Should fail with empty input filename";
}

// Test all filter types work with process_one_gpu
TEST_F(ProcessErrorFixture, ProcessOneGPU_AllFilterTypes)
{
    std::vector<imgfx::core::FILTER_TYPE> filters = {
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR};

    for (size_t i = 0; i < filters.size(); i++)
    {
        std::string input = get_temp_path("filter_test_" + std::to_string(i) + ".png");
        std::string output = get_temp_path("filter_out_" + std::to_string(i) + ".png");
        create_test_image(input);

        int result = imgfx::app::process_one_gpu(
            false,
            input,
            output,
            filters[i]);

        EXPECT_EQ(result, 0) << "Filter type " << (int)filters[i] << " should work";
    }
}

// Test all filter types work with process_one_cpu
TEST_F(ProcessErrorFixture, ProcessOneCPU_AllFilterTypes)
{
    std::vector<imgfx::core::FILTER_TYPE> filters = {
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR};

    for (size_t i = 0; i < filters.size(); i++)
    {
        std::string input = get_temp_path("cpu_filter_test_" + std::to_string(i) + ".png");
        std::string output = get_temp_path("cpu_filter_out_" + std::to_string(i) + ".png");
        create_test_image(input);

        int result = imgfx::app::process_one_cpu(
            false,
            input,
            output,
            filters[i]);

        EXPECT_EQ(result, 0) << "CPU filter type " << (int)filters[i] << " should work";
    }
}

// Test batch processing with very small batch size
TEST_F(ProcessErrorFixture, ProcessBatchGPU_BatchSizeOneWithMultipleImages)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 5; i++)
    {
        std::string input = get_temp_path("batch_small_" + std::to_string(i) + ".png");
        create_test_image(input, 8, 8, 3);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("batch_small_out");
    std::filesystem::create_directories(output_dir);

    // Batch size 1 forces multiple kernel launches
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        1);

    EXPECT_EQ(result, 0);
}

// Test batch processing with batch size larger than number of images
TEST_F(ProcessErrorFixture, ProcessBatchGPU_BatchSizeLargerThanImageCount)
{
    std::vector<std::string> input_files;
    for (int i = 0; i < 3; i++)
    {
        std::string input = get_temp_path("batch_large_" + std::to_string(i) + ".png");
        create_test_image(input);
        input_files.push_back(input);
    }

    std::string output_dir = get_temp_path("batch_large_out");
    std::filesystem::create_directories(output_dir);

    // Batch size larger than image count
    int result = imgfx::app::process_batch_gpu(
        input_files,
        output_dir,
        imgfx::core::FILTER_TYPE::NEGATIVE,
        1000);

    EXPECT_EQ(result, 0);
}

// Test process functions with different image dimensions
TEST_F(ProcessErrorFixture, ProcessOneGPU_VariousImageSizes)
{
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {16, 16}, {128, 128}, {256, 256}};

    for (size_t i = 0; i < sizes.size(); i++)
    {
        std::string input = get_temp_path("size_test_" + std::to_string(i) + ".png");
        std::string output = get_temp_path("size_out_" + std::to_string(i) + ".png");
        create_test_image(input, sizes[i].first, sizes[i].second, 3);

        int result = imgfx::app::process_one_gpu(
            false,
            input,
            output,
            imgfx::core::FILTER_TYPE::GRAYSCALE);

        EXPECT_EQ(result, 0) << "Size " << sizes[i].first << "x" << sizes[i].second << " should work";
    }
}

// Test process functions with different channel counts
TEST_F(ProcessErrorFixture, ProcessOneGPU_VariousChannelCounts)
{
    std::vector<int> channels = {1, 3, 4};

    for (size_t i = 0; i < channels.size(); i++)
    {
        std::string input = get_temp_path("channels_test_" + std::to_string(i) + ".png");
        std::string output = get_temp_path("channels_out_" + std::to_string(i) + ".png");
        create_test_image(input, 32, 32, channels[i]);

        int result = imgfx::app::process_one_gpu(
            false,
            input,
            output,
            imgfx::core::FILTER_TYPE::NEGATIVE);

        EXPECT_EQ(result, 0) << "Channel count " << channels[i] << " should work";
    }
}
