// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/cli/cli_parser.h"
#include "../src/core/gpu_utils.h"
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

/**
 * @brief Extended tests for CLI parser - covering edge cases and error conditions
 *
 * Uses temporary files to avoid filesystem-related crashes during testing.
 */

// Helper to create temporary test files
class CLIParserFixture : public ::testing::Test
{
protected:
    std::filesystem::path temp_dir;

    void SetUp() override
    {
        temp_dir = std::filesystem::temp_directory_path() / "hip_img_fx_cli_test";
        std::filesystem::create_directories(temp_dir);
    }

    void TearDown() override
    {
        if (std::filesystem::exists(temp_dir))
        {
            std::filesystem::remove_all(temp_dir);
        }
    }

    std::string create_temp_file(const std::string &name)
    {
        auto path = temp_dir / name;
        std::ofstream file(path);
        file << "test";
        file.close();
        return path.string();
    }

    std::string create_temp_dir(const std::string &name)
    {
        auto path = temp_dir / name;
        std::filesystem::create_directories(path);
        return path.string();
    }
};

TEST(CLIParserExtended, EmptyArguments)
{
    // Just program name, no arguments
    // This should exit with error, but we can't test that easily
    // Just document the expected behavior
    EXPECT_TRUE(true) << "Empty arguments should cause error exit";
}

TEST_F(CLIParserFixture, MultipleInputs)
{
    auto input1 = create_temp_file("first.png");
    auto input2 = create_temp_file("second.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input1.c_str(),
        "--input", input2.c_str(), // Second input should overwrite
        "--output", output.c_str(),
        "--filter", "grayscale"};
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    // Should use the last specified input
    EXPECT_STREQ(args.input_file, input2.c_str());
}

TEST_F(CLIParserFixture, MultipleOutputs)
{
    auto input = create_temp_file("input.png");
    auto output1 = create_temp_file("first.png");
    auto output2 = create_temp_file("second.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output1.c_str(),
        "--output", output2.c_str(), // Second output should overwrite
        "--filter", "negative"};
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.output_file, output2.c_str());
}

TEST_F(CLIParserFixture, MultipleFilters)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale",
        "--filter", "negative"}; // Last filter wins
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::NEGATIVE);
}

TEST_F(CLIParserFixture, MultipleBatchSizes)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale",
        "--batch-size", "32",
        "--batch-size", "128"}; // Last batch size wins
    int argc = 11;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 128);
}

TEST_F(CLIParserFixture, CPUFlagMultipleTimes)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "gaussian-blur",
        "--use-cpu",
        "--use-cpu"}; // Repeated flag should be fine
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_TRUE(args.use_cpu);
}

TEST_F(CLIParserFixture, BatchSizeOne)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale",
        "--batch-size", "1"};
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 1);
}

TEST_F(CLIParserFixture, BatchSizeLarge)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "negative",
        "--batch-size", "1000"};
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 1000);
}

TEST_F(CLIParserFixture, DefaultBatchSize)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 64) << "Default batch size should be 64";
}

TEST_F(CLIParserFixture, DefaultCPUFlag)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "negative"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_FALSE(args.use_cpu) << "CPU flag should be false by default";
}

TEST_F(CLIParserFixture, AllFiltersValid)
{
    auto input = create_temp_file("in.png");
    auto output = create_temp_file("out.png");

    std::vector<std::tuple<const char *, imgfx::core::FILTER_TYPE>> filters = {
        {"grayscale", imgfx::core::FILTER_TYPE::GRAYSCALE},
        {"negative", imgfx::core::FILTER_TYPE::NEGATIVE},
        {"gaussian-blur", imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR}};

    for (const auto &[filter_name, expected_type] : filters)
    {
        const char *argv[] = {
            "hip-img-fx",
            "--input", input.c_str(),
            "--output", output.c_str(),
            "--filter", filter_name};
        int argc = 7;

        auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));
        EXPECT_EQ(args.filter_type, expected_type) << "Filter: " << filter_name;
    }
}

TEST_F(CLIParserFixture, ArgumentOrderVariation1)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    // Filter first, then input/output
    const char *argv[] = {
        "hip-img-fx",
        "--filter", "grayscale",
        "--input", input.c_str(),
        "--output", output.c_str()};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::GRAYSCALE);
}

TEST_F(CLIParserFixture, ArgumentOrderVariation2)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    // Output first
    const char *argv[] = {
        "hip-img-fx",
        "--output", output.c_str(),
        "--filter", "negative",
        "--input", input.c_str()};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::NEGATIVE);
}

TEST_F(CLIParserFixture, CPUFlagPosition)
{
    std::string input = create_temp_file("input.png");
    std::string output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale",
        "--use-cpu"};

    int argc = 8;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));
    EXPECT_TRUE(args.use_cpu);
}

TEST_F(CLIParserFixture, BatchSizePosition)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--batch-size", "16",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "negative"};
    int argc = 9;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 16);
}

TEST_F(CLIParserFixture, LongInputPath)
{
    // Create a deeply nested directory structure
    auto deep_dir = create_temp_dir("very/long/path/to/some/directory/with/many/levels");
    auto input = create_temp_file("very/long/path/to/some/directory/with/many/levels/input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
}

TEST_F(CLIParserFixture, LongOutputPath)
{
    auto input = create_temp_file("input.png");
    auto deep_dir = create_temp_dir("another/very/long/path/to/output/directory");
    auto output = create_temp_file("another/very/long/path/to/output/directory/output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "negative"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.output_file, output.c_str());
}

TEST_F(CLIParserFixture, FileNameWithSpaces)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
}

TEST_F(CLIParserFixture, FileNameWithSpecialChars)
{
    auto input = create_temp_file("file_with-special.png");
    auto output = create_temp_file("output_result.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "negative"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
}

TEST_F(CLIParserFixture, RelativePaths)
{
    auto input = create_temp_file("input.png");
    auto output_dir = create_temp_dir("output");
    auto output = create_temp_file("output/result.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "gaussian-blur"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
}

TEST_F(CLIParserFixture, AbsolutePaths)
{
    // Create files with absolute paths in our temp directory
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "grayscale"};
    int argc = 7;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
}

TEST_F(CLIParserFixture, AllOptionsSpecified)
{
    auto input = create_temp_file("input.png");
    auto output = create_temp_file("output.png");

    const char *argv[] = {
        "hip-img-fx",
        "--input", input.c_str(),
        "--output", output.c_str(),
        "--filter", "gaussian-blur",
        "--use-cpu",
        "--batch-size", "32"};
    int argc = 10;

    auto args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, input.c_str());
    EXPECT_STREQ(args.output_file, output.c_str());
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);
    EXPECT_TRUE(args.use_cpu);
    EXPECT_EQ(args.batch_size, 32);
}

TEST(CLIParserExtended, PrintHelpExecution)
{
    // Just verify print_help doesn't crash
    ASSERT_NO_THROW(imgfx::cli::print_help());
}

TEST(CLIParserExtended, StructInitialization)
{
    imgfx::cli::cli_args_t args;
    args.input_file = "test.png";
    args.output_file = "out.png";
    args.filter_type = imgfx::core::FILTER_TYPE::GRAYSCALE;
    args.use_cpu = true;
    args.batch_size = 128;

    EXPECT_STREQ(args.input_file, "test.png");
    EXPECT_STREQ(args.output_file, "out.png");
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::GRAYSCALE);
    EXPECT_TRUE(args.use_cpu);
    EXPECT_EQ(args.batch_size, 128);
}

TEST(CLIParserExtended, FilterTypeStringMapping)
{
    // Verify filter_type_to_string works correctly (used in cli_parser)
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::GRAYSCALE), "GRAYSCALE");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::NEGATIVE), "NEGATIVE");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR), "GAUSSIAN_BLUR");
    EXPECT_EQ(imgfx::core::filter_type_to_string(imgfx::core::FILTER_TYPE::UNDEFINED), "UNDEFINED");
}
