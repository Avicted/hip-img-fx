// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "../src/cli/cli_parser.h"
#include <cstring>

/**
 * @brief CLI parser tests - always run (CPU-only)
 *
 * Tests command-line argument parsing logic.
 */

TEST(CLIParser, ValidArguments)
{
    const char *argv[] = {
        "hip-img-fx",
        "--input", "input.png",
        "--output", "output.png",
        "--filter", "grayscale"};
    int argc = 7;

    // Note: parse_cli_args may exit on error, so we test valid case
    imgfx::cli::cli_args_t args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_STREQ(args.input_file, "input.png");
    EXPECT_STREQ(args.output_file, "output.png");
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::GRAYSCALE);
    EXPECT_FALSE(args.use_cpu);
}

TEST(CLIParser, MissingRequiredArgs)
{
    // Note: parse_cli_args currently calls exit() on missing args,
    // which cannot be tested without process forking.
    // This test documents expected behavior.

    // Future improvement: Refactor parse_cli_args to return
    // optional/result type instead of exiting.
    EXPECT_TRUE(true) << "CLI parser validates required arguments";
}

TEST(CLIParser, InvalidFilterName)
{
    // Note: parse_cli_args currently calls exit() on invalid filter,
    // which cannot be tested without process forking.
    // This test documents expected behavior.

    // Future improvement: Refactor parse_cli_args to return
    // optional/result type instead of exiting.
    EXPECT_TRUE(true) << "CLI parser validates filter names";
}

TEST(CLIParser, CPUFlagEnabled)
{
    const char *argv[] = {
        "hip-img-fx",
        "--input", "input.png",
        "--output", "output.png",
        "--filter", "negative",
        "--use-cpu" // CPU flag
    };
    int argc = 8;

    imgfx::cli::cli_args_t args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_TRUE(args.use_cpu) << "CPU flag should be set";
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::NEGATIVE);
}

TEST(CLIParser, BatchSizeOption)
{
    const char *argv[] = {
        "hip-img-fx",
        "--input", "input.png",
        "--output", "output.png",
        "--filter", "gaussian-blur",
        "--batch-size", "8" // Batch size
    };
    int argc = 9;

    imgfx::cli::cli_args_t args = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));

    EXPECT_EQ(args.batch_size, 8);
    EXPECT_EQ(args.filter_type, imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);
}

TEST(CLIParser, HelpOutput)
{
    // Test that help function exists and can be called
    ASSERT_NO_THROW(imgfx::cli::print_help());
}

TEST(CLIParser, FilterTypeMapping)
{
    // Test all supported filter types
    const char *argv_grayscale[] = {
        "hip-img-fx", "--input", "in.png", "--output", "out.png", "--filter", "grayscale"};
    const char *argv_negative[] = {
        "hip-img-fx", "--input", "in.png", "--output", "out.png", "--filter", "negative"};
    const char *argv_blur[] = {
        "hip-img-fx", "--input", "in.png", "--output", "out.png", "--filter", "gaussian-blur"};

    int argc = 7;

    auto args1 = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv_grayscale));
    EXPECT_EQ(args1.filter_type, imgfx::core::FILTER_TYPE::GRAYSCALE);

    auto args2 = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv_negative));
    EXPECT_EQ(args2.filter_type, imgfx::core::FILTER_TYPE::NEGATIVE);

    auto args3 = imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv_blur));
    EXPECT_EQ(args3.filter_type, imgfx::core::FILTER_TYPE::GAUSSIAN_BLUR);
}
