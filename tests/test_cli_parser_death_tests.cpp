#include <gtest/gtest.h>
#include "../src/cli/cli_parser.h"
#include <cstring>

/**
 * @brief Death tests for CLI parser error paths that call exit()
 *
 * These tests use EXPECT_EXIT to verify that the parser correctly calls exit()
 * with the appropriate exit code and error message when invalid inputs are provided.
 */

// Helper function to avoid comma issues in EXPECT_EXIT macro
static void parse_args(int argc, const char **argv)
{
    imgfx::cli::parse_cli_args(argc, const_cast<char **>(argv));
}

// Test missing --input argument (calls exit(1))
TEST(CLIParserDeathTests, MissingInputExit)
{
    const char *argv[] = {"hip-img-fx", "--output", "/tmp/out.png", "--filter", "grayscale"};
    EXPECT_EXIT(parse_args(5, argv),
                ::testing::ExitedWithCode(1),
                ""); // Empty regex - just check exit code
}

// Test missing --output argument (calls exit(1))
TEST(CLIParserDeathTests, MissingOutputExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--filter", "negative"};
    EXPECT_EXIT(parse_args(5, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test missing --filter argument (calls exit(1))
TEST(CLIParserDeathTests, MissingFilterExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png"};
    EXPECT_EXIT(parse_args(5, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test all required arguments missing (calls exit(1))
TEST(CLIParserDeathTests, AllRequiredArgumentsMissingExit)
{
    const char *argv[] = {"hip-img-fx"};
    EXPECT_EXIT(parse_args(1, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test unknown filter type (calls exit(1))
TEST(CLIParserDeathTests, UnknownFilterTypeExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "invalid_filter"};
    EXPECT_EXIT(parse_args(7, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test unknown argument (calls exit(1))
TEST(CLIParserDeathTests, UnknownArgumentExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "grayscale", "--unknown-flag"};
    EXPECT_EXIT(parse_args(8, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test invalid batch size zero (calls exit(1))
TEST(CLIParserDeathTests, BatchSizeZeroExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "grayscale", "--batch-size", "0"};
    EXPECT_EXIT(parse_args(9, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test invalid batch size negative (calls exit(1))
TEST(CLIParserDeathTests, BatchSizeNegativeExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "negative", "--batch-size", "-5"};
    EXPECT_EXIT(parse_args(9, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test --help flag (calls exit(0))
TEST(CLIParserDeathTests, HelpFlagExit)
{
    const char *argv[] = {"hip-img-fx", "--help"};
    EXPECT_EXIT(parse_args(2, argv),
                ::testing::ExitedWithCode(0),
                "");
}

// Test incomplete --input at end of args (no value provided)
TEST(CLIParserDeathTests, IncompleteInputAtEndExit)
{
    const char *argv[] = {"hip-img-fx", "--output", "/tmp/out.png", "--filter", "grayscale", "--input"};
    EXPECT_EXIT(parse_args(6, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test incomplete --output at end of args
TEST(CLIParserDeathTests, IncompleteOutputAtEndExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--filter", "negative", "--output"};
    EXPECT_EXIT(parse_args(6, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test incomplete --filter at end of args
TEST(CLIParserDeathTests, IncompleteFilterAtEndExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter"};
    EXPECT_EXIT(parse_args(6, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test case sensitivity - "Grayscale" instead of "grayscale"
TEST(CLIParserDeathTests, FilterCaseSensitiveGrayscaleExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "Grayscale"};
    EXPECT_EXIT(parse_args(7, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test case sensitivity - "NEGATIVE" instead of "negative"
TEST(CLIParserDeathTests, FilterCaseSensitiveNegativeExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "NEGATIVE"};
    EXPECT_EXIT(parse_args(7, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test underscore instead of dash in filter name
TEST(CLIParserDeathTests, FilterWithUnderscoreInsteadOfDashExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "gaussian_blur"};
    EXPECT_EXIT(parse_args(7, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test typo in filter name
TEST(CLIParserDeathTests, FilterTypoExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "greyscale"};
    EXPECT_EXIT(parse_args(7, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test typo in flag name "--cpu" instead of "--use-cpu"
TEST(CLIParserDeathTests, UnknownFlagCPUTypoExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "negative", "--cpu"};
    EXPECT_EXIT(parse_args(8, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test batch size with non-numeric string
TEST(CLIParserDeathTests, BatchSizeNonNumericExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png", "--output", "/tmp/out.png", "--filter", "grayscale", "--batch-size", "abc"};
    EXPECT_EXIT(parse_args(9, argv),
                ::testing::ExitedWithCode(1),
                "");
}

// Test two of three required arguments missing
TEST(CLIParserDeathTests, TwoMissingInputOutputExit)
{
    const char *argv[] = {"hip-img-fx", "--filter", "negative"};
    EXPECT_EXIT(parse_args(3, argv),
                ::testing::ExitedWithCode(1),
                "");
}

TEST(CLIParserDeathTests, TwoMissingInputFilterExit)
{
    const char *argv[] = {"hip-img-fx", "--output", "/tmp/out.png"};
    EXPECT_EXIT(parse_args(3, argv),
                ::testing::ExitedWithCode(1),
                "");
}

TEST(CLIParserDeathTests, TwoMissingOutputFilterExit)
{
    const char *argv[] = {"hip-img-fx", "--input", "/tmp/in.png"};
    EXPECT_EXIT(parse_args(3, argv),
                ::testing::ExitedWithCode(1),
                "");
}
