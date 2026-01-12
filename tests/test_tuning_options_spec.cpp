#include <gtest/gtest.h>
#include "../../include/hip-img-fx/autotune/types.h"

/**
 * @file test_tuning_options_spec.cpp
 * @brief Specification tests for TuningOptions public API
 *
 * Category: tests/spec/ - Authoritative behavioral tests
 *
 * Tests the contract and invariants of TuningOptions, not implementation details.
 * Follows testing SKILL.md guidelines:
 * - Encode contracts, not implementation
 * - Negative tests required
 * - Break-the-code validation
 */

using namespace imgfx::core::autotune;

// ============================================================================
// POSITIVE TESTS - Valid Usage
// ============================================================================

TEST(TuningOptionsSpec, DefaultOptionsAreValid)
{
    TuningOptions opts = TuningOptions::defaults();

    EXPECT_TRUE(opts.validate()) << "Default options must be valid";
    EXPECT_GT(opts.warmup_runs, 0) << "Default warmup must be positive";
    EXPECT_GT(opts.timing_runs, 0) << "Default timing runs must be positive";
    EXPECT_TRUE(opts.verbose) << "Default should be verbose";
    EXPECT_FALSE(opts.force_retune) << "Default should not force retune";
    EXPECT_TRUE(opts.enable_early_exit) << "Default should enable early exit";
}

TEST(TuningOptionsSpec, QuietOptionsAreValid)
{
    TuningOptions opts = TuningOptions::quiet();

    EXPECT_TRUE(opts.validate()) << "Quiet options must be valid";
    EXPECT_FALSE(opts.verbose) << "Quiet options must disable verbose";
}

TEST(TuningOptionsSpec, ConservativeOptionsAreValid)
{
    TuningOptions opts = TuningOptions::conservative();

    EXPECT_TRUE(opts.validate()) << "Conservative options must be valid";
    EXPECT_FALSE(opts.enable_early_exit) << "Conservative must disable early exit";
}

TEST(TuningOptionsSpec, AggressiveOptionsAreValid)
{
    TuningOptions opts = TuningOptions::aggressive();

    EXPECT_TRUE(opts.validate()) << "Aggressive options must be valid";
    EXPECT_TRUE(opts.enable_early_exit) << "Aggressive must enable early exit";
    EXPECT_LT(opts.early_exit_threshold, 1.15) << "Aggressive threshold should be tighter";
}

TEST(TuningOptionsSpec, CanModifyWarmupRuns)
{
    TuningOptions opts;
    opts.warmup_runs = 10;

    EXPECT_TRUE(opts.validate());
    EXPECT_EQ(opts.warmup_runs, 10);
}

TEST(TuningOptionsSpec, CanModifyTimingRuns)
{
    TuningOptions opts;
    opts.timing_runs = 20;

    EXPECT_TRUE(opts.validate());
    EXPECT_EQ(opts.timing_runs, 20);
}

TEST(TuningOptionsSpec, CanToggleVerbose)
{
    TuningOptions opts;
    opts.verbose = false;
    EXPECT_TRUE(opts.validate());

    opts.verbose = true;
    EXPECT_TRUE(opts.validate());
}

TEST(TuningOptionsSpec, CanToggleForceRetune)
{
    TuningOptions opts;
    opts.force_retune = true;
    EXPECT_TRUE(opts.validate());

    opts.force_retune = false;
    EXPECT_TRUE(opts.validate());
}

TEST(TuningOptionsSpec, CanToggleEarlyExit)
{
    TuningOptions opts;
    opts.enable_early_exit = false;
    EXPECT_TRUE(opts.validate());

    opts.enable_early_exit = true;
    EXPECT_TRUE(opts.validate());
}

TEST(TuningOptionsSpec, CanModifyEarlyExitThreshold)
{
    TuningOptions opts;
    opts.early_exit_threshold = 1.25;

    EXPECT_TRUE(opts.validate());
    EXPECT_DOUBLE_EQ(opts.early_exit_threshold, 1.25);
}

TEST(TuningOptionsSpec, CanModifyEarlyExitMinCoverage)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = 0.5;

    EXPECT_TRUE(opts.validate());
    EXPECT_DOUBLE_EQ(opts.early_exit_min_coverage, 0.5);
}

TEST(TuningOptionsSpec, ZeroWarmupIsValid)
{
    TuningOptions opts;
    opts.warmup_runs = 0; // Allow zero warmup

    EXPECT_TRUE(opts.validate()) << "Zero warmup should be allowed";
}

TEST(TuningOptionsSpec, MinimalTimingRunsIsOne)
{
    TuningOptions opts;
    opts.timing_runs = 1;

    EXPECT_TRUE(opts.validate()) << "Single timing run should be valid";
}

TEST(TuningOptionsSpec, ThresholdExactlyOne)
{
    TuningOptions opts;
    opts.early_exit_threshold = 1.0; // Exactly 1.0 means no tolerance

    EXPECT_TRUE(opts.validate()) << "Threshold of 1.0 should be valid (no tolerance)";
}

TEST(TuningOptionsSpec, FullCoverageIsValid)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = 1.0; // Test all candidates

    EXPECT_TRUE(opts.validate()) << "100% coverage should be valid";
}

TEST(TuningOptionsSpec, MinimalCoverageIsValid)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = 0.01; // Test at least 1% of candidates

    EXPECT_TRUE(opts.validate()) << "1% min coverage should be valid";
}

// ============================================================================
// NEGATIVE TESTS - Invalid Usage (Required by SKILL.md)
// ============================================================================

TEST(TuningOptionsSpec, NegativeWarmupIsInvalid)
{
    TuningOptions opts;
    opts.warmup_runs = -1;

    EXPECT_FALSE(opts.validate()) << "Negative warmup must be rejected";
}

TEST(TuningOptionsSpec, NegativeTimingRunsIsInvalid)
{
    TuningOptions opts;
    opts.timing_runs = -5;

    EXPECT_FALSE(opts.validate()) << "Negative timing runs must be rejected";
}

TEST(TuningOptionsSpec, ZeroTimingRunsIsInvalid)
{
    TuningOptions opts;
    opts.timing_runs = 0;

    EXPECT_FALSE(opts.validate()) << "Zero timing runs must be rejected (need at least 1)";
}

TEST(TuningOptionsSpec, ThresholdBelowOneIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_threshold = 0.99;

    EXPECT_FALSE(opts.validate()) << "Threshold < 1.0 makes no sense (slower than best)";
}

TEST(TuningOptionsSpec, ThresholdZeroIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_threshold = 0.0;

    EXPECT_FALSE(opts.validate()) << "Zero threshold must be rejected";
}

TEST(TuningOptionsSpec, NegativeThresholdIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_threshold = -0.5;

    EXPECT_FALSE(opts.validate()) << "Negative threshold must be rejected";
}

TEST(TuningOptionsSpec, ZeroCoverageIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = 0.0;

    EXPECT_FALSE(opts.validate()) << "Zero coverage must be rejected (need to test something)";
}

TEST(TuningOptionsSpec, NegativeCoverageIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = -0.1;

    EXPECT_FALSE(opts.validate()) << "Negative coverage must be rejected";
}

TEST(TuningOptionsSpec, CoverageAboveOneIsInvalid)
{
    TuningOptions opts;
    opts.early_exit_min_coverage = 1.5;

    EXPECT_FALSE(opts.validate()) << "Coverage > 100% must be rejected";
}

// ============================================================================
// BOUNDARY TESTS - Edge Cases
// ============================================================================

TEST(TuningOptionsSpec, VeryLargeWarmupIsValid)
{
    TuningOptions opts;
    opts.warmup_runs = 10000;

    EXPECT_TRUE(opts.validate()) << "Large warmup should be valid (user choice)";
}

TEST(TuningOptionsSpec, VeryLargeTimingRunsIsValid)
{
    TuningOptions opts;
    opts.timing_runs = 10000;

    EXPECT_TRUE(opts.validate()) << "Large timing runs should be valid";
}

TEST(TuningOptionsSpec, VeryLargeThresholdIsValid)
{
    TuningOptions opts;
    opts.early_exit_threshold = 100.0; // 100x slower than best

    EXPECT_TRUE(opts.validate()) << "Large threshold should be valid (very permissive)";
}

// ============================================================================
// COMBINATION TESTS - Multiple Invalid Fields
// ============================================================================

TEST(TuningOptionsSpec, MultipleInvalidFieldsStillInvalid)
{
    TuningOptions opts;
    opts.warmup_runs = -1;
    opts.timing_runs = -1;
    opts.early_exit_threshold = 0.5;
    opts.early_exit_min_coverage = -0.5;

    EXPECT_FALSE(opts.validate()) << "Multiple invalid fields must fail validation";
}

// ============================================================================
// PRESET CONSISTENCY TESTS
// ============================================================================

TEST(TuningOptionsSpec, ConservativeDisablesEarlyExit)
{
    TuningOptions opts = TuningOptions::conservative();

    EXPECT_FALSE(opts.enable_early_exit) << "Conservative preset must disable early exit";
}

TEST(TuningOptionsSpec, AggressiveHasTighterThreshold)
{
    TuningOptions defaults = TuningOptions::defaults();
    TuningOptions aggressive = TuningOptions::aggressive();

    EXPECT_LT(aggressive.early_exit_threshold, defaults.early_exit_threshold)
        << "Aggressive should have tighter threshold than defaults";
}

TEST(TuningOptionsSpec, AggressiveHasLowerMinCoverage)
{
    TuningOptions defaults = TuningOptions::defaults();
    TuningOptions aggressive = TuningOptions::aggressive();

    EXPECT_LT(aggressive.early_exit_min_coverage, defaults.early_exit_min_coverage)
        << "Aggressive exits even earlier with fewer candidates tested";
}

// ============================================================================
// INVARIANT TESTS - Contract Enforcement
// ============================================================================

TEST(TuningOptionsSpec, ValidateReturnsTrueForValidOptions)
{
    TuningOptions opts;
    opts.warmup_runs = 5;
    opts.timing_runs = 10;
    opts.early_exit_threshold = 1.2;
    opts.early_exit_min_coverage = 0.5;

    EXPECT_TRUE(opts.validate()) << "Valid options must pass validation";
}

TEST(TuningOptionsSpec, ValidateReturnsFalseForInvalidOptions)
{
    TuningOptions opts;
    opts.timing_runs = -1;

    EXPECT_FALSE(opts.validate()) << "Invalid options must fail validation";
}

// ============================================================================
// ASSERTION TESTS - Debug Mode Behavior
// ============================================================================

#ifndef NDEBUG
TEST(TuningOptionsSpec, DefaultsAssertsValidation)
{
    // In debug mode, defaults() should assert if validation fails
    // We can't directly test assertion death here without forking,
    // but we document that default presets must be valid
    EXPECT_TRUE(TuningOptions::defaults().validate());
}

TEST(TuningOptionsSpec, QuietAssertsValidation)
{
    EXPECT_TRUE(TuningOptions::quiet().validate());
}

TEST(TuningOptionsSpec, ConservativeAssertsValidation)
{
    EXPECT_TRUE(TuningOptions::conservative().validate());
}

TEST(TuningOptionsSpec, AggressiveAssertsValidation)
{
    EXPECT_TRUE(TuningOptions::aggressive().validate());
}
#endif

// ============================================================================
// REGRESSION TESTS - Historical Bugs
// ============================================================================

// Add regression tests here when bugs are discovered
// Example: TEST(TuningOptionsSpec, RegressionIssue42_ThresholdClampedProperly) { ... }
