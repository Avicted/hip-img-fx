#include <gtest/gtest.h>
#include "../../include/hip-img-fx/autotune/tuning_config.h"

/**
 * @file test_tuning_config_spec.cpp
 * @brief Specification tests for TuningConfig public API
 *
 * Category: tests/spec/ - Authoritative behavioral tests
 *
 * Tests the contract and invariants of TuningConfig, focusing on:
 * - Type safety
 * - Error handling
 * - Method contracts (remove, clear, keys, total_threads)
 * - Consistency invariants
 *
 * Follows testing SKILL.md guidelines.
 */

using namespace imgfx::core::autotune;

// ============================================================================
// SIZE & EMPTY TESTS
// ============================================================================

TEST(TuningConfigSpec, NewConfigIsEmpty)
{
    TuningConfig cfg;

    EXPECT_TRUE(cfg.empty());
    EXPECT_EQ(cfg.size(), 0);
}

TEST(TuningConfigSpec, SizeIncrementsWithSet)
{
    TuningConfig cfg;

    EXPECT_EQ(cfg.size(), 0);
    cfg.set("param1", 1);
    EXPECT_EQ(cfg.size(), 1);
    cfg.set("param2", 2);
    EXPECT_EQ(cfg.size(), 2);
}

TEST(TuningConfigSpec, SetSameKeyDoesNotIncreaseSize)
{
    TuningConfig cfg;
    cfg.set("param", 1);
    EXPECT_EQ(cfg.size(), 1);

    cfg.set("param", 2); // Overwrite
    EXPECT_EQ(cfg.size(), 1) << "Overwriting should not increase size";
}

TEST(TuningConfigSpec, EmptyIsFalseAfterSet)
{
    TuningConfig cfg;
    EXPECT_TRUE(cfg.empty());

    cfg.set("param", 1);
    EXPECT_FALSE(cfg.empty());
}

// ============================================================================
// TOTAL_THREADS TESTS
// ============================================================================

TEST(TuningConfigSpec, TotalThreadsDefaultDimensions)
{
    TuningConfig cfg;
    // Defaults: block_x=256, block_y=1, block_z=1

    EXPECT_EQ(cfg.total_threads(), 256) << "Default total threads";
}

TEST(TuningConfigSpec, TotalThreadsCustomDimensions)
{
    TuningConfig cfg;
    cfg.set_block_dims(128, 2, 4);

    EXPECT_EQ(cfg.total_threads(), 128 * 2 * 4) << "Total threads = block_x * block_y * block_z";
}

TEST(TuningConfigSpec, TotalThreadsOneDimension)
{
    TuningConfig cfg;
    cfg.set_block_dims(512, 1, 1);

    EXPECT_EQ(cfg.total_threads(), 512);
}

TEST(TuningConfigSpec, TotalThreadsTwoDimensions)
{
    TuningConfig cfg;
    cfg.set_block_dims(16, 16, 1);

    EXPECT_EQ(cfg.total_threads(), 256);
}

TEST(TuningConfigSpec, TotalThreadsThreeDimensions)
{
    TuningConfig cfg;
    cfg.set_block_dims(8, 8, 8);

    EXPECT_EQ(cfg.total_threads(), 512);
}

// ============================================================================
// BLOCK DIMENSION ACCESSORS TESTS
// ============================================================================

TEST(TuningConfigSpec, BlockDimensionDefaults)
{
    TuningConfig cfg;

    EXPECT_EQ(cfg.block_x(), 256) << "Default block_x";
    EXPECT_EQ(cfg.block_y(), 1) << "Default block_y";
    EXPECT_EQ(cfg.block_z(), 1) << "Default block_z";
}

TEST(TuningConfigSpec, SetBlockDimsUpdatesAllDimensions)
{
    TuningConfig cfg;
    cfg.set_block_dims(128, 4, 2);

    EXPECT_EQ(cfg.block_x(), 128);
    EXPECT_EQ(cfg.block_y(), 4);
    EXPECT_EQ(cfg.block_z(), 2);
}

TEST(TuningConfigSpec, SetBlockDimsWithDefaultZ)
{
    TuningConfig cfg;
    cfg.set_block_dims(64, 8); // z defaults to 1

    EXPECT_EQ(cfg.block_x(), 64);
    EXPECT_EQ(cfg.block_y(), 8);
    EXPECT_EQ(cfg.block_z(), 1);
}

TEST(TuningConfigSpec, BlockAccessorsUseGetOr)
{
    TuningConfig cfg;
    // Don't set block_x explicitly - should use default

    int x = cfg.block_x();
    EXPECT_EQ(x, 256) << "Accessor should use default when param not set";
}

// ============================================================================
// EQUALITY TESTS
// ============================================================================

TEST(TuningConfigSpec, EmptyConfigsAreEqual)
{
    TuningConfig cfg1, cfg2;

    EXPECT_TRUE(cfg1 == cfg2);
    EXPECT_FALSE(cfg1 != cfg2);
}

TEST(TuningConfigSpec, IdenticalConfigsAreEqual)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param", 100);
    cfg2.set("param", 100);

    EXPECT_TRUE(cfg1 == cfg2);
}

TEST(TuningConfigSpec, DifferentValuesAreNotEqual)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param", 100);
    cfg2.set("param", 200);

    EXPECT_FALSE(cfg1 == cfg2);
    EXPECT_TRUE(cfg1 != cfg2);
}

TEST(TuningConfigSpec, DifferentKeysAreNotEqual)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param1", 100);
    cfg2.set("param2", 100);

    EXPECT_FALSE(cfg1 == cfg2);
}

TEST(TuningConfigSpec, DifferentSizesAreNotEqual)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param1", 100);
    cfg1.set("param2", 200);
    cfg2.set("param1", 100);

    EXPECT_FALSE(cfg1 == cfg2);
}

TEST(TuningConfigSpec, OrderDoesNotAffectEquality)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("alpha", 1);
    cfg1.set("beta", 2);

    cfg2.set("beta", 2);
    cfg2.set("alpha", 1);

    EXPECT_TRUE(cfg1 == cfg2) << "Order of insertion should not affect equality";
}

TEST(TuningConfigSpec, DifferentTypesAreNotEqual)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param", 100);    // int
    cfg2.set("param", 100.0f); // float

    EXPECT_FALSE(cfg1 == cfg2) << "Same value but different type must not be equal";
}

// ============================================================================
// TYPE SAFETY TESTS
// ============================================================================

TEST(TuningConfigSpec, IntFloatBoolAreDifferentTypes)
{
    TuningConfig cfg;
    cfg.set("param", 100);

    EXPECT_THROW(cfg.get<float>("param"), std::runtime_error) << "Getting int as float must throw";
    EXPECT_THROW(cfg.get<bool>("param"), std::runtime_error) << "Getting int as bool must throw";
}

TEST(TuningConfigSpec, CanStoreAllSupportedTypes)
{
    TuningConfig cfg;
    cfg.set("int_param", 42);
    cfg.set("float_param", 3.14f);
    cfg.set("bool_param", true);

    EXPECT_EQ(cfg.get<int>("int_param"), 42);
    EXPECT_FLOAT_EQ(cfg.get<float>("float_param"), 3.14f);
    EXPECT_TRUE(cfg.get<bool>("bool_param"));
}

// ============================================================================
// INVARIANT TESTS
// ============================================================================

TEST(TuningConfigSpec, HasIsTrueOnlyAfterSet)
{
    TuningConfig cfg;
    EXPECT_FALSE(cfg.has("param"));

    cfg.set("param", 100);
    EXPECT_TRUE(cfg.has("param"));
}

TEST(TuningConfigSpec, HasReturnsFalseForNonExistent)
{
    TuningConfig cfg;
    cfg.set("param1", 100);

    EXPECT_TRUE(cfg.has("param1"));
    EXPECT_FALSE(cfg.has("param2")) << "Non-existent param must return false";
}

// ============================================================================
// NEGATIVE TESTS
// ============================================================================

TEST(TuningConfigSpec, GetThrowsOnMissingParameter)
{
    TuningConfig cfg;

    EXPECT_THROW(cfg.get<int>("nonexistent"), std::runtime_error)
        << "Getting non-existent parameter must throw";
}

TEST(TuningConfigSpec, GetThrowsOnTypeMismatch)
{
    TuningConfig cfg;
    cfg.set("param", 42); // int

    EXPECT_THROW(cfg.get<bool>("param"), std::runtime_error)
        << "Getting with wrong type must throw";
}

TEST(TuningConfigSpec, GetOrDoesNotThrowOnMissing)
{
    TuningConfig cfg;

    EXPECT_NO_THROW(cfg.get_or<int>("missing", 999))
        << "get_or must not throw on missing parameter";
}

TEST(TuningConfigSpec, GetOrReturnsDefaultOnMissing)
{
    TuningConfig cfg;

    int value = cfg.get_or<int>("missing", 42);
    EXPECT_EQ(value, 42) << "get_or must return default for missing parameter";
}

TEST(TuningConfigSpec, GetOrReturnsDefaultOnTypeMismatch)
{
    TuningConfig cfg;
    cfg.set("param", 100); // int

    bool value = cfg.get_or<bool>("param", true);
    EXPECT_TRUE(value) << "get_or must return default on type mismatch";
}

// ============================================================================
// STRESS TESTS
// ============================================================================

TEST(TuningConfigSpec, ManyParameters)
{
    TuningConfig cfg;
    const int N = 1000;

    for (int i = 0; i < N; ++i)
    {
        cfg.set("param_" + std::to_string(i), i);
    }

    EXPECT_EQ(cfg.size(), N);

    for (int i = 0; i < N; ++i)
    {
        EXPECT_EQ(cfg.get<int>("param_" + std::to_string(i)), i);
    }
}

TEST(TuningConfigSpec, RepeatedSetAndOverwrite)
{
    TuningConfig cfg;

    for (int iteration = 0; iteration < 100; ++iteration)
    {
        cfg.set("param", iteration);
        EXPECT_EQ(cfg.get<int>("param"), iteration);
        EXPECT_EQ(cfg.size(), 1) << "Size should remain 1 when overwriting";
    }
}

// ============================================================================
// ADDITIONAL BRANCH COVERAGE TESTS
// ============================================================================

// Note: TuningConfig doesn't have remove() or clear() methods
// These tests focus on what's actually available in the API

TEST(TuningConfigSpec, ToStringProducesHumanReadableFormat)
{
    TuningConfig cfg;
    cfg.set("block_x", 256);
    cfg.set("block_y", 1);

    std::string str = cfg.to_string();

    // Should contain both parameters
    EXPECT_TRUE(str.find("block_x") != std::string::npos);
    EXPECT_TRUE(str.find("block_y") != std::string::npos);
    EXPECT_TRUE(str.find("256") != std::string::npos);
    EXPECT_TRUE(str.find("1") != std::string::npos);
}

TEST(TuningConfigSpec, ToStringForEmptyConfig)
{
    TuningConfig cfg;

    std::string str = cfg.to_string();

    EXPECT_FALSE(str.empty());
    // Should indicate it's empty or have minimal content
}

TEST(TuningConfigSpec, GetOrWithTypeMismatchReturnsDefault)
{
    TuningConfig cfg;
    cfg.set("param", 42); // int

    // Try to get as float with default
    float result = cfg.get_or<float>("param", 3.14f);
    EXPECT_FLOAT_EQ(result, 3.14f);
}

TEST(TuningConfigSpec, SetBlockDimsWithZeroValues)
{
    TuningConfig cfg;
    cfg.set_block_dims(0, 0, 0);

    EXPECT_EQ(cfg.block_x(), 0);
    EXPECT_EQ(cfg.block_y(), 0);
    EXPECT_EQ(cfg.block_z(), 0);
    EXPECT_EQ(cfg.total_threads(), 0);
}

TEST(TuningConfigSpec, SetBlockDimsWithLargeValues)
{
    TuningConfig cfg;
    cfg.set_block_dims(1024, 1024, 64);

    EXPECT_EQ(cfg.block_x(), 1024);
    EXPECT_EQ(cfg.block_y(), 1024);
    EXPECT_EQ(cfg.block_z(), 64);
    EXPECT_EQ(cfg.total_threads(), 1024 * 1024 * 64);
}

TEST(TuningConfigSpec, InequalityOperatorOppositeOfEquality)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("param", 10);
    cfg2.set("param", 20);

    EXPECT_TRUE(cfg1 != cfg2);
    EXPECT_FALSE(cfg1 == cfg2);
}

TEST(TuningConfigSpec, EqualityWithSameTypesSameValues)
{
    TuningConfig cfg1, cfg2;
    cfg1.set("int_val", 42);
    cfg1.set("float_val", 3.14f);
    cfg1.set("bool_val", true);

    cfg2.set("int_val", 42);
    cfg2.set("float_val", 3.14f);
    cfg2.set("bool_val", true);

    EXPECT_TRUE(cfg1 == cfg2);
}

TEST(TuningConfigSpec, SetOverwritesWithDifferentType)
{
    TuningConfig cfg;
    cfg.set("param", 42); // int

    EXPECT_EQ(cfg.get<int>("param"), 42);

    cfg.set("param", 3.14f); // float (overwrites)

    // Old int should be gone, new float should be there
    EXPECT_THROW(cfg.get<int>("param"), std::runtime_error);
    EXPECT_FLOAT_EQ(cfg.get<float>("param"), 3.14f);
}

TEST(TuningConfigSpec, BooleanParameterHandling)
{
    TuningConfig cfg;
    cfg.set("flag_true", true);
    cfg.set("flag_false", false);

    EXPECT_TRUE(cfg.get<bool>("flag_true"));
    EXPECT_FALSE(cfg.get<bool>("flag_false"));

    EXPECT_TRUE(cfg.get_or<bool>("flag_true", false));
    EXPECT_FALSE(cfg.get_or<bool>("flag_false", true));
}

TEST(TuningConfigSpec, ZeroAndNegativeIntegerValues)
{
    TuningConfig cfg;
    cfg.set("zero", 0);
    cfg.set("negative", -100);

    EXPECT_EQ(cfg.get<int>("zero"), 0);
    EXPECT_EQ(cfg.get<int>("negative"), -100);
}

TEST(TuningConfigSpec, FloatingPointEdgeCases)
{
    TuningConfig cfg;
    cfg.set("zero", 0.0f);
    cfg.set("negative", -3.14f);
    cfg.set("small", 0.0001f);

    EXPECT_FLOAT_EQ(cfg.get<float>("zero"), 0.0f);
    EXPECT_FLOAT_EQ(cfg.get<float>("negative"), -3.14f);
    EXPECT_FLOAT_EQ(cfg.get<float>("small"), 0.0001f);
}
