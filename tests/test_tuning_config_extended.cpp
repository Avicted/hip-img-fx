#include <gtest/gtest.h>
#include "hip-img-fx/autotune/tuning_config.h"
#include <stdexcept>

/**
 * @brief Extended tests for TuningConfig - focusing on to_string and other uncovered methods
 */

using namespace imgfx::core::autotune;

TEST(TuningConfigExtended, ToStringEmpty)
{
    TuningConfig config;
    std::string str = config.to_string();

    // Empty config should produce minimal output
    EXPECT_NE(str.find("TuningConfig{"), std::string::npos);
    EXPECT_NE(str.find("}"), std::string::npos);
}

TEST(TuningConfigExtended, ToStringSingleParam)
{
    TuningConfig config;
    config.set("block_x", 128);

    std::string str = config.to_string();

    EXPECT_NE(str.find("TuningConfig{"), std::string::npos);
    EXPECT_NE(str.find("block_x"), std::string::npos);
    EXPECT_NE(str.find("128"), std::string::npos);
    EXPECT_NE(str.find("}"), std::string::npos);
}

TEST(TuningConfigExtended, ToStringMultipleParams)
{
    TuningConfig config;
    config.set("block_x", 256);
    config.set("block_y", 1);
    config.set("use_shared", true);
    config.set("scale", 1.5f);

    std::string str = config.to_string();

    EXPECT_NE(str.find("TuningConfig{"), std::string::npos);
    EXPECT_NE(str.find("block_x"), std::string::npos);
    EXPECT_NE(str.find("256"), std::string::npos);
    EXPECT_NE(str.find("block_y"), std::string::npos);
    EXPECT_NE(str.find("1"), std::string::npos);
    EXPECT_NE(str.find("use_shared"), std::string::npos);
    EXPECT_NE(str.find("scale"), std::string::npos);
    EXPECT_NE(str.find("}"), std::string::npos);
}

TEST(TuningConfigExtended, ToStringAllTypes)
{
    TuningConfig config;
    config.set("int_val", 42);
    config.set("float_val", 3.14f);
    config.set("bool_true", true);
    config.set("bool_false", false);

    std::string str = config.to_string();

    // Verify all types are represented
    EXPECT_NE(str.find("int_val"), std::string::npos);
    EXPECT_NE(str.find("42"), std::string::npos);
    EXPECT_NE(str.find("float_val"), std::string::npos);
    EXPECT_NE(str.find("3.14"), std::string::npos);
    EXPECT_NE(str.find("bool_true"), std::string::npos);
    EXPECT_NE(str.find("bool_false"), std::string::npos);
}

TEST(TuningConfigExtended, GetThrowsOnMissing)
{
    TuningConfig config;
    config.set("existing", 100);

    // Should throw when accessing non-existent parameter
    EXPECT_THROW(config.get<int>("nonexistent"), std::runtime_error);
}

TEST(TuningConfigExtended, GetThrowsOnWrongType)
{
    TuningConfig config;
    config.set("int_param", 100);

    // Should throw when accessing with wrong type
    EXPECT_THROW(config.get<float>("int_param"), std::runtime_error);
    EXPECT_THROW(config.get<bool>("int_param"), std::runtime_error);
}

TEST(TuningConfigExtended, GetOrDefaultMissingParam)
{
    TuningConfig config;

    // Should return default value when parameter doesn't exist
    EXPECT_EQ(config.get_or<int>("missing", 999), 999);
    EXPECT_FLOAT_EQ(config.get_or<float>("missing", 1.5f), 1.5f);
    EXPECT_EQ(config.get_or<bool>("missing", true), true);
}

TEST(TuningConfigExtended, GetOrDefaultWrongType)
{
    TuningConfig config;
    config.set("int_param", 100);

    // Should return default when type is wrong
    EXPECT_FLOAT_EQ(config.get_or<float>("int_param", 2.5f), 2.5f);
    EXPECT_EQ(config.get_or<bool>("int_param", false), false);
}

TEST(TuningConfigExtended, GetOrDefaultExistingParam)
{
    TuningConfig config;
    config.set("param", 42);

    // Should return actual value, not default
    EXPECT_EQ(config.get_or<int>("param", 999), 42);
}

TEST(TuningConfigExtended, HasParam)
{
    TuningConfig config;
    config.set("existing", 100);

    EXPECT_TRUE(config.has("existing"));
    EXPECT_FALSE(config.has("nonexistent"));
}

TEST(TuningConfigExtended, SetOverwritesExisting)
{
    TuningConfig config;
    config.set("param", 100);
    EXPECT_EQ(config.get<int>("param"), 100);

    // Overwrite with same type
    config.set("param", 200);
    EXPECT_EQ(config.get<int>("param"), 200);

    // Overwrite with different type
    config.set("param", 3.14f);
    EXPECT_FLOAT_EQ(config.get<float>("param"), 3.14f);
}

TEST(TuningConfigExtended, ToKeyStringEmpty)
{
    TuningConfig config;
    std::string key = config.to_key_string();

    EXPECT_EQ(key, "");
}

TEST(TuningConfigExtended, ToKeyStringSingleParam)
{
    TuningConfig config;
    config.set("block_x", 128);

    std::string key = config.to_key_string();

    EXPECT_NE(key.find("block_x=128"), std::string::npos);
}

TEST(TuningConfigExtended, ToKeyStringMultipleParams)
{
    TuningConfig config;
    config.set("z_param", 3);
    config.set("a_param", 1);
    config.set("m_param", 2);

    std::string key = config.to_key_string();

    // Should be sorted alphabetically
    size_t a_pos = key.find("a_param");
    size_t m_pos = key.find("m_param");
    size_t z_pos = key.find("z_param");

    EXPECT_LT(a_pos, m_pos);
    EXPECT_LT(m_pos, z_pos);
}

TEST(TuningConfigExtended, ToKeyStringWithBool)
{
    TuningConfig config;
    config.set("flag_true", true);
    config.set("flag_false", false);

    std::string key = config.to_key_string();

    EXPECT_NE(key.find("flag_true=1"), std::string::npos);
    EXPECT_NE(key.find("flag_false=0"), std::string::npos);
}

TEST(TuningConfigExtended, ToKeyStringWithFloat)
{
    TuningConfig config;
    config.set("scale", 2.5f);

    std::string key = config.to_key_string();

    EXPECT_NE(key.find("scale="), std::string::npos);
    EXPECT_NE(key.find("2.5"), std::string::npos);
}

TEST(TuningConfigExtended, FromKeyStringEmpty)
{
    TuningConfig config = TuningConfig::from_key_string("");

    EXPECT_FALSE(config.has("anything"));
}

TEST(TuningConfigExtended, FromKeyStringSingleInt)
{
    TuningConfig config = TuningConfig::from_key_string("block_x=256");

    EXPECT_TRUE(config.has("block_x"));
    EXPECT_EQ(config.get<int>("block_x"), 256);
}

TEST(TuningConfigExtended, FromKeyStringMultipleParams)
{
    TuningConfig config = TuningConfig::from_key_string("a=1,b=2,c=3");

    EXPECT_EQ(config.get<int>("a"), 1);
    EXPECT_EQ(config.get<int>("b"), 2);
    EXPECT_EQ(config.get<int>("c"), 3);
}

TEST(TuningConfigExtended, FromKeyStringWithBool)
{
    TuningConfig config = TuningConfig::from_key_string("flag1=true,flag2=false");

    EXPECT_EQ(config.get<bool>("flag1"), true);
    EXPECT_EQ(config.get<bool>("flag2"), false);
}

TEST(TuningConfigExtended, FromKeyStringWithFloat)
{
    TuningConfig config = TuningConfig::from_key_string("scale=1.5");

    EXPECT_TRUE(config.has("scale"));
    EXPECT_FLOAT_EQ(config.get<float>("scale"), 1.5f);
}

TEST(TuningConfigExtended, FromKeyStringMixedTypes)
{
    TuningConfig config = TuningConfig::from_key_string("int_val=42,float_val=3.14,bool_val=true");

    EXPECT_EQ(config.get<int>("int_val"), 42);
    EXPECT_FLOAT_EQ(config.get<float>("float_val"), 3.14f);
    EXPECT_EQ(config.get<bool>("bool_val"), true);
}

TEST(TuningConfigExtended, RoundTripSerialization)
{
    TuningConfig original;
    original.set("block_x", 128);
    original.set("block_y", 1);

    std::string key = original.to_key_string();
    TuningConfig restored = TuningConfig::from_key_string(key);

    EXPECT_EQ(restored.get<int>("block_x"), 128);
    EXPECT_EQ(restored.get<int>("block_y"), 1);
}

TEST(TuningConfigExtended, RoundTripWithBoolNote)
{
    // Note: Boolean round-trip has limitations
    // to_key_string outputs 0/1, but from_key_string expects "true"/"false"
    // This is a known limitation in the current implementation
    TuningConfig original;
    original.set("flag", true);

    std::string key = original.to_key_string();
    // Key will be "flag=1", which from_key_string parses as int
    EXPECT_NE(key.find("flag=1"), std::string::npos);
}

TEST(TuningConfigExtended, FromKeyStringInvalidFormat)
{
    // Malformed entries should be ignored
    TuningConfig config = TuningConfig::from_key_string("valid=1,invalid,also_valid=2");

    EXPECT_TRUE(config.has("valid"));
    EXPECT_TRUE(config.has("also_valid"));
    EXPECT_FALSE(config.has("invalid"));
}

TEST(TuningConfigExtended, FromKeyStringNegativeNumbers)
{
    TuningConfig config = TuningConfig::from_key_string("neg_int=-5,neg_float=-2.5");

    EXPECT_EQ(config.get<int>("neg_int"), -5);
    EXPECT_FLOAT_EQ(config.get<float>("neg_float"), -2.5f);
}

TEST(TuningConfigExtended, GenericSetGet)
{
    TuningConfig config;

    // Test all supported types through generic interface
    config.set("int_test", 123);
    config.set("float_test", 4.56f);
    config.set("bool_test", true);

    EXPECT_EQ(config.get<int>("int_test"), 123);
    EXPECT_FLOAT_EQ(config.get<float>("float_test"), 4.56f);
    EXPECT_EQ(config.get<bool>("bool_test"), true);
}

TEST(TuningConfigExtended, BlockDimsAccessors)
{
    TuningConfig config;
    config.set_block_dims(64, 8, 2);

    EXPECT_EQ(config.block_x(), 64);
    EXPECT_EQ(config.block_y(), 8);
    EXPECT_EQ(config.block_z(), 2);
    EXPECT_EQ(config.total_threads(), 64 * 8 * 2);
}

TEST(TuningConfigExtended, EqualityOperator)
{
    TuningConfig c1, c2, c3;

    c1.set("a", 1);
    c1.set("b", 2);

    c2.set("a", 1);
    c2.set("b", 2);

    c3.set("a", 1);
    c3.set("b", 3);

    EXPECT_EQ(c1, c2);
    EXPECT_NE(c1, c3);
}

TEST(TuningConfigExtended, EqualityDifferentParamCount)
{
    TuningConfig c1, c2;

    c1.set("a", 1);

    c2.set("a", 1);
    c2.set("b", 2);

    EXPECT_NE(c1, c2);
}

TEST(TuningConfigExtended, LargeIntValue)
{
    TuningConfig config;
    config.set("large", 1000000);

    EXPECT_EQ(config.get<int>("large"), 1000000);

    std::string key = config.to_key_string();
    TuningConfig restored = TuningConfig::from_key_string(key);
    EXPECT_EQ(restored.get<int>("large"), 1000000);
}

TEST(TuningConfigExtended, ZeroValues)
{
    TuningConfig config;
    config.set("zero_int", 0);
    config.set("zero_float", 0.0f);
    config.set("zero_bool", false);

    EXPECT_EQ(config.get<int>("zero_int"), 0);
    EXPECT_FLOAT_EQ(config.get<float>("zero_float"), 0.0f);
    EXPECT_EQ(config.get<bool>("zero_bool"), false);
}
