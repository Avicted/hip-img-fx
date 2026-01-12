#include <gtest/gtest.h>
#include "hip-img-fx/autotune/tuning_config.h"
#include "hip-img-fx/autotune/cache_store.h"
#include <fstream>
#include <filesystem>

/**
 * @brief Extended autotune tests for better coverage
 *
 * Tests for TuningConfig serialization edge cases and CacheStore operations.
 */

using namespace imgfx::core::autotune;

namespace fs = std::filesystem;

class AutotuneExtendedTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_dir_ = fs::path("/tmp/hip_img_fx_autotune_tests");
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

// TuningConfig Tests

TEST_F(AutotuneExtendedTest, TuningConfigEmpty)
{
    TuningConfig config;

    std::string key = config.to_key_string();
    EXPECT_EQ(key, "");
}

TEST_F(AutotuneExtendedTest, TuningConfigSingleParameter)
{
    TuningConfig config;
    config.set("threads", 256);

    std::string key = config.to_key_string();
    EXPECT_EQ(key, "threads=256");
}

TEST_F(AutotuneExtendedTest, TuningConfigMultipleParameters)
{
    TuningConfig config;
    config.set("threads", 512);
    config.set("blocks", 128);
    config.set("use_shared", true);

    std::string key = config.to_key_string();

    // Should have all three parameters (order is sorted alphabetically)
    EXPECT_NE(key.find("threads=512"), std::string::npos);
    EXPECT_NE(key.find("blocks=128"), std::string::npos);
    EXPECT_NE(key.find("use_shared=1"), std::string::npos) << "Actual key: " << key;
}

TEST_F(AutotuneExtendedTest, TuningConfigFromKeyStringEmpty)
{
    std::string empty_key = "";
    auto config = TuningConfig::from_key_string(empty_key);

    EXPECT_EQ(config.to_key_string(), "");
}

TEST_F(AutotuneExtendedTest, TuningConfigFromKeyStringSingle)
{
    std::string key = "threads=256";
    auto config = TuningConfig::from_key_string(key);

    EXPECT_EQ(config.get_or<int>("threads", 0), 256);
}

TEST_F(AutotuneExtendedTest, TuningConfigFromKeyStringMultiple)
{
    std::string key = "blocks=64,threads=128,use_shared=true";
    auto config = TuningConfig::from_key_string(key);

    EXPECT_EQ(config.get_or<int>("blocks", 0), 64);
    EXPECT_EQ(config.get_or<int>("threads", 0), 128);
    EXPECT_EQ(config.get_or<bool>("use_shared", false), true);
}

TEST_F(AutotuneExtendedTest, TuningConfigFromKeyStringWithFloat)
{
    std::string key = "alpha=0.5,beta=1.25";
    auto config = TuningConfig::from_key_string(key);

    float alpha = config.get_or<float>("alpha", 0.0f);
    float beta = config.get_or<float>("beta", 0.0f);

    EXPECT_NEAR(alpha, 0.5f, 0.001f);
    EXPECT_NEAR(beta, 1.25f, 0.001f);
}

TEST_F(AutotuneExtendedTest, TuningConfigFromKeyStringWithBool)
{
    std::string key = "enabled=true,disabled=false";
    auto config = TuningConfig::from_key_string(key);

    EXPECT_EQ(config.get_or<bool>("enabled", false), true);
    EXPECT_EQ(config.get_or<bool>("disabled", true), false);
}

TEST_F(AutotuneExtendedTest, TuningConfigGetNonExistent)
{
    TuningConfig config;
    config.set("threads", 256);

    // Get non-existent key with default
    int value = config.get_or<int>("blocks", 999);
    EXPECT_EQ(value, 999);
}

TEST_F(AutotuneExtendedTest, TuningConfigRoundTrip)
{
    TuningConfig original;
    original.set("threads", 512);
    original.set("blocks", 256);
    original.set("warp_size", 32);

    std::string key = original.to_key_string();
    auto restored = TuningConfig::from_key_string(key);

    EXPECT_EQ(restored.get_or<int>("threads", 0), 512);
    EXPECT_EQ(restored.get_or<int>("blocks", 0), 256);
    EXPECT_EQ(restored.get_or<int>("warp_size", 0), 32);
}

// CacheStore Tests

TEST_F(AutotuneExtendedTest, CacheStoreEmpty)
{
    CacheStore store;

    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), 0);
}

TEST_F(AutotuneExtendedTest, CacheStoreSingleEntry)
{
    CacheStore store;

    TuningConfig config;
    config.set("threads", 256);

    CacheKey key("gfx1030", "grayscale_kernel", "64x64x3");
    store.insert(key, config, 1.5f);

    EXPECT_FALSE(store.empty());
    EXPECT_EQ(store.size(), 1);
    EXPECT_TRUE(store.contains(key));
}

TEST_F(AutotuneExtendedTest, CacheStoreMultipleEntries)
{
    CacheStore store;

    TuningConfig config1;
    config1.set("threads", 256);

    TuningConfig config2;
    config2.set("threads", 512);

    CacheKey key1("gfx1030", "kernel1", "context1");
    CacheKey key2("gfx1030", "kernel2", "context2");

    store.insert(key1, config1, 1.0f);
    store.insert(key2, config2, 2.0f);

    EXPECT_EQ(store.size(), 2);
    EXPECT_TRUE(store.contains(key1));
    EXPECT_TRUE(store.contains(key2));
}

TEST_F(AutotuneExtendedTest, CacheStoreLookupSuccess)
{
    CacheStore store;

    TuningConfig config;
    config.set("threads", 256);
    config.set("blocks", 128);

    CacheKey key("gfx1030", "test_kernel", "100x100x3");
    store.insert(key, config, 2.5f);

    auto result = store.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get_or<int>("threads", 0), 256);
    EXPECT_EQ(result->get_or<int>("blocks", 0), 128);
}

TEST_F(AutotuneExtendedTest, CacheStoreLookupNotFound)
{
    CacheStore store;

    CacheKey key("gfx1030", "nonexistent", "context");
    auto result = store.lookup(key);
    EXPECT_FALSE(result.has_value());
}

TEST_F(AutotuneExtendedTest, CacheStoreSerializeDeserialize)
{
    CacheStore store1;

    TuningConfig config;
    config.set("threads", 512);
    config.set("blocks", 256);

    CacheKey key("gfx1030", "kernel_a", "128x128x4");
    store1.insert(key, config, 3.14f);

    fs::path cache_file = test_dir_ / "test_save_load.json";

    // Save to file
    bool saved = store1.save(cache_file.string());
    ASSERT_TRUE(saved);

    // Load into new store
    CacheStore store2;
    bool loaded = store2.load(cache_file.string());
    ASSERT_TRUE(loaded);

    auto result = store2.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get_or<int>("threads", 0), 512);
    EXPECT_EQ(result->get_or<int>("blocks", 0), 256);
}

TEST_F(AutotuneExtendedTest, CacheStoreFileOperations)
{
    CacheStore store1;

    TuningConfig config;
    config.set("threads", 1024);

    CacheKey key("gfx1030", "my_kernel", "256x256x3");
    store1.insert(key, config, 5.0f);

    fs::path cache_file = test_dir_ / "test_cache.json";

    // Save to file
    bool saved = store1.save(cache_file.string());
    ASSERT_TRUE(saved);
    EXPECT_TRUE(fs::exists(cache_file));

    // Load from file
    CacheStore store2;
    bool loaded = store2.load(cache_file.string());
    ASSERT_TRUE(loaded);

    auto result = store2.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get_or<int>("threads", 0), 1024);
}

TEST_F(AutotuneExtendedTest, CacheStoreLoadNonExistentFile)
{
    CacheStore store;

    fs::path nonexistent = test_dir_ / "does_not_exist.json";
    bool loaded = store.load(nonexistent.string());

    EXPECT_FALSE(loaded);
}

TEST_F(AutotuneExtendedTest, CacheStoreClearOperation)
{
    CacheStore store;

    TuningConfig config;
    config.set("threads", 256);

    CacheKey key1("gfx1030", "kernel1", "context1");
    CacheKey key2("gfx1030", "kernel2", "context2");

    store.insert(key1, config);
    store.insert(key2, config);

    EXPECT_EQ(store.size(), 2);

    store.clear();

    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), 0);
    EXPECT_FALSE(store.contains(key1));
    EXPECT_FALSE(store.contains(key2));
}

TEST_F(AutotuneExtendedTest, CacheStoreOverwriteEntry)
{
    CacheStore store;

    TuningConfig config1;
    config1.set("threads", 256);

    TuningConfig config2;
    config2.set("threads", 512);

    CacheKey key("gfx1030", "kernel", "context");

    // Add first entry
    store.insert(key, config1, 1.0f);

    // Add second entry with same key (should overwrite)
    store.insert(key, config2, 2.0f);

    EXPECT_EQ(store.size(), 1); // Should still be 1 entry
    auto result = store.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get_or<int>("threads", 0), 512); // Should have second value
}

TEST_F(AutotuneExtendedTest, CacheStoreDifferentGPUArchs)
{
    CacheStore store;

    TuningConfig config1;
    config1.set("threads", 256);

    TuningConfig config2;
    config2.set("threads", 512);

    CacheKey key1("gfx1030", "kernel", "context");
    CacheKey key2("gfx1100", "kernel", "context");

    store.insert(key1, config1, 1.0f);
    store.insert(key2, config2, 2.0f);

    auto result1 = store.lookup(key1);
    auto result2 = store.lookup(key2);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result1->get_or<int>("threads", 0), 256);
    EXPECT_EQ(result2->get_or<int>("threads", 0), 512);
}

TEST_F(AutotuneExtendedTest, CacheStoreEntriesMethod)
{
    CacheStore store;

    TuningConfig config1;
    config1.set("threads", 256);

    TuningConfig config2;
    config2.set("threads", 512);

    CacheKey key1("gfx1030", "kernel1", "context1");
    CacheKey key2("gfx1030", "kernel2", "context2");

    store.insert(key1, config1);
    store.insert(key2, config2);

    auto entries = store.entries();
    EXPECT_EQ(entries.size(), 2);

    // Verify entries are present
    bool found1 = false, found2 = false;
    for (const auto &entry : entries)
    {
        if (entry.key == key1)
        {
            found1 = true;
        }
        if (entry.key == key2)
        {
            found2 = true;
        }
    }
    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

TEST_F(AutotuneExtendedTest, CacheStoreModifiedFlag)
{
    CacheStore store;

    TuningConfig config;
    config.set("threads", 256);

    CacheKey key("gfx1030", "kernel", "context");

    // Initially not modified
    EXPECT_FALSE(store.is_modified());

    // Insert should mark as modified
    store.insert(key, config);
    EXPECT_TRUE(store.is_modified());

    // Reset modified flag
    store.reset_modified();
    EXPECT_FALSE(store.is_modified());

    // Clear should mark as modified
    store.clear();
    EXPECT_TRUE(store.is_modified());
}

TEST_F(AutotuneExtendedTest, CacheKeyEquality)
{
    CacheKey key1("gfx1030", "kernel_a", "context_1");
    CacheKey key2("gfx1030", "kernel_a", "context_1");
    CacheKey key3("gfx1100", "kernel_a", "context_1");

    EXPECT_EQ(key1, key2);
    EXPECT_NE(key1, key3);
}

TEST_F(AutotuneExtendedTest, CacheKeyToString)
{
    CacheKey key("gfx1030", "my_kernel", "my_context");
    std::string str = key.to_string();

    EXPECT_EQ(str, "gfx1030:my_kernel:my_context");
}

TEST_F(AutotuneExtendedTest, CacheStoreRemoveIf)
{
    CacheStore store;

    TuningConfig config;
    config.set("threads", 256);

    CacheKey key1("gfx1030", "kernel1", "context1");
    CacheKey key2("gfx1100", "kernel2", "context2");
    CacheKey key3("gfx1030", "kernel3", "context3");

    store.insert(key1, config);
    store.insert(key2, config);
    store.insert(key3, config);

    EXPECT_EQ(store.size(), 3);

    // Remove all gfx1030 entries
    store.remove_if([](const CacheEntry &entry)
                    { return entry.key.gpu_arch == "gfx1030"; });

    EXPECT_EQ(store.size(), 1);
    EXPECT_FALSE(store.contains(key1));
    EXPECT_TRUE(store.contains(key2));
    EXPECT_FALSE(store.contains(key3));
}
