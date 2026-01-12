/**
 * @file test_cache_store_spec.cpp
 * @brief Specification tests for CacheStore public API (cache_store.h)
 *
 * Following SKILL.md guidelines:
 * - Encode contracts not implementation
 * - Negative tests required for every positive case
 * - Break-the-code validation
 */

#include <gtest/gtest.h>
#include "hip-img-fx/autotune/cache_store.h"
#include <filesystem>
#include <fstream>

using namespace imgfx::core::autotune;

// ============================================================================
// TEST SUITE: CacheKey Specification
// ============================================================================

TEST(CacheKeySpec, DefaultConstructorCreatesEmptyKey)
{
    CacheKey key;
    EXPECT_TRUE(key.gpu_arch.empty());
    EXPECT_TRUE(key.kernel_name.empty());
    EXPECT_TRUE(key.context.empty());
}

TEST(CacheKeySpec, ParameterizedConstructorStoresValues)
{
    CacheKey key("gfx1100", "grayscale", "small");
    EXPECT_EQ(key.gpu_arch, "gfx1100");
    EXPECT_EQ(key.kernel_name, "grayscale");
    EXPECT_EQ(key.context, "small");
}

TEST(CacheKeySpec, ToStringProducesStableSerializedForm)
{
    CacheKey key("gfx1100", "grayscale", "small");
    std::string expected = "gfx1100:grayscale:small";
    EXPECT_EQ(key.to_string(), expected);

    // Stability: calling twice should give same result
    EXPECT_EQ(key.to_string(), key.to_string());
}

TEST(CacheKeySpec, ToStringWithEmptyFieldsProducesColons)
{
    CacheKey key("", "", "");
    EXPECT_EQ(key.to_string(), "::");
}

TEST(CacheKeySpec, EqualityIsReflexive)
{
    CacheKey key("gfx1100", "grayscale", "small");
    EXPECT_TRUE(key == key);
}

TEST(CacheKeySpec, EqualityIsSymmetric)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "grayscale", "small");
    EXPECT_TRUE(key1 == key2);
    EXPECT_TRUE(key2 == key1);
}

TEST(CacheKeySpec, EqualityIsTransitive)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "grayscale", "small");
    CacheKey key3("gfx1100", "grayscale", "small");
    EXPECT_TRUE(key1 == key2);
    EXPECT_TRUE(key2 == key3);
    EXPECT_TRUE(key1 == key3);
}

TEST(CacheKeySpec, DifferentArchMeansNotEqual)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1030", "grayscale", "small");
    EXPECT_FALSE(key1 == key2);
    EXPECT_TRUE(key1 != key2);
}

TEST(CacheKeySpec, DifferentKernelMeansNotEqual)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "blur", "small");
    EXPECT_FALSE(key1 == key2);
    EXPECT_TRUE(key1 != key2);
}

TEST(CacheKeySpec, DifferentContextMeansNotEqual)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "grayscale", "large");
    EXPECT_FALSE(key1 == key2);
    EXPECT_TRUE(key1 != key2);
}

TEST(CacheKeySpec, HashFunctionProducesStableHashes)
{
    CacheKey key("gfx1100", "grayscale", "small");
    std::hash<CacheKey> hasher;
    size_t hash1 = hasher(key);
    size_t hash2 = hasher(key);
    EXPECT_EQ(hash1, hash2);
}

TEST(CacheKeySpec, EqualKeysHaveSameHash)
{
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "grayscale", "small");
    std::hash<CacheKey> hasher;
    EXPECT_EQ(hasher(key1), hasher(key2));
}

TEST(CacheKeySpec, DifferentKeysShouldHaveDifferentHashes)
{
    // Note: hash collisions are possible, but extremely unlikely for this test
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1030", "blur", "large");
    std::hash<CacheKey> hasher;
    // We can't guarantee different hashes, but for vastly different keys, they should differ
    EXPECT_NE(hasher(key1), hasher(key2));
}

// ============================================================================
// TEST SUITE: CacheEntry Specification
// ============================================================================

TEST(CacheEntrySpec, DefaultConstructorCreatesEmptyEntry)
{
    CacheEntry entry;
    EXPECT_TRUE(entry.key.gpu_arch.empty());
    EXPECT_TRUE(entry.config.empty());
    EXPECT_EQ(entry.benchmark_time_ms, 0.0f);
}

TEST(CacheEntrySpec, ParameterizedConstructorStoresValues)
{
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;
    config.set("block_x", 256);
    config.set("block_y", 1);

    CacheEntry entry(key, config, 1.234f);

    EXPECT_EQ(entry.key, key);
    EXPECT_EQ(entry.config.block_x(), 256);
    EXPECT_EQ(entry.benchmark_time_ms, 1.234f);
}

TEST(CacheEntrySpec, ConstructorWithoutTimeDefaultsToZero)
{
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;

    CacheEntry entry(key, config);

    EXPECT_EQ(entry.benchmark_time_ms, 0.0f);
}

// ============================================================================
// TEST SUITE: CacheStore Basic Operations
// ============================================================================

TEST(CacheStoreSpec, DefaultConstructorCreatesEmptyStore)
{
    CacheStore store;
    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), 0u);
}

TEST(CacheStoreSpec, InsertIncreasesSize)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;
    config.set("block_x", 256);

    store.insert(key, config);

    EXPECT_EQ(store.size(), 1u);
    EXPECT_FALSE(store.empty());
}

TEST(CacheStoreSpec, InsertThenLookupRetrievesValue)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;
    config.set("block_x", 256);

    store.insert(key, config);
    auto result = store.lookup(key);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->block_x(), 256);
}

TEST(CacheStoreSpec, LookupOnMissingKeyReturnsNullopt)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");

    auto result = store.lookup(key);

    EXPECT_FALSE(result.has_value());
}

TEST(CacheStoreSpec, ContainsReturnsTrueForExistingKey)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;

    store.insert(key, config);

    EXPECT_TRUE(store.contains(key));
}

TEST(CacheStoreSpec, ContainsReturnsFalseForMissingKey)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");

    EXPECT_FALSE(store.contains(key));
}

TEST(CacheStoreSpec, InsertWithSameKeyUpdatesValue)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");

    TuningConfig config1;
    config1.set("block_x", 256);
    store.insert(key, config1);

    TuningConfig config2;
    config2.set("block_x", 512);
    store.insert(key, config2);

    EXPECT_EQ(store.size(), 1u); // Still one entry
    auto result = store.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->block_x(), 512); // Updated value
}

TEST(CacheStoreSpec, ClearRemovesAllEntries)
{
    CacheStore store;
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1030", "blur", "large");
    TuningConfig config;

    store.insert(key1, config);
    store.insert(key2, config);
    EXPECT_EQ(store.size(), 2u);

    store.clear();

    EXPECT_EQ(store.size(), 0u);
    EXPECT_TRUE(store.empty());
}

TEST(CacheStoreSpec, EntriesReturnsAllCachedEntries)
{
    CacheStore store;
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1030", "blur", "large");
    TuningConfig config;

    store.insert(key1, config, 1.0f);
    store.insert(key2, config, 2.0f);

    auto entries = store.entries();

    EXPECT_EQ(entries.size(), 2u);
}

TEST(CacheStoreSpec, RemoveIfWithPredicateRemovesMatchingEntries)
{
    CacheStore store;
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1100", "blur", "large");
    CacheKey key3("gfx1030", "grayscale", "small");
    TuningConfig config;

    store.insert(key1, config);
    store.insert(key2, config);
    store.insert(key3, config);

    // Remove all entries for gfx1100
    store.remove_if([](const CacheEntry &e)
                    { return e.key.gpu_arch == "gfx1100"; });

    EXPECT_EQ(store.size(), 1u);
    EXPECT_TRUE(store.contains(key3));
    EXPECT_FALSE(store.contains(key1));
    EXPECT_FALSE(store.contains(key2));
}

// ============================================================================
// TEST SUITE: CacheStore Persistence
// ============================================================================

TEST(CacheStoreSpec, SaveToNonExistentPathCreatesFile)
{
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_cache_save.json";

    // Clean up if exists
    std::filesystem::remove(temp_path);

    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;
    config.set("block_x", 256);
    store.insert(key, config);

    bool success = store.save(temp_path.string());

    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(temp_path));

    // Cleanup
    std::filesystem::remove(temp_path);
}

TEST(CacheStoreSpec, LoadFromExistingFilePopulatesCache)
{
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_cache_load.json";

    // Create cache and save
    {
        CacheStore store;
        CacheKey key("gfx1100", "grayscale", "small");
        TuningConfig config;
        config.set("block_x", 256);
        store.insert(key, config);
        store.save(temp_path.string());
    }

    // Load into new cache
    {
        CacheStore store;
        bool success = store.load(temp_path.string());

        EXPECT_TRUE(success);
        EXPECT_EQ(store.size(), 1u);

        CacheKey key("gfx1100", "grayscale", "small");
        auto result = store.lookup(key);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->block_x(), 256);
    }

    // Cleanup
    std::filesystem::remove(temp_path);
}

TEST(CacheStoreSpec, LoadFromNonExistentFileReturnsFalse)
{
    CacheStore store;
    bool success = store.load("/nonexistent/path/to/cache.json");
    EXPECT_FALSE(success);
}

TEST(CacheStoreSpec, LoadFromStringPopulatesCache)
{
    const char *json_content = R"({
        "version": "2.0",
        "entries": [
            {
                "gpu_arch": "gfx1100",
                "kernel_name": "grayscale",
                "context": "small",
                "config": "block_x=256,block_y=1",
                "benchmark_time_ms": 1.5,
                "timestamp": "2026-01-01T00:00:00"
            }
        ]
    })";

    CacheStore store;
    bool success = store.load_from_string(json_content);

    EXPECT_TRUE(success);
    EXPECT_EQ(store.size(), 1u);

    CacheKey key("gfx1100", "grayscale", "small");
    auto result = store.lookup(key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->block_x(), 256);
}

TEST(CacheStoreSpec, LoadFromInvalidJsonReturnsFalse)
{
    const char *invalid_json = "{ this is not valid json }";

    CacheStore store;
    bool success = store.load_from_string(invalid_json);

    EXPECT_FALSE(success);
    EXPECT_TRUE(store.empty());
}

TEST(CacheStoreSpec, SaveAndLoadRoundTripPreservesData)
{
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_cache_roundtrip.json";

    // Create cache with multiple entries
    CacheKey key1("gfx1100", "grayscale", "small");
    CacheKey key2("gfx1030", "blur", "large");
    TuningConfig config1, config2;
    config1.set("block_x", 256);
    config1.set("block_y", 1);
    config2.set("block_x", 512);
    config2.set("block_y", 2);

    // Save
    {
        CacheStore store;
        store.insert(key1, config1, 1.234f);
        store.insert(key2, config2, 5.678f);
        store.save(temp_path.string());
    }

    // Load and verify
    {
        CacheStore store;
        store.load(temp_path.string());

        EXPECT_EQ(store.size(), 2u);

        auto result1 = store.lookup(key1);
        ASSERT_TRUE(result1.has_value());
        EXPECT_EQ(result1->block_x(), 256);
        EXPECT_EQ(result1->block_y(), 1);

        auto result2 = store.lookup(key2);
        ASSERT_TRUE(result2.has_value());
        EXPECT_EQ(result2->block_x(), 512);
        EXPECT_EQ(result2->block_y(), 2);
    }

    // Cleanup
    std::filesystem::remove(temp_path);
}

// ============================================================================
// TEST SUITE: CacheStore Modified Flag
// ============================================================================

TEST(CacheStoreSpec, ModifiedFlagIsFalseInitially)
{
    CacheStore store;
    EXPECT_FALSE(store.is_modified());
}

TEST(CacheStoreSpec, InsertSetsModifiedFlag)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;

    store.insert(key, config);

    EXPECT_TRUE(store.is_modified());
}

TEST(CacheStoreSpec, ClearSetsModifiedFlag)
{
    CacheStore store;
    store.clear();
    EXPECT_TRUE(store.is_modified());
}

TEST(CacheStoreSpec, ResetModifiedClearsFlag)
{
    CacheStore store;
    CacheKey key("gfx1100", "grayscale", "small");
    TuningConfig config;
    store.insert(key, config);

    EXPECT_TRUE(store.is_modified());
    store.reset_modified();
    EXPECT_FALSE(store.is_modified());
}

TEST(CacheStoreSpec, LoadDoesNotSetModifiedFlag)
{
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_cache_modified.json";

    // Create and save cache
    {
        CacheStore store;
        CacheKey key("gfx1100", "grayscale", "small");
        TuningConfig config;
        store.insert(key, config);
        store.save(temp_path.string());
    }

    // Load and check modified flag
    {
        CacheStore store;
        store.load(temp_path.string());
        EXPECT_FALSE(store.is_modified());
    }

    // Cleanup
    std::filesystem::remove(temp_path);
}

// ============================================================================
// TEST SUITE: CacheStore Stress Tests
// ============================================================================

TEST(CacheStoreSpec, ManyEntriesCanBeStoredAndRetrieved)
{
    CacheStore store;
    const int num_entries = 1000;

    // Insert many entries
    for (int i = 0; i < num_entries; ++i)
    {
        CacheKey key("gfx1100", "kernel_" + std::to_string(i), "ctx");
        TuningConfig config;
        config.set("block_x", i);
        store.insert(key, config);
    }

    EXPECT_EQ(store.size(), static_cast<size_t>(num_entries));

    // Verify all entries can be retrieved
    for (int i = 0; i < num_entries; ++i)
    {
        CacheKey key("gfx1100", "kernel_" + std::to_string(i), "ctx");
        auto result = store.lookup(key);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->block_x(), i);
    }
}

TEST(CacheStoreSpec, LargeConfigurationsCanBeStored)
{
    CacheStore store;
    CacheKey key("gfx1100", "test", "large_config");

    TuningConfig config;
    for (int i = 0; i < 100; ++i)
    {
        config.set("param_" + std::to_string(i), i);
    }

    store.insert(key, config);
    auto result = store.lookup(key);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 100u);
}
