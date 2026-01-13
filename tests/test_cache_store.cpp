// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include "hip-img-fx/autotune/cache_store.h"
#include <fstream>
#include <cstdio>

/**
 * @brief Cache store tests - always run (CPU-only)
 *
 * Tests JSON persistence and cache key generation.
 */

TEST(CacheStore, SaveAndLoadJSON)
{
    using namespace imgfx::core::autotune;

    const char *test_path = "/tmp/test_cache_store.json";

    // Create a cache with some entries
    CacheStore cache;

    TuningConfig config1;
    config1.set_block_dims(256, 1, 1);

    CacheKey key1("gfx1030", "grayscale", "small");
    cache.insert(key1, config1, 1.5f);

    // Save to file
    bool save_success = cache.save(test_path);
    EXPECT_TRUE(save_success) << "Failed to save cache to file";

    // Load from file
    CacheStore loaded_cache;
    bool load_success = loaded_cache.load(test_path);
    EXPECT_TRUE(load_success) << "Failed to load cache from file";

    // Verify entry exists
    EXPECT_TRUE(loaded_cache.contains(key1));
    auto retrieved = loaded_cache.lookup(key1);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->block_x(), 256);

    // Cleanup
    std::remove(test_path);
}

TEST(CacheStore, EmptyCache)
{
    using namespace imgfx::core::autotune;

    CacheStore empty_cache;

    // Empty cache should be valid
    EXPECT_TRUE(empty_cache.empty());
    EXPECT_EQ(empty_cache.size(), 0);

    const char *test_path = "/tmp/test_empty_cache.json";
    bool saved = empty_cache.save(test_path);
    EXPECT_TRUE(saved) << "Empty cache should save successfully";

    std::remove(test_path);
}

TEST(CacheStore, InvalidJSON)
{
    using namespace imgfx::core::autotune;

    const char *test_path = "/tmp/test_invalid_cache.json";

    // Write invalid JSON
    {
        std::ofstream file(test_path);
        file << "{ invalid json content !!!";
    }

    CacheStore cache;
    bool load_success = cache.load(test_path);

    // Should handle invalid JSON gracefully (may return false or use default)
    EXPECT_FALSE(load_success) << "Loading invalid JSON should fail gracefully";

    std::remove(test_path);
}

TEST(CacheStore, CacheKeyGeneration)
{
    using namespace imgfx::core::autotune;

    // Test that cache keys are generated consistently
    CacheKey key1("gfx1030", "grayscale", "small");
    CacheKey key2("gfx1030", "grayscale", "small");
    CacheKey key3("gfx1100", "grayscale", "small");

    // Same keys should be equal
    EXPECT_EQ(key1, key2);
    EXPECT_EQ(key1.to_string(), key2.to_string());

    // Different keys should not be equal
    EXPECT_NE(key1, key3);
    EXPECT_NE(key1.to_string(), key3.to_string());

    // Test CacheStore operations with keys
    CacheStore cache;
    TuningConfig config;
    config.set_block_dims(128, 1, 1);

    cache.insert(key1, config);
    EXPECT_TRUE(cache.contains(key1));
    EXPECT_TRUE(cache.contains(key2));  // Same key
    EXPECT_FALSE(cache.contains(key3)); // Different key
}
