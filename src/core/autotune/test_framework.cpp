/**
 * @file test_autotune_framework.cpp
 * @brief Unit-test-style validation for Phase 1 autotuning framework
 *
 * This is NOT a comprehensive test suite, but a quick validation
 * that the core components work as expected.
 */

#include "../src/core/autotune/tuning_config.h"
#include "../src/core/autotune/cache_store.h"
#include <iostream>
#include <cassert>
#include <cstdio>

using namespace imgfx::core::autotune;

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name)                 \
    printf("\n[TEST] %s\n", name); \
    bool test_passed = true;

#define ASSERT(condition, message)            \
    if (!(condition))                         \
    {                                         \
        printf("  ❌ FAILED: %s\n", message); \
        test_passed = false;                  \
        tests_failed++;                       \
    }

#define END_TEST()               \
    if (test_passed)             \
    {                            \
        printf("  ✅ PASSED\n"); \
        tests_passed++;          \
    }

void test_tuning_config_basic()
{
    TEST("TuningConfig - Basic operations");

    TuningConfig cfg;

    // Test setting and getting values
    cfg.set("block_x", 128);
    cfg.set("block_y", 2);
    cfg.set("use_smem", true);
    cfg.set("threshold", 0.5f);

    ASSERT(cfg.get<int>("block_x") == 128, "get block_x");
    ASSERT(cfg.get<int>("block_y") == 2, "get block_y");
    ASSERT(cfg.get<bool>("use_smem") == true, "get use_smem");
    ASSERT(cfg.get<float>("threshold") == 0.5f, "get threshold");

    // Test has()
    ASSERT(cfg.has("block_x"), "has block_x");
    ASSERT(!cfg.has("nonexistent"), "!has nonexistent");

    // Test get_or()
    ASSERT(cfg.get_or("block_x", 64) == 128, "get_or existing");
    ASSERT(cfg.get_or("missing", 64) == 64, "get_or missing");

    // Test size
    ASSERT(cfg.size() == 4, "size");

    END_TEST();
}

void test_tuning_config_convenience()
{
    TEST("TuningConfig - Convenience accessors");

    TuningConfig cfg;
    cfg.set_block_dims(256, 1, 1);

    ASSERT(cfg.block_x() == 256, "block_x()");
    ASSERT(cfg.block_y() == 1, "block_y()");
    ASSERT(cfg.block_z() == 1, "block_z()");
    ASSERT(cfg.total_threads() == 256, "total_threads()");

    END_TEST();
}

void test_tuning_config_serialization()
{
    TEST("TuningConfig - Serialization");

    TuningConfig cfg;
    cfg.set("block_x", 128);
    cfg.set("block_y", 2);
    cfg.set("vec_width", 4);

    // to_key_string should be deterministic (sorted keys)
    std::string key = cfg.to_key_string();
    ASSERT(!key.empty(), "to_key_string not empty");
    ASSERT(key.find("block_x=128") != std::string::npos, "contains block_x");
    ASSERT(key.find("block_y=2") != std::string::npos, "contains block_y");
    ASSERT(key.find("vec_width=4") != std::string::npos, "contains vec_width");

    // Round-trip test
    TuningConfig cfg2 = TuningConfig::from_key_string(key);
    ASSERT(cfg2.get<int>("block_x") == 128, "round-trip block_x");
    ASSERT(cfg2.get<int>("block_y") == 2, "round-trip block_y");
    ASSERT(cfg2.get<int>("vec_width") == 4, "round-trip vec_width");

    // Test equality
    ASSERT(cfg == cfg2, "equality after round-trip");

    END_TEST();
}

void test_tuning_config_to_string()
{
    TEST("TuningConfig - Debug string");

    TuningConfig cfg;
    cfg.set("block_x", 128);
    cfg.set("enabled", true);

    std::string str = cfg.to_string();
    ASSERT(!str.empty(), "to_string not empty");
    ASSERT(str.find("TuningConfig{") != std::string::npos, "contains header");
    printf("  Debug string: %s\n", str.c_str());

    END_TEST();
}

void test_cache_key()
{
    TEST("CacheKey - Basic operations");

    CacheKey key1{"gfx1030", "grayscale", "small"};
    CacheKey key2{"gfx1030", "grayscale", "small"};
    CacheKey key3{"gfx1030", "grayscale", "large"};

    ASSERT(key1 == key2, "equality");
    ASSERT(key1 != key3, "inequality");

    std::string str = key1.to_string();
    ASSERT(str == "gfx1030:grayscale:small", "to_string format");

    // Test hashing (for unordered_map)
    std::hash<CacheKey> hasher;
    size_t h1 = hasher(key1);
    size_t h2 = hasher(key2);
    size_t h3 = hasher(key3);

    ASSERT(h1 == h2, "hash equality for equal keys");
    ASSERT(h1 != h3, "hash inequality for different keys");

    END_TEST();
}

void test_cache_store_basic()
{
    TEST("CacheStore - Basic operations");

    CacheStore cache;

    CacheKey key{"gfx1030", "test_kernel", "default"};
    TuningConfig cfg;
    cfg.set_block_dims(128, 1);

    // Initially empty
    ASSERT(cache.empty(), "initially empty");
    ASSERT(cache.size() == 0, "size 0");
    ASSERT(!cache.contains(key), "!contains before insert");

    // Insert
    cache.insert(key, cfg, 1.5f);
    ASSERT(!cache.empty(), "not empty after insert");
    ASSERT(cache.size() == 1, "size 1");
    ASSERT(cache.contains(key), "contains after insert");

    // Lookup
    auto result = cache.lookup(key);
    ASSERT(result.has_value(), "lookup returns value");
    ASSERT(result->block_x() == 128, "lookup correct block_x");

    // Lookup non-existent
    CacheKey missing_key{"gfx1030", "nonexistent", "default"};
    auto missing = cache.lookup(missing_key);
    ASSERT(!missing.has_value(), "lookup missing returns nullopt");

    END_TEST();
}

void test_cache_store_persistence()
{
    TEST("CacheStore - Save/Load");

    const char *test_file = "/tmp/test_autotune_cache.json";

    // Create and save
    {
        CacheStore cache;

        CacheKey key1{"gfx1030", "kernel1", "small"};
        TuningConfig cfg1;
        cfg1.set_block_dims(128, 1);
        cache.insert(key1, cfg1, 1.2f);

        CacheKey key2{"gfx1030", "kernel2", "large"};
        TuningConfig cfg2;
        cfg2.set_block_dims(256, 2);
        cache.insert(key2, cfg2, 2.5f);

        bool saved = cache.save(test_file);
        ASSERT(saved, "save succeeded");
    }

    // Load and verify
    {
        CacheStore cache;
        bool loaded = cache.load(test_file);
        ASSERT(loaded, "load succeeded");
        ASSERT(cache.size() == 2, "loaded 2 entries");

        CacheKey key1{"gfx1030", "kernel1", "small"};
        auto cfg1 = cache.lookup(key1);
        ASSERT(cfg1.has_value(), "key1 found");
        ASSERT(cfg1->block_x() == 128, "key1 block_x");

        CacheKey key2{"gfx1030", "kernel2", "large"};
        auto cfg2 = cache.lookup(key2);
        ASSERT(cfg2.has_value(), "key2 found");
        ASSERT(cfg2->block_x() == 256, "key2 block_x");
    }

    // Cleanup
    std::remove(test_file);

    END_TEST();
}

void test_cache_store_entries()
{
    TEST("CacheStore - Get entries");

    CacheStore cache;

    CacheKey key1{"gfx1030", "kernel1", "small"};
    TuningConfig cfg1;
    cfg1.set_block_dims(128, 1);
    cache.insert(key1, cfg1);

    CacheKey key2{"gfx1030", "kernel2", "large"};
    TuningConfig cfg2;
    cfg2.set_block_dims(256, 1);
    cache.insert(key2, cfg2);

    auto entries = cache.entries();
    ASSERT(entries.size() == 2, "entries size");

    // Verify entries contain correct data
    bool found_kernel1 = false;
    bool found_kernel2 = false;
    for (const auto &entry : entries)
    {
        if (entry.key.kernel_name == "kernel1")
        {
            found_kernel1 = true;
            ASSERT(entry.config.block_x() == 128, "kernel1 config");
        }
        if (entry.key.kernel_name == "kernel2")
        {
            found_kernel2 = true;
            ASSERT(entry.config.block_x() == 256, "kernel2 config");
        }
    }
    ASSERT(found_kernel1 && found_kernel2, "found both kernels");

    END_TEST();
}

void test_cache_store_remove_if()
{
    TEST("CacheStore - Remove if");

    CacheStore cache;

    cache.insert(CacheKey{"gfx1030", "kernel1", "small"}, TuningConfig{});
    cache.insert(CacheKey{"gfx1030", "kernel2", "small"}, TuningConfig{});
    cache.insert(CacheKey{"gfx1100", "kernel3", "small"}, TuningConfig{});

    ASSERT(cache.size() == 3, "initial size");

    // Remove all gfx1030 entries
    cache.remove_if([](const CacheEntry &e)
                    { return e.key.gpu_arch == "gfx1030"; });

    ASSERT(cache.size() == 1, "size after remove_if");
    ASSERT(cache.contains(CacheKey{"gfx1100", "kernel3", "small"}), "gfx1100 entry remains");
    ASSERT(!cache.contains(CacheKey{"gfx1030", "kernel1", "small"}), "gfx1030 entry removed");

    END_TEST();
}

void test_cache_store_clear()
{
    TEST("CacheStore - Clear");

    CacheStore cache;
    cache.insert(CacheKey{"gfx1030", "kernel1", "small"}, TuningConfig{});
    cache.insert(CacheKey{"gfx1030", "kernel2", "small"}, TuningConfig{});

    ASSERT(cache.size() == 2, "size before clear");

    cache.clear();

    ASSERT(cache.empty(), "empty after clear");
    ASSERT(cache.size() == 0, "size 0 after clear");

    END_TEST();
}

int main()
{
    printf("===========================================\n");
    printf("Phase 1 Autotuning Framework Validation\n");
    printf("===========================================\n");

    // TuningConfig tests
    test_tuning_config_basic();
    test_tuning_config_convenience();
    test_tuning_config_serialization();
    test_tuning_config_to_string();

    // CacheKey tests
    test_cache_key();

    // CacheStore tests
    test_cache_store_basic();
    test_cache_store_persistence();
    test_cache_store_entries();
    test_cache_store_remove_if();
    test_cache_store_clear();

    // Summary
    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("===========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
