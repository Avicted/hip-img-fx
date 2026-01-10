#pragma once

/**
 * @file cache_store.h
 * @brief On-disk cache management for tuned configurations
 */

#include <string>
#include <unordered_map>
#include <optional>
#include <vector>
#include <functional>

#include "tuning_config.h"

namespace imgfx::core::autotune
{
    /**
     * @brief Cache key for looking up tuned configurations
     */
    struct CacheKey
    {
        std::string gpu_arch;    ///< GPU architecture (e.g., "gfx1100")
        std::string kernel_name; ///< Kernel identifier (e.g., "grayscale")
        std::string context;     ///< Context string (e.g., "small", "blur_amount_5")

        CacheKey() = default;

        CacheKey(std::string arch, std::string name, std::string ctx)
            : gpu_arch(std::move(arch)),
              kernel_name(std::move(name)),
              context(std::move(ctx)) {}

        /**
         * @brief Serialize to string for hashing
         */
        std::string to_string() const
        {
            return gpu_arch + ":" + kernel_name + ":" + context;
        }

        /**
         * @brief Equality comparison
         */
        bool operator==(const CacheKey &other) const
        {
            return gpu_arch == other.gpu_arch &&
                   kernel_name == other.kernel_name &&
                   context == other.context;
        }

        bool operator!=(const CacheKey &other) const
        {
            return !(*this == other);
        }
    };

} // namespace imgfx::core::autotune

// Hash function for CacheKey
namespace std
{
    template <>
    struct hash<imgfx::core::autotune::CacheKey>
    {
        size_t operator()(const imgfx::core::autotune::CacheKey &k) const
        {
            // Simple hash combination
            size_t h1 = std::hash<std::string>{}(k.gpu_arch);
            size_t h2 = std::hash<std::string>{}(k.kernel_name);
            size_t h3 = std::hash<std::string>{}(k.context);

            // Combine hashes (boost hash_combine algorithm)
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

namespace imgfx::core::autotune
{
    /**
     * @brief Cache entry for serialization
     */
    struct CacheEntry
    {
        CacheKey key;
        TuningConfig config;
        float benchmark_time_ms; ///< Reference timing (optional)
        std::string timestamp;   ///< When entry was created (optional)

        CacheEntry() : benchmark_time_ms(0.0f) {}

        CacheEntry(CacheKey k, TuningConfig cfg, float time = 0.0f)
            : key(std::move(k)), config(std::move(cfg)), benchmark_time_ms(time) {}
    };

    /**
     * @brief On-disk cache management
     *
     * Provides O(1) lookup for tuned configurations with JSON persistence.
     * Thread-safe for reads; writes should be externally synchronized.
     */
    class CacheStore
    {
    public:
        CacheStore() = default;

        /**
         * @brief Load cache from disk
         *
         * @param path Path to cache file
         * @return true if cache was loaded successfully
         */
        bool load(const std::string &path = ".autotune_cache.json");

        /**
         * @brief Load cache from embedded JSON string
         *
         * @param json_string JSON content in v2.0 format
         * @return true if JSON was parsed successfully
         */
        bool load_from_string(const char *json_string);

        /**
         * @brief Save cache to disk
         *
         * @param path Path to cache file
         * @return true if cache was saved successfully
         */
        bool save(const std::string &path = ".autotune_cache.json") const;

        /**
         * @brief Lookup configuration in cache
         *
         * @param key Cache key
         * @return Configuration if found, std::nullopt otherwise
         */
        std::optional<TuningConfig> lookup(const CacheKey &key) const
        {
            auto it = cache_.find(key);
            if (it != cache_.end())
            {
                return it->second.config;
            }
            return std::nullopt;
        }

        /**
         * @brief Insert or update configuration in cache
         *
         * @param key Cache key
         * @param config Configuration to store
         * @param time_ms Benchmark time (optional, for reference)
         */
        void insert(const CacheKey &key, const TuningConfig &config, float time_ms = 0.0f)
        {
            cache_[key] = CacheEntry(key, config, time_ms);
        }

        /**
         * @brief Check if cache contains key
         *
         * @param key Cache key
         * @return true if key exists in cache
         */
        bool contains(const CacheKey &key) const
        {
            return cache_.find(key) != cache_.end();
        }

        /**
         * @brief Get number of cached entries
         */
        size_t size() const { return cache_.size(); }

        /**
         * @brief Check if cache is empty
         */
        bool empty() const { return cache_.empty(); }

        /**
         * @brief Clear all cached entries
         */
        void clear() { cache_.clear(); }

        /**
         * @brief Get all cache entries (for inspection/debugging)
         */
        std::vector<CacheEntry> entries() const
        {
            std::vector<CacheEntry> result;
            result.reserve(cache_.size());
            for (const auto &[_, entry] : cache_)
            {
                result.push_back(entry);
            }
            return result;
        }

        /**
         * @brief Remove entries matching predicate
         *
         * Example: Remove entries for specific GPU
         *     cache.remove_if([](const CacheEntry& e) {
         *         return e.key.gpu_arch == "gfx1030";
         *     });
         */
        template <typename Predicate>
        void remove_if(Predicate pred)
        {
            for (auto it = cache_.begin(); it != cache_.end();)
            {
                if (pred(it->second))
                {
                    it = cache_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

    private:
        std::unordered_map<CacheKey, CacheEntry> cache_;

        // JSON serialization helpers (implementation in .cpp)
        std::string serialize() const;
        void deserialize(const std::string &json_content);
    };

} // namespace imgfx::core::autotune
