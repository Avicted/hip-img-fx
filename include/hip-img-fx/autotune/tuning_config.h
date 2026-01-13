// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#pragma once

/**
 * @file tuning_config.h
 * @brief Extensible configuration container for tunable parameters
 */

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace imgfx::core::autotune
{
    /**
     * @brief Extensible kernel configuration with arbitrary parameters
     *
     * Supports adding new tunable parameters without modifying core framework.
     * Uses variant-based storage for type safety.
     *
     * Example usage:
     *     TuningConfig cfg;
     *     cfg.set("block_x", 128);
     *     cfg.set("block_y", 1);
     *     cfg.set("vec_width", 4);
     *     cfg.set("use_shared_mem", true);
     *
     *     int bx = cfg.block_x();  // Type-safe accessor
     *     int vw = cfg.get<int>("vec_width");  // Generic accessor
     */
    class TuningConfig
    {
    public:
        using ParamValue = std::variant<int, float, bool>;

        TuningConfig() = default;

        /**
         * @brief Set a parameter value
         *
         * @param name Parameter name
         * @param value Parameter value (int, float, or bool)
         */
        void set(const std::string &name, ParamValue value)
        {
            params_[name] = value;
        }

        /**
         * @brief Get a parameter value with type checking
         *
         * @tparam T Expected parameter type
         * @param name Parameter name
         * @return Parameter value
         * @throws std::runtime_error if parameter not found or wrong type
         */
        template <typename T>
        T get(const std::string &name) const
        {
            auto it = params_.find(name);
            if (it == params_.end())
            {
                throw std::runtime_error("Parameter '" + name + "' not found");
            }

            try
            {
                return std::get<T>(it->second);
            }
            catch (const std::bad_variant_access &)
            {
                throw std::runtime_error("Parameter '" + name + "' has wrong type");
            }
        }

        /**
         * @brief Get a parameter value with default fallback
         *
         * @tparam T Parameter type
         * @param name Parameter name
         * @param default_value Value to return if parameter not found
         * @return Parameter value or default
         */
        template <typename T>
        T get_or(const std::string &name, T default_value) const
        {
            auto it = params_.find(name);
            if (it == params_.end())
            {
                return default_value;
            }

            try
            {
                return std::get<T>(it->second);
            }
            catch (const std::bad_variant_access &)
            {
                return default_value;
            }
        }

        /**
         * @brief Check if parameter exists
         *
         * @param name Parameter name
         * @return true if parameter is set
         */
        bool has(const std::string &name) const
        {
            return params_.find(name) != params_.end();
        }

        /**
         * @brief Get number of parameters
         */
        size_t size() const { return params_.size(); }

        /**
         * @brief Check if config is empty
         */
        bool empty() const { return params_.empty(); }

        // Common parameter accessors (for convenience)

        int block_x() const { return get_or<int>("block_x", 256); }
        int block_y() const { return get_or<int>("block_y", 1); }
        int block_z() const { return get_or<int>("block_z", 1); }

        void set_block_dims(int x, int y, int z = 1)
        {
            set("block_x", x);
            set("block_y", y);
            set("block_z", z);
        }

        /**
         * @brief Total threads per block
         */
        int total_threads() const
        {
            return block_x() * block_y() * block_z();
        }

        /**
         * @brief Convert to string for cache key
         *
         * Format: "key1=val1,key2=val2,..."
         * Keys are sorted alphabetically for consistency.
         */
        std::string to_key_string() const;

        /**
         * @brief Parse from cache key string
         *
         * Format: "key1=val1,key2=val2,..."
         *
         * @param str Key string
         * @return Parsed TuningConfig
         */
        static TuningConfig from_key_string(const std::string &str);

        /**
         * @brief Equality comparison
         */
        bool operator==(const TuningConfig &other) const
        {
            if (params_.size() != other.params_.size())
                return false;

            for (const auto &[key, value] : params_)
            {
                auto it = other.params_.find(key);
                if (it == other.params_.end() || it->second != value)
                {
                    return false;
                }
            }

            return true;
        }

        bool operator!=(const TuningConfig &other) const
        {
            return !(*this == other);
        }

        /**
         * @brief Get all parameter names
         */
        std::vector<std::string> keys() const
        {
            std::vector<std::string> result;
            result.reserve(params_.size());
            for (const auto &[key, _] : params_)
            {
                result.push_back(key);
            }
            return result;
        }

        /**
         * @brief Debug string representation
         */
        std::string to_string() const;

    private:
        std::unordered_map<std::string, ParamValue> params_;
    };

} // namespace imgfx::core::autotune
