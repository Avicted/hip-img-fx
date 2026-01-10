#include "hip-img-fx/autotune/tuning_config.h"
#include <sstream>

namespace imgfx::core::autotune
{
    std::string TuningConfig::to_key_string() const
    {
        if (params_.empty())
        {
            return "";
        }

        // Sort keys for consistent ordering
        std::vector<std::string> keys_vec;
        keys_vec.reserve(params_.size());
        for (const auto &[key, _] : params_)
        {
            keys_vec.push_back(key);
        }
        std::sort(keys_vec.begin(), keys_vec.end());

        // Build key string
        std::ostringstream oss;
        bool first = true;
        for (const auto &key : keys_vec)
        {
            if (!first)
                oss << ",";
            first = false;

            oss << key << "=";

            const auto &value = params_.at(key);
            std::visit([&oss](auto &&arg)
                       { oss << arg; },
                       value);
        }

        return oss.str();
    }

    TuningConfig TuningConfig::from_key_string(const std::string &str)
    {
        TuningConfig config;

        if (str.empty())
        {
            return config;
        }

        // Split by comma
        size_t pos = 0;
        while (pos < str.length())
        {
            size_t comma = str.find(',', pos);
            if (comma == std::string::npos)
                comma = str.length();

            std::string pair = str.substr(pos, comma - pos);

            // Split by equals
            size_t eq = pair.find('=');
            if (eq != std::string::npos)
            {
                std::string key = pair.substr(0, eq);
                std::string value_str = pair.substr(eq + 1);

                // Try to parse as int, then bool, then float
                if (value_str == "true" || value_str == "false")
                {
                    config.set(key, value_str == "true");
                }
                else if (value_str.find('.') != std::string::npos)
                {
                    try
                    {
                        config.set(key, std::stof(value_str));
                    }
                    catch (...)
                    {
                        // If float parsing fails, ignore this parameter
                    }
                }
                else
                {
                    try
                    {
                        config.set(key, std::stoi(value_str));
                    }
                    catch (...)
                    {
                        // If int parsing fails, ignore this parameter
                    }
                }
            }

            pos = comma + 1;
        }

        return config;
    }

    std::string TuningConfig::to_string() const
    {
        std::ostringstream oss;
        oss << "TuningConfig{";

        bool first = true;
        for (const auto &[key, value] : params_)
        {
            if (!first)
                oss << ", ";
            first = false;

            oss << key << "=";
            std::visit([&oss](auto &&arg)
                       { oss << arg; },
                       value);
        }

        oss << "}";
        return oss.str();
    }

} // namespace imgfx::core::autotune
