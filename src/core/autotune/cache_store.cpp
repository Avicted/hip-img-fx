#include "cache_store.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <cstdio>

namespace imgfx::core::autotune
{
    // Simple JSON serialization helpers
    namespace
    {
        std::string escape_json_string(const std::string &s)
        {
            std::string result;
            for (char c : s)
            {
                if (c == '"' || c == '\\')
                {
                    result += '\\';
                }
                result += c;
            }
            return result;
        }

        std::string find_json_value(const std::string &json, const std::string &key)
        {
            std::string search_key = "\"" + key + "\":";
            size_t pos = json.find(search_key);
            if (pos == std::string::npos)
                return "";

            pos += search_key.length();
            while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t'))
                pos++;

            if (pos >= json.length())
                return "";

            if (json[pos] == '"')
            {
                // String value
                size_t start = pos + 1;
                size_t end = json.find('"', start);
                if (end == std::string::npos)
                    return "";
                return json.substr(start, end - start);
            }
            else
            {
                // Numeric value
                size_t start = pos;
                size_t end = start;
                while (end < json.length() &&
                       (std::isdigit(json[end]) || json[end] == '.' ||
                        json[end] == '-' || json[end] == '+' ||
                        json[end] == 'e' || json[end] == 'E'))
                    end++;
                return json.substr(start, end - start);
            }
        }
    }

    std::string CacheStore::serialize() const
    {
        std::ostringstream oss;

        oss << "{\n";
        oss << "  \"version\": \"2.0\",\n";
        oss << "  \"entries\": [\n";

        bool first = true;
        for (const auto &[key, entry] : cache_)
        {
            if (!first)
                oss << ",\n";
            first = false;

            oss << "    {\n";
            oss << "      \"gpu_arch\": \"" << escape_json_string(entry.key.gpu_arch) << "\",\n";
            oss << "      \"kernel_name\": \"" << escape_json_string(entry.key.kernel_name) << "\",\n";
            oss << "      \"context\": \"" << escape_json_string(entry.key.context) << "\",\n";
            oss << "      \"config\": \"" << escape_json_string(entry.config.to_key_string()) << "\",\n";
            oss << "      \"benchmark_time_ms\": " << entry.benchmark_time_ms << ",\n";
            oss << "      \"timestamp\": \"" << escape_json_string(entry.timestamp) << "\"\n";
            oss << "    }";
        }

        oss << "\n  ]\n";
        oss << "}\n";

        return oss.str();
    }

    void CacheStore::deserialize(const std::string &json_content)
    {
        cache_.clear();

        // Find array of entries
        size_t array_start = json_content.find('[');
        size_t array_end = json_content.rfind(']');

        if (array_start == std::string::npos || array_end == std::string::npos)
            return;

        std::string array_content = json_content.substr(array_start + 1, array_end - array_start - 1);

        // Split by object boundaries
        size_t pos = 0;
        while (pos < array_content.length())
        {
            size_t obj_start = array_content.find('{', pos);
            if (obj_start == std::string::npos)
                break;

            size_t obj_end = array_content.find('}', obj_start);
            if (obj_end == std::string::npos)
                break;

            std::string obj_content = array_content.substr(obj_start, obj_end - obj_start + 1);

            // Parse entry
            std::string gpu_arch = find_json_value(obj_content, "gpu_arch");
            std::string kernel_name = find_json_value(obj_content, "kernel_name");
            std::string context = find_json_value(obj_content, "context");
            std::string config_str = find_json_value(obj_content, "config");
            std::string time_str = find_json_value(obj_content, "benchmark_time_ms");
            std::string timestamp = find_json_value(obj_content, "timestamp");

            if (!gpu_arch.empty() && !kernel_name.empty() && !config_str.empty())
            {
                CacheKey key{gpu_arch, kernel_name, context};
                TuningConfig config = TuningConfig::from_key_string(config_str);

                float time_ms = 0.0f;
                if (!time_str.empty())
                {
                    try
                    {
                        time_ms = std::stof(time_str);
                    }
                    catch (...)
                    {
                        time_ms = 0.0f;
                    }
                }

                CacheEntry entry{key, config, time_ms};
                entry.timestamp = timestamp;
                cache_[key] = entry;
            }

            pos = obj_end + 1;
        }
    }

    bool CacheStore::load(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            // Cache file doesn't exist yet - not an error
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        deserialize(content);

        if (!cache_.empty())
        {
            printf("[AutoTuner] Loaded %zu cached configurations from '%s'\n",
                   cache_.size(), path.c_str());
        }

        return !cache_.empty();
    }

    bool CacheStore::load_from_string(const char *json_string)
    {
        if (json_string == nullptr || json_string[0] == '\0')
        {
            return false;
        }

        size_t prev_size = cache_.size();
        deserialize(std::string(json_string));

        if (cache_.size() > prev_size)
        {
            printf("[AutoTuner] Loaded %zu embedded default configurations\n",
                   cache_.size() - prev_size);
        }

        return cache_.size() > prev_size;
    }

    bool CacheStore::save(const std::string &path) const
    {
        if (cache_.empty())
        {
            return true; // Nothing to save
        }

        std::ofstream file(path);
        if (!file.is_open())
        {
            fprintf(stderr, "[AutoTuner] Warning: Failed to open cache file '%s' for writing\n",
                    path.c_str());
            return false;
        }

        file << serialize();
        file.close();

        printf("[AutoTuner] Saved %zu configurations to cache '%s'\n",
               cache_.size(), path.c_str());

        return true;
    }

} // namespace imgfx::core::autotune
