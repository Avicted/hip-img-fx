#include "autotuning.h"
#include "gpu_utils.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace imgfx::core
{
    // Simple JSON parser/writer for cache persistence
    namespace json_utils
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

        std::string to_json(const TunedConfigCache &entry)
        {
            std::ostringstream oss;
            oss << "  {\n";
            oss << "    \"gpu_arch\": \"" << escape_json_string(entry.gpu_arch) << "\",\n";
            oss << "    \"kernel_name\": \"" << escape_json_string(entry.kernel_name) << "\",\n";
            oss << "    \"image_size_cat\": \"" << escape_json_string(entry.image_size_cat) << "\",\n";
            oss << "    \"block_x\": " << entry.config.block_x << ",\n";
            oss << "    \"block_y\": " << entry.config.block_y << ",\n";
            oss << "    \"avg_time_ms\": " << entry.avg_time_ms << "\n";
            oss << "  }";
            return oss.str();
        }

        bool parse_cache_entry(const std::string &json, TunedConfigCache &entry)
        {
            // Simple parser for our specific JSON format
            auto find_value = [&](const std::string &key) -> std::string
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
                    while (end < json.length() && (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+' || json[end] == 'e' || json[end] == 'E'))
                        end++;
                    return json.substr(start, end - start);
                }
            };

            entry.gpu_arch = find_value("gpu_arch");
            entry.kernel_name = find_value("kernel_name");
            entry.image_size_cat = find_value("image_size_cat");

            std::string block_x_str = find_value("block_x");
            std::string block_y_str = find_value("block_y");
            std::string avg_time_str = find_value("avg_time_ms");

            if (block_x_str.empty() || block_y_str.empty())
                return false;

            try
            {
                entry.config.block_x = std::stoi(block_x_str);
                entry.config.block_y = std::stoi(block_y_str);
                entry.avg_time_ms = avg_time_str.empty() ? 0.0f : std::stof(avg_time_str);
            }
            catch (...)
            {
                return false;
            }

            return true;
        }

        std::vector<TunedConfigCache> parse_cache_file(const std::string &content)
        {
            std::vector<TunedConfigCache> result;

            // Find array of entries
            size_t array_start = content.find('[');
            size_t array_end = content.rfind(']');

            if (array_start == std::string::npos || array_end == std::string::npos)
                return result;

            std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);

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
                TunedConfigCache entry;
                if (parse_cache_entry(obj_content, entry))
                {
                    result.push_back(entry);
                }

                pos = obj_end + 1;
            }

            return result;
        }
    } // namespace json_utils

    AutoTuner::AutoTuner()
    {
        m_gpu_arch = query_gpu_arch();
    }

    AutoTuner::~AutoTuner()
    {
        // Auto-save cache on destruction
        save_cache();
    }

    std::string AutoTuner::query_gpu_arch()
    {
        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, 0);
        if (err != hipSuccess)
        {
            fprintf(stderr, "Warning: Failed to query GPU properties for autotuning\n");
            return "unknown";
        }

        // AMD GPUs use gcnArchName (e.g., "gfx1100")
        std::string arch_name = prop.gcnArchName;
        if (arch_name.empty())
        {
            // Fallback: use device name
            arch_name = prop.name;
        }

        return arch_name;
    }

    std::vector<KernelConfig> AutoTuner::get_candidate_configs() const
    {
        // AMD-friendly block sizes (wavefront = 64)
        // Total threads: 128-256 per block
        // Mix of 1D and 2D shapes
        return {
            KernelConfig(64, 1),  // 64 threads (1 wavefront)
            KernelConfig(128, 1), // 128 threads (2 wavefronts)
            KernelConfig(256, 1), // 256 threads (4 wavefronts)
            KernelConfig(16, 8),  // 128 threads (2D)
            KernelConfig(16, 16), // 256 threads (2D)
            KernelConfig(32, 8),  // 256 threads (2D, wider)
        };
    }

    BenchmarkResult AutoTuner::benchmark_config(
        const KernelConfig &config,
        LaunchFunc launch_func,
        void *args,
        hipStream_t stream,
        int warmup_runs,
        int timing_runs)
    {
        // Warmup phase
        for (int i = 0; i < warmup_runs; ++i)
        {
            launch_func(config, stream, args);
        }
        HIP_ERRCHK(hipStreamSynchronize(stream));

        // Timing phase with HIP events
        HIPEvent start_event, end_event;
        if (!start_event.is_valid() || !end_event.is_valid())
        {
            fprintf(stderr, "Warning: Failed to create HIP events for benchmarking\n");
            return BenchmarkResult();
        }

        start_event.record(stream);

        for (int i = 0; i < timing_runs; ++i)
        {
            launch_func(config, stream, args);
        }

        end_event.record(stream);
        end_event.synchronize();

        float total_ms = HIPEvent::elapsed_time(start_event, end_event);
        float avg_ms = total_ms / timing_runs;

        return BenchmarkResult(config, avg_ms);
    }

    KernelConfig AutoTuner::autotune_kernel(
        const std::string &kernel_name,
        LaunchFunc launch_func,
        void *args,
        int warmup_runs,
        int timing_runs)
    {
        printf("[AutoTuner] Tuning kernel '%s' for GPU arch '%s'...\n",
               kernel_name.c_str(), m_gpu_arch.c_str());

        // Create dedicated stream for benchmarking
        hipStream_t stream;
        HIP_ERRCHK(hipStreamCreate(&stream));

        std::vector<KernelConfig> candidates = get_candidate_configs();
        std::vector<BenchmarkResult> results;

        // Benchmark each candidate
        for (const auto &config : candidates)
        {
            BenchmarkResult result = benchmark_config(
                config, launch_func, args, stream, warmup_runs, timing_runs);

            if (result.valid)
            {
                results.push_back(result);
                printf("  [%dx%d] = %.4f ms\n",
                       config.block_x, config.block_y, result.avg_time_ms);
            }
        }

        HIP_ERRCHK(hipStreamDestroy(stream));

        if (results.empty())
        {
            fprintf(stderr, "[AutoTuner] Warning: No valid benchmark results, using default config\n");
            return KernelConfig(256, 1);
        }

        // Select fastest configuration
        auto best = std::min_element(results.begin(), results.end(),
                                     [](const BenchmarkResult &a, const BenchmarkResult &b)
                                     {
                                         return a.avg_time_ms < b.avg_time_ms;
                                     });

        printf("[AutoTuner] Selected config [%dx%d] with avg time %.4f ms\n",
               best->config.block_x, best->config.block_y, best->avg_time_ms);

        // Add to cache
        TunedConfigCache cache_entry;
        cache_entry.gpu_arch = m_gpu_arch;
        cache_entry.kernel_name = kernel_name;
        cache_entry.image_size_cat = "";
        cache_entry.config = best->config;
        cache_entry.avg_time_ms = best->avg_time_ms;

        // Remove existing entry for this kernel if present
        m_cache.erase(
            std::remove_if(m_cache.begin(), m_cache.end(),
                           [&](const TunedConfigCache &e)
                           {
                               return e.gpu_arch == m_gpu_arch && e.kernel_name == kernel_name;
                           }),
            m_cache.end());

        m_cache.push_back(cache_entry);

        return best->config;
    }

    KernelConfig AutoTuner::get_config(
        const std::string &kernel_name,
        LaunchFunc launch_func,
        void *args,
        int warmup_runs,
        int timing_runs)
    {
        // Check cache first
        for (const auto &entry : m_cache)
        {
            if (entry.gpu_arch == m_gpu_arch && entry.kernel_name == kernel_name)
            {
                // printf("[AutoTuner] Using cached config [%dx%d] for kernel '%s'\n", entry.config.block_x, entry.config.block_y, kernel_name.c_str());
                return entry.config;
            }
        }

        // Not in cache - perform autotuning
        return autotune_kernel(kernel_name, launch_func, args, warmup_runs, timing_runs);
    }

    bool AutoTuner::has_cached_config(const std::string &kernel_name) const
    {
        for (const auto &entry : m_cache)
        {
            if (entry.gpu_arch == m_gpu_arch && entry.kernel_name == kernel_name)
            {
                return true;
            }
        }
        return false;
    }

    std::string AutoTuner::get_gpu_arch() const
    {
        return m_gpu_arch;
    }

    bool AutoTuner::load_cache(const std::string &cache_path)
    {
        std::ifstream file(cache_path);
        if (!file.is_open())
        {
            // Cache file doesn't exist yet - not an error
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        m_cache = json_utils::parse_cache_file(content);

        printf("[AutoTuner] Loaded %zu cached configurations from '%s'\n",
               m_cache.size(), cache_path.c_str());

        return !m_cache.empty();
    }

    bool AutoTuner::save_cache(const std::string &cache_path) const
    {
        if (m_cache.empty())
        {
            return true; // Nothing to save
        }

        std::ofstream file(cache_path);
        if (!file.is_open())
        {
            fprintf(stderr, "[AutoTuner] Warning: Failed to open cache file '%s' for writing\n",
                    cache_path.c_str());
            return false;
        }

        file << "{\n";
        file << "  \"version\": \"1.0\",\n";
        file << "  \"entries\": [\n";

        for (size_t i = 0; i < m_cache.size(); ++i)
        {
            file << json_utils::to_json(m_cache[i]);
            if (i < m_cache.size() - 1)
            {
                file << ",";
            }
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";
        file.close();

        printf("[AutoTuner] Saved %zu configurations to cache '%s'\n",
               m_cache.size(), cache_path.c_str());

        return true;
    }

} // namespace imgfx::core
