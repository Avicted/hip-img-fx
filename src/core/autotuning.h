#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace imgfx::core
{
    /**
     * @brief Kernel launch configuration for block dimensions
     */
    struct KernelConfig
    {
        int block_x;
        int block_y;

        KernelConfig(int bx = 256, int by = 1) : block_x(bx), block_y(by) {}

        int total_threads() const { return block_x * block_y; }

        bool operator==(const KernelConfig &other) const
        {
            return block_x == other.block_x && block_y == other.block_y;
        }
    };

    /**
     * @brief Benchmark result for a specific kernel configuration
     */
    struct BenchmarkResult
    {
        KernelConfig config;
        float avg_time_ms;
        bool valid;

        BenchmarkResult() : config(), avg_time_ms(0.0f), valid(false) {}
        BenchmarkResult(const KernelConfig &cfg, float time)
            : config(cfg), avg_time_ms(time), valid(true) {}
    };

    /**
     * @brief Cache entry for storing tuned kernel configurations
     */
    struct TunedConfigCache
    {
        std::string gpu_arch;       // e.g., "gfx1100"
        std::string kernel_name;    // e.g., "grayscale"
        std::string image_size_cat; // Optional: width x height in pixels
        KernelConfig config;
        float avg_time_ms;

        TunedConfigCache() : config(), avg_time_ms(0.0f) {}
    };

    /**
     * @brief Autotuner for HIP kernel launch configurations
     *
     * Performs empirical benchmarking to select optimal block dimensions
     * for memory-bound kernels. Results are cached per GPU architecture.
     */
    class AutoTuner
    {
    public:
        /**
         * @brief Kernel launch wrapper function type
         *
         * @param config Block configuration to use
         * @param stream HIP stream for kernel launch
         * @param args Additional kernel-specific arguments
         */
        using LaunchFunc = std::function<void(const KernelConfig &config, hipStream_t stream, void *args)>;

        AutoTuner();
        ~AutoTuner();

        // Disable copy, allow move
        AutoTuner(const AutoTuner &) = delete;
        AutoTuner &operator=(const AutoTuner &) = delete;
        AutoTuner(AutoTuner &&) noexcept = default;
        AutoTuner &operator=(AutoTuner &&) noexcept = default;

        /**
         * @brief Get or compute optimal kernel configuration
         *
         * Checks cache first; if not found, performs autotuning.
         *
         * @param kernel_name Name of the kernel (e.g., "grayscale")
         * @param launch_func Kernel launch wrapper
         * @param args Arguments to pass to launch function
         * @param warmup_runs Number of warmup iterations (default: 5)
         * @param timing_runs Number of timed iterations (default: 10)
         * @return Optimal KernelConfig
         */
        KernelConfig get_config(
            const std::string &kernel_name,
            LaunchFunc launch_func,
            void *args,
            int warmup_runs = 5,
            int timing_runs = 10);

        /**
         * @brief Check if a tuned configuration exists in cache
         *
         * @param kernel_name Name of the kernel
         * @return true if cached config exists
         */
        bool has_cached_config(const std::string &kernel_name) const;

        /**
         * @brief Get the GPU architecture string (e.g., "gfx1100")
         *
         * @return GPU architecture identifier
         */
        std::string get_gpu_arch() const;

        /**
         * @brief Load cache from disk
         *
         * @param cache_path Path to cache file (default: ".autotune_cache.json")
         * @return true if cache was loaded successfully
         */
        bool load_cache(const std::string &cache_path = ".autotune_cache.json");

        /**
         * @brief Save cache to disk
         *
         * @param cache_path Path to cache file (default: ".autotune_cache.json")
         * @return true if cache was saved successfully
         */
        bool save_cache(const std::string &cache_path = ".autotune_cache.json") const;

    private:
        /**
         * @brief Perform autotuning for a specific kernel
         *
         * Benchmarks all candidate configurations and selects the fastest.
         *
         * @param kernel_name Name of the kernel
         * @param launch_func Kernel launch wrapper
         * @param args Arguments to pass to launch function
         * @param warmup_runs Number of warmup iterations
         * @param timing_runs Number of timed iterations
         * @return Optimal KernelConfig
         */
        KernelConfig autotune_kernel(
            const std::string &kernel_name,
            LaunchFunc launch_func,
            void *args,
            int warmup_runs,
            int timing_runs);

        /**
         * @brief Benchmark a single kernel configuration
         *
         * @param config Configuration to benchmark
         * @param launch_func Kernel launch wrapper
         * @param args Arguments to pass to launch function
         * @param stream HIP stream for kernel execution
         * @param warmup_runs Number of warmup iterations
         * @param timing_runs Number of timed iterations
         * @return BenchmarkResult with average execution time
         */
        BenchmarkResult benchmark_config(
            const KernelConfig &config,
            LaunchFunc launch_func,
            void *args,
            hipStream_t stream,
            int warmup_runs,
            int timing_runs);

        /**
         * @brief Get candidate block configurations for tuning
         *
         * Returns AMD-friendly block sizes optimized for wavefront size = 64.
         * Includes both 1D and 2D configurations with 128-256 threads.
         *
         * @return Vector of candidate configurations
         */
        std::vector<KernelConfig> get_candidate_configs() const;

        /**
         * @brief Extract GPU architecture from device properties
         *
         * Queries HIP device 0 and formats architecture string.
         *
         * @return Architecture string (e.g., "gfx1100")
         */
        std::string query_gpu_arch();

        std::string m_gpu_arch;
        std::vector<TunedConfigCache> m_cache;
    };

} // namespace imgfx::core
