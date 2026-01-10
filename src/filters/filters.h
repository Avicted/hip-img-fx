#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include <string>

#include "../core/gpu_utils.h"
#include "../core/autotune/tuning_config.h"

namespace imgfx::filters
{
    extern "C" __global__ void grayscale_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images);

    extern "C" __global__ void negative_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images);

    extern "C" __global__ void gaussian_blur_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        int blurAmount);

    void grayscale_cpu(
        const unsigned char *input_image,
        unsigned char *output_image,
        int width,
        int height,
        int channels);

    void gaussian_blur_cpu(
        const unsigned char *input_image,
        unsigned char *output_image,
        int width,
        int height,
        int channels,
        int blurAmount);

    void negative_cpu(
        const unsigned char *input_image,
        unsigned char *output_image,
        int width,
        int height,
        int channels);

    // ========================================================================
    // AUTOTUNING FRAMEWORK
    // ========================================================================

    /**
     * @brief Kernel traits for grayscale filter
     *
     * Implements the traits-based interface for TuningOrchestrator.
     */
    struct GrayscaleKernelTraits
    {
        // 1. Kernel name (unique identifier for caching)
        static constexpr const char *name() { return "grayscale"; }

        // 2. Kernel launch arguments (type-safe container)
        struct Args
        {
            const unsigned char *input;
            unsigned char *output;
            const imgfx::core::image_meta_t *metas;
            int num_images;
            size_t max_image_bytes;
        };

        // 3. Cache context (for different workload categories)
        struct Context
        {
            size_t image_bytes;

            std::string cache_key() const
            {
                // Categorize by image size for cache differentiation
                if (image_bytes < 1024 * 1024)
                    return "small"; // < 1MB
                if (image_bytes < 10 * 1024 * 1024)
                    return "medium"; // 1-10MB
                return "large";      // > 10MB
            }
        };

        // 4. Generate candidate configurations
        static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
        {
            using imgfx::core::autotune::TuningConfig;
            std::vector<TuningConfig> configs;

            // Test 1D configurations (common for memory-bound kernels)
            for (int bx : {64, 128, 256, 512})
            {
                TuningConfig cfg;
                cfg.set("block_x", bx);
                cfg.set("block_y", 1);
                configs.push_back(cfg);
            }

            // Test 2D configurations (can improve spatial locality)
            for (int bx : {16, 32})
            {
                for (int by : {4, 8, 16})
                {
                    TuningConfig cfg;
                    cfg.set("block_x", bx);
                    cfg.set("block_y", by);
                    configs.push_back(cfg);
                }
            }

            return configs;
        }

        // 5. Validate configuration
        static bool is_valid_config(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args & /*args*/)
        {
            int threads = cfg.block_x() * cfg.block_y();

            // Wavefront alignment (AMD GPUs use 64-wide wavefronts)
            if (threads % 64 != 0)
                return false;

            // Reasonable thread limits
            if (threads < 64 || threads > 1024)
                return false;

            return true;
        }

        // 6. Launch kernel with configuration
        static void launch(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args &args,
            hipStream_t stream)
        {
            // Calculate grid dimensions
            int threads_per_block = cfg.block_x() * cfg.block_y();
            int blocks_x = (args.max_image_bytes + threads_per_block - 1) / threads_per_block;

            dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
            dim3 grid_dim(blocks_x, args.num_images, 1);

            // Launch kernel with specified configuration
            hipLaunchKernelGGL(
                grayscale_kernel,
                grid_dim,
                block_dim,
                0,
                stream,
                args.input,
                args.output,
                args.metas,
                args.num_images);
        }
    };

    // ========================================================================
    // NEGATIVE FILTER - New Autotuning Framework
    // ========================================================================

    /**
     * @brief Kernel traits for negative filter (new autotuning framework)
     */
    struct NegativeKernelTraits
    {
        static constexpr const char *name() { return "negative"; }

        struct Args
        {
            const unsigned char *input;
            unsigned char *output;
            const imgfx::core::image_meta_t *metas;
            int num_images;
            size_t max_image_bytes;
        };

        struct Context
        {
            size_t image_bytes;

            std::string cache_key() const
            {
                if (image_bytes < 1024 * 1024)
                    return "small";
                if (image_bytes < 10 * 1024 * 1024)
                    return "medium";
                return "large";
            }
        };

        static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
        {
            using imgfx::core::autotune::TuningConfig;
            std::vector<TuningConfig> configs;

            // 1D configurations
            for (int bx : {64, 128, 256, 512})
            {
                TuningConfig cfg;
                cfg.set("block_x", bx);
                cfg.set("block_y", 1);
                configs.push_back(cfg);
            }

            // 2D configurations
            for (int bx : {16, 32})
            {
                for (int by : {4, 8, 16})
                {
                    TuningConfig cfg;
                    cfg.set("block_x", bx);
                    cfg.set("block_y", by);
                    configs.push_back(cfg);
                }
            }

            return configs;
        }

        static bool is_valid_config(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args & /*args*/)
        {
            int threads = cfg.block_x() * cfg.block_y();
            if (threads % 64 != 0)
                return false;
            if (threads < 64 || threads > 1024)
                return false;
            return true;
        }

        static void launch(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args &args,
            hipStream_t stream)
        {
            int threads_per_block = cfg.block_x() * cfg.block_y();
            int blocks_x = (args.max_image_bytes + threads_per_block - 1) / threads_per_block;

            dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
            dim3 grid_dim(blocks_x, args.num_images, 1);

            hipLaunchKernelGGL(
                negative_kernel,
                grid_dim,
                block_dim,
                0,
                stream,
                args.input,
                args.output,
                args.metas,
                args.num_images);
        }
    };

    // ========================================================================
    // GAUSSIAN BLUR FILTER - New Autotuning Framework
    // ========================================================================

    /**
     * @brief Kernel traits for Gaussian blur filter (new autotuning framework)
     */
    struct GaussianBlurKernelTraits
    {
        static constexpr const char *name() { return "gaussian_blur"; }

        struct Args
        {
            const unsigned char *input;
            unsigned char *output;
            const imgfx::core::image_meta_t *metas;
            int num_images;
            size_t max_image_bytes;
            int blur_amount;
        };

        struct Context
        {
            size_t image_bytes;
            int blur_amount;

            std::string cache_key() const
            {
                std::string size_key;
                if (image_bytes < 1024 * 1024)
                    size_key = "small";
                else if (image_bytes < 10 * 1024 * 1024)
                    size_key = "medium";
                else
                    size_key = "large";

                // Blur amount affects performance
                std::string blur_key = blur_amount < 5 ? "blur_small" : "blur_large";
                return size_key + "_" + blur_key;
            }
        };

        static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
        {
            using imgfx::core::autotune::TuningConfig;
            std::vector<TuningConfig> configs;

            // 2D configurations (better for blur due to 2D memory access)
            for (int bx : {8, 16, 32})
            {
                for (int by : {8, 16, 32})
                {
                    if (bx * by >= 64 && bx * by <= 1024)
                    {
                        TuningConfig cfg;
                        cfg.set("block_x", bx);
                        cfg.set("block_y", by);
                        configs.push_back(cfg);
                    }
                }
            }

            // Also test some 1D configurations
            for (int bx : {128, 256, 512})
            {
                TuningConfig cfg;
                cfg.set("block_x", bx);
                cfg.set("block_y", 1);
                configs.push_back(cfg);
            }

            return configs;
        }

        static bool is_valid_config(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args & /*args*/)
        {
            int threads = cfg.block_x() * cfg.block_y();
            if (threads % 64 != 0)
                return false;
            if (threads < 64 || threads > 1024)
                return false;
            return true;
        }

        static void launch(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args &args,
            hipStream_t stream)
        {
            int threads_per_block = cfg.block_x() * cfg.block_y();
            int blocks_x = (args.max_image_bytes + threads_per_block - 1) / threads_per_block;

            dim3 block_dim(cfg.block_x(), cfg.block_y(), 1);
            dim3 grid_dim(blocks_x, args.num_images, 1);

            hipLaunchKernelGGL(
                gaussian_blur_kernel,
                grid_dim,
                block_dim,
                0,
                stream,
                args.input,
                args.output,
                args.metas,
                args.num_images,
                args.blur_amount);
        }
    };

    // ========================================================================
    // V2 AUTOTUNED FUNCTIONS (New Framework)
    // ========================================================================

    /**
     * @brief Apply grayscale filter using autotuning framework
     */
    void apply_grayscale_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        hipStream_t stream);

    /**
     * @brief Apply negative filter using autotuning framework
     */
    void apply_negative_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        hipStream_t stream);

    /**
     * @brief Apply Gaussian blur filter using new autotuning framework
     */
    void apply_gaussian_blur_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        int blur_amount,
        hipStream_t stream);

}
