/**
 * @file grayscale_kernel_traits_example.h
 * @brief Example of migrated grayscale kernel using new autotuning framework
 *
 * This shows how the grayscale kernel would be defined using the
 * traits-based architecture, eliminating ~50 lines of boilerplate.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include "../core/autotune/tuning_config.h"
#include "../core/image.h"

namespace imgfx::filters
{
    // Forward declare the actual kernel (unchanged)
    extern "C" __global__ void grayscale_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images);

    /**
     * @brief Kernel traits for grayscale filter
     *
     * Defines all autotuning parameters for the grayscale kernel.
     * This replaces the old GrayscaleLaunchArgs struct, launch_grayscale_kernel
     * function, and most of apply_grayscale_autotuned.
     */
    struct GrayscaleKernelTraits
    {
        //
        // REQUIRED TRAIT INTERFACE
        //

        /**
         * @brief Unique kernel identifier
         */
        static constexpr const char *name() { return "grayscale"; }

        /**
         * @brief Type-safe kernel arguments
         *
         * Replaces void* casting with proper types.
         */
        struct Args
        {
            const unsigned char *input;
            unsigned char *output;
            const imgfx::core::image_meta_t *metas;
            int num_images;
            size_t max_image_bytes;
        };

        /**
         * @brief Context for cache key generation
         *
         * Allows different tuning for different image sizes.
         */
        struct Context
        {
            size_t image_bytes;

            /**
             * @brief Generate context string for cache key
             *
             * Categorizes images into size buckets for better tuning.
             */
            std::string cache_key() const
            {
                // Small: < 1MB
                // Medium: 1MB - 10MB
                // Large: > 10MB
                if (image_bytes < 1024 * 1024)
                {
                    return "small";
                }
                else if (image_bytes < 10 * 1024 * 1024)
                {
                    return "medium";
                }
                else
                {
                    return "large";
                }
            }
        };

        /**
         * @brief Generate candidate configurations to benchmark
         *
         * Returns all configurations to test during autotuning.
         * Can be extended to include additional parameters like:
         * - vec_width (1, 2, 4 for vectorization)
         * - work_per_thread (1, 2, 4 for workload distribution)
         * - use_shared_mem (true/false)
         */
        static std::vector<imgfx::core::autotune::TuningConfig> generate_candidates()
        {
            using imgfx::core::autotune::TuningConfig;
            std::vector<TuningConfig> configs;

            // 1D block configurations
            // AMD-friendly sizes (multiples of wavefront size = 64)
            for (int block_x : {64, 128, 256})
            {
                TuningConfig cfg;
                cfg.set("block_x", block_x);
                cfg.set("block_y", 1);
                configs.push_back(cfg);
            }

            // 2D block configurations
            // Good for spatial locality in image processing
            std::vector<std::pair<int, int>> block_2d = {
                {16, 8},  // 128 threads (narrow)
                {16, 16}, // 256 threads (square)
                {32, 8}   // 256 threads (wide)
            };

            for (auto [bx, by] : block_2d)
            {
                TuningConfig cfg;
                cfg.set("block_x", bx);
                cfg.set("block_y", by);
                configs.push_back(cfg);
            }

            // FUTURE EXTENSION: Vectorization
            // Uncomment when vectorized kernels are implemented:
            //
            // for (int block_x : {64, 128, 256}) {
            //     for (int vec_width : {1, 2, 4}) {
            //         TuningConfig cfg;
            //         cfg.set("block_x", block_x);
            //         cfg.set("block_y", 1);
            //         cfg.set("vec_width", vec_width);
            //         configs.push_back(cfg);
            //     }
            // }

            return configs;
        }

        /**
         * @brief Validate configuration against kernel constraints
         *
         * Checks if a configuration is valid for this kernel and arguments.
         *
         * @param cfg Configuration to validate
         * @param args Kernel arguments (for size-dependent validation)
         * @return true if configuration is valid
         */
        static bool is_valid_config(
            const imgfx::core::autotune::TuningConfig &cfg,
            const Args &args)
        {
            int threads = cfg.block_x() * cfg.block_y();

            // Must be multiple of wavefront size (64) for efficiency
            if (threads % 64 != 0)
            {
                return false;
            }

            // Reasonable thread count limits
            if (threads < 64 || threads > 1024)
            {
                return false;
            }

            // Ensure we have enough blocks to cover the work
            // (This is a soft constraint - kernel will still work, just inefficient)
            size_t total_work = args.max_image_bytes * args.num_images;
            if (total_work > 0 && threads > total_work)
            {
                // Block size larger than work - inefficient
                return true; // Still valid, just not optimal
            }

            return true;
        }

        /**
         * @brief Launch kernel with given configuration
         *
         * Calculates grid dimensions and invokes the kernel.
         * This replaces the old launch_grayscale_kernel function.
         *
         * @param cfg Tuning configuration
         * @param args Kernel arguments
         * @param stream HIP stream for execution
         */
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

            // Launch kernel
            hipLaunchKernelGGL(
                grayscale_kernel,
                grid_dim,
                block_dim,
                0, // shared memory size
                stream,
                args.input,
                args.output,
                args.metas,
                args.num_images);
        }

        //
        // OPTIONAL EXTENSIONS (for future work)
        //

        /**
         * @brief Estimate shared memory usage for a configuration
         *
         * Used by framework to check device limits.
         */
        static size_t shared_memory_bytes(const imgfx::core::autotune::TuningConfig &cfg)
        {
            // Grayscale doesn't use shared memory currently
            return 0;

            // FUTURE: If we add shared memory tiling:
            // int smem_tiles = cfg.get_or("smem_tiles", 0);
            // return smem_tiles * cfg.block_x() * cfg.block_y() * sizeof(uchar3);
        }

        /**
         * @brief Estimate occupancy for a configuration
         *
         * Higher occupancy generally means better performance.
         */
        static float estimate_occupancy(const imgfx::core::autotune::TuningConfig &cfg)
        {
            // Simple heuristic: more threads per block = higher occupancy
            // (up to a point, limited by register usage)
            int threads = cfg.total_threads();

            if (threads <= 64)
                return 0.5f;
            else if (threads <= 128)
                return 0.75f;
            else if (threads <= 256)
                return 1.0f;
            else
                return 0.75f; // Larger blocks may reduce occupancy due to register pressure
        }
    };

    //
    // PUBLIC API (called by application code)
    //

    /**
     * @brief Apply grayscale filter with autotuned kernel configuration
     *
     * NEW IMPLEMENTATION: Uses traits-based orchestrator.
     * This is much simpler than the old implementation.
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param autotuner DEPRECATED: Old AutoTuner (ignored in new implementation)
     * @param stream HIP stream for kernel execution
     */
    void apply_grayscale_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        imgfx::core::AutoTuner & /* deprecated_autotuner */,
        hipStream_t stream);

    // IMPLEMENTATION (would go in .cpp file):
    //
    // void apply_grayscale_autotuned(
    //     const unsigned char* input,
    //     unsigned char* output,
    //     const imgfx::core::image_meta_t* metas,
    //     int num_images,
    //     size_t max_image_bytes,
    //     imgfx::core::AutoTuner& /* deprecated */,
    //     hipStream_t stream)
    // {
    //     using namespace imgfx::core::autotune;
    //
    //     // Static orchestrator - initialized once, cached thereafter
    //     static TuningOrchestrator<GrayscaleKernelTraits> orchestrator;
    //
    //     // Prepare arguments (type-safe, no void* casting!)
    //     GrayscaleKernelTraits::Args args;
    //     args.input = input;
    //     args.output = output;
    //     args.metas = metas;
    //     args.num_images = num_images;
    //     args.max_image_bytes = max_image_bytes;
    //
    //     // Prepare context for cache key
    //     GrayscaleKernelTraits::Context ctx;
    //     ctx.image_bytes = max_image_bytes;
    //
    //     // Execute with autotuning (gets cached config or tunes if needed)
    //     orchestrator.execute(args, ctx, stream);
    // }

} // namespace imgfx::filters
