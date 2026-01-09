#include "filters.h"
#include "../core/autotuning.h"

namespace imgfx::filters
{
    /**
     * @brief Arguments structure for negative kernel launch
     */
    struct NegativeLaunchArgs
    {
        const unsigned char *input;
        unsigned char *output;
        const imgfx::core::image_meta_t *metas;
        int num_images;
        size_t max_image_bytes;
    };

    /**
     * @brief Launch wrapper for negative kernel with configurable block dimensions
     *
     * @param config Kernel configuration (block dimensions)
     * @param stream HIP stream for kernel execution
     * @param args Pointer to NegativeLaunchArgs structure
     */
    void launch_negative_kernel(
        const imgfx::core::KernelConfig &config,
        hipStream_t stream,
        void *args)
    {
        NegativeLaunchArgs *launch_args = static_cast<NegativeLaunchArgs *>(args);

        int threads_per_block = config.block_x * config.block_y;
        int blocks_x = (launch_args->max_image_bytes + threads_per_block - 1) / threads_per_block;

        dim3 block_dim(config.block_x, config.block_y, 1);
        dim3 grid_dim(blocks_x, launch_args->num_images, 1);

        hipLaunchKernelGGL(
            negative_kernel,
            grid_dim,
            block_dim,
            0,
            stream,
            launch_args->input,
            launch_args->output,
            launch_args->metas,
            launch_args->num_images);
    }

    /**
     * @brief Apply negative filter with autotuned kernel configuration
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param autotuner AutoTuner instance
     * @param stream HIP stream for kernel execution
     */
    void apply_negative_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        imgfx::core::AutoTuner &autotuner,
        hipStream_t stream)
    {
        NegativeLaunchArgs launch_args;
        launch_args.input = input;
        launch_args.output = output;
        launch_args.metas = metas;
        launch_args.num_images = num_images;
        launch_args.max_image_bytes = max_image_bytes;

        const int warmup_runs = 5;
        const int timing_runs = 10;
        imgfx::core::KernelConfig config = autotuner.get_config(
            "negative",
            launch_negative_kernel,
            &launch_args,
            warmup_runs, timing_runs);

        launch_negative_kernel(config, stream, &launch_args);
    }

} // namespace imgfx::filters
