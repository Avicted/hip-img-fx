#include "filters.h"
#include "../core/autotuning.h"

namespace imgfx::filters
{
    /**
     * @brief Arguments structure for gaussian_blur kernel launch
     */
    struct GaussianBlurLaunchArgs
    {
        const unsigned char *input;
        unsigned char *output;
        const imgfx::core::image_meta_t *metas;
        int num_images;
        size_t max_image_bytes;
        int blur_amount;
        size_t shared_bytes;
    };

    /**
     * @brief Launch wrapper for gaussian_blur kernel with configurable block dimensions
     *
     * @param config Kernel configuration (block dimensions)
     * @param stream HIP stream for kernel execution
     * @param args Pointer to GaussianBlurLaunchArgs structure
     */
    void launch_gaussian_blur_kernel(
        const imgfx::core::KernelConfig &config,
        hipStream_t stream,
        void *args)
    {
        GaussianBlurLaunchArgs *launch_args = static_cast<GaussianBlurLaunchArgs *>(args);

        int threads_per_block = config.block_x * config.block_y;
        int blocks_x = (launch_args->max_image_bytes + threads_per_block - 1) / threads_per_block;

        dim3 block_dim(config.block_x, config.block_y, 1);
        dim3 grid_dim(blocks_x, launch_args->num_images, 1);

        hipLaunchKernelGGL(
            gaussian_blur_kernel,
            grid_dim,
            block_dim,
            launch_args->shared_bytes,
            stream,
            launch_args->input,
            launch_args->output,
            launch_args->metas,
            launch_args->num_images,
            launch_args->blur_amount);
    }

    /**
     * @brief Apply gaussian blur filter with autotuned kernel configuration
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param blur_amount Blur kernel size (must be odd)
     * @param autotuner AutoTuner instance
     * @param stream HIP stream for kernel execution
     */
    void apply_gaussian_blur_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        int blur_amount,
        imgfx::core::AutoTuner &autotuner,
        hipStream_t stream)
    {
        size_t shared_bytes = sizeof(float) * blur_amount * blur_amount;

        GaussianBlurLaunchArgs launch_args;
        launch_args.input = input;
        launch_args.output = output;
        launch_args.metas = metas;
        launch_args.num_images = num_images;
        launch_args.max_image_bytes = max_image_bytes;
        launch_args.blur_amount = blur_amount;
        launch_args.shared_bytes = shared_bytes;

        const int warmup_runs = 5;
        const int timing_runs = 10;
        imgfx::core::KernelConfig config = autotuner.get_config(
            "gaussian_blur",
            launch_gaussian_blur_kernel,
            &launch_args,
            warmup_runs, timing_runs);

        launch_gaussian_blur_kernel(config, stream, &launch_args);
    }

} // namespace imgfx::filters
