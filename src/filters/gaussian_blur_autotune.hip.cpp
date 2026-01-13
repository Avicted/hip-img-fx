// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include "filters.h"
#include "hip-img-fx/autotune/orchestrator.h"

namespace imgfx::filters
{
    /**
     * @brief Apply Gaussian blur filter using new autotuning framework (v2)
     *
     * This implementation uses the new TuningOrchestrator with GaussianBlurKernelTraits.
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param blur_amount Blur kernel radius (must be odd)
     * @param stream HIP stream for kernel execution
     */
    void apply_gaussian_blur_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        int blur_amount,
        hipStream_t stream)
    {
        using namespace imgfx::core::autotune;

        // Static orchestrator (initialized once per process)
        static TuningOrchestrator<GaussianBlurKernelTraits> orchestrator;

        // Prepare kernel arguments
        GaussianBlurKernelTraits::Args args;
        args.input = input;
        args.output = output;
        args.metas = metas;
        args.num_images = num_images;
        args.max_image_bytes = max_image_bytes;
        args.blur_amount = blur_amount;

        // Prepare context for caching
        GaussianBlurKernelTraits::Context ctx;
        ctx.image_bytes = max_image_bytes;
        ctx.blur_amount = blur_amount;

        // Execute with optimal configuration (cached or autotuned)
        orchestrator.execute(args, ctx, stream);
    }

} // namespace imgfx::filters
