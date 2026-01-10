#include "filters.h"
#include "../core/autotune/orchestrator.h"

namespace imgfx::filters
{
    /**
     * @brief Apply negative filter using new autotuning framework (v2)
     *
     * This implementation uses the new TuningOrchestrator with NegativeKernelTraits.
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param stream HIP stream for kernel execution
     */
    void apply_negative_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        hipStream_t stream)
    {
        using namespace imgfx::core::autotune;

        // Static orchestrator (initialized once per process)
        static TuningOrchestrator<NegativeKernelTraits> orchestrator;

        // Prepare kernel arguments
        NegativeKernelTraits::Args args;
        args.input = input;
        args.output = output;
        args.metas = metas;
        args.num_images = num_images;
        args.max_image_bytes = max_image_bytes;

        // Prepare context for caching
        NegativeKernelTraits::Context ctx;
        ctx.image_bytes = max_image_bytes;

        // Execute with optimal configuration (cached or autotuned)
        orchestrator.execute(args, ctx, stream);
    }

} // namespace imgfx::filters
