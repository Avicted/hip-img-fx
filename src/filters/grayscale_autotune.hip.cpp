#include "filters.h"
#include "../core/autotune/orchestrator.h"

namespace imgfx::filters
{
    /**
     * @brief Apply grayscale filter using autotuning framework
     *
     * This implementation uses the TuningOrchestrator with GrayscaleKernelTraits.
     *
     * Key features:
     * - Type-safe arguments through traits
     * - Three-tier caching (thread-local, persistent, tuning)
     * - Cleaner separation of concerns
     * - Better testability
     *
     * @param input Device input buffer
     * @param output Device output buffer
     * @param metas Device metadata buffer
     * @param num_images Number of images in batch
     * @param max_image_bytes Maximum image size in bytes
     * @param stream HIP stream for kernel execution
     */
    void apply_grayscale_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        hipStream_t stream)
    {
        using namespace imgfx::core::autotune;

        // Static orchestrator (initialized once per process)
        // Cache is loaded on first use, saved on destruction
        static TuningOrchestrator<GrayscaleKernelTraits> orchestrator;

        // Prepare kernel arguments
        GrayscaleKernelTraits::Args args;
        args.input = input;
        args.output = output;
        args.metas = metas;
        args.num_images = num_images;
        args.max_image_bytes = max_image_bytes;

        // Prepare context for caching
        GrayscaleKernelTraits::Context ctx;
        ctx.image_bytes = max_image_bytes;

        // Execute with optimal configuration (cached or autotuned)
        orchestrator.execute(args, ctx, stream);
    }

} // namespace imgfx::filters
