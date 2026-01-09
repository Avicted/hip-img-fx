#pragma once

#include <hip/hip_runtime.h>

#include "../core/gpu_utils.h"

// Forward declaration to avoid circular dependency
namespace imgfx::core
{
    class AutoTuner;
}

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

    // Autotuned kernel launches
    void apply_grayscale_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        imgfx::core::AutoTuner &autotuner,
        hipStream_t stream);

    void apply_negative_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        imgfx::core::AutoTuner &autotuner,
        hipStream_t stream);

    void apply_gaussian_blur_autotuned(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        size_t max_image_bytes,
        int blur_amount,
        imgfx::core::AutoTuner &autotuner,
        hipStream_t stream);
}
