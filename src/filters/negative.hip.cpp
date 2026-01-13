// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include "filters.h"

namespace imgfx::filters
{
    extern "C" __global__ void negative_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images)
    {
        const int img_idx = (int)blockIdx.y;
        if (img_idx >= num_images)
        {
            return;
        }

        const imgfx::core::image_meta_t meta = metas[img_idx];
        // Support both 1D and 2D block configurations
        const size_t threads_per_block = (size_t)blockDim.x * (size_t)blockDim.y;
        const size_t thread_idx_in_block = (size_t)threadIdx.y * (size_t)blockDim.x + (size_t)threadIdx.x;
        const size_t idx_in_image = (size_t)blockIdx.x * threads_per_block + thread_idx_in_block;
        const size_t total_bytes = (size_t)meta.width * (size_t)meta.height * (size_t)meta.channels;
        if (idx_in_image >= total_bytes)
        {
            return;
        }

        const size_t global_idx = (size_t)meta.offset + idx_in_image;
        output[global_idx] = 255 - input[global_idx];
    }

    void negative_cpu(
        const unsigned char *input_image,
        unsigned char *output_image,
        int width,
        int height,
        int channels)
    {
        const int total_pixels = width * height;

#pragma omp parallel for
        for (int i = 0; i < total_pixels; i++)
        {
            int idx = i * channels;

            for (int c = 0; c < channels; c++)
            {
                output_image[idx + c] = 255 - input_image[idx + c];
            }
        }
    }
}
