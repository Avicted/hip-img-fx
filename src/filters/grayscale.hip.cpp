#include "filters.h"

namespace imgfx::filters
{
    extern "C" __global__ void grayscale_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Total pixels = sum of width*height*channels for all images
        // We'll compute which image this pixel belongs to
        int img_idx = 0;
        while (img_idx < num_images && idx >= metas[img_idx].offset + metas[img_idx].width * metas[img_idx].height * metas[img_idx].channels)
        {
            img_idx++;
        }

        if (img_idx >= num_images)
        {
            return;
        }

        const imgfx::core::image_meta_t &meta = metas[img_idx];
        int pixel_idx = idx - meta.offset;

        int c = pixel_idx % meta.channels;
        int pixel_base = idx - c;

        unsigned char r = input[pixel_base];
        unsigned char g = input[pixel_base + 1];
        unsigned char b = input[pixel_base + 2];

        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        output[pixel_base] = gray;
        output[pixel_base + 1] = gray;
        output[pixel_base + 2] = gray;

        if (meta.channels == 4)
        {
            output[pixel_base + 3] = input[pixel_base + 3]; // copy alpha
        }
    }

    void grayscale_cpu(
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

            unsigned char r = input_image[idx + 0];
            unsigned char g = input_image[idx + 1];
            unsigned char b = input_image[idx + 2];

            unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

            output_image[idx + 0] = gray;
            output_image[idx + 1] = gray;
            output_image[idx + 2] = gray;

            if (channels == 4)
            {
                output_image[idx + 3] = input_image[idx + 3];
            }
        }
    }
}
