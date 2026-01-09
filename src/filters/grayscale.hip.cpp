#include "filters.h"

namespace imgfx::filters
{
    extern "C" __global__ void grayscale_kernel(
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
        const size_t idx_in_image = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
        const size_t image_bytes = (size_t)meta.width * (size_t)meta.height * (size_t)meta.channels;
        if (idx_in_image >= image_bytes)
        {
            return;
        }

        const size_t channels = (size_t)meta.channels;
        const size_t c = idx_in_image % channels;
        const size_t pixel_base_in_image = idx_in_image - c;
        if (pixel_base_in_image + 2 >= image_bytes)
        {
            return;
        }

        const size_t pixel_base = (size_t)meta.offset + pixel_base_in_image;

        unsigned char r = input[pixel_base + 0];
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
