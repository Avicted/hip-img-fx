#include "filters.h"

extern "C" __global__ void negative_kernel(
    const unsigned char *input,
    unsigned char *output,
    const image_meta_t *metas,
    int num_images)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine which image this pixel belongs to
    int img_idx = 0;
    while (img_idx < num_images &&
           idx >= metas[img_idx].offset + metas[img_idx].width * metas[img_idx].height * metas[img_idx].channels)
    {
        img_idx++;
    }

    if (img_idx >= num_images)
    {
        return;
    }

    const image_meta_t &meta = metas[img_idx];
    int pixel_idx = idx - meta.offset;

    output[meta.offset + pixel_idx] = 255 - input[meta.offset + pixel_idx];
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
