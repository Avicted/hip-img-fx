#include "gpu_utils.h"

extern "C" __global__ void grayscale_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        unsigned char r = input_image[idx];
        unsigned char g = input_image[idx + 1];
        unsigned char b = input_image[idx + 2];

        // Compute grayscale value using luminosity method
        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        output_image[idx] = gray;
        output_image[idx + 1] = gray;
        output_image[idx + 2] = gray;

        // If there is an alpha channel, copy it as is
        if (channels == 4)
        {
            output_image[idx + 3] = input_image[idx + 3];
        }
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
