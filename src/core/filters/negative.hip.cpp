#include "gpu_utils.h"

extern "C" __global__ void negative_kernel(
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

        for (int c = 0; c < channels; c++)
        {
            output_image[idx + c] = 255 - input_image[idx + c];
        }
    }
}
