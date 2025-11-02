#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cmath>
#include <vector>

#include "image.h"

extern "C" __global__ void gaussian_blur_kernel(
    const unsigned char *input,
    unsigned char *output,
    const image_meta_t *metas,
    int num_images,
    int blurAmount) // must be odd
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine which image this pixel belongs to
    int img_idx = 0;
    while (img_idx < num_images &&
           idx >= metas[img_idx].offset + metas[img_idx].width * metas[img_idx].height * metas[img_idx].channels)
        img_idx++;

    if (img_idx >= num_images)
        return;

    const image_meta_t &meta = metas[img_idx];
    int pixel_idx = idx - meta.offset; // index inside this image

    int c = pixel_idx % meta.channels;
    int pos_in_image = pixel_idx / meta.channels;
    int x = pos_in_image % meta.width;
    int y = pos_in_image / meta.width;

    extern __shared__ float kernel[];
    int radius = blurAmount / 2;

    // compute Gaussian kernel once per block
    if (threadIdx.x == 0)
    {
        float sigma = blurAmount / 2.0f;
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ky++)
            for (int kx = -radius; kx <= radius; kx++)
            {
                float v = expf(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
                kernel[(ky + radius) * blurAmount + (kx + radius)] = v;
                sum += v;
            }
        for (int i = 0; i < blurAmount * blurAmount; i++)
            kernel[i] /= sum;
    }
    __syncthreads();

    float pixel_value = 0.0f;

    for (int ky = -radius; ky <= radius; ky++)
    {
        int ny = min(max(y + ky, 0), meta.height - 1);
        for (int kx = -radius; kx <= radius; kx++)
        {
            int nx = min(max(x + kx, 0), meta.width - 1);
            int nidx = (ny * meta.width + nx) * meta.channels + c;
            pixel_value += input[meta.offset + nidx] * kernel[(ky + radius) * blurAmount + (kx + radius)];
        }
    }

    // clamp and write
    int outv = static_cast<int>(pixel_value + 0.5f);
    if (outv < 0)
        outv = 0;
    if (outv > 255)
        outv = 255;
    output[meta.offset + pixel_idx] = static_cast<unsigned char>(outv);
}

void gaussian_blur_cpu(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    int blurAmount)
{
    const int kernelRadius = blurAmount / 2;
    const int kernelSize = blurAmount * blurAmount;

    // build kernel once like in GPU block(0,0)
    std::vector<float> kernel(kernelSize);

    float sigma = blurAmount / 2.0f;
    float sum = 0.0f;

    for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
    {
        for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
        {
            float v = std::exp(-(float)(kx * kx + ky * ky) / (2.0f * sigma * sigma));
            kernel[(ky + kernelRadius) * blurAmount + (kx + kernelRadius)] = v;
            sum += v;
        }
    }

    if (sum <= 0.0f)
    {
        sum = 1.0f;
    }

    for (int i = 0; i < kernelSize; ++i)
    {
        kernel[i] /= sum;
    }

    // convolution
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float pixel_value = 0.0f;

                for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
                {
                    int ny = y + ky;
                    if (ny < 0)
                    {
                        ny = 0;
                    }
                    else if (ny >= height)
                    {
                        ny = height - 1;
                    }

                    for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                    {
                        int nx = x + kx;
                        if (nx < 0)
                        {
                            nx = 0;
                        }
                        else if (nx >= width)
                        {
                            nx = width - 1;
                        }

                        int nidx = (ny * width + nx) * channels + c;
                        float in_val = (float)input_image[nidx];
                        float k = kernel[(ky + kernelRadius) * blurAmount + (kx + kernelRadius)];
                        pixel_value += in_val * k;
                    }
                }

                int idx = (y * width + x) * channels + c;
                int outv = (int)(pixel_value + 0.5f);
                if (outv < 0)
                {
                    outv = 0;
                }
                else if (outv > 255)
                {
                    outv = 255;
                }
                output_image[idx] = (unsigned char)outv;
            }
        }
    }
}
