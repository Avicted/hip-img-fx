// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include "filters.h"

namespace imgfx::filters
{
    extern "C" __global__ void gaussian_blur_kernel(
        const unsigned char *input,
        unsigned char *output,
        const imgfx::core::image_meta_t *metas,
        int num_images,
        int blurAmount)
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

        const size_t pixel_idx = idx_in_image; // index inside this image (byte index)

        const size_t channels = (size_t)meta.channels;
        int c = (int)(pixel_idx % channels);
        size_t pos_in_image = pixel_idx / channels;
        int x = (int)(pos_in_image % (size_t)meta.width);
        int y = (int)(pos_in_image / (size_t)meta.width);

        extern __shared__ float kernel[];
        int radius = blurAmount / 2;

        // compute Gaussian kernel once per block (only thread 0,0 in 2D blocks)
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            float sigma = blurAmount / 2.0f;
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    float v = expf(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
                    kernel[(ky + radius) * blurAmount + (kx + radius)] = v;
                    sum += v;
                }
            }

            for (int i = 0; i < blurAmount * blurAmount; i++)
            {
                kernel[i] /= sum;
            }
        }
        __syncthreads();

        float pixel_value = 0.0f;

        for (int ky = -radius; ky <= radius; ky++)
        {
            int ny = min(max(y + ky, 0), meta.height - 1);
            for (int kx = -radius; kx <= radius; kx++)
            {
                int nx = min(max(x + kx, 0), meta.width - 1);
                size_t nidx = ((size_t)ny * (size_t)meta.width + (size_t)nx) * channels + (size_t)c;
                pixel_value += input[(size_t)meta.offset + nidx] * kernel[(ky + radius) * blurAmount + (kx + radius)];
            }
        }

        // clamp and write
        int outv = static_cast<int>(pixel_value + 0.5f);
        if (outv < 0)
        {
            outv = 0;
        }
        if (outv > 255)
        {
            outv = 255;
        }

        output[(size_t)meta.offset + pixel_idx] = static_cast<unsigned char>(outv);
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

        // Build kernel
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

// Convolution (parallelized)
#pragma omp parallel for collapse(2)
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    float pixel_value = 0.0f;

                    for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
                    {
                        int ny = std::clamp(y + ky, 0, height - 1);

                        for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                        {
                            int nx = std::clamp(x + kx, 0, width - 1);

                            int nidx = (ny * width + nx) * channels + c;
                            float in_val = (float)input_image[nidx];
                            float k = kernel[(ky + kernelRadius) * blurAmount + (kx + kernelRadius)];
                            pixel_value += in_val * k;
                        }
                    }

                    int idx = (y * width + x) * channels + c;
                    int outv = (int)(pixel_value + 0.5f);
                    output_image[idx] = (unsigned char)std::clamp(outv, 0, 255);
                }
            }
        }
    }
}
