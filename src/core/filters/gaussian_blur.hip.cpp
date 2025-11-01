#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cmath>
#include <vector>

extern "C" __global__ void gaussian_blur_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    int blurAmount // must be odd, e.g. 3, 5, 7
)
{
    extern __shared__ float kernel[]; // dynamically allocated shared memory

    const int kernelRadius = blurAmount / 2;
    const int kernelSize = blurAmount * blurAmount;

    // Compute kernel in shared memory once per block (thread 0,0)
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float sigma = blurAmount / 2.0f;
        float sum = 0.0f;

        // compute unnormalized kernel
        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
        {
            for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
            {
                float v = expf(-((float)(kx * kx + ky * ky)) / (2.0f * sigma * sigma));
                kernel[(ky + kernelRadius) * blurAmount + (kx + kernelRadius)] = v;
                sum += v;
            }
        }

        // normalize (guard against sum == 0)
        if (sum <= 0.0f)
        {
            sum = 1.0f;
        }
        for (int i = 0; i < kernelSize; ++i)
        {
            kernel[i] /= sum;
        }
    }
    __syncthreads(); // wait until kernel[] is ready

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int c = 0; c < channels; ++c)
    {
        float pixel_value = 0.0f;

        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
        {
            int ny = y + ky;

            // clamp manually to [0, height-1]
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
                float in_val = static_cast<float>(input_image[nidx]);
                float k = kernel[(ky + kernelRadius) * blurAmount + (kx + kernelRadius)];
                pixel_value += in_val * k;
            }
        }

        // clamp to [0,255] and write
        int idx = (y * width + x) * channels + c;
        int outv = static_cast<int>(pixel_value + 0.5f); // round
        if (outv < 0)
        {
            outv = 0;
        }
        else if (outv > 255)
        {
            outv = 255;
        }

        output_image[idx] = static_cast<unsigned char>(outv);
    }
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
