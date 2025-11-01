#pragma once

#include <hip/hip_runtime.h>

extern "C" __global__ void grayscale_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

extern "C" __global__ void negative_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

extern "C" __global__ void gaussian_blur_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    int blurAmount);
