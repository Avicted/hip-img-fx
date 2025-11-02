#pragma once

#include <hip/hip_runtime.h>

extern "C" __global__ void grayscale_kernel(
    const unsigned char *input,
    unsigned char *output,
    const image_meta_t *metas,
    int num_images);

extern "C" __global__ void negative_kernel(
    const unsigned char *input,
    unsigned char *output,
    const image_meta_t *metas,
    int num_images);

extern "C" __global__ void gaussian_blur_kernel(
    const unsigned char *input,
    unsigned char *output,
    const image_meta_t *metas,
    int num_images,
    int blurAmount);

void grayscale_cpu(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

void gaussian_blur_cpu(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    int blurAmount);

void negative_cpu(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);
