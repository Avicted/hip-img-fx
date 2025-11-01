#pragma once

#include <hip/hip_runtime.h>

enum class FILTER_TYPE
{
    UNDEFINED,
    GRAYSCALE,
    NEGATIVE,
};

inline void hip_errchk(hipError_t err, const char *file, int line);
int get_hip_devices(void);

std::string filter_type_to_string(FILTER_TYPE type);

int apply_filter(
    FILTER_TYPE filter_type,
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

hipError_t apply_grayscale_filter(
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

__global__ void grayscale_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

hipError_t apply_negative_filter(
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

__global__ void negative_kernel(
    const unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);