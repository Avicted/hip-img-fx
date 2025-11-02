#pragma once

#include <hip/hip_runtime.h>
#include <omp.h>

#include <vector>
#include <string>
#include <filesystem>

#include "image.h"
#include "filters/filters.h"

#define GAUSSIAN_BLUR_AMOUNT 11 // Has to be an odd number

#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__))

inline void hip_errchk(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        fprintf(stderr, "\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

enum class FILTER_TYPE
{
    UNDEFINED,
    GRAYSCALE,
    NEGATIVE,
    GAUSSIAN_BLUR,
};

struct DeviceBuffer
{
    unsigned char *ptr = nullptr;
    size_t size = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&o) noexcept
    {
        ptr = o.ptr;
        size = o.size;
        o.ptr = nullptr;
        o.size = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept
    {
        if (ptr)
        {
            HIP_ERRCHK(hipFree(ptr));
        }

        ptr = o.ptr;
        size = o.size;
        o.ptr = nullptr;
        o.size = 0;
        return *this;
    }
    ~DeviceBuffer()
    {
        if (ptr)
        {
            hipError_t err = hipFree(ptr);
            if (err != hipSuccess && err != hipErrorInvalidValue)
            {
                fprintf(stderr, "hipFree failed: %s\n", hipGetErrorString(err));
            }
            ptr = nullptr;
        }
    }

    hipError_t allocate(size_t bytes)
    {
        size = bytes;
        return hipMalloc(reinterpret_cast<void **>(&ptr), bytes);
    }
};

struct DeviceBatch
{
    image_t *d_images = nullptr;
    unsigned char *d_pixels = nullptr; // all images concatenated
    size_t total_bytes = 0;
    int N = 0;
};

inline void hip_errchk(hipError_t err, const char *file, int line);
int get_hip_devices(void);
std::string filter_type_to_string(FILTER_TYPE type);

hipError_t apply_filter_cpu(
    FILTER_TYPE filter_type,
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

hipError_t apply_filter_gpu(
    FILTER_TYPE filter_type,
    std::vector<image_t> &input_images,
    std::vector<image_t> &output_images);
