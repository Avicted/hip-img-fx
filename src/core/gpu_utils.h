#pragma once

#include <hip/hip_runtime.h>
#include <omp.h>

#include <vector>
#include <string>
#include <filesystem>
#include <cstdio>
#include <cstdlib>

#include "image.h"
#include "../filters/filters.h"

namespace imgfx::core
{
    constexpr int GAUSSIAN_BLUR_AMOUNT = 11; // Must be odd

    // Error checking helper
    inline void hip_errchk(hipError_t err, const char *file, int line)
    {
        if (err != hipSuccess)
        {
            std::fprintf(stderr, "\nHIP error: %s in %s at line %d\n",
                         hipGetErrorString(err), file, line);
            std::exit(EXIT_FAILURE);
        }
    }

#define HIP_ERRCHK(err) (imgfx::core::hip_errchk(err, __FILE__, __LINE__))

    enum class FILTER_TYPE
    {
        UNDEFINED,
        GRAYSCALE,
        NEGATIVE,
        GAUSSIAN_BLUR,
    };

    // Device memory wrapper
    struct DeviceBuffer
    {
        unsigned char *ptr = nullptr;
        size_t size = 0;

        DeviceBuffer() = default;

        // Disable copy
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        // Move semantics
        DeviceBuffer(DeviceBuffer &&other) noexcept
            : ptr(other.ptr), size(other.size)
        {
            other.ptr = nullptr;
            other.size = 0;
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (ptr)
            {
                HIP_ERRCHK(hipFree(ptr));
            }
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
            return *this;
        }

        // Destructor safely frees GPU memory
        ~DeviceBuffer()
        {
            if (ptr)
            {
                hipError_t err = hipFree(ptr);
                if (err != hipSuccess && err != hipErrorInvalidValue)
                {
                    std::fprintf(stderr, "hipFree failed: %s\n", hipGetErrorString(err));
                }
                ptr = nullptr;
            }
        }

        // Allocate GPU memory
        hipError_t allocate(size_t bytes)
        {
            size = bytes;
            return hipMalloc(reinterpret_cast<void **>(&ptr), bytes);
        }
    };

    // Batch of device images
    struct DeviceBatch
    {
        image_t *d_images = nullptr;
        unsigned char *d_pixels = nullptr; // concatenated pixel data
        size_t total_bytes = 0;
        int N = 0;
    };

    // Function declarations
    int get_hip_devices() noexcept;

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

}
