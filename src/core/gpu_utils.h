#pragma once

#include <hip/hip_runtime.h>

#include "filters/filters.h"

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
            (void)hipFree(ptr);
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
            (void)hipFree(ptr);
        }
    }
    hipError_t allocate(size_t bytes)
    {
        size = bytes;
        return hipMalloc(reinterpret_cast<void **>(&ptr), bytes);
    }
};

inline void hip_errchk(hipError_t err, const char *file, int line);
inline dim3 compute_grid(int width, int height, const dim3 &block = dim3(16, 16));

hipError_t prepare_device_buffers(
    unsigned char *input_image,
    DeviceBuffer &d_input,
    DeviceBuffer &d_output,
    size_t image_bytes,
    int device_id = 0);

hipError_t copy_back_and_finish(unsigned char *output_image, DeviceBuffer &d_output, size_t image_bytes);

hipError_t apply_filter(
    bool use_cpu,
    FILTER_TYPE filter_type,
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels);

template <typename Launcher>
hipError_t apply_filter_generic_templated(
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    Launcher &&launch_kernel,
    dim3 blockSize = dim3(16, 16),
    int device_id = 0,
    size_t shared_bytes = 0);

int get_hip_devices(void);

std::string filter_type_to_string(FILTER_TYPE type);
