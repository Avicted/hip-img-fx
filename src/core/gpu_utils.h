#pragma once

#include <hip/hip_runtime.h>
#include <omp.h>

#include <vector>
#include <string>

#include "image.h"
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
            hipFree(ptr);
        ptr = o.ptr;
        size = o.size;
        o.ptr = nullptr;
        o.size = 0;
        return *this;
    }
    ~DeviceBuffer()
    {
        if (ptr)
            hipFree(ptr);
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
inline dim3 compute_grid(int width, int height, const dim3 &block = dim3(16, 16));

hipError_t prepare_device_batch(
    std::vector<image_t> &host_in,
    std::vector<image_t> &host_out,
    DeviceBatch &batch,
    int device_id = 0);

hipError_t copy_back_batch(
    std::vector<image_t> &host_out,
    DeviceBatch &batch);

void free_batch(DeviceBatch &batch);

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

template <typename Launcher>
hipError_t apply_filter_generic_templated(
    std::vector<image_t> &input_images,
    std::vector<image_t> &output_images,
    Launcher &&launch_kernel,
    dim3 blockSize = dim3(16, 16),
    int device_id = 0,
    size_t shared_bytes = 0);

int get_hip_devices(void);

std::string filter_type_to_string(FILTER_TYPE type);
