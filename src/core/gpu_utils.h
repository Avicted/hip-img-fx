#pragma once

#include <hip/hip_runtime.h>
#include <omp.h>

#include <vector>
#include <string>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "image.h"
#include "../filters/filters.h"

namespace imgfx::core
{
    constexpr int GAUSSIAN_BLUR_AMOUNT = 11; // Must be odd

    // Error checking helper (defined first for use in classes below)
    inline void hip_errchk(hipError_t err, const char *file, int line)
    {
        if (err != hipSuccess)
        {
            std::fprintf(stderr, "\nHIP error: %s in %s at line %d\n",
                         hipGetErrorString(err), file, line);
            std::exit(EXIT_FAILURE);
        }
    }

#ifndef HIP_ERRCHK
#define HIP_ERRCHK(err) (imgfx::core::hip_errchk(err, __FILE__, __LINE__))
#endif

    // GPU Timing Infrastructure
    struct GPUTimings
    {
        float h2d_ms = 0.0f;    // Host to Device transfer time
        float kernel_ms = 0.0f; // Kernel execution time
        float d2h_ms = 0.0f;    // Device to Host transfer time
        float total_ms = 0.0f;  // Total GPU pipeline time

        void print() const
        {
            printf("GPU Timings:\n");
            printf("  H2D:    %8.3f ms\n", h2d_ms);
            printf("  Kernel: %8.3f ms\n", kernel_ms);
            printf("  D2H:    %8.3f ms\n", d2h_ms);
            printf("  Total:  %8.3f ms\n", total_ms);
        }
    };

    // RAII wrapper for HIP events with stream support
    class HIPEvent
    {
    private:
        hipEvent_t event;
        bool valid;

    public:
        HIPEvent() : valid(false)
        {
            hipError_t err = hipEventCreate(&event);
            if (err == hipSuccess)
            {
                valid = true;
            }
        }

        ~HIPEvent()
        {
            if (valid)
            {
                (void)hipEventDestroy(event);
            }
        }

        // Disable copy
        HIPEvent(const HIPEvent &) = delete;
        HIPEvent &operator=(const HIPEvent &) = delete;

        // Enable move
        HIPEvent(HIPEvent &&other) noexcept : event(other.event), valid(other.valid)
        {
            other.valid = false;
        }

        hipEvent_t get() const { return event; }
        bool is_valid() const { return valid; }

        void record(hipStream_t stream = 0)
        {
            if (valid)
            {
                HIP_ERRCHK(hipEventRecord(event, stream));
            }
        }

        void synchronize()
        {
            if (valid)
            {
                HIP_ERRCHK(hipEventSynchronize(event));
            }
        }

        static float elapsed_time(const HIPEvent &start, const HIPEvent &end)
        {
            float ms = 0.0f;
            if (start.is_valid() && end.is_valid())
            {
                HIP_ERRCHK(hipEventElapsedTime(&ms, start.get(), end.get()));
            }
            return ms;
        }
    };

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

    // Single image processing with optional timing
    hipError_t apply_filter_gpu(
        FILTER_TYPE filter_type,
        image_t &input_image,
        image_t &output_image,
        bool enable_timing = false,
        GPUTimings *timings = nullptr);

    // Batch processing for multiple images (with optional timing)
    hipError_t apply_filter_gpu(
        FILTER_TYPE filter_type,
        std::vector<image_t> &input_images,
        std::vector<image_t> &output_images,
        bool enable_timing = false,
        GPUTimings *timings = nullptr);

}
