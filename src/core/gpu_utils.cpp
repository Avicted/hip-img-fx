#include "gpu_utils.h"

namespace imgfx::core
{
    int get_hip_devices(void) noexcept
    {
        int deviceCount = 0;
        hipError_t error = hipGetDeviceCount(&deviceCount);

        if (error != hipSuccess)
        {
            printf("    Failed to get HIP Device Count: %s\n", hipGetErrorString(error));
            return -1;
        }
        else
        {
            printf("    HIP Device Count: %d\n", deviceCount);
        }

        for (int i = 0; i < deviceCount; i++)
        {
            hipDeviceProp_t prop;
            hipError_t error = hipGetDeviceProperties(&prop, i);

            if (error != hipSuccess)
            {
                printf("    Failed to get HIP Device Properties: %s\n", hipGetErrorString(error));
                continue;
            }
            else
            {
                printf("    Device %d: %s\n", i, prop.name);
                printf("        Compute Capability: ------------ = %d.%d\n", prop.major, prop.minor);
                printf("        Total Global Memory: ----------- = %lu\n", prop.totalGlobalMem);
                printf("        Shared Memory per Block: ------- = %lu\n", prop.sharedMemPerBlock);
                printf("        Registers per Block: ----------- = %d\n", prop.regsPerBlock);
                printf("        Warp Size: --------------------- = %d\n", prop.warpSize);
                printf("        Max Threads per Block: --------- = %d\n", prop.maxThreadsPerBlock);
                printf("        Max Threads Dimension: --------- = (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
                printf("        Max Grid Size: ----------------- = (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
                printf("        Clock Rate: -------------------- = %d\n", prop.clockRate);
                printf("        Total Constant Memory: --------- = %lu\n", prop.totalConstMem);
                printf("        Multiprocessor Count: ---------- = %d\n", prop.multiProcessorCount);
                printf("        L2 Cache Size: ----------------- = %d\n", prop.l2CacheSize);
                printf("        Max Threads per Multiprocessor:  = %d\n", prop.maxThreadsPerMultiProcessor);
                printf("        Unified Addressing: ------------ = %d\n", prop.unifiedAddressing);
                printf("        Memory Clock Rate: ------------- = %d\n", prop.memoryClockRate);
                printf("        Memory Bus Width: -------------- = %d\n", prop.memoryBusWidth);
                printf("        Peak Memory Bandwidth: --------- = %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
            }
        }

        return deviceCount;
    }

    std::string filter_type_to_string(FILTER_TYPE type)
    {
        switch (type)
        {
        case FILTER_TYPE::UNDEFINED:
            return "UNDEFINED";
        case FILTER_TYPE::GRAYSCALE:
            return "GRAYSCALE";
        case FILTER_TYPE::NEGATIVE:
            return "NEGATIVE";
        case FILTER_TYPE::GAUSSIAN_BLUR:
            return "GAUSSIAN_BLUR";
        default:
            return "UNKNOWN";
        }
    }

    hipError_t apply_filter_cpu(
        FILTER_TYPE filter_type,
        unsigned char *input_image,
        unsigned char *output_image,
        int width,
        int height,
        int channels)
    {
        switch (filter_type)
        {
        case FILTER_TYPE::GRAYSCALE:
        {
            imgfx::filters::grayscale_cpu(input_image, output_image, width, height, channels);
            return hipSuccess;
        }
        case FILTER_TYPE::NEGATIVE:
        {
            imgfx::filters::negative_cpu(input_image, output_image, width, height, channels);
            return hipSuccess;
        }
        case FILTER_TYPE::GAUSSIAN_BLUR:
        {
            if (GAUSSIAN_BLUR_AMOUNT % 2 == 0)
            {
                fprintf(stderr, "ERROR: blurAmount must be an odd number. You chose: %d.\n", GAUSSIAN_BLUR_AMOUNT);
                return hipErrorInvalidValue;
            }

            imgfx::filters::gaussian_blur_cpu(input_image, output_image, width, height, channels, GAUSSIAN_BLUR_AMOUNT);
            return hipSuccess;
        }
        default:
            printf("ERROR: Unsupported CPU filter type\n");
            return hipErrorInvalidValue;
        }
    }

    hipError_t apply_filter_gpu(
        FILTER_TYPE filter_type,
        image_t &input_image,
        image_t &output_image,
        bool enable_timing,
        GPUTimings *timings)
    {
        // Create timing events if requested
        HIPEvent start_h2d, end_h2d, start_kernel, end_kernel, start_d2h, end_d2h;

        if (enable_timing && timings)
        {
            timings->h2d_ms = 0.0f;
            timings->kernel_ms = 0.0f;
            timings->d2h_ms = 0.0f;
            timings->total_ms = 0.0f;
        }

        // Calculate image size
        size_t total_bytes = size_t(input_image.width) * input_image.height * input_image.channels;

        // Prepare metadata
        image_meta_t meta;
        meta.width = input_image.width;
        meta.height = input_image.height;
        meta.channels = input_image.channels;
        meta.offset = 0;

        // Allocate device buffers
        DeviceBuffer d_input, d_output, d_meta;
        HIP_ERRCHK(hipMalloc(&d_input.ptr, total_bytes));
        HIP_ERRCHK(hipMalloc(&d_output.ptr, total_bytes));
        HIP_ERRCHK(hipMalloc(&d_meta.ptr, sizeof(image_meta_t)));

        // === H2D Transfer ===
        if (enable_timing)
            start_h2d.record();

        // Copy image to device
        HIP_ERRCHK(hipMemcpy(d_input.ptr, input_image.data, total_bytes, hipMemcpyHostToDevice));
        HIP_ERRCHK(hipMemcpy(d_meta.ptr, &meta, sizeof(image_meta_t), hipMemcpyHostToDevice));

        if (enable_timing)
        {
            end_h2d.record();
            end_h2d.synchronize();
        }

        // === Kernel Execution ===
        if (enable_timing)
            start_kernel.record();

        int threads = 512;
        int blocks = (total_bytes + threads - 1) / threads;

        switch (filter_type)
        {
        case FILTER_TYPE::GRAYSCALE:
        {
            hipLaunchKernelGGL(imgfx::filters::grayscale_kernel, dim3(blocks), dim3(threads), 0, 0,
                               (unsigned char *)d_input.ptr,
                               (unsigned char *)d_output.ptr,
                               (image_meta_t *)d_meta.ptr,
                               1);
            break;
        }
        case FILTER_TYPE::NEGATIVE:
        {
            hipLaunchKernelGGL(imgfx::filters::negative_kernel, dim3(blocks), dim3(threads), 0, 0,
                               (unsigned char *)d_input.ptr,
                               (unsigned char *)d_output.ptr,
                               (image_meta_t *)d_meta.ptr,
                               1);
            break;
        }
        case FILTER_TYPE::GAUSSIAN_BLUR:
        {
            if (GAUSSIAN_BLUR_AMOUNT % 2 == 0)
            {
                fprintf(stderr, "ERROR: blurAmount must be an odd number. You chose: %d.\n", GAUSSIAN_BLUR_AMOUNT);
                return hipErrorInvalidValue;
            }

            size_t shared_bytes = sizeof(float) * GAUSSIAN_BLUR_AMOUNT * GAUSSIAN_BLUR_AMOUNT;

            hipLaunchKernelGGL(
                imgfx::filters::gaussian_blur_kernel,
                dim3(blocks),
                dim3(threads),
                shared_bytes,
                0,
                (unsigned char *)d_input.ptr,
                (unsigned char *)d_output.ptr,
                (image_meta_t *)d_meta.ptr,
                1,
                GAUSSIAN_BLUR_AMOUNT);
            break;
        }
        default:
        {
            printf("ERROR: Unsupported GPU filter type\n");
            return hipErrorInvalidValue;
        }
        }

        if (enable_timing)
        {
            end_kernel.record();
        }

        HIP_ERRCHK(hipDeviceSynchronize());

        if (enable_timing)
        {
            end_kernel.synchronize();
        }

        // === D2H Transfer ===
        if (enable_timing)
            start_d2h.record();

        // Copy result back to host
        HIP_ERRCHK(hipMemcpy(output_image.data, d_output.ptr, total_bytes, hipMemcpyDeviceToHost));

        if (enable_timing)
        {
            end_d2h.record();
            end_d2h.synchronize();
        }

        // Calculate timings
        if (enable_timing && timings)
        {
            timings->h2d_ms = HIPEvent::elapsed_time(start_h2d, end_h2d);
            timings->kernel_ms = HIPEvent::elapsed_time(start_kernel, end_kernel);
            timings->d2h_ms = HIPEvent::elapsed_time(start_d2h, end_d2h);
            timings->total_ms = timings->h2d_ms + timings->kernel_ms + timings->d2h_ms;
        }

        // Free device memory
        HIP_ERRCHK(hipFree(d_input.ptr));
        HIP_ERRCHK(hipFree(d_output.ptr));
        HIP_ERRCHK(hipFree(d_meta.ptr));

        return hipSuccess;
    }
}
