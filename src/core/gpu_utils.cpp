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

    // Batch processing for multiple images (with optional timing)
    hipError_t apply_filter_gpu(
        FILTER_TYPE filter_type,
        std::vector<image_t> &input_images,
        std::vector<image_t> &output_images,
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

        // Compute total bytes and metadata
        size_t total_bytes = 0;
        size_t max_image_bytes = 0;
        std::vector<image_meta_t> metas(input_images.size());

        for (size_t i = 0; i < input_images.size(); ++i)
        {
            metas[i].width = input_images[i].width;
            metas[i].height = input_images[i].height;
            metas[i].channels = input_images[i].channels;
            metas[i].offset = total_bytes;
            const size_t bytes = size_t(input_images[i].width) * input_images[i].height * input_images[i].channels;
            total_bytes += bytes;
            if (bytes > max_image_bytes)
            {
                max_image_bytes = bytes;
            }
        }

        // Allocate contiguous device buffers
        DeviceBuffer d_input, d_output, d_metas;
        HIP_ERRCHK(hipMalloc(&d_input.ptr, total_bytes));
        HIP_ERRCHK(hipMalloc(&d_output.ptr, total_bytes));
        HIP_ERRCHK(hipMalloc(&d_metas.ptr, sizeof(image_meta_t) * metas.size()));

        // H2D Transfer
        if (enable_timing)
        {
            start_h2d.record();
        }

        // Copy pixels to device
        size_t pos = 0;
        for (size_t i = 0; i < input_images.size(); ++i)
        {
            size_t bytes = size_t(input_images[i].width) * input_images[i].height * input_images[i].channels;
            HIP_ERRCHK(hipMemcpy((unsigned char *)d_input.ptr + pos, input_images[i].data, bytes, hipMemcpyHostToDevice));
            pos += bytes;
        }

        // Copy metadata
        HIP_ERRCHK(hipMemcpy(d_metas.ptr, metas.data(), sizeof(image_meta_t) * metas.size(), hipMemcpyHostToDevice));

        if (enable_timing)
        {
            end_h2d.record();
            end_h2d.synchronize();
        }

        // Kernel Execution
        if (enable_timing)
        {
            start_kernel.record();
        }

        // Launch kernel: grid.x covers bytes within an image; grid.y selects the image.
        const int threads = 512;
        const int blocks_x = (max_image_bytes + threads - 1) / threads;
        const dim3 grid((unsigned int)blocks_x, (unsigned int)input_images.size(), 1);

        switch (filter_type)
        {
        case FILTER_TYPE::GRAYSCALE:
        {
            hipLaunchKernelGGL(imgfx::filters::grayscale_kernel, grid, dim3(threads), 0, 0,
                               (unsigned char *)d_input.ptr,
                               (unsigned char *)d_output.ptr,
                               (image_meta_t *)d_metas.ptr,
                               (int)input_images.size());
            break;
        }
        case FILTER_TYPE::NEGATIVE:
        {
            hipLaunchKernelGGL(imgfx::filters::negative_kernel, grid, dim3(threads), 0, 0,
                               (unsigned char *)d_input.ptr,
                               (unsigned char *)d_output.ptr,
                               (image_meta_t *)d_metas.ptr,
                               (int)input_images.size());
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
                grid,
                dim3(threads),
                shared_bytes,
                0,
                (unsigned char *)d_input.ptr,
                (unsigned char *)d_output.ptr,
                (image_meta_t *)d_metas.ptr,
                (int)input_images.size(),
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

        // D2H Transfer
        if (enable_timing)
        {
            start_d2h.record();
        }

        // Copy back flattened output to original image buffers
        pos = 0;
        for (size_t i = 0; i < output_images.size(); ++i)
        {
            size_t bytes = size_t(output_images[i].width) * output_images[i].height * output_images[i].channels;
            HIP_ERRCHK(hipMemcpy(output_images[i].data, (unsigned char *)d_output.ptr + pos, bytes, hipMemcpyDeviceToHost));
            pos += bytes;
        }

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
        HIP_ERRCHK(hipFree(d_metas.ptr));

        return hipSuccess;
    }

    // Single image wrapper - calls batch version with 1-element vector
    hipError_t apply_filter_gpu(
        FILTER_TYPE filter_type,
        image_t &input_image,
        image_t &output_image,
        bool enable_timing,
        GPUTimings *timings)
    {
        std::vector<image_t> input_vec = {input_image};
        std::vector<image_t> output_vec = {output_image};

        return apply_filter_gpu(filter_type, input_vec, output_vec, enable_timing, timings);
    }
}
