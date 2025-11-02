#include "gpu_utils.h"

int get_hip_devices(void)
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
        grayscale_cpu(input_image, output_image, width, height, channels);
        return hipSuccess;
    }
    case FILTER_TYPE::NEGATIVE:
    {
        negative_cpu(input_image, output_image, width, height, channels);
        return hipSuccess;
    }
    case FILTER_TYPE::GAUSSIAN_BLUR:
    {
        int blurAmount = 11; // example fixed blur amount
        if (blurAmount % 2 == 0)
        {
            fprintf(stderr, "ERROR: blurAmount must be an odd number. You chose: %d.\n", blurAmount);
            return hipErrorInvalidValue;
        }

        gaussian_blur_cpu(input_image, output_image, width, height, channels, blurAmount);
        return hipSuccess;
    }
    default:
        printf("ERROR: Unsupported CPU filter type\n");
        return hipErrorInvalidValue;
    }
}

hipError_t apply_filter_gpu(
    FILTER_TYPE filter_type,
    std::vector<image_t> &input_images,
    std::vector<image_t> &output_images)
{
    // Compute total pixels and metadata
    size_t total_bytes = 0;
    std::vector<image_meta_t> metas(input_images.size());

    for (size_t i = 0; i < input_images.size(); ++i)
    {
        metas[i].width = input_images[i].width;
        metas[i].height = input_images[i].height;
        metas[i].channels = input_images[i].channels;
        metas[i].offset = total_bytes / sizeof(unsigned char);
        total_bytes += size_t(input_images[i].width) * input_images[i].height * input_images[i].channels;
    }

    // Allocate contiguous device buffers
    DeviceBuffer d_input, d_output, d_metas;
    HIP_ERRCHK(hipMalloc(&d_input.ptr, total_bytes));
    HIP_ERRCHK(hipMalloc(&d_output.ptr, total_bytes));
    HIP_ERRCHK(hipMalloc(&d_metas.ptr, sizeof(image_meta_t) * metas.size()));

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

    // Launch kernel
    // printf("Launching GPU filter kernel: %s\n", filter_type_to_string(filter_type).c_str());

    int threads = 512;
    int blocks = (total_bytes + threads - 1) / threads;

    switch (filter_type)
    {
    case FILTER_TYPE::GRAYSCALE:
    {
        hipLaunchKernelGGL(grayscale_kernel, dim3(blocks), dim3(threads), 0, 0,
                           (unsigned char *)d_input.ptr,
                           (unsigned char *)d_output.ptr,
                           (image_meta_t *)d_metas.ptr,
                           (int)input_images.size());
        break;
    }
    case FILTER_TYPE::NEGATIVE:
    {
        hipLaunchKernelGGL(negative_kernel, dim3(blocks), dim3(threads), 0, 0,
                           (unsigned char *)d_input.ptr,
                           (unsigned char *)d_output.ptr,
                           (image_meta_t *)d_metas.ptr,
                           (int)input_images.size());
        break;
    }
    case FILTER_TYPE::GAUSSIAN_BLUR:
    {
        int blurAmount = 11; // example fixed blur amount
        if (blurAmount % 2 == 0)
        {
            fprintf(stderr, "ERROR: blurAmount must be an odd number. You chose: %d.\n", blurAmount);
            return hipErrorInvalidValue;
        }

        size_t shared_bytes = sizeof(float) * blurAmount * blurAmount;

        hipLaunchKernelGGL(
            gaussian_blur_kernel,         // kernel
            dim3(blocks),                 // grid
            dim3(threads),                // block
            shared_bytes,                 // dynamic shared memory
            0,                            // stream
            (unsigned char *)d_input.ptr, // kernel args start here
            (unsigned char *)d_output.ptr,
            (image_meta_t *)d_metas.ptr,
            (int)input_images.size(),
            blurAmount);
        break;
    }
    default:
    {
        printf("ERROR: Unsupported GPU filter type\n");
        return hipErrorInvalidValue;
    }
    }

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy back flattened output to original image buffers
    pos = 0;
    for (size_t i = 0; i < output_images.size(); ++i)
    {
        size_t bytes = size_t(output_images[i].width) * output_images[i].height * output_images[i].channels;
        HIP_ERRCHK(hipMemcpy(output_images[i].data, (unsigned char *)d_output.ptr + pos, bytes, hipMemcpyDeviceToHost));
        pos += bytes;
    }

    // Free device memory
    HIP_ERRCHK(hipFree(d_input.ptr));
    HIP_ERRCHK(hipFree(d_output.ptr));
    HIP_ERRCHK(hipFree(d_metas.ptr));

    return hipSuccess;
}
