#include "gpu_utils.h"

// HIP error handling macro
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__))

inline void hip_errchk(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

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

int apply_filter(
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
        return apply_filter_generic_templated(input_image, output_image, width, height, channels,
                                              [&](unsigned char *d_in, unsigned char *d_out, dim3 grid, dim3 block, size_t shared_bytes)
                                              {
                                                  hipLaunchKernelGGL(
                                                      grayscale_kernel,
                                                      grid,
                                                      block,
                                                      shared_bytes,
                                                      0,
                                                      d_in,
                                                      d_out,
                                                      width,
                                                      height,
                                                      channels);
                                              });
    }
    case FILTER_TYPE::NEGATIVE:
    {
        return apply_filter_generic_templated(input_image, output_image, width, height, channels,
                                              [&](unsigned char *d_in, unsigned char *d_out, dim3 grid, dim3 block, size_t shared_bytes)
                                              {
                                                  hipLaunchKernelGGL(
                                                      negative_kernel,
                                                      grid,
                                                      block,
                                                      shared_bytes,
                                                      0,
                                                      d_in,
                                                      d_out,
                                                      width,
                                                      height,
                                                      channels);
                                              });
    }
    case FILTER_TYPE::GAUSSIAN_BLUR:
    {
        const int blurAmount = 11; // must be odd
        if (blurAmount % 2 == 0)
        {
            return hipErrorInvalidValue;
        }

        size_t shared_bytes = sizeof(float) * (size_t)blurAmount * (size_t)blurAmount;

        return apply_filter_generic_templated(input_image, output_image, width, height, channels, [&](unsigned char *d_in, unsigned char *d_out, dim3 grid, dim3 block, size_t sb)
                                              { hipLaunchKernelGGL(
                                                    gaussian_blur_kernel,
                                                    grid,
                                                    block,
                                                    sb,
                                                    0,
                                                    d_in,
                                                    d_out,
                                                    width,
                                                    height,
                                                    channels,
                                                    blurAmount); }, dim3(16, 16), 0, shared_bytes);
    }
    default:
    {
        printf("ERROR: Unsupported filter type!\n");
        return -1;
    }
    }
}

inline dim3 compute_grid(int width, int height, const dim3 &block)
{
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

hipError_t prepare_device_buffers(
    unsigned char *input_image,
    DeviceBuffer &d_input,
    DeviceBuffer &d_output,
    size_t image_bytes,
    int device_id)
{
    int hip_device_count = get_hip_devices();
    if (hip_device_count < 1)
    {
        fprintf(stderr, "ERROR: Could not find any HIP device!\n");
        return hipErrorNoDevice;
    }
    HIP_ERRCHK(hipSetDevice(device_id));
    HIP_ERRCHK(hipMalloc(reinterpret_cast<void **>(&d_input.ptr), image_bytes));
    HIP_ERRCHK(hipMalloc(reinterpret_cast<void **>(&d_output.ptr), image_bytes));
    d_input.size = image_bytes;
    d_output.size = image_bytes;
    HIP_ERRCHK(hipMemcpy(d_input.ptr, input_image, image_bytes, hipMemcpyHostToDevice));
    return hipSuccess;
}

hipError_t copy_back_and_finish(unsigned char *output_image, DeviceBuffer &d_output, size_t image_bytes)
{
    HIP_ERRCHK(hipMemcpy(output_image, d_output.ptr, image_bytes, hipMemcpyDeviceToHost));
    return hipSuccess;
}

template <typename Launcher>
hipError_t apply_filter_generic_templated(
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels,
    Launcher &&launch_kernel,
    dim3 blockSize,
    int device_id,
    size_t shared_bytes)
{
    size_t image_bytes = (size_t)width * (size_t)height * (size_t)channels * sizeof(unsigned char);
    DeviceBuffer d_input, d_output;

    HIP_ERRCHK(prepare_device_buffers(input_image, d_input, d_output, image_bytes, device_id));

    dim3 gridSize = compute_grid(width, height, blockSize);

    printf("====================================================================\n");
    printf("    Using GPU %d\n", device_id);

    launch_kernel(d_input.ptr, d_output.ptr, gridSize, blockSize, shared_bytes);

    HIP_ERRCHK(hipDeviceSynchronize());
    HIP_ERRCHK(copy_back_and_finish(output_image, d_output, image_bytes));

    return hipSuccess;
}
