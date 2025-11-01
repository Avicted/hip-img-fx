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
        return apply_grayscale_filter(input_image, output_image, width, height, channels);
    default:
        printf("ERROR: Unsupported filter type!\n");
        return -1;
    }
}

hipError_t apply_grayscale_filter(
    unsigned char *input_image,
    unsigned char *output_image,
    int width,
    int height,
    int channels)
{
    int hip_device_count = get_hip_devices();
    if (hip_device_count < 1)
    {
        fprintf(stderr, "ERROR: Could not find any HIP device!\n");
        return hipErrorNoDevice;
    }
    printf("hip_device_count: %d\n", hip_device_count);

    hipError_t err;

    size_t image_bytes = width * height * channels * sizeof(unsigned char);

    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;

    // Allocate device memory
    HIP_ERRCHK(hipMalloc(reinterpret_cast<void **>(&d_input), image_bytes));
    HIP_ERRCHK(hipMalloc(reinterpret_cast<void **>(&d_output), image_bytes));

    // Copy data to device memory
    HIP_ERRCHK(hipMemcpy(
        d_input,
        input_image,
        image_bytes,
        hipMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    HIP_ERRCHK(hipSetDevice(0));
    printf("====================================================================\n");
    printf("    Using GPU 0\n");

    hipLaunchKernelGGL(
        grayscale_kernel,
        gridSize,
        blockSize,
        0,
        0,
        d_input,
        d_output,
        width,
        height,
        channels);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(
        output_image,
        d_output,
        image_bytes,
        hipMemcpyDeviceToHost));

    // Free device memory
    HIP_ERRCHK(hipFree(d_input));
    HIP_ERRCHK(hipFree(d_output));

    return hipSuccess;
}
