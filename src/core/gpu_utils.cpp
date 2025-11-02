#include "gpu_utils.h"
#include <cstdio>
#include <cstdlib>

// HIP error handling macro
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__))

inline void hip_errchk(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        printf("\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
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

inline dim3 compute_grid(int width, int height, const dim3 &block)
{
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

// ------------------------ batch helpers ------------------------

hipError_t prepare_device_batch(
    std::vector<image_t> &host_in,
    std::vector<image_t> &host_out,
    DeviceBatch &batch,
    int device_id)
{
    batch.N = (int)host_in.size();
    int N = batch.N;
    HIP_ERRCHK(hipSetDevice(device_id));

    // compute total bytes
    batch.total_bytes = 0;
    for (auto &img : host_in)
        batch.total_bytes += size_t(img.width) * img.height * img.channels;

    // allocate single pixel buffer
    HIP_ERRCHK(hipMalloc(&batch.d_pixels, batch.total_bytes));

    // allocate image_t array
    HIP_ERRCHK(hipMalloc(&batch.d_images, sizeof(image_t) * N));

    // copy image_t structs and patch .data pointers
    size_t offset = 0;
    std::vector<image_t> tmp_images(N);
    for (int i = 0; i < N; ++i)
    {
        size_t bytes = size_t(host_in[i].width) * host_in[i].height * host_in[i].channels;
        unsigned char *d_ptr = batch.d_pixels + offset;
        HIP_ERRCHK(hipMemcpy(d_ptr, host_in[i].data, bytes, hipMemcpyHostToDevice));

        image_t tmp = host_in[i];
        tmp.data = d_ptr;
        tmp_images[i] = tmp;

        offset += bytes;
    }

    // copy structs to device
    HIP_ERRCHK(hipMemcpy(batch.d_images, tmp_images.data(), sizeof(image_t) * N, hipMemcpyHostToDevice));

    return hipSuccess;
}

hipError_t copy_back_batch(
    std::vector<image_t> &host_out,
    DeviceBatch &batch)
{
    size_t offset = 0;
    for (int i = 0; i < batch.N; ++i)
    {
        size_t bytes = size_t(host_out[i].width) * host_out[i].height * host_out[i].channels;
        HIP_ERRCHK(hipMemcpy(host_out[i].data, batch.d_images[i].data + offset, bytes, hipMemcpyDeviceToHost));
        offset += bytes;
    }
    return hipSuccess;
}

void free_batch(DeviceBatch &batch)
{
    if (batch.d_images)
        hipFree(batch.d_images);
    if (batch.d_pixels)
        hipFree(batch.d_pixels);
    batch = DeviceBatch{};
}

// ------------------------ generic templated GPU call ------------------------

template <typename Launcher>
hipError_t apply_filter_generic_templated(
    std::vector<image_t> &input_images,
    std::vector<image_t> &output_images,
    Launcher &&launch_kernel,
    dim3 blockSize,
    int device_id,
    size_t shared_bytes)
{
    DeviceBatch batch;
    HIP_ERRCHK(prepare_device_batch(input_images, output_images, batch, device_id));

    // compute grid based on first image
    int w = input_images[0].width;
    int h = input_images[0].height;

    dim3 grid = compute_grid(w, h, blockSize);
    grid.z = batch.N; // batch dimension

    launch_kernel(batch.d_images, batch.d_images, grid, blockSize, shared_bytes);

    HIP_ERRCHK(hipDeviceSynchronize());

    HIP_ERRCHK(copy_back_batch(output_images, batch));

    free_batch(batch);

    return hipSuccess;
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
        grayscale_cpu(input_image, output_image, width, height, channels);
        return hipSuccess;
    case FILTER_TYPE::NEGATIVE:
        negative_cpu(input_image, output_image, width, height, channels);
        return hipSuccess;
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
    hipFree(d_input.ptr);
    hipFree(d_output.ptr);
    hipFree(d_metas.ptr);

    return hipSuccess;
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
