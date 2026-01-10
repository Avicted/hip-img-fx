/**
 * @file benchmark_grayscale_migration.cpp
 * @brief Performance benchmark for grayscale autotuned implementation
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>

#include "../src/core/gpu_utils.h"
#include "../src/filters/filters.h"

#define HIP_CHECK(call)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = call;                                          \
        if (err != hipSuccess)                                          \
        {                                                               \
            fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err)); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

using namespace std::chrono;

void benchmark_cached_performance(int width, int height, int channels, int iterations)
{
    size_t bytes = width * height * channels;

    // Allocate device memory
    unsigned char *d_input, *d_output;
    imgfx::core::image_meta_t *d_metas;

    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMalloc(&d_metas, sizeof(imgfx::core::image_meta_t)));

    // Setup metadata
    imgfx::core::image_meta_t meta;
    meta.width = width;
    meta.height = height;
    meta.channels = channels;
    meta.offset = 0;
    HIP_CHECK(hipMemcpy(d_metas, &meta, sizeof(meta), hipMemcpyHostToDevice));

    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warm up
    imgfx::filters::apply_grayscale_autotuned_v2(
        d_input, d_output, d_metas, 1, bytes, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark implementation
    std::vector<double> times;
    for (int i = 0; i < iterations; i++)
    {
        auto start = high_resolution_clock::now();
        imgfx::filters::apply_grayscale_autotuned_v2(
            d_input, d_output, d_metas, 1, bytes, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

    // Calculate statistics
    double sum = 0, min = times[0], max = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min)
            min = t;
        if (t > max)
            max = t;
    }
    double avg = sum / times.size();

    double var_sum = 0;
    for (double t : times)
    {
        var_sum += (t - avg) * (t - avg);
    }
    double stddev = sqrt(var_sum / times.size());

    std::cout << "\n"
              << width << "x" << height << " (" << bytes << " bytes)\n";
    std::cout << "  Time: " << avg << " ± " << stddev << " ms "
              << "(min: " << min << ", max: " << max << ")\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_metas));
    HIP_CHECK(hipStreamDestroy(stream));
}

int main()
{
    std::cout << "================================================\n";
    std::cout << "  Grayscale Autotuned Performance Benchmark\n";
    std::cout << "================================================\n";

    const int iterations = 100;

    std::cout << "\nRunning " << iterations << " iterations per configuration...\n";

    // Delete cache to ensure we test fresh tuning
    system("rm -f .autotune_cache.json");

    // Test various sizes
    benchmark_cached_performance(512, 512, 3, iterations);
    benchmark_cached_performance(1024, 768, 3, iterations);
    benchmark_cached_performance(2048, 1536, 3, iterations);
    benchmark_cached_performance(4096, 3072, 3, iterations);

    std::cout << "\n================================================\n";

    return 0;
}
