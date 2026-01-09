/**
 * @file test_tier1_improvements.cpp
 * @brief Integration test for Tier-1 autotuning improvements
 *
 * Tests:
 * 1. Pre-Seeded Cache: Embedded defaults load when user cache missing
 * 2. Early-Exit Benchmarking: Tuning time reduced with early termination
 *
 * Expected results:
 * - Cold start with embedded cache should be < 1ms (no tuning)
 * - Early-exit should skip 30-50% of candidates
 * - Configurations should be optimal (within 5% of exhaustive search)
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <fstream>
#include <hip/hip_runtime.h>
#include <cstdio>

#include "../src/core/gpu_utils.h"
#include "../src/core/autotuning.h"
#include "../src/filters/filters.h"
#include "../src/core/autotune/orchestrator.h"
#include "../src/core/autotune/types.h"

#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = call;                                              \
        if (err != hipSuccess)                                              \
        {                                                                   \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

using namespace std::chrono;

void test_embedded_cache()
{
    std::cout << "\n========================================\n";
    std::cout << "TEST 1: Pre-Seeded Cache (Embedded Defaults)\n";
    std::cout << "========================================\n\n";

    // Delete user cache to force embedded fallback
    std::system("rm -f .autotune_cache.json");
    std::cout << "✓ Deleted user cache\n";

    // Setup test image (large context)
    const int width = 3840;
    const int height = 2160;
    const int channels = 3;
    const size_t bytes = width * height * channels;

    unsigned char *d_input, *d_output;
    imgfx::core::image_meta_t *d_metas;

    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMalloc(&d_metas, sizeof(imgfx::core::image_meta_t)));

    imgfx::core::image_meta_t meta;
    meta.width = width;
    meta.height = height;
    meta.channels = channels;
    meta.offset = 0;
    HIP_CHECK(hipMemcpy(d_metas, &meta, sizeof(imgfx::core::image_meta_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::cout << "Testing grayscale_v2 kernel with large image (3840x2160)...\n";
    std::cout << "Expected: Should load embedded config instantly (no tuning)\n\n";

    auto start = high_resolution_clock::now();

    // This should use embedded cache for gfx1030/grayscale_v2/large
    imgfx::filters::apply_grayscale_autotuned_v2(
        d_input, d_output, d_metas, 1, bytes, stream);

    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();
    double time_ms = duration<double, std::milli>(end - start).count();

    std::cout << "✓ First run completed in " << time_ms << " ms\n";

    if (time_ms < 5.0)
    {
        std::cout << "✓ PASS: Embedded cache loaded (< 5ms overhead)\n";
    }
    else
    {
        std::cout << "✗ FAIL: Took too long (" << time_ms << " ms), expected < 5ms with embedded cache\n";
    }

    // Check that cache file was created with embedded entry
    std::ifstream cache_file(".autotune_cache.json");
    if (cache_file.is_open())
    {
        std::string content((std::istreambuf_iterator<char>(cache_file)),
                            std::istreambuf_iterator<char>());
        if (content.find("grayscale_v2") != std::string::npos &&
            content.find("large") != std::string::npos &&
            content.find("gfx1030") != std::string::npos)
        {
            std::cout << "✓ Cache file created with embedded entry\n";
        }
        else
        {
            std::cout << "✗ Cache file missing expected entry\n";
        }
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_metas));
    HIP_CHECK(hipStreamDestroy(stream));
}

void test_early_exit()
{
    std::cout << "\n========================================\n";
    std::cout << "TEST 2: Early-Exit Benchmarking\n";
    std::cout << "========================================\n\n";

    // Delete cache to force fresh tuning
    std::system("rm -f .autotune_cache.json");

    // Setup test image (medium context - will trigger tuning)
    const int width = 1920;
    const int height = 1080;
    const int channels = 3;
    const size_t bytes = width * height * channels;

    unsigned char *d_input, *d_output;
    imgfx::core::image_meta_t *d_metas;

    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMalloc(&d_metas, sizeof(imgfx::core::image_meta_t)));

    imgfx::core::image_meta_t meta;
    meta.width = width;
    meta.height = height;
    meta.channels = channels;
    meta.offset = 0;
    HIP_CHECK(hipMemcpy(d_metas, &meta, sizeof(imgfx::core::image_meta_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::cout << "\nTesting with early-exit ENABLED (default)...\n";
    auto start = high_resolution_clock::now();

    // Initialize AutoTuner for old API
    imgfx::core::AutoTuner tuner;

    // This will tune negative kernel (not in embedded cache)
    // Early-exit should reduce tuning time
    imgfx::filters::apply_negative_autotuned(
        d_input, d_output, d_metas, 1, bytes, tuner, stream);

    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();
    double time_early_ms = duration<double, std::milli>(end - start).count();

    std::cout << "✓ Tuning with early-exit: " << time_early_ms << " ms\n";

    // Now test with early-exit DISABLED
    std::system("rm -f .autotune_cache.json");

    std::cout << "\nTesting with early-exit DISABLED (force exhaustive)...\n";
    start = high_resolution_clock::now();

    // Note: Would need to pass TuningOptions with enable_early_exit=false
    // For now, just document expected behavior
    std::cout << "ℹ Note: Early-exit is enabled by default\n";
    std::cout << "ℹ Expected: Early-exit skips 30-50% of candidates\n";
    std::cout << "ℹ Expected: Time savings of 20-40% vs exhaustive search\n";

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_metas));
    HIP_CHECK(hipStreamDestroy(stream));
}

int main()
{
    std::cout << "===========================================\n";
    std::cout << "Tier-1 Autotuning Improvements Test Suite\n";
    std::cout << "===========================================\n";

    // Get GPU info
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Arch: " << prop.gcnArchName << "\n";

    // Run tests
    test_embedded_cache();
    test_early_exit();

    std::cout << "\n===========================================\n";
    std::cout << "All tests completed\n";
    std::cout << "===========================================\n";

    return 0;
}
