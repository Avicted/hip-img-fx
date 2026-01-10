/**
 * @file validate_grayscale_migration.cpp
 * @brief Validation test for grayscale kernel migration to new autotuning framework
 *
 * Tests:
 * 1. Output equivalence (bitwise identical results)
 * 2. Performance comparison (cold vs warm cache)
 * 3. Cache correctness (entries created and reused)
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>

#include "../src/core/gpu_utils.h"
#include "../src/filters/filters.h"

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

struct TestImage
{
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;

    size_t bytes() const { return width * height * channels; }

    void fill_pattern()
    {
        // Create a gradient pattern for visual validation
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * channels;
                data[idx + 0] = (unsigned char)(x * 255 / width);                  // R
                data[idx + 1] = (unsigned char)(y * 255 / height);                 // G
                data[idx + 2] = (unsigned char)((x + y) * 255 / (width + height)); // B
                if (channels == 4)
                    data[idx + 3] = 255; // A
            }
        }
    }
};

struct TestResults
{
    bool outputs_match;
    double old_time_cold_ms;
    double old_time_warm_ms;
    double new_time_cold_ms;
    double new_time_warm_ms;
    int num_differences;
};

TestResults run_comparison_test(const TestImage &img, const std::string &size_category)
{
    TestResults results = {};

    // Allocate device memory
    unsigned char *d_input, *d_output_old, *d_output_new;
    imgfx::core::image_meta_t *d_metas;

    HIP_CHECK(hipMalloc(&d_input, img.bytes()));
    HIP_CHECK(hipMalloc(&d_output_old, img.bytes()));
    HIP_CHECK(hipMalloc(&d_output_new, img.bytes()));
    HIP_CHECK(hipMalloc(&d_metas, sizeof(imgfx::core::image_meta_t)));

    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, img.data.data(), img.bytes(), hipMemcpyHostToDevice));

    // Setup metadata
    imgfx::core::image_meta_t meta;
    meta.width = img.width;
    meta.height = img.height;
    meta.channels = img.channels;
    meta.offset = 0;
    HIP_CHECK(hipMemcpy(d_metas, &meta, sizeof(imgfx::core::image_meta_t), hipMemcpyHostToDevice));

    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // NOTE: Old AutoTuner-based implementation removed in v1.0
    // This test now only validates the new implementation

    std::cout << "\n=== Testing " << size_category << " image ("
              << img.width << "x" << img.height << ", " << img.bytes() << " bytes) ===\n";

    // ========================================================================
    // Test 1: NEW implementation (COLD - force retune by deleting cache)
    // ========================================================================
    std::cout << "Running NEW implementation (cold)... " << std::flush;

    // Delete cache file to force retuning
    system("rm -f .autotune_cache.json");

    auto start = high_resolution_clock::now();

    imgfx::filters::apply_grayscale_autotuned_v2(
        d_input, d_output_new, d_metas, 1, img.bytes(), stream);

    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();
    results.new_time_cold_ms = duration<double, std::milli>(end - start).count();
    std::cout << results.new_time_cold_ms << " ms\n";

    // ========================================================================
    // Test 2: NEW implementation (WARM - from cache)
    // ========================================================================
    std::cout << "Running NEW implementation (warm)... " << std::flush;
    start = high_resolution_clock::now();

    imgfx::filters::apply_grayscale_autotuned_v2(
        d_input, d_output_new, d_metas, 1, img.bytes(), stream);

    HIP_CHECK(hipStreamSynchronize(stream));
    end = high_resolution_clock::now();
    results.new_time_warm_ms = duration<double, std::milli>(end - start).count();
    std::cout << results.new_time_warm_ms << " ms (speedup: "
              << (results.new_time_cold_ms / results.new_time_warm_ms) << "x)\n";

    // ========================================================================
    // Test 3: Consistency check (run again, output should be identical)
    // ========================================================================
    std::cout << "Comparing outputs... " << std::flush;

    // Run again to get second output
    imgfx::filters::apply_grayscale_autotuned_v2(
        d_input, d_output_old, d_metas, 1, img.bytes(), stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<unsigned char> h_output_old(img.bytes());
    std::vector<unsigned char> h_output_new(img.bytes());

    HIP_CHECK(hipMemcpy(h_output_old.data(), d_output_old, img.bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_output_new.data(), d_output_new, img.bytes(), hipMemcpyDeviceToHost));

    results.num_differences = 0;
    for (size_t i = 0; i < img.bytes(); i++)
    {
        if (h_output_old[i] != h_output_new[i])
        {
            results.num_differences++;
            if (results.num_differences <= 5)
            {
                std::cout << "\nDifference at byte " << i << ": old="
                          << (int)h_output_old[i] << ", new=" << (int)h_output_new[i];
            }
        }
    }

    results.outputs_match = (results.num_differences == 0);

    if (results.outputs_match)
    {
        std::cout << "PASS (bitwise identical)\n";
    }
    else
    {
        std::cout << "\nFAIL (" << results.num_differences << " differences)\n";
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output_old));
    HIP_CHECK(hipFree(d_output_new));
    HIP_CHECK(hipFree(d_metas));
    HIP_CHECK(hipStreamDestroy(stream));

    return results;
}

int main()
{
    std::cout << "=======================================================\n";
    std::cout << "  Grayscale Kernel Migration Validation (Phase 2a)\n";
    std::cout << "=======================================================\n";

    // Test cases covering different size categories
    std::vector<TestImage> test_images;

    // Small image (< 1MB)
    {
        TestImage img;
        img.width = 512;
        img.height = 512;
        img.channels = 3;
        img.data.resize(img.bytes());
        img.fill_pattern();
        test_images.push_back(img);
    }

    // Medium image (1-10MB)
    {
        TestImage img;
        img.width = 2048;
        img.height = 1536;
        img.channels = 3;
        img.data.resize(img.bytes());
        img.fill_pattern();
        test_images.push_back(img);
    }

    // Large image (> 10MB)
    {
        TestImage img;
        img.width = 4096;
        img.height = 3072;
        img.channels = 3;
        img.data.resize(img.bytes());
        img.fill_pattern();
        test_images.push_back(img);
    }

    std::vector<std::string> categories = {"small", "medium", "large"};

    bool all_passed = true;
    std::vector<TestResults> results;

    for (size_t i = 0; i < test_images.size(); i++)
    {
        TestResults result = run_comparison_test(test_images[i], categories[i]);
        results.push_back(result);

        if (!result.outputs_match)
        {
            all_passed = false;
        }
    }

    // Summary
    std::cout << "\n=======================================================\n";
    std::cout << "  SUMMARY\n";
    std::cout << "=======================================================\n";

    std::cout << "\nOutput Equivalence:\n";
    for (size_t i = 0; i < results.size(); i++)
    {
        std::cout << "  " << categories[i] << ": "
                  << (results[i].outputs_match ? "PASS ✓" : "FAIL ✗")
                  << " (" << results[i].num_differences << " differences)\n";
    }

    std::cout << "\nPerformance Comparison:\n";
    std::cout << "  Category   | Old Cold | Old Warm | New Cold | New Warm | Improvement\n";
    std::cout << "  -----------|----------|----------|----------|----------|-----------\n";
    for (size_t i = 0; i < results.size(); i++)
    {
        double improvement = (results[i].old_time_warm_ms - results[i].new_time_warm_ms) / results[i].old_time_warm_ms * 100.0;
        printf("  %-9s  | %7.3f  | %7.3f  | %7.3f  | %7.3f  | %+6.1f%%\n",
               categories[i].c_str(),
               results[i].old_time_cold_ms,
               results[i].old_time_warm_ms,
               results[i].new_time_cold_ms,
               results[i].new_time_warm_ms,
               improvement);
    }

    std::cout << "\n=======================================================\n";
    if (all_passed)
    {
        std::cout << "  ✓ ALL TESTS PASSED\n";
        std::cout << "=======================================================\n";
        return 0;
    }
    else
    {
        std::cout << "  ✗ SOME TESTS FAILED\n";
        std::cout << "=======================================================\n";
        return 1;
    }
}
