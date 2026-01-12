/**
 * @file test_benchmarker_spec.cpp
 * @brief Specification tests for Benchmarker public API (benchmarker.h)
 *
 * Following SKILL.md guidelines:
 * - Encode contracts not implementation
 * - Negative tests required for every positive case
 * - Break-the-code validation
 *
 * Note: Benchmarker is a template class, so we test with actual kernel traits
 */

#include <gtest/gtest.h>
#include "hip-img-fx/autotune/benchmarker.h"
#include "hip-img-fx/autotune/tuning_config.h"
#include "hip-img-fx/autotune/types.h"
#include "../src/filters/filters.h"
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <hip/hip_runtime.h>

using namespace imgfx::core::autotune;
using imgfx::filters::GrayscaleKernelTraits;

// Helper function to initialize GPU memory for testing
static void setup_gpu_test_memory(unsigned char **d_input, unsigned char **d_output,
                                  imgfx::core::image_meta_t **d_metas, size_t size,
                                  int width, int height, int channels)
{
    hipError_t err = hipMalloc(d_input, size);
    if (err != hipSuccess)
        return;
    (void)hipMalloc(d_output, size);
    (void)hipMalloc(d_metas, sizeof(imgfx::core::image_meta_t));

    // Initialize metadata on host
    imgfx::core::image_meta_t meta;
    meta.offset = 0;
    meta.width = width;
    meta.height = height;
    meta.channels = channels;

    // Copy metadata to device
    (void)hipMemcpy(*d_metas, &meta, sizeof(imgfx::core::image_meta_t), hipMemcpyHostToDevice);
}

// ============================================================================
// TEST SUITE: BenchmarkResult Specification
// ============================================================================

TEST(BenchmarkResultSpec, DefaultConstructorCreatesInvalidResult)
{
    BenchmarkResult result;
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.avg_time_ms, 0.0f);
    EXPECT_EQ(result.stddev_ms, 0.0f);
    EXPECT_EQ(result.min_time_ms, 0.0f);
    EXPECT_EQ(result.max_time_ms, 0.0f);
}

TEST(BenchmarkResultSpec, ParameterizedConstructorStoresValues)
{
    TuningConfig config;
    config.set("block_x", 256);

    BenchmarkResult result(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.config.block_x(), 256);
    EXPECT_EQ(result.avg_time_ms, 1.5f);
    EXPECT_EQ(result.stddev_ms, 0.1f);
    EXPECT_EQ(result.min_time_ms, 1.4f);
    EXPECT_EQ(result.max_time_ms, 1.6f);
}

TEST(BenchmarkResultSpec, ComparisonOperatorComparesAverageTime)
{
    TuningConfig config1, config2;
    config1.set("block_x", 256);
    config2.set("block_x", 512);

    BenchmarkResult faster(config1, 1.0f, 0.1f, 0.9f, 1.1f);
    BenchmarkResult slower(config2, 2.0f, 0.1f, 1.9f, 2.1f);

    EXPECT_TRUE(faster < slower);
    EXPECT_FALSE(slower < faster);
}

TEST(BenchmarkResultSpec, ComparisonWithEqualTimesIsStable)
{
    TuningConfig config;
    BenchmarkResult result1(config, 1.5f, 0.1f, 1.4f, 1.6f);
    BenchmarkResult result2(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_FALSE(result1 < result2);
    EXPECT_FALSE(result2 < result1);
}

// ============================================================================
// TEST SUITE: Benchmarker Basic Operations
// ============================================================================

TEST(BenchmarkerSpec, BenchmarkWithInvalidConfigReturnsInvalidResult)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    TuningConfig invalid_config;
    invalid_config.set("block_x", 13); // Not wavefront aligned
    invalid_config.set("block_y", 1);

    GrayscaleKernelTraits::Args args{};
    TuningOptions options = TuningOptions::defaults();

    auto result = benchmarker.benchmark(invalid_config, args, 0, options);

    EXPECT_FALSE(result.valid);
}

TEST(BenchmarkerSpec, BenchmarkWithValidConfigReturnsValidResult)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    // Setup GPU memory
    int width = 1024, height = 1024, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    if (d_input == nullptr)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    TuningConfig valid_config;
    valid_config.set("block_x", 256);
    valid_config.set("block_y", 1);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 2;

    auto result = benchmarker.benchmark(valid_config, args, 0, options);

    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.avg_time_ms, 0.0f);
    EXPECT_GE(result.min_time_ms, 0.0f);
    EXPECT_GE(result.max_time_ms, result.min_time_ms);
    EXPECT_GE(result.stddev_ms, 0.0f);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

TEST(BenchmarkerSpec, BenchmarkComputesReasonableStatistics)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    // Setup GPU memory
    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    TuningConfig config;
    config.set("block_x", 256);
    config.set("block_y", 1);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 2;
    options.timing_runs = 5;

    auto result = benchmarker.benchmark(config, args, 0, options);

    ASSERT_TRUE(result.valid);

    // Average should be between min and max
    EXPECT_GE(result.avg_time_ms, result.min_time_ms);
    EXPECT_LE(result.avg_time_ms, result.max_time_ms);

    // Timing should be reasonable (< 1 second for 1MB image)
    EXPECT_LT(result.avg_time_ms, 1000.0f);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

// ============================================================================
// TEST SUITE: Benchmarker Warmup and Timing Runs
// ============================================================================

TEST(BenchmarkerSpec, ZeroWarmupRunsIsAllowed)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    TuningConfig config;
    config.set("block_x", 256);
    config.set("block_y", 1);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 0;
    options.timing_runs = 1;

    auto result = benchmarker.benchmark(config, args, 0, options);

    EXPECT_TRUE(result.valid);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

TEST(BenchmarkerSpec, SingleTimingRunProducesZeroStddev)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    TuningConfig config;
    config.set("block_x", 256);
    config.set("block_y", 1);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 1;

    auto result = benchmarker.benchmark(config, args, 0, options);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.stddev_ms, 0.0f);
    EXPECT_EQ(result.min_time_ms, result.max_time_ms);
    EXPECT_EQ(result.avg_time_ms, result.min_time_ms);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

// ============================================================================
// TEST SUITE: Benchmarker Multiple Candidates
// ============================================================================

TEST(BenchmarkerSpec, BenchmarkAllWithEmptyCandidatesReturnsEmpty)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;
    std::vector<TuningConfig> empty_candidates;

    GrayscaleKernelTraits::Args args{};
    TuningOptions options = TuningOptions::quiet();

    auto results = benchmarker.benchmark_all(empty_candidates, args, 0, options);

    EXPECT_TRUE(results.empty());
}

TEST(BenchmarkerSpec, BenchmarkAllSkipsInvalidConfigurations)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;

    // Mix of valid and invalid configs
    std::vector<TuningConfig> candidates;
    TuningConfig invalid;
    invalid.set("block_x", 13); // Invalid
    invalid.set("block_y", 1);
    candidates.push_back(invalid);

    TuningConfig valid;
    valid.set("block_x", 256); // Valid
    valid.set("block_y", 1);
    candidates.push_back(valid);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 1;

    auto results = benchmarker.benchmark_all(candidates, args, 0, options);

    // Only the valid config should produce a result
    EXPECT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].valid);
    EXPECT_EQ(results[0].config.block_x(), 256);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

TEST(BenchmarkerSpec, BenchmarkAllReturnsResultsInOrder)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;

    std::vector<TuningConfig> candidates;
    TuningConfig config1;
    config1.set("block_x", 256);
    config1.set("block_y", 1);
    candidates.push_back(config1);

    TuningConfig config2;
    config2.set("block_x", 512);
    config2.set("block_y", 1);
    candidates.push_back(config2);

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 1;
    options.enable_early_exit = false; // Disable early exit for this test

    auto results = benchmarker.benchmark_all(candidates, args, 0, options);

    EXPECT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].config.block_x(), 256);
    EXPECT_EQ(results[1].config.block_x(), 512);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

// ============================================================================
// TEST SUITE: Benchmarker Early Exit Logic
// ============================================================================

TEST(BenchmarkerSpec, EarlyExitDisabledTestsAllCandidates)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;

    // Create 5 valid candidates
    std::vector<TuningConfig> candidates;
    for (int block_x : {64, 128, 256, 512, 1024})
    {
        TuningConfig config;
        config.set("block_x", block_x);
        config.set("block_y", 1);
        candidates.push_back(config);
    }

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 1;
    options.enable_early_exit = false;

    auto results = benchmarker.benchmark_all(candidates, args, 0, options);

    // Should test all 5 candidates
    EXPECT_EQ(results.size(), 5u);

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

TEST(BenchmarkerSpec, EarlyExitEnabledMaySkipCandidates)
{
    if (imgfx::core::get_hip_devices() == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }

    int width = 100, height = 100, channels = 3;
    size_t size = width * height * channels;
    unsigned char *d_input;
    unsigned char *d_output;
    imgfx::core::image_meta_t *d_metas;

    setup_gpu_test_memory(&d_input, &d_output, &d_metas, size, width, height, channels);
    hipError_t err = (d_input == nullptr) ? hipErrorMemoryAllocation : hipSuccess;
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    Benchmarker<GrayscaleKernelTraits> benchmarker;

    // Create many candidates
    std::vector<TuningConfig> candidates;
    for (int block_x : {64, 128, 256, 512, 1024, 64, 128, 256, 512, 1024})
    {
        TuningConfig config;
        config.set("block_x", block_x);
        config.set("block_y", 1);
        candidates.push_back(config);
    }

    GrayscaleKernelTraits::Args args{};
    args.input = d_input;
    args.output = d_output;
    args.metas = d_metas;
    args.num_images = 1;
    args.max_image_bytes = size;

    TuningOptions options = TuningOptions::quiet();
    options.warmup_runs = 1;
    options.timing_runs = 1;
    options.enable_early_exit = true;
    options.early_exit_threshold = 1.5;    // Exit if 50% slower
    options.early_exit_min_coverage = 0.3; // Test at least 30%

    auto results = benchmarker.benchmark_all(candidates, args, 0, options);

    // May test fewer than all candidates (early exit can trigger)
    EXPECT_GE(results.size(), 3u); // At least minimum coverage
    EXPECT_LE(results.size(), candidates.size());

    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipFree(d_metas);
}

// ============================================================================
// TEST SUITE: Benchmarker Result Comparison
// ============================================================================

TEST(BenchmarkerSpec, ResultsCanBeSortedByAverageTime)
{
    TuningConfig config1, config2, config3;
    config1.set("block_x", 64);
    config2.set("block_x", 128);
    config3.set("block_x", 256);

    BenchmarkResult result1(config1, 2.0f, 0.1f, 1.9f, 2.1f);
    BenchmarkResult result2(config2, 1.0f, 0.1f, 0.9f, 1.1f);
    BenchmarkResult result3(config3, 3.0f, 0.1f, 2.9f, 3.1f);

    std::vector<BenchmarkResult> results = {result1, result2, result3};
    std::sort(results.begin(), results.end());

    EXPECT_EQ(results[0].avg_time_ms, 1.0f);
    EXPECT_EQ(results[1].avg_time_ms, 2.0f);
    EXPECT_EQ(results[2].avg_time_ms, 3.0f);
}

TEST(BenchmarkerSpec, MinElementFindsTestConfig)
{
    TuningConfig config1, config2, config3;
    config1.set("block_x", 64);
    config2.set("block_x", 128);
    config3.set("block_x", 256);

    BenchmarkResult result1(config1, 2.0f, 0.1f, 1.9f, 2.1f);
    BenchmarkResult result2(config2, 1.0f, 0.1f, 0.9f, 1.1f);
    BenchmarkResult result3(config3, 3.0f, 0.1f, 2.9f, 3.1f);

    std::vector<BenchmarkResult> results = {result1, result2, result3};
    auto best = std::min_element(results.begin(), results.end());

    EXPECT_EQ(best->config.block_x(), 128);
    EXPECT_EQ(best->avg_time_ms, 1.0f);
}

// ============================================================================
// TEST SUITE: Benchmarker Invariants
// ============================================================================

TEST(BenchmarkerSpec, MinTimeNeverGreaterThanMaxTime)
{
    TuningConfig config;
    config.set("block_x", 256);

    BenchmarkResult result(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_LE(result.min_time_ms, result.max_time_ms);
}

TEST(BenchmarkerSpec, AverageTimeWithinMinMaxRange)
{
    TuningConfig config;
    config.set("block_x", 256);

    BenchmarkResult result(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_GE(result.avg_time_ms, result.min_time_ms);
    EXPECT_LE(result.avg_time_ms, result.max_time_ms);
}

TEST(BenchmarkerSpec, StandardDeviationNeverNegative)
{
    TuningConfig config;
    config.set("block_x", 256);

    BenchmarkResult result(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_GE(result.stddev_ms, 0.0f);
}

TEST(BenchmarkerSpec, InvalidResultHasFalseValidFlag)
{
    BenchmarkResult result; // Default constructor

    EXPECT_FALSE(result.valid);
}

TEST(BenchmarkerSpec, ValidResultHasTrueValidFlag)
{
    TuningConfig config;
    config.set("block_x", 256);

    BenchmarkResult result(config, 1.5f, 0.1f, 1.4f, 1.6f);

    EXPECT_TRUE(result.valid);
}
