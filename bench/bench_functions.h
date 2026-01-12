/**
 * @file bench_functions.h
 * @brief Testable interface for benchmark runner functions
 *
 * Exposes internal functions from run_bench.cpp for unit testing.
 * This header should only be included in test files.
 */

#pragma once

#include <vector>

// Benchmark-specific image structure (local to benchmark, not core image_t)
struct bench_image_t
{
    unsigned char *data;
    int width;
    int height;
    int channels;
};

/**
 * @brief Generate a test image with gradient pattern
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param channels Number of color channels (3 for RGB, 4 for RGBA)
 * @return Generated image structure
 */
bench_image_t generate_test_image(int width, int height, int channels);

/**
 * @brief Free memory allocated for test image
 * @param img Image to free (data pointer will be set to nullptr)
 */
void free_test_image(bench_image_t &img);

/**
 * @brief Calculate sample standard deviation
 * @param values Vector of timing values
 * @param mean Pre-calculated mean of the values
 * @return Sample standard deviation (using n-1 denominator)
 */
double calculate_std_dev(const std::vector<double> &values, double mean);
