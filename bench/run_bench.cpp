/*
 * Production-Grade GPU Benchmark Harness
 *
 * Benchmarks CPU (single-threaded, OpenMP) vs GPU performance
 * Sweeps image resolutions for single-image processing
 * Outputs CSV for analysis
 *
 * Compile: Part of meson build system
 * Usage: ./build/hip-img-fx-bench [--warmup N] [--iterations N] [--output results.csv]
 */

#include "../src/core/gpu_utils.h"
#include "../src/core/image.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace imgfx::core;
using namespace std::chrono;

struct BenchConfig
{
    int warmup_iterations = 3;
    int bench_iterations = 10;
    std::string output_csv = "bench/results/benchmark_results.csv";
    bool verbose = false;
};

struct BenchResult
{
    std::string filter_name;
    int resolution;
    int batch_size;
    int channels;

    // CPU timings (single-threaded)
    double cpu_single_ms_avg = 0.0;
    double cpu_single_ms_std = 0.0;

    // CPU timings (OpenMP)
    double cpu_omp_ms_avg = 0.0;
    double cpu_omp_ms_std = 0.0;

    // GPU timings
    double gpu_h2d_ms_avg = 0.0;
    double gpu_kernel_ms_avg = 0.0;
    double gpu_d2h_ms_avg = 0.0;
    double gpu_total_ms_avg = 0.0;
    double gpu_total_ms_std = 0.0;

    // Performance metrics
    double speedup_vs_single = 0.0;
    double speedup_vs_omp = 0.0;
    double bandwidth_gb_s = 0.0;
};

image_t generate_test_image(int width, int height, int channels)
{
    image_t img;
    img.width = width;
    img.height = height;
    img.channels = channels;

    size_t total_bytes = width * height * channels;
    img.data = (unsigned char *)malloc(total_bytes);

    if (!img.data)
    {
        fprintf(stderr, "Failed to allocate test image\n");
        return img;
    }

    // Fill with gradient pattern (more realistic than zeros)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * channels;
            img.data[idx + 0] = (x * 255) / width;                  // R gradient
            img.data[idx + 1] = (y * 255) / height;                 // G gradient
            img.data[idx + 2] = ((x + y) * 255) / (width + height); // B gradient
            if (channels == 4)
            {
                img.data[idx + 3] = 255; // Alpha
            }
        }
    }

    return img;
}

static void free_test_image(image_t &img)
{
    if (img.data)
    {
        free(img.data);
        img.data = nullptr;
    }
}

double calculate_std_dev(const std::vector<double> &values, double mean)
{
    if (values.size() <= 1)
    {
        return 0.0;
    }

    double sum_sq_diff = 0.0;
    for (double val : values)
    {
        double diff = val - mean;
        sum_sq_diff += diff * diff;
    }

    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

double bench_cpu_single(FILTER_TYPE filter, image_t &input, image_t &output, int iterations)
{
    // Disable OpenMP for single-threaded test
    int original_threads = omp_get_max_threads();
    omp_set_num_threads(1);

    std::vector<double> times;

    for (int i = 0; i < iterations; i++)
    {
        auto start = high_resolution_clock::now();

        (void)apply_filter_cpu(filter, input.data, output.data, input.width, input.height, input.channels);

        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(ms);
    }

    // Restore OpenMP threads
    omp_set_num_threads(original_threads);

    double sum = 0.0;
    for (double t : times)
    {
        sum += t;
    }
    return sum / times.size();
}

double bench_cpu_omp(FILTER_TYPE filter, image_t &input, image_t &output, int iterations)
{
    std::vector<double> times;

    for (int i = 0; i < iterations; i++)
    {
        auto start = high_resolution_clock::now();

        (void)apply_filter_cpu(filter, input.data, output.data, input.width, input.height, input.channels);

        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(ms);
    }

    double sum = 0.0;
    for (double t : times)
    {
        sum += t;
    }
    return sum / times.size();
}

void run_benchmark_suite(const BenchConfig &config)
{
    printf("====================================================================\n");
    printf("   HIP Image FX - Production GPU Benchmark Suite\n");
    printf("====================================================================\n");
    printf("Configuration:\n");
    printf("  Warmup iterations: %d\n", config.warmup_iterations);
    printf("  Benchmark iterations: %d\n", config.bench_iterations);
    printf("  Output CSV: %s\n", config.output_csv.c_str());
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
    printf("====================================================================\n\n");

    // Test configurations
    std::vector<int> resolutions = {512, 1024, 2048, 4096};
    std::vector<int> batch_sizes = {1, 8, 16, 32, 64};
    std::vector<std::pair<FILTER_TYPE, std::string>> filters = {
        {FILTER_TYPE::GRAYSCALE, "grayscale"},
        {FILTER_TYPE::NEGATIVE, "negative"},
        {FILTER_TYPE::GAUSSIAN_BLUR, "gaussian_blur"}};

    int channels = 3; // RGB

    std::vector<BenchResult> all_results;

    printf("Detecting GPU...\n");
    get_hip_devices();
    printf("\n");

    int test_num = 0;
    int total_tests = resolutions.size() * batch_sizes.size() * filters.size();

    for (int batch_size : batch_sizes)
    {
        printf("============================================================\n");
        printf("Testing Batch Size: %d\n", batch_size);
        printf("============================================================\n");

        for (int res : resolutions)
        {
            printf("------------------------------------------------------------\n");
            printf("Testing: %dx%d resolution\n", res, res);
            printf("------------------------------------------------------------\n");

            for (auto &[filter_type, filter_name] : filters)
            {
                test_num++;
                printf("\n[%d/%d] Filter: %s\n", test_num, total_tests, filter_name.c_str());

                // CPU baselines are measured per-image on a single synthetic image.
                // GPU is measured as a batched call (batch_size images) and then converted
                // to per-image time by dividing by batch_size.
                image_t cpu_input = generate_test_image(res, res, channels);
                image_t cpu_output = generate_test_image(res, res, channels);

                std::vector<image_t> gpu_inputs;
                std::vector<image_t> gpu_outputs;
                gpu_inputs.reserve(batch_size);
                gpu_outputs.reserve(batch_size);
                for (int i = 0; i < batch_size; ++i)
                {
                    gpu_inputs.push_back(generate_test_image(res, res, channels));
                    gpu_outputs.push_back(generate_test_image(res, res, channels));
                }

                if (config.verbose)
                {
                    printf("  Warming up GPU...\n");
                }

                for (int i = 0; i < config.warmup_iterations; i++)
                {
                    hipError_t err = apply_filter_gpu(filter_type, gpu_inputs, gpu_outputs, false, nullptr);
                    if (err != hipSuccess)
                    {
                        fprintf(stderr, "GPU warmup failed\n");
                        break;
                    }
                }

                if (config.verbose)
                {
                    printf("  Benchmarking GPU...\n");
                }

                BenchResult result;
                result.filter_name = filter_name;
                result.resolution = res;
                result.batch_size = batch_size;
                result.channels = channels;

                std::vector<double> total_times, h2d_times, kernel_times, d2h_times;

                for (int i = 0; i < config.bench_iterations; i++)
                {
                    GPUTimings timings;
                    hipError_t err = apply_filter_gpu(filter_type, gpu_inputs, gpu_outputs, true, &timings);
                    if (err != hipSuccess)
                    {
                        fprintf(stderr, "GPU benchmark failed\n");
                        continue;
                    }

                    // Record per-image times (timings are for the full batch)
                    const double denom = double(batch_size);
                    total_times.push_back(timings.total_ms / denom);
                    h2d_times.push_back(timings.h2d_ms / denom);
                    kernel_times.push_back(timings.kernel_ms / denom);
                    d2h_times.push_back(timings.d2h_ms / denom);
                }

                for (double t : total_times)
                {
                    result.gpu_total_ms_avg += t;
                }
                result.gpu_total_ms_avg /= total_times.size();

                for (double t : h2d_times)
                {
                    result.gpu_h2d_ms_avg += t;
                }
                result.gpu_h2d_ms_avg /= h2d_times.size();

                for (double t : kernel_times)
                {
                    result.gpu_kernel_ms_avg += t;
                }
                result.gpu_kernel_ms_avg /= kernel_times.size();

                for (double t : d2h_times)
                {
                    result.gpu_d2h_ms_avg += t;
                }
                result.gpu_d2h_ms_avg /= d2h_times.size();

                result.gpu_total_ms_std = calculate_std_dev(total_times, result.gpu_total_ms_avg);

                size_t bytes_per_image = res * res * channels * sizeof(unsigned char);
                size_t total_bytes = bytes_per_image * 2; // read + write
                result.bandwidth_gb_s = (total_bytes / 1e9) / (result.gpu_total_ms_avg / 1000.0);

                if (config.verbose)
                {
                    printf("  Benchmarking CPU single-threaded...\n");
                }

                result.cpu_single_ms_avg = bench_cpu_single(filter_type, cpu_input, cpu_output,
                                                            config.bench_iterations);

                if (config.verbose)
                {
                    printf("  Benchmarking CPU OpenMP...\n");
                }

                result.cpu_omp_ms_avg = bench_cpu_omp(filter_type, cpu_input, cpu_output,
                                                      config.bench_iterations);

                result.speedup_vs_single = result.cpu_single_ms_avg / result.gpu_total_ms_avg;
                result.speedup_vs_omp = result.cpu_omp_ms_avg / result.gpu_total_ms_avg;

                printf("  Results:\n");
                printf("    CPU (single):  %8.2f ms\n", result.cpu_single_ms_avg);
                printf("    CPU (OpenMP):  %8.2f ms\n", result.cpu_omp_ms_avg);
                printf("    GPU H2D:       %8.2f ms\n", result.gpu_h2d_ms_avg);
                printf("    GPU Kernel:    %8.2f ms\n", result.gpu_kernel_ms_avg);
                printf("    GPU D2H:       %8.2f ms\n", result.gpu_d2h_ms_avg);
                printf("    GPU Total:     %8.2f ms (±%.2f)\n", result.gpu_total_ms_avg, result.gpu_total_ms_std);
                printf("    Speedup vs Single: %.2fx\n", result.speedup_vs_single);
                printf("    Speedup vs OpenMP: %.2fx\n", result.speedup_vs_omp);
                printf("    Bandwidth:     %8.2f GB/s\n", result.bandwidth_gb_s);

                all_results.push_back(result);

                free_test_image(cpu_input);
                free_test_image(cpu_output);
                for (auto &img : gpu_inputs)
                {
                    free_test_image(img);
                }
                for (auto &img : gpu_outputs)
                {
                    free_test_image(img);
                }
            }
        }
    } // end batch_size loop

    printf("\n====================================================================\n");
    printf("Writing results to: %s\n", config.output_csv.c_str());

    std::ofstream csv(config.output_csv);
    if (!csv.is_open())
    {
        fprintf(stderr, "ERROR: Could not open output file: %s\n", config.output_csv.c_str());
        return;
    }

    // CSV header
    csv << "filter,resolution,batch_size,channels,"
        << "cpu_single_ms,cpu_omp_ms,"
        << "gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_total_ms,gpu_std_ms,"
        << "speedup_vs_single,speedup_vs_omp,bandwidth_gb_s\n";

    // CSV data
    for (const auto &r : all_results)
    {
        csv << std::fixed << std::setprecision(3)
            << r.filter_name << ","
            << r.resolution << ","
            << r.batch_size << ","
            << r.channels << ","
            << r.cpu_single_ms_avg << ","
            << r.cpu_omp_ms_avg << ","
            << r.gpu_h2d_ms_avg << ","
            << r.gpu_kernel_ms_avg << ","
            << r.gpu_d2h_ms_avg << ","
            << r.gpu_total_ms_avg << ","
            << r.gpu_total_ms_std << ","
            << r.speedup_vs_single << ","
            << r.speedup_vs_omp << ","
            << r.bandwidth_gb_s << "\n";
    }

    csv.close();
    printf("Benchmark complete!\n");
    printf("====================================================================\n");
}

void print_usage(const char *prog_name)
{
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --warmup N        Number of warmup iterations (default: 3)\n");
    printf("  --iterations N    Number of benchmark iterations (default: 10)\n");
    printf("  --output FILE     Output CSV file (default: bench/results/benchmark_results.csv)\n");
    printf("  --verbose         Enable verbose output\n");
    printf("  --help            Show this help message\n");
}

int main(int argc, char **argv)
{
    BenchConfig config;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            config.warmup_iterations = std::atoi(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            config.bench_iterations = std::atoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            config.output_csv = argv[++i];
        }
        else if (arg == "--verbose")
        {
            config.verbose = true;
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    run_benchmark_suite(config);

    return 0;
}
