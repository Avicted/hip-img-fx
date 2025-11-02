#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <omp.h>

#include "gpu_utils.h"
#include "cli_parser.h"
#include "image.h"

namespace fs = std::filesystem;

// Supported image extensions (lowercase)
const std::vector<std::string> supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};

bool has_supported_ext(const fs::path &p)
{
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(supported_exts.begin(), supported_exts.end(), ext) != supported_exts.end();
}

int process_batch(const std::vector<std::string> &input_files,
                  const std::string &output_path,
                  FILTER_TYPE filter_type)
{
    printf("omp in parallel: %d\n", omp_in_parallel());
    printf("num threads: %d\n", omp_get_max_threads());

    std::vector<image_t> input_images(input_files.size());
    std::vector<image_t> output_images(input_files.size());

    // Load all images in parallel
#pragma omp parallel for shared(input_images, output_images, input_files)
    for (size_t i = 0; i < input_files.size(); ++i)
    {
        const std::string &input_file = input_files[i];
        image_t img = load_image(input_file.c_str());
        if (img.data == nullptr)
        {
#pragma omp critical
            {
                fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_file.c_str());
            }
        }
        input_images[i] = img;

        if (img.data != nullptr)
        {
            image_t out_img;
            out_img.width = img.width;
            out_img.height = img.height;
            out_img.channels = img.channels;
            size_t bytes = size_t(img.width) * img.height * img.channels;
            out_img.data = (unsigned char *)malloc(bytes);
            if (out_img.data == nullptr)
            {
#pragma omp critical
                {
                    fprintf(stderr, "ERROR: Could not allocate memory for output image: %s\n", input_file.c_str());
                }
            }
            output_images[i] = out_img;
        }
    }

    printf("Loaded %zu images for batch processing.\n", input_images.size());

    // Launch GPU kernel in chunks
    const size_t chunk_size = 64; // tune based on GPU memory
    size_t total_images = input_images.size();

    for (size_t offset = 0; offset < total_images; offset += chunk_size)
    {
        size_t current_chunk = std::min(chunk_size, total_images - offset);
        std::vector<image_t> input_chunk(input_images.begin() + offset, input_images.begin() + offset + current_chunk);
        std::vector<image_t> output_chunk(output_images.begin() + offset, output_images.begin() + offset + current_chunk);

        printf("Launching GPU filter kernel: %s (images %zu to %zu)\n",
               filter_type_to_string(filter_type).c_str(), offset, offset + current_chunk - 1);

        hipError_t err = apply_filter_gpu(filter_type, input_chunk, output_chunk);
        if (err != hipSuccess)
        {
            fprintf(stderr, "ERROR: Failed to apply GPU filter on chunk starting at image %zu\n", offset);
            return -1;
        }
    }

    // Save all output images in parallel
#pragma omp parallel for shared(input_images, output_images, input_files, output_path)
    for (size_t i = 0; i < input_files.size(); ++i)
    {
        fs::path in_path(input_files[i]);
        fs::path out_file = fs::path(output_path) / in_path.filename();

        if (!save_image(out_file.string().c_str(), &output_images[i]))
        {
#pragma omp critical
            {
                fprintf(stderr, "ERROR: Could not save output image: %s\n", out_file.string().c_str());
            }
        }

        free_image(&input_images[i]);
        free_image(&output_images[i]);
    }

    printf("Batch processing complete: %zu images processed.\n", input_images.size());

    return 0;
}

int process_one(bool use_cpu, const std::string &input_path, const std::string &output_path, FILTER_TYPE filter_type)
{
    image_t image = load_image(input_path.c_str());
    if (image.data == nullptr)
    {
        fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_path.c_str());
        return -1;
    }

    // print_image_info(&image);

    image_t output_image;
    output_image.width = image.width;
    output_image.height = image.height;
    output_image.channels = image.channels;
    size_t image_size = image.width * image.height * image.channels * sizeof(unsigned char);
    output_image.data = (unsigned char *)malloc(image_size);
    if (output_image.data == nullptr)
    {
        fprintf(stderr, "ERROR: Could not allocate memory for output image!\n");
        free_image(&image);
        return -1;
    }

    if (apply_filter_cpu(
            filter_type,
            image.data,
            output_image.data,
            image.width,
            image.height,
            image.channels) != 0)
    {
        fprintf(stderr, "ERROR: Failed to apply filter!\n");
        free_image(&image);
        free_image(&output_image);
        return -1;
    }

    if (!save_image(output_path.c_str(), &output_image))
    {
        fprintf(stderr, "ERROR: Could not save output image: %s\n", output_path.c_str());
        free_image(&image);
        free_image(&output_image);
        return -1;
    }

    printf("Successfully saved output image: %s\n", output_path.c_str());
    printf("------------------------------------------------------------\n");

    free_image(&image);
    free_image(&output_image);
    return 0;
}

int main(int argc, char **argv)
{
    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();

    printf("Running HIP image fx...\n");
    bool use_cpu = false;

    cli_args_t args = parse_cli_args(argc, argv);
    printf("Input: %s\n", args.input_file);
    printf("Output: %s\n", args.output_file);
    printf("Filter Type: %s\n", filter_type_to_string(args.filter_type).c_str());
    use_cpu = args.use_cpu;
    printf("Using %s for processing.\n", (use_cpu ? "CPU" : "GPU"));

    if (!use_cpu)
    {
        int hip_device_count = get_hip_devices();
        if (hip_device_count < 1)
        {
            fprintf(stderr, "ERROR: Could not find any HIP device!\n");
            printf("Falling back to CPU processing...\n");
            use_cpu = true;
        }
    }

    fs::path input_path(args.input_file);
    fs::path output_path(args.output_file);

    bool input_is_dir = fs::is_directory(input_path);
    bool output_is_dir = fs::is_directory(output_path);

    int ret = 0;
    if (input_is_dir && output_is_dir)
    {
        // Batch mode
        std::vector<std::string> input_files = {};
        int processed = 0;
        int failed = 0;

        for (const auto &entry : fs::directory_iterator(input_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (!has_supported_ext(entry.path()))
            {
                continue;
            }

            input_files.push_back(entry.path().string());

            // fs::path out_file = output_path / entry.path().filename();
            // printf("\nProcessing: %s -> %s\n", entry.path().string().c_str(), out_file.string().c_str());
            // int res = process_one(use_cpu, entry.path().string(), out_file.string(), args.filter_type);
            // int res = process_batch(...);
            // if (res == 0)
            // {
            //     processed++;
            // }
            // else
            // {
            //     failed++;
            // }
        }

        for (const std::string &file_name : input_files)
        {
            // printf("\nProcessing: %s\n", file_name.c_str());
        }

        int res = process_batch(input_files, output_path.string(), args.filter_type);

        printf("\nBatch processing complete. Success: %d, Failed: %d\n", processed, failed);
        ret = (failed == 0 ? 0 : 1);
    }
    else if (!input_is_dir && !output_is_dir)
    {
        // Single file mode
        ret = process_one(use_cpu, args.input_file, args.output_file, args.filter_type);
    }
    else
    {
        fprintf(stderr, "ERROR: Both --input and --output must be either files or directories.\n");
        ret = -1;
    }

    auto end_time = clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    int minutes = static_cast<int>(elapsed / 60000);
    int seconds = static_cast<int>((elapsed % 60000) / 1000);
    int millis = static_cast<int>(elapsed % 1000);
    printf("Total processing time: %02dm %02ds %03dms\n", minutes, seconds, millis);

    return ret;
}
