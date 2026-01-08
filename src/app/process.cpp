#include "process.h"

namespace imgfx::app
{
    namespace fs = std::filesystem;

    int process_batch_gpu(const std::vector<std::string> &input_files,
                          const std::string &output_path,
                          imgfx::core::FILTER_TYPE filter_type)
    {
        printf("num threads: %d\n", omp_get_max_threads());
        printf("Processing %zu images individually on GPU\n", input_files.size());

        // Process each image one at a time for optimal performance
        for (size_t i = 0; i < input_files.size(); ++i)
        {
            const std::string &input_file = input_files[i];

            // Load image
            imgfx::core::image_t input_image = imgfx::core::load_image(input_file.c_str());
            if (input_image.data == nullptr)
            {
                fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_file.c_str());
                continue;
            }

            // Allocate output image
            imgfx::core::image_t output_image;
            output_image.width = input_image.width;
            output_image.height = input_image.height;
            output_image.channels = input_image.channels;
            size_t bytes = size_t(input_image.width) * input_image.height * input_image.channels;
            output_image.data = (unsigned char *)malloc(bytes);

            if (output_image.data == nullptr)
            {
                fprintf(stderr, "ERROR: Could not allocate memory for output image: %s\n", input_file.c_str());
                free_image(&input_image);
                continue;
            }

            // Apply filter on GPU
            hipError_t err = apply_filter_gpu(filter_type, input_image, output_image, false, nullptr);

            if (err != hipSuccess)
            {
                fprintf(stderr, "ERROR: Failed to apply GPU filter on image: %s\n", input_file.c_str());
                free_image(&input_image);
                free_image(&output_image);
                continue;
            }

            // Save output image
            fs::path in_path(input_file);
            fs::path out_file = fs::path(output_path) / in_path.filename();

            if (!save_image(out_file.string().c_str(), &output_image))
            {
                fprintf(stderr, "ERROR: Could not save output image: %s\n", out_file.string().c_str());
            }

            // Cleanup
            free_image(&input_image);
            free_image(&output_image);
        }

        printf("Batch processing complete: %zu images processed.\n", input_files.size());

        return 0;
    }

    int process_one_cpu(
        bool running_as_batch,
        const std::string &input_path,
        const std::string &output_path,
        imgfx::core::FILTER_TYPE filter_type)
    {
        imgfx::core::image_t image = imgfx::core::load_image(input_path.c_str());
        if (image.data == nullptr)
        {
            fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_path.c_str());
            return -1;
        }

        // print_image_info(&image);

        imgfx::core::image_t output_image;
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

        if (!running_as_batch)
        {
            printf("Successfully saved output image: %s\n", output_path.c_str());
            printf("------------------------------------------------------------\n");
        }

        free_image(&image);
        free_image(&output_image);
        return 0;
    }

    int process_batch_cpu(const std::vector<std::string> &input_files,
                          const std::string &output_path,
                          imgfx::core::FILTER_TYPE filter_type)
    {
        printf("num threads: %d\n", omp_get_max_threads());

#pragma omp parallel for
        for (size_t i = 0; i < input_files.size(); ++i)
        {
            const fs::path in_path(input_files[i]);
            const fs::path out_file = fs::path(output_path) / in_path.filename();
            const bool running_as_batch = true;

            if (process_one_cpu(running_as_batch, input_files[i].c_str(), out_file.string(), filter_type) != 0)
            {
            }
        }

        printf("Batch processing complete: %zu images processed.\n", input_files.size());

        return 0;
    }
}
