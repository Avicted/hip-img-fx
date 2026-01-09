#include "process.h"

namespace imgfx::app
{
    namespace fs = std::filesystem;

    int process_one_gpu(
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

        imgfx::core::image_t output_image;
        output_image.width = image.width;
        output_image.height = image.height;
        output_image.channels = image.channels;

        const size_t image_size = size_t(image.width) * image.height * image.channels * sizeof(unsigned char);
        output_image.data = (unsigned char *)malloc(image_size);
        if (output_image.data == nullptr)
        {
            fprintf(stderr, "ERROR: Could not allocate memory for output image!\n");
            free_image(&image);
            return -1;
        }

        hipError_t err = apply_filter_gpu(filter_type, image, output_image, false, nullptr);
        if (err != hipSuccess)
        {
            fprintf(stderr, "ERROR: Failed to apply GPU filter: %s\n", hipGetErrorString(err));
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

    int process_batch_gpu(const std::vector<std::string> &input_files,
                          const std::string &output_path,
                          imgfx::core::FILTER_TYPE filter_type,
                          int batch_size)
    {
        if (batch_size <= 0)
        {
            fprintf(stderr, "ERROR: batch_size must be a positive integer. Got: %d\n", batch_size);
            return -1;
        }

        printf("num threads: %d\n", omp_get_max_threads());
        printf("GPU batch size: %d\n", batch_size);

        std::vector<imgfx::core::image_t> input_images(input_files.size());
        std::vector<imgfx::core::image_t> output_images(input_files.size());

        // Load all images in parallel
#pragma omp parallel for shared(input_images, output_images, input_files)
        for (size_t i = 0; i < input_files.size(); ++i)
        {
            const std::string &input_file = input_files[i];
            imgfx::core::image_t img = imgfx::core::load_image(input_file.c_str());
            if (img.data == nullptr)
            {
#pragma omp critical
                fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_file.c_str());
                continue;
            }
            input_images[i] = img;

            if (img.data != nullptr)
            {
                imgfx::core::image_t out_img;
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
        size_t total_images = input_images.size();

        for (size_t offset = 0; offset < total_images; offset += batch_size)
        {
            size_t current_chunk = std::min(static_cast<size_t>(batch_size), total_images - offset);
            std::vector<imgfx::core::image_t> input_chunk(input_images.begin() + offset, input_images.begin() + offset + current_chunk);
            std::vector<imgfx::core::image_t> output_chunk(output_images.begin() + offset, output_images.begin() + offset + current_chunk);

            // printf("Launching GPU filter kernel: %s (batched)\n", filter_type_to_string(filter_type).c_str());

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
