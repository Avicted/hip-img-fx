#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

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

int process_one(const std::string &input_path, const std::string &output_path, FILTER_TYPE filter_type)
{
    image_t image = load_image(input_path.c_str());
    if (image.data == nullptr)
    {
        fprintf(stderr, "ERROR: Failed to load input image: %s\n", input_path.c_str());
        return -1;
    }

    print_image_info(&image);

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

    if (apply_filter(
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

    free_image(&image);
    free_image(&output_image);
    return 0;
}

int main(int argc, char **argv)
{
    printf("Running HIP image fx...\n");

    cli_args_t args = parse_cli_args(argc, argv);
    printf("Input: %s\n", args.input_file);
    printf("Output: %s\n", args.output_file);
    printf("Filter Type: %s\n", filter_type_to_string(args.filter_type).c_str());

    fs::path input_path(args.input_file);
    fs::path output_path(args.output_file);

    bool input_is_dir = fs::is_directory(input_path);
    bool output_is_dir = fs::is_directory(output_path);

    if (input_is_dir && output_is_dir)
    {
        // Batch mode
        int processed = 0, failed = 0;
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

            fs::path out_file = output_path / entry.path().filename();
            printf("\nProcessing: %s -> %s\n", entry.path().string().c_str(), out_file.string().c_str());
            int res = process_one(entry.path().string(), out_file.string(), args.filter_type);
            if (res == 0)
            {
                processed++;
            }
            else
            {
                failed++;
            }
        }
        printf("\nBatch processing complete. Success: %d, Failed: %d\n", processed, failed);
        return failed == 0 ? 0 : 1;
    }
    else if (!input_is_dir && !output_is_dir)
    {
        // Single file mode
        return process_one(args.input_file, args.output_file, args.filter_type);
    }
    else
    {
        fprintf(stderr, "ERROR: Both --input and --output must be either files or directories.\n");
        return -1;
    }
}
