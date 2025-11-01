#include <stdio.h>

#include "gpu_utils.h"
#include "cli_parser.h"
#include "image.h"

int main(int argc, char **argv)
{
    printf("Running HIP image fx...\n");

    cli_args_t args = parse_cli_args(argc, argv);
    printf("Input File: %s\n", args.input_file);
    printf("Output File: %s\n", args.output_file);
    printf("Filter Type: %s\n", filter_type_to_string(args.filter_type).c_str());

    image_t image = load_image(args.input_file);
    if (image.data == nullptr)
    {
        fprintf(stderr, "ERROR: Failed to load input image!\n");
        return -1;
    }

    print_image_info(&image);

    // Allocate output image
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
            args.filter_type,
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

    // Save output image
    if (!save_image(args.output_file, &output_image))
    {
        fprintf(stderr, "ERROR: Could not save output image!\n");
        free_image(&image);
        free_image(&output_image);
        return -1;
    }

    printf("Successfully saved output image: %s\n", args.output_file);

    free_image(&image);
    free_image(&output_image);

    return 0;
}
