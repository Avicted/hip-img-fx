#include "cli_parser.h"

void print_help()
{
    printf("\nUsage: hip-img-fx [options]\n");
    printf("Options:\n");
    printf("  --input <input_file|input_dir>     Specifies the input file or directory path.\n");
    printf("  --output <output_file|output_dir>  Specifies the output file or directory path.\n");
    printf("  --filter <filter_type>             Specifies the type of filter to apply (e.g., \"grayscale\", \"negative\", \"gaussian-blur\").\n");
    printf("  --help                             Displays this help information.\n");
    printf("\n");
    printf("Notes:\n");
    printf("  - For batch processing, specify both --input and --output as directories.\n");
    printf("  - For single image processing, specify both as files.\n");
    printf("  - Supported filters: grayscale, negative, gaussian-blur\n");
}

cli_args_t parse_cli_args(int argc, char **argv)
{
    cli_args_t args = {
        .input_file = nullptr,
        .output_file = nullptr,
        .filter_type = FILTER_TYPE::UNDEFINED,
    };

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
        {
            args.input_file = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            args.output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc)
        {
            if (strcmp(argv[++i], "grayscale") == 0)
            {
                args.filter_type = FILTER_TYPE::GRAYSCALE;
            }
            else if (strcmp(argv[i], "negative") == 0)
            {
                args.filter_type = FILTER_TYPE::NEGATIVE;
            }
            else if (strcmp(argv[i], "gaussian-blur") == 0)
            {
                args.filter_type = FILTER_TYPE::GAUSSIAN_BLUR;
            }
            else
            {
                printf("Unknown filter type: %s\n", argv[i]);
                print_help();
                exit(1);
            }
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            print_help();
            exit(0);
        }
        else
        {
            printf("Unknown argument: %s\n", argv[i]);
            print_help();
            exit(1);
        }
    }

    bool has_error = false;
    if (args.input_file == nullptr)
    {
        printf("Error: --input argument is required.\n");
        has_error = true;
    }
    if (args.output_file == nullptr)
    {
        printf("Error: --output argument is required.\n");
        has_error = true;
    }
    if (args.filter_type == FILTER_TYPE::UNDEFINED)
    {
        printf("Error: --filter argument is required.\n");
        has_error = true;
    }
    if (has_error)
    {
        print_help();
        exit(1);
    }

    return args;
}
