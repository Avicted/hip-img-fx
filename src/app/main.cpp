#include <chrono>

#include "../cli/cli_parser.h"
#include "process.h"

int main(int argc, char **argv)
{
    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();

    printf("Running HIP image fx...\n");
    bool use_cpu = false;

    cli_args_t args = parse_cli_args(argc, argv);
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
    int processed = 0;
    int failed = 0;
    if (input_is_dir && output_is_dir)
    {
        std::vector<std::string> input_files = {};

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
        }

        if (use_cpu)
        {
            process_batch_cpu(input_files, output_path.string(), args.filter_type) == 0 ? processed++ : failed++;
        }
        else
        {
            process_batch_gpu(input_files, output_path.string(), args.filter_type) == 0 ? processed++ : failed++;
        }

        ret = (failed == 0 ? 0 : 1);
    }
    else if (!input_is_dir && !output_is_dir)
    {
        const bool running_as_batch = false;
        ret = process_one_cpu(running_as_batch, args.input_file, args.output_file, args.filter_type);
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
