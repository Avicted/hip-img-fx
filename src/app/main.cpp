#include <chrono>
#include <filesystem>
#include <cstdio>
#include <vector>
#include <string>

#include "../cli/cli_parser.h"
#include "process.h"

namespace cli = imgfx::cli;
namespace core = imgfx::core;
namespace app = imgfx::app;
namespace fs = std::filesystem;

inline std::vector<std::string> gather_files(const fs::path &input);
inline void print_elapsed_time(const std::chrono::steady_clock::time_point &start);

int main(int argc, char **argv)
{
    const auto start_time = std::chrono::steady_clock::now();
    printf("============================\n");
    printf("Running HIP Image FX v%s\n", HIP_IMG_FX_VERSION);
    printf("============================\n");

    const cli::cli_args_t args = cli::parse_cli_args(argc, argv);
    bool use_cpu = args.use_cpu;
    printf("Using %s for processing.\n", (use_cpu ? "CPU" : "GPU"));

    if (!use_cpu && core::get_hip_devices() < 1)
    {
        fprintf(stderr, "WARNING: No HIP devices found. Falling back to CPU.\n");
        use_cpu = true;
    }

    fs::path input_path(args.input_file);
    fs::path output_path(args.output_file);

    if (fs::is_directory(input_path) != fs::is_directory(output_path))
    {
        fprintf(stderr, "ERROR: Both --input and --output must be files or directories.\n");
        return -1;
    }

    const auto files = gather_files(input_path);
    if (files.empty())
    {
        fprintf(stderr, "No supported files found in the specified input path.\n");
        return -1;
    }

    int ret_code = 0;
    if (fs::is_directory(input_path))
    {
        bool success = false;
        if (use_cpu)
        {
            success = (app::process_batch_cpu(files, output_path.string(), args.filter_type) == 0);
        }
        else
        {
            success = (app::process_batch_gpu(files, output_path.string(), args.filter_type, args.batch_size) == 0);
        }

        if (!success)
        {
            fprintf(stderr, "Failed to process batch of files.\n");
            ret_code = 1;
        }
    }
    else
    {
        const bool running_as_batch = false;
        const int rc = use_cpu
                           ? app::process_one_cpu(running_as_batch, files[0], output_path.string(), args.filter_type)
                           : app::process_one_gpu(running_as_batch, files[0], output_path.string(), args.filter_type);
        if (rc != 0)
        {
            fprintf(stderr, "Failed to process file: %s\n", files[0].c_str());
            ret_code = 1;
        }
    }

    print_elapsed_time(start_time);
    return ret_code;
}

inline std::vector<std::string> gather_files(const fs::path &input)
{
    std::vector<std::string> files;
    if (fs::is_directory(input))
    {
        for (const auto &entry : fs::directory_iterator(input))
        {
            if (entry.is_regular_file() && core::has_supported_ext(entry.path()))
            {
                files.push_back(entry.path().string());
            }
        }
    }
    else
    {
        files.push_back(input.string());
    }
    return files;
}

inline void print_elapsed_time(const std::chrono::steady_clock::time_point &start)
{
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    printf("Total processing time: %02ldm %02lds %03ldms\n",
           elapsed_ms / 60000,
           (elapsed_ms % 60000) / 1000,
           elapsed_ms % 1000);
}
