#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <filesystem>

#include "../core/gpu_utils.h"

namespace imgfx::cli
{
    typedef struct
    {
        const char *input_file;
        const char *output_file;
        imgfx::core::FILTER_TYPE filter_type;
        bool use_cpu;
        int batch_size;
    } cli_args_t;

    void print_help();

    cli_args_t parse_cli_args(int argc, char **argv);
}
