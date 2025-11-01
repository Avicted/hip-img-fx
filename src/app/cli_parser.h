#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <filesystem>

#include "gpu_utils.h"

typedef struct
{
    const char *input_file;
    const char *output_file;
    FILTER_TYPE filter_type;
} cli_args_t;

void print_help();

cli_args_t parse_cli_args(int argc, char **argv);