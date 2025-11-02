#pragma once

#include "gpu_utils.h"

#define GPU_CHUNK_SIZE 64 // tune based on GPU memory

int process_batch_gpu(const std::vector<std::string> &input_files,
                      const std::string &output_path,
                      FILTER_TYPE filter_type);

int process_one_cpu(bool running_as_batch,
                    const std::string &input_path,
                    const std::string &output_path,
                    FILTER_TYPE filter_type);

int process_batch_cpu(const std::vector<std::string> &input_files,
                      const std::string &output_path,
                      FILTER_TYPE filter_type);
