#pragma once

#include "../core/gpu_utils.h"

namespace imgfx::app
{
    int process_batch_gpu(const std::vector<std::string> &input_files,
                          const std::string &output_path,
                          imgfx::core::FILTER_TYPE filter_type,
                          int batch_size = 64);

    int process_one_gpu(bool running_as_batch,
                        const std::string &input_path,
                        const std::string &output_path,
                        imgfx::core::FILTER_TYPE filter_type);

    int process_one_cpu(bool running_as_batch,
                        const std::string &input_path,
                        const std::string &output_path,
                        imgfx::core::FILTER_TYPE filter_type);

    int process_batch_cpu(const std::vector<std::string> &input_files,
                          const std::string &output_path,
                          imgfx::core::FILTER_TYPE filter_type);
}
