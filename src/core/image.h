#pragma once

#include <filesystem>
#include <vector>
#include <algorithm>

namespace imgfx::core
{
    namespace fs = std::filesystem;

    typedef struct
    {
        unsigned char *data;
        int width;
        int height;
        int channels;
    } image_t;

    typedef struct
    {
        int offset; // start index of the image in the flattened buffer
        int width;
        int height;
        int channels;
    } image_meta_t;

    bool has_supported_ext(const fs::path &p);
    image_t load_image(const char *filename);
    void print_image_info(const image_t *img);
    void free_image(image_t *img);
    bool save_image(const char *filename, const image_t *img);
}
