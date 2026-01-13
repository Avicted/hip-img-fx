// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor_stb.h"

namespace imgfx::core
{
    const std::vector<std::string> supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};

    bool has_supported_ext(const fs::path &p)
    {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return std::find(supported_exts.begin(), supported_exts.end(), ext) != supported_exts.end();
    }

    image_t load_image(const char *filename)
    {
        image_t img;
        img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);
        if (img.data == nullptr)
        {
            img.width = 0;
            img.height = 0;
            img.channels = 0;
        }

        return img;
    }

    void print_image_info(const image_t *img)
    {
        printf("Image Info:\n");
        printf("  Width: %d\n", img->width);
        printf("  Height: %d\n", img->height);
        printf("  Channels: %d\n", img->channels);
        printf("  Size (megabytes): %.2f MB\n", (img->width * img->height * img->channels) / (1024.0 * 1024.0));
    }

    void free_image(image_t *img)
    {
        if (img->data)
        {
            stbi_image_free(img->data);
            img->data = nullptr;
        }
    }

    bool save_image(const char *filename, const image_t *img)
    {
        int success = 0;
        const char *ext = strrchr(filename, '.');
        if (ext)
        {
            if (strcmp(ext, ".png") == 0)
            {
                success = stbi_write_png(filename, img->width, img->height, img->channels, img->data, img->width * img->channels);
            }
            else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)
            {
                success = stbi_write_jpg(filename, img->width, img->height, img->channels, img->data, 90); // Quality set to 90
            }
            else if (strcmp(ext, ".bmp") == 0)
            {
                success = stbi_write_bmp(filename, img->width, img->height, img->channels, img->data);
            }
            else if (strcmp(ext, ".tga") == 0)
            {
                success = stbi_write_tga(filename, img->width, img->height, img->channels, img->data);
            }
            else
            {
                fprintf(stderr, "ERROR: Unsupported image format: %s\n", ext);
                return false;
            }
        }
        else
        {
            fprintf(stderr, "ERROR: No file extension found in filename: %s\n", filename);
            return false;
        }

        return success != 0;
    }
}
