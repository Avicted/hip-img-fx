#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

image_t load_image(const char *filename)
{
    image_t img;
    img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);
    if (img.data == nullptr)
    {
        fprintf(stderr, "ERROR: Could not load image: %s\n", filename);
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
