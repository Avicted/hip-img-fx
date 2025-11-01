#pragma once

typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels;
} image_t;

image_t load_image(const char *filename);
void print_image_info(const image_t *img);
void free_image(image_t *img);
bool save_image(const char *filename, const image_t *img);