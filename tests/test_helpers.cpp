#include "test_helpers.h"
#include <algorithm>
#include <random>

namespace test_helpers
{
    std::vector<unsigned char> generate_solid_color_image(
        int width, int height, int channels,
        unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        size_t total_size = width * height * channels;
        std::vector<unsigned char> image(total_size);

        for (int i = 0; i < width * height; ++i)
        {
            image[i * channels + 0] = r;
            image[i * channels + 1] = g;
            image[i * channels + 2] = b;
            if (channels == 4)
            {
                image[i * channels + 3] = a;
            }
        }

        return image;
    }

    std::vector<unsigned char> generate_gradient_image(
        int width, int height, int channels)
    {
        size_t total_size = width * height * channels;
        std::vector<unsigned char> image(total_size);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                // Horizontal gradient from 0 to 255
                unsigned char value = static_cast<unsigned char>((x * 255) / std::max(1, width - 1));

                int idx = (y * width + x) * channels;
                image[idx + 0] = value;
                image[idx + 1] = value;
                image[idx + 2] = value;
                if (channels == 4)
                {
                    image[idx + 3] = 255;
                }
            }
        }

        return image;
    }

    std::vector<unsigned char> generate_checkerboard_image(
        int width, int height, int channels, int block_size)
    {
        size_t total_size = width * height * channels;
        std::vector<unsigned char> image(total_size);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                // Determine if this pixel is in a "white" or "black" square
                bool is_white = ((x / block_size) + (y / block_size)) % 2 == 0;
                unsigned char value = is_white ? 255 : 0;

                int idx = (y * width + x) * channels;
                image[idx + 0] = value;
                image[idx + 1] = value;
                image[idx + 2] = value;
                if (channels == 4)
                {
                    image[idx + 3] = 255;
                }
            }
        }

        return image;
    }

    int compare_images_with_tolerance(
        const unsigned char *img1,
        const unsigned char *img2,
        size_t size,
        int tolerance)
    {
        int mismatches = 0;

        for (size_t i = 0; i < size; ++i)
        {
            int diff = std::abs(static_cast<int>(img1[i]) - static_cast<int>(img2[i]));
            if (diff > tolerance)
            {
                mismatches++;
            }
        }

        return mismatches;
    }

    void fill_random_image(unsigned char *data, size_t size, unsigned int seed)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);

        for (size_t i = 0; i < size; ++i)
        {
            data[i] = static_cast<unsigned char>(dist(rng));
        }
    }

    double calculate_image_variance(
        const unsigned char *img,
        int width,
        int height,
        int channels)
    {
        if (width <= 0 || height <= 0 || channels <= 0 || img == nullptr)
        {
            return 0.0;
        }

        const int pixel_count = width * height;
        const size_t stride = static_cast<size_t>(channels);

        // Calculate mean (using first channel only)
        double sum = 0.0;
        for (int i = 0; i < pixel_count; ++i)
        {
            sum += img[i * stride];
        }
        const double mean = sum / pixel_count;

        // Calculate variance
        double variance = 0.0;
        for (int i = 0; i < pixel_count; ++i)
        {
            const double diff = img[i * stride] - mean;
            variance += diff * diff;
        }

        return variance / pixel_count;
    }

} // namespace test_helpers
