#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>

namespace test_helpers
{
    /**
     * @brief Check if GPU is available for testing
     * @return true if at least one HIP-compatible GPU is available
     */
    inline bool has_gpu_available()
    {
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);
        return (error == hipSuccess && device_count > 0);
    }

    /**
     * @brief Generate a solid color image
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param channels Number of channels (3 or 4)
     * @param r Red value (0-255)
     * @param g Green value (0-255)
     * @param b Blue value (0-255)
     * @param a Alpha value (0-255), only used if channels == 4
     * @return Vector containing image data
     */
    std::vector<unsigned char> generate_solid_color_image(
        int width, int height, int channels,
        unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255);

    /**
     * @brief Generate a gradient image (horizontal gradient from black to white)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param channels Number of channels (3 or 4)
     * @return Vector containing image data
     */
    std::vector<unsigned char> generate_gradient_image(
        int width, int height, int channels);

    /**
     * @brief Generate a checkerboard pattern image
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param channels Number of channels (3 or 4)
     * @param block_size Size of each checkerboard square in pixels
     * @return Vector containing image data
     */
    std::vector<unsigned char> generate_checkerboard_image(
        int width, int height, int channels, int block_size = 32);

    /**
     * @brief Compare two images pixel-by-pixel with tolerance
     * @param img1 First image data
     * @param img2 Second image data
     * @param size Total number of bytes to compare
     * @param tolerance Maximum allowed difference per pixel (default: 1)
     * @return Number of pixels exceeding tolerance
     */
    int compare_images_with_tolerance(
        const unsigned char *img1,
        const unsigned char *img2,
        size_t size,
        int tolerance = 1);

    /**
     * @brief Fill image with random pixel values (for testing)
     * @param data Output buffer
     * @param size Total number of bytes
     * @param seed Random seed for reproducibility
     */
    void fill_random_image(unsigned char *data, size_t size, unsigned int seed = 42);

    /**
     * @brief Calculate pixel variance for an image (uses first channel)
     * @param img Image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels per pixel
     * @return Variance value (0 = uniform, higher = more variation)
     */
    double calculate_image_variance(
        const unsigned char *img,
        int width,
        int height,
        int channels);

} // namespace test_helpers
