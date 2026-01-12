#include <gtest/gtest.h>
#include "../src/core/gpu_utils.h"
#include "test_helpers.h"
#include <vector>

/**
 * @brief GPU integration tests - skip if no GPU available
 *
 * Tests for GPU memory management, batch processing, and error handling.
 */

class GPUIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!test_helpers::has_gpu_available())
        {
            GTEST_SKIP() << "GPU not available - skipping GPU integration test";
        }
    }
};

TEST_F(GPUIntegrationTest, BatchProcessing)
{
    constexpr int width = 128;
    constexpr int height = 128;
    constexpr int channels = 3;
    constexpr int num_images = 4;

    // Create multiple input images
    std::vector<std::vector<unsigned char>> inputs;
    std::vector<std::vector<unsigned char>> outputs;
    std::vector<imgfx::core::image_t> input_imgs;
    std::vector<imgfx::core::image_t> output_imgs;

    for (int i = 0; i < num_images; ++i)
    {
        inputs.push_back(test_helpers::generate_gradient_image(width, height, channels));
        outputs.push_back(std::vector<unsigned char>(width * height * channels));

        input_imgs.push_back({inputs[i].data(), width, height, channels});
        output_imgs.push_back({outputs[i].data(), width, height, channels});
    }

    // Process batch
    hipError_t err = imgfx::core::apply_filter_gpu(
        imgfx::core::FILTER_TYPE::GRAYSCALE,
        input_imgs, output_imgs,
        false, nullptr);

    EXPECT_EQ(err, hipSuccess) << "Batch processing should succeed";

    // Verify all images were processed
    for (int i = 0; i < num_images; ++i)
    {
        bool has_output = false;
        for (size_t j = 0; j < outputs[i].size(); j += channels)
        {
            if (outputs[i][j] > 0)
            {
                has_output = true;
                break;
            }
        }
        EXPECT_TRUE(has_output) << "Image " << i << " should have non-zero output";
    }
}

TEST_F(GPUIntegrationTest, DeviceMemoryLifetime)
{
    using namespace imgfx::core;

    constexpr size_t alloc_size = 1024 * 1024; // 1 MB

    // Test DeviceBuffer RAII
    {
        DeviceBuffer buffer;
        hipError_t err = buffer.allocate(alloc_size);

        EXPECT_EQ(err, hipSuccess) << "Device allocation should succeed";
        EXPECT_NE(buffer.ptr, nullptr) << "Device pointer should not be null";
        EXPECT_EQ(buffer.size, alloc_size);

        // Buffer should auto-free on destruction
    }

    // Allocate again to verify previous free worked
    {
        DeviceBuffer buffer2;
        hipError_t err = buffer2.allocate(alloc_size);
        EXPECT_EQ(err, hipSuccess) << "Second allocation should succeed";
    }
}

TEST_F(GPUIntegrationTest, HostToDeviceTransfer)
{
    using namespace imgfx::core;

    constexpr size_t data_size = 1024;
    std::vector<unsigned char> host_data(data_size);

    // Fill with known pattern
    for (size_t i = 0; i < data_size; ++i)
    {
        host_data[i] = static_cast<unsigned char>(i % 256);
    }

    // Allocate device memory
    DeviceBuffer device_buffer;
    hipError_t err = device_buffer.allocate(data_size);
    ASSERT_EQ(err, hipSuccess);

    // Copy to device
    err = hipMemcpy(device_buffer.ptr, host_data.data(), data_size, hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess) << "H2D transfer should succeed";

    // Copy back to verify
    std::vector<unsigned char> readback(data_size);
    err = hipMemcpy(readback.data(), device_buffer.ptr, data_size, hipMemcpyDeviceToHost);
    EXPECT_EQ(err, hipSuccess) << "D2H transfer should succeed";

    // Verify data integrity
    int mismatches = test_helpers::compare_images_with_tolerance(
        host_data.data(), readback.data(), data_size, 0);
    EXPECT_EQ(mismatches, 0) << "Data should match exactly after round-trip";
}

TEST_F(GPUIntegrationTest, DeviceToHostTransfer)
{
    using namespace imgfx::core;

    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 3;
    constexpr size_t total_size = width * height * channels;

    auto input = test_helpers::generate_solid_color_image(width, height, channels,
                                                          100, 150, 200, 0);
    std::vector<unsigned char> output(total_size);

    // Process on GPU
    image_t in_img{input.data(), width, height, channels};
    image_t out_img{output.data(), width, height, channels};

    hipError_t err = apply_filter_gpu(
        FILTER_TYPE::NEGATIVE,
        in_img, out_img,
        false, nullptr);

    ASSERT_EQ(err, hipSuccess);

    // Verify data was transferred back
    EXPECT_EQ(output[0], 255 - 100) << "Output should have inverted values";
    EXPECT_EQ(output[1], 255 - 150);
    EXPECT_EQ(output[2], 255 - 200);
}

TEST_F(GPUIntegrationTest, ErrorHandling_InvalidDevice)
{
    // Attempt to set an invalid device
    hipError_t err = hipSetDevice(9999);

    EXPECT_NE(err, hipSuccess) << "Setting invalid device should fail";

    // Reset to device 0
    err = hipSetDevice(0);
    EXPECT_EQ(err, hipSuccess) << "Should be able to reset to valid device";
}

TEST_F(GPUIntegrationTest, GPUTimings)
{
    using namespace imgfx::core;

    constexpr int width = 512;
    constexpr int height = 512;
    constexpr int channels = 3;

    auto input = test_helpers::generate_gradient_image(width, height, channels);
    std::vector<unsigned char> output(input.size());

    image_t in_img{input.data(), width, height, channels};
    image_t out_img{output.data(), width, height, channels};

    GPUTimings timings;

    hipError_t err = apply_filter_gpu(
        FILTER_TYPE::GRAYSCALE,
        in_img, out_img,
        true, &timings);

    ASSERT_EQ(err, hipSuccess);

    // Verify timing data was captured
    EXPECT_GT(timings.h2d_ms, 0.0f) << "H2D timing should be positive";
    EXPECT_GT(timings.kernel_ms, 0.0f) << "Kernel timing should be positive";
    EXPECT_GT(timings.d2h_ms, 0.0f) << "D2H timing should be positive";
    EXPECT_GT(timings.total_ms, 0.0f) << "Total timing should be positive";

    // Total should be approximately sum of components
    float sum = timings.h2d_ms + timings.kernel_ms + timings.d2h_ms;
    EXPECT_NEAR(timings.total_ms, sum, 1.0f) << "Total should equal sum of components";
}
