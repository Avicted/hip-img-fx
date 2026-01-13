// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Victor Anderssén

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>

/**
 * @brief Global test environment for HIP/GPU availability detection
 *
 * This environment checks for GPU availability once at the start of test execution.
 * GPU-dependent tests can check the gpu_available flag and skip gracefully if needed.
 */
class HIPTestEnvironment : public ::testing::Environment
{
public:
    static bool gpu_available;
    static std::string gpu_name;
    static int device_count;

    void SetUp() override
    {
        device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);
        gpu_available = (error == hipSuccess && device_count > 0);

        if (gpu_available)
        {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
            {
                gpu_name = prop.name;
                std::cout << "=== GPU detected: " << gpu_name
                          << " (Device count: " << device_count << ") ===" << std::endl;
            }
        }
        else
        {
            std::cout << "=== No GPU detected - GPU tests will be skipped ===" << std::endl;
            if (error != hipSuccess)
            {
                std::cout << "    HIP Error: " << hipGetErrorString(error) << std::endl;
            }
        }
    }

    void TearDown() override
    {
        // Clean up if needed
    }
};

// Initialize static members
bool HIPTestEnvironment::gpu_available = false;
std::string HIPTestEnvironment::gpu_name = "none";
int HIPTestEnvironment::device_count = 0;

/**
 * @brief Main test entry point
 *
 * Initializes GoogleTest and registers the global HIP environment.
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new HIPTestEnvironment);
    return RUN_ALL_TESTS();
}
