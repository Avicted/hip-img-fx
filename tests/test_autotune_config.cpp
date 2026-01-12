#include <gtest/gtest.h>
#include "hip-img-fx/autotune/tuning_config.h"

/**
 * @brief Autotuning configuration tests - always run (CPU-only)
 *
 * Tests TuningConfig validation and serialization logic.
 */

TEST(AutotuneConfig, ValidationRanges)
{
    using namespace imgfx::core::autotune;

    // Valid configuration
    TuningConfig valid_config;
    valid_config.set_block_dims(256, 1, 1);

    EXPECT_EQ(valid_config.block_x(), 256);
    EXPECT_EQ(valid_config.block_y(), 1);
    EXPECT_EQ(valid_config.block_z(), 1);
    EXPECT_EQ(valid_config.total_threads(), 256);
    // Zero block size - config allows this but would fail at kernel launch
    TuningConfig zero_config;
    zero_config.set_block_dims(0, 1, 1);
    EXPECT_EQ(zero_config.total_threads(), 0);

    // Large block size - config allows this but would exceed GPU limits
    TuningConfig large_config;
    large_config.set_block_dims(2048, 1, 1);
    EXPECT_EQ(large_config.total_threads(), 2048);
    EXPECT_GT(large_config.total_threads(), 1024) << "Config exceeds typical GPU max";
    using namespace imgfx::core::autotune;

    TuningConfig config;
    config.set_block_dims(16, 16, 1);

    EXPECT_EQ(config.total_threads(), 256) << "16 * 16 * 1 = 256";

    config.set_block_dims(32, 8, 1);

    EXPECT_EQ(config.total_threads(), 256) << "32 * 8 * 1 = 256";

    TuningConfig config1;
    config1.set_block_dims(128, 1, 1);

    TuningConfig config2;
    config2.set_block_dims(128, 1, 1);

    TuningConfig config3;
    config3.set_block_dims(256, 1, 1);

    // Configs with same values should be equal
    EXPECT_EQ(config1, config2);
    EXPECT_EQ(config1.block_x(), config2.block_x());
    EXPECT_EQ(config1.block_y(), config2.block_y());
    EXPECT_EQ(config1.block_z(), config2.block_z());

    // Different configs should not be equal
    EXPECT_NE(config1, config3);
    EXPECT_NE(config1.block_x(), config3.block_x());
}
