/**
 * @file test_hip_event_spec.cpp
 * @brief Specification tests for HIPEvent public API (hip_event.h)
 *
 * Following SKILL.md guidelines:
 * - Encode contracts not implementation
 * - Negative tests required for every positive case
 * - Break-the-code validation
 */

#include <gtest/gtest.h>
#include "hip-img-fx/autotune/hip_event.h"
#include <hip/hip_runtime.h>

using namespace imgfx::core::autotune;

// ============================================================================
// TEST SUITE: HIPEvent Construction and Destruction
// ============================================================================

TEST(HIPEventSpec, DefaultConstructorCreatesValidEvent)
{
    HIPEvent event;
    EXPECT_TRUE(event.is_valid());
}

TEST(HIPEventSpec, GetReturnsNonNullHandle)
{
    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    hipEvent_t handle = event.get();
    EXPECT_NE(handle, nullptr);
}

TEST(HIPEventSpec, MultipleEventsCanCoexist)
{
    HIPEvent event1;
    HIPEvent event2;
    HIPEvent event3;

    EXPECT_TRUE(event1.is_valid());
    EXPECT_TRUE(event2.is_valid());
    EXPECT_TRUE(event3.is_valid());

    // Events should have different handles
    EXPECT_NE(event1.get(), event2.get());
    EXPECT_NE(event2.get(), event3.get());
}

// ============================================================================
// TEST SUITE: HIPEvent Move Semantics
// ============================================================================

TEST(HIPEventSpec, MoveConstructorTransfersOwnership)
{
    HIPEvent event1;
    ASSERT_TRUE(event1.is_valid());
    hipEvent_t original_handle = event1.get();

    HIPEvent event2(std::move(event1));

    EXPECT_FALSE(event1.is_valid()); // Moved-from object is invalid
    EXPECT_TRUE(event2.is_valid());
    EXPECT_EQ(event2.get(), original_handle); // Same handle
}

TEST(HIPEventSpec, MovedFromObjectCanBeDestroyed)
{
    HIPEvent event1;
    HIPEvent event2(std::move(event1));

    EXPECT_FALSE(event1.is_valid());
    // Destruction of event1 should not crash
}

TEST(HIPEventSpec, MovedToObjectCanBeUsed)
{
    HIPEvent event1;
    HIPEvent event2(std::move(event1));

    ASSERT_TRUE(event2.is_valid());

    // Should be able to record
    event2.record();
    event2.synchronize();
}

// ============================================================================
// TEST SUITE: HIPEvent Recording
// ============================================================================

TEST(HIPEventSpec, RecordOnDefaultStreamSucceeds)
{
    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    // Should not throw or crash
    event.record();
    event.synchronize();
}

TEST(HIPEventSpec, RecordTwiceIsAllowed)
{
    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    event.record();
    event.synchronize();

    event.record();
    event.synchronize();
}

TEST(HIPEventSpec, RecordOnInvalidEventDoesNotCrash)
{
    HIPEvent event1;
    HIPEvent event2(std::move(event1)); // event1 is now invalid

    ASSERT_FALSE(event1.is_valid());

    // Should not crash (implementation just skips if invalid)
    event1.record();
}

// ============================================================================
// TEST SUITE: HIPEvent Synchronization
// ============================================================================

TEST(HIPEventSpec, SynchronizeWaitsForEvent)
{
    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    event.record();
    event.synchronize(); // Should block until event is complete
}

TEST(HIPEventSpec, SynchronizeOnInvalidEventDoesNotCrash)
{
    HIPEvent event1;
    HIPEvent event2(std::move(event1));

    ASSERT_FALSE(event1.is_valid());

    // Should not crash
    event1.synchronize();
}

// ============================================================================
// TEST SUITE: HIPEvent Timing
// ============================================================================

TEST(HIPEventSpec, ElapsedTimeReturnsNonNegativeValue)
{
    HIPEvent start, end;
    ASSERT_TRUE(start.is_valid());
    ASSERT_TRUE(end.is_valid());

    start.record();
    end.record();
    end.synchronize();

    float elapsed = HIPEvent::elapsed_time(start, end);
    EXPECT_GE(elapsed, 0.0f);
}

TEST(HIPEventSpec, ElapsedTimeForSameEventIsNearZero)
{
    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    event.record();
    event.synchronize();

    float elapsed = HIPEvent::elapsed_time(event, event);
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_LT(elapsed, 0.1f); // Should be very small
}

TEST(HIPEventSpec, ElapsedTimeWithInvalidStartReturnsZero)
{
    HIPEvent valid_event;
    HIPEvent invalid_event1;
    HIPEvent invalid_event2(std::move(invalid_event1));

    ASSERT_FALSE(invalid_event1.is_valid());
    ASSERT_TRUE(valid_event.is_valid());

    valid_event.record();
    valid_event.synchronize();

    float elapsed = HIPEvent::elapsed_time(invalid_event1, valid_event);
    EXPECT_EQ(elapsed, 0.0f);
}

TEST(HIPEventSpec, ElapsedTimeWithInvalidEndReturnsZero)
{
    HIPEvent valid_event;
    HIPEvent invalid_event1;
    HIPEvent invalid_event2(std::move(invalid_event1));

    ASSERT_TRUE(valid_event.is_valid());
    ASSERT_FALSE(invalid_event1.is_valid());

    valid_event.record();
    valid_event.synchronize();

    float elapsed = HIPEvent::elapsed_time(valid_event, invalid_event1);
    EXPECT_EQ(elapsed, 0.0f);
}

TEST(HIPEventSpec, ElapsedTimeWithBothInvalidReturnsZero)
{
    HIPEvent event1, event2;
    HIPEvent moved1(std::move(event1));
    HIPEvent moved2(std::move(event2));

    ASSERT_FALSE(event1.is_valid());
    ASSERT_FALSE(event2.is_valid());

    float elapsed = HIPEvent::elapsed_time(event1, event2);
    EXPECT_EQ(elapsed, 0.0f);
}

// ============================================================================
// TEST SUITE: HIPEvent Timing Sequences
// ============================================================================

TEST(HIPEventSpec, MultipleTimingIntervalsCanBeMeasured)
{
    HIPEvent event1, event2, event3;

    event1.record();
    event2.record();
    event3.record();
    event3.synchronize();

    float time_1_to_2 = HIPEvent::elapsed_time(event1, event2);
    float time_2_to_3 = HIPEvent::elapsed_time(event2, event3);
    float time_1_to_3 = HIPEvent::elapsed_time(event1, event3);

    EXPECT_GE(time_1_to_2, 0.0f);
    EXPECT_GE(time_2_to_3, 0.0f);
    EXPECT_GE(time_1_to_3, 0.0f);

    // Total time should be >= sum of parts (due to precision)
    EXPECT_GE(time_1_to_3, time_1_to_2 - 0.01f); // Allow small tolerance
    EXPECT_GE(time_1_to_3, time_2_to_3 - 0.01f);
}

TEST(HIPEventSpec, TimingWithActualKernelWorkProducesNonZeroTime)
{
    // Allocate device memory
    int *d_data;
    size_t size = 1024 * 1024 * sizeof(int);
    hipError_t err = hipMalloc(&d_data, size);
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "GPU memory allocation failed";
    }

    HIPEvent start, end;

    start.record();

    // Do some work
    (void)hipMemset(d_data, 0, size);

    end.record();
    end.synchronize();

    float elapsed = HIPEvent::elapsed_time(start, end);

    EXPECT_GT(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1000.0f); // Should be less than 1 second

    (void)hipFree(d_data);
}

// ============================================================================
// TEST SUITE: HIPEvent RAII Guarantees
// ============================================================================

TEST(HIPEventSpec, DestructorReleasesResources)
{
    // Create and destroy many events in a loop
    // If destructor doesn't properly clean up, this will leak memory
    for (int i = 0; i < 100; ++i)
    {
        HIPEvent event;
        EXPECT_TRUE(event.is_valid());
        event.record();
        event.synchronize();
        // Destructor called here
    }
}

TEST(HIPEventSpec, ScopeBasedLifetimeWorks)
{
    {
        HIPEvent inner_event;
        EXPECT_TRUE(inner_event.is_valid());
    }
    // inner_event destroyed here
}

// ============================================================================
// TEST SUITE: HIPEvent Copy Semantics (Disabled)
// ============================================================================

TEST(HIPEventSpec, CopyConstructorIsDeleted)
{
    // This test verifies that the copy constructor is deleted
    // If it compiles, the test passes (compilation check)
    EXPECT_FALSE(std::is_copy_constructible_v<HIPEvent>);
}

TEST(HIPEventSpec, CopyAssignmentIsDeleted)
{
    // This test verifies that the copy assignment is deleted
    EXPECT_FALSE(std::is_copy_assignable_v<HIPEvent>);
}

// ============================================================================
// TEST SUITE: HIPEvent Stream Support
// ============================================================================

TEST(HIPEventSpec, RecordOnCustomStreamSucceeds)
{
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "Failed to create stream";
    }

    HIPEvent event;
    ASSERT_TRUE(event.is_valid());

    event.record(stream);
    event.synchronize();

    (void)hipStreamDestroy(stream);
}

TEST(HIPEventSpec, TimingOnCustomStreamWorks)
{
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess)
    {
        GTEST_SKIP() << "Failed to create stream";
    }

    HIPEvent start, end;

    start.record(stream);
    end.record(stream);
    end.synchronize();

    float elapsed = HIPEvent::elapsed_time(start, end);
    EXPECT_GE(elapsed, 0.0f);

    (void)hipStreamDestroy(stream);
}

// ============================================================================
// TEST SUITE: HIPEvent Invariants
// ============================================================================

TEST(HIPEventSpec, ValidEventStaysValidUntilMoved)
{
    HIPEvent event;

    EXPECT_TRUE(event.is_valid());
    event.record();
    EXPECT_TRUE(event.is_valid());
    event.synchronize();
    EXPECT_TRUE(event.is_valid());

    HIPEvent moved_event(std::move(event));
    EXPECT_FALSE(event.is_valid());
    EXPECT_TRUE(moved_event.is_valid());
}

TEST(HIPEventSpec, InvalidEventStaysInvalid)
{
    HIPEvent event1;
    HIPEvent event2(std::move(event1));

    EXPECT_FALSE(event1.is_valid());
    event1.record();
    EXPECT_FALSE(event1.is_valid());
    event1.synchronize();
    EXPECT_FALSE(event1.is_valid());
}
