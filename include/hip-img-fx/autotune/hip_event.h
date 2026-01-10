#pragma once

/**
 * @file hip_event.h
 * @brief RAII wrapper for HIP events
 */

#include <hip/hip_runtime.h>

// Helper macro for HIP error checking
#ifndef HIP_ERRCHK
#define HIP_ERRCHK(call)                                                    \
    do                                                                      \
    {                                                                       \
        hipError_t err = call;                                              \
        if (err != hipSuccess)                                              \
        {                                                                   \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
#endif

namespace imgfx::core::autotune
{
    /**
     * @brief RAII wrapper for HIP events with stream support
     *
     * Provides safe management of hipEvent_t resources and timing utilities.
     */
    class HIPEvent
    {
    private:
        hipEvent_t event;
        bool valid;

    public:
        HIPEvent() : valid(false)
        {
            hipError_t err = hipEventCreate(&event);
            if (err == hipSuccess)
            {
                valid = true;
            }
        }

        ~HIPEvent()
        {
            if (valid)
            {
                (void)hipEventDestroy(event);
            }
        }

        // Disable copy
        HIPEvent(const HIPEvent &) = delete;
        HIPEvent &operator=(const HIPEvent &) = delete;

        // Enable move
        HIPEvent(HIPEvent &&other) noexcept : event(other.event), valid(other.valid)
        {
            other.valid = false;
        }

        hipEvent_t get() const { return event; }
        bool is_valid() const { return valid; }

        void record(hipStream_t stream = 0)
        {
            if (valid)
            {
                HIP_ERRCHK(hipEventRecord(event, stream));
            }
        }

        void synchronize()
        {
            if (valid)
            {
                HIP_ERRCHK(hipEventSynchronize(event));
            }
        }

        static float elapsed_time(const HIPEvent &start, const HIPEvent &end)
        {
            float ms = 0.0f;
            if (start.is_valid() && end.is_valid())
            {
                HIP_ERRCHK(hipEventElapsedTime(&ms, start.get(), end.get()));
            }
            return ms;
        }
    };

} // namespace imgfx::core::autotune
