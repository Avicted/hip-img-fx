#pragma once

#include <hip/hip_runtime.h>

inline void hip_errchk(hipError_t err, const char *file, int line);
int get_hip_devices(void);
