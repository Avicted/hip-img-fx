#include <stdio.h>

#include "gpu_utils.h"

int main(int argc, char **argv)
{
    printf("Running HIP image fx...\n");

    int hip_device_count = get_hip_devices();
    if (hip_device_count < 1)
    {
        fprintf(stderr, "ERROR: Could not find any HIP device!\n");
        return -1;
    }

    printf("hip_device_count: %d\n", hip_device_count);

    return 0;
}
