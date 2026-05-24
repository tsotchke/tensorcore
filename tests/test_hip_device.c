/*
 * tests/test_hip_device.c - optional HIP/chipStar runtime probe.
 *
 * Builds when TC_ENABLE_HIP=ON. Exits 77 when the runtime is present at
 * build time but no usable device is visible at runtime.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/hip.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "tc_init failed\n");
        return 1;
    }

    tc_status_t s = tc_hip_init(ctx);
    if (s != TC_OK) {
        printf("[skip] no HIP/chipStar runtime device available: %s\n",
               tc_status_string(s));
        tc_shutdown(ctx);
        return 77;
    }

    const int count = tc_hip_device_count();
    if (count <= 0) {
        fprintf(stderr, "tc_hip_init succeeded but device_count=%d\n", count);
        tc_shutdown(ctx);
        return 1;
    }

    tc_hip_device_info info;
    memset(&info, 0, sizeof(info));
    if (tc_hip_device_at(0, &info) != TC_OK ||
        info.device_name[0] == '\0' ||
        info.global_memory_bytes == 0 ||
        info.compute_units == 0) {
        fprintf(stderr, "HIP device info is incomplete\n");
        tc_shutdown(ctx);
        return 1;
    }
    if (tc_hip_device_info_get(ctx, &info) != TC_OK) {
        fprintf(stderr, "tc_hip_device_info_get failed\n");
        tc_shutdown(ctx);
        return 1;
    }
    if (strcmp(tc_hip_last_kernel_name(), "none") != 0) {
        fprintf(stderr, "HIP device probe should not report a kernel\n");
        tc_shutdown(ctx);
        return 1;
    }

    printf("HIP runtime OK: %s vendor=%d devices=%d cu=%u\n",
           info.device_name, (int)info.vendor, count, info.compute_units);
    tc_shutdown(ctx);
    return 0;
}
