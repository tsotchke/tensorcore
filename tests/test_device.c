/*
 * Smoke test: tc_init / tc_device_info_get / tc_shutdown.
 * Prints what we detected so users can sanity-check device + family.
 */

#include <stdio.h>
#include "tensorcore/tensorcore.h"

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    tc_device_info info;
    if (tc_device_info_get(ctx, &info) != TC_OK) {
        fprintf(stderr, "tc_device_info_get failed\n");
        return 2;
    }
    printf("tensorcore: %s\n", tc_version());
    printf("  device           : %s\n", info.name);
    printf("  family           : Apple%d\n", (int)info.family);
    printf("  unified memory   : %s\n", info.unified_memory ? "yes" : "no");
    printf("  max buffer       : %.1f GB\n",
           (double)info.max_buffer_bytes / (1024.0*1024.0*1024.0));
    printf("  working set      : %.1f GB\n",
           (double)info.recommended_working_set_bytes / (1024.0*1024.0*1024.0));
    printf("  max TG mem       : %u KB\n", info.max_threadgroup_memory / 1024);
    printf("  bf16 simdgroup   : %s\n", info.supports_bf16_simdgroup ? "yes" : "no");
    printf("  i8 simdgroup     : %s\n", info.supports_i8_simdgroup   ? "yes" : "no");
    printf("  tensorops (M5)   : %s\n", info.supports_tensorops_m5   ? "yes" : "no");

    if (info.family < TC_FAMILY_APPLE7) {
        fprintf(stderr, "warning: pre-M1 GPU detected — simdgroup_matrix unavailable\n");
    }

    tc_context* ctx2 = NULL;
    s = tc_init(&ctx2);
    if (s != TC_ERR_ALREADY_INITIALIZED || ctx2 != ctx) {
        fprintf(stderr, "second tc_init did not return shared initialized context\n");
        return 3;
    }

    if (tc_shutdown(ctx) != TC_OK) {
        fprintf(stderr, "first tc_shutdown failed\n");
        return 4;
    }
    if (tc_device_info_get(ctx2, &info) != TC_OK) {
        fprintf(stderr, "context was destroyed before final shutdown\n");
        return 5;
    }
    if (tc_shutdown(ctx2) != TC_OK) {
        fprintf(stderr, "second tc_shutdown failed\n");
        return 6;
    }
    return 0;
}
