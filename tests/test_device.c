/*
 * Smoke test: tc_init / tc_device_info_get / tc_shutdown.
 * Prints what we detected so users can sanity-check device + family.
 */

#include <stdio.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

static int check_named_family(const tc_device_info* info) {
    struct expected_family {
        const char* chip;
        tc_family_t family;
        bool bf16;
    } expected[] = {
        {"Apple M1", TC_FAMILY_APPLE7, false},
        {"Apple M2", TC_FAMILY_APPLE8, false},
        {"Apple M3", TC_FAMILY_APPLE9, true},
        {"Apple M4", TC_FAMILY_APPLE9, true},
        {"Apple M5", TC_FAMILY_APPLE10, true},
    };
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        if (!strstr(info->name, expected[i].chip)) continue;
        if (info->family != expected[i].family ||
            info->supports_bf16_simdgroup != expected[i].bf16 ||
            info->supports_i8_simdgroup) {
            fprintf(stderr,
                    "chip capability mismatch: device=%s family=Apple%d "
                    "bf16_sg=%d i8_sg=%d expected_family=Apple%d expected_bf16=%d\n",
                    info->name, (int)info->family,
                    info->supports_bf16_simdgroup ? 1 : 0,
                    info->supports_i8_simdgroup ? 1 : 0,
                    (int)expected[i].family, expected[i].bf16 ? 1 : 0);
            return 1;
        }
        printf("  family policy    : %s=Apple%d verified\n",
               expected[i].chip, (int)expected[i].family);
        return 0;
    }
    return 0;
}

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
    if (check_named_family(&info) != 0) return 7;

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
