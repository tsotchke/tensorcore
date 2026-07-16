#include <cstdio>
#include <initializer_list>

#include "core/apple_family.h"

static int expect_family(tc_apple_family_support support, tc_family_t expected) {
    const tc_family_t actual = tc_apple_family_select(support);
    if (actual == expected) return 0;
    std::fprintf(stderr, "family selection mismatch: expected=%d actual=%d\n",
                 static_cast<int>(expected), static_cast<int>(actual));
    return 1;
}

int main() {
    int failures = 0;
    failures += expect_family({false, false, false, false}, TC_FAMILY_UNKNOWN);
    failures += expect_family({true, false, false, false}, TC_FAMILY_APPLE7);
    failures += expect_family({true, true, false, false}, TC_FAMILY_APPLE8);
    failures += expect_family({true, true, true, false}, TC_FAMILY_APPLE9);
    failures += expect_family({true, true, true, true}, TC_FAMILY_APPLE10);
    failures += expect_family({false, false, false, true}, TC_FAMILY_APPLE10);

    if (TC_MTL_GPU_FAMILY_APPLE7_RAW != 1007 ||
        TC_MTL_GPU_FAMILY_APPLE8_RAW != 1008 ||
        TC_MTL_GPU_FAMILY_APPLE9_RAW != 1009 ||
        TC_MTL_GPU_FAMILY_APPLE10_RAW != 1010) {
        std::fprintf(stderr, "MTLGPUFamily raw-value contract changed\n");
        failures++;
    }
    if (tc_apple_family_supports_bf16_simdgroup(TC_FAMILY_APPLE8) ||
        !tc_apple_family_supports_bf16_simdgroup(TC_FAMILY_APPLE9) ||
        !tc_apple_family_supports_bf16_simdgroup(TC_FAMILY_APPLE10) ||
        tc_apple_family_supports_bf16_simdgroup(TC_FAMILY_APPLE11)) {
        std::fprintf(stderr, "BF16 family policy mismatch\n");
        failures++;
    }
    for (tc_family_t family : {TC_FAMILY_UNKNOWN, TC_FAMILY_APPLE7,
                               TC_FAMILY_APPLE8, TC_FAMILY_APPLE9,
                               TC_FAMILY_APPLE10, TC_FAMILY_APPLE11}) {
        if (tc_apple_family_supports_i8_simdgroup(family)) {
            std::fprintf(stderr, "integer simdgroup_matrix must remain disabled\n");
            failures++;
        }
    }
    if (tc_apple_family_supports_tensorops_m5(TC_FAMILY_APPLE9) ||
        !tc_apple_family_supports_tensorops_m5(TC_FAMILY_APPLE10) ||
        tc_apple_family_supports_tensorops_m5(TC_FAMILY_APPLE11)) {
        std::fprintf(stderr, "M5 TensorOps family policy mismatch\n");
        failures++;
    }

    if (failures == 0) {
        std::puts("Apple family policy OK: M4=Apple9 M5=Apple10 Apple11=reserved i8_sg=false");
    }
    return failures == 0 ? 0 : 1;
}
