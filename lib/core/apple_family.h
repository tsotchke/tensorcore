#ifndef TENSORCORE_APPLE_FAMILY_H
#define TENSORCORE_APPLE_FAMILY_H

#include "tensorcore/device.h"

/* MTLGPUFamily uses stable raw values 1000 + the Apple family number.
 * Apple10 is absent from older SDK headers, so the runtime probe uses the
 * vendor-defined raw value while remaining source-compatible with them. */
enum {
    TC_MTL_GPU_FAMILY_APPLE7_RAW = 1007,
    TC_MTL_GPU_FAMILY_APPLE8_RAW = 1008,
    TC_MTL_GPU_FAMILY_APPLE9_RAW = 1009,
    TC_MTL_GPU_FAMILY_APPLE10_RAW = 1010,
};

typedef struct {
    bool apple7;
    bool apple8;
    bool apple9;
    bool apple10;
} tc_apple_family_support;

static inline tc_family_t tc_apple_family_select(tc_apple_family_support support) {
    if (support.apple10) return TC_FAMILY_APPLE10;
    if (support.apple9) return TC_FAMILY_APPLE9;
    if (support.apple8) return TC_FAMILY_APPLE8;
    if (support.apple7) return TC_FAMILY_APPLE7;
    return TC_FAMILY_UNKNOWN;
}

static inline bool tc_apple_family_supports_bf16_simdgroup(tc_family_t family) {
    return family == TC_FAMILY_APPLE9 || family == TC_FAMILY_APPLE10;
}

static inline bool tc_apple_family_supports_i8_simdgroup(tc_family_t family) {
    (void)family;
    /* The public Metal 4 MSL contract permits half, bfloat, and float
     * simdgroup_matrix elements. Integer element types are not public. */
    return false;
}

static inline bool tc_apple_family_supports_tensorops_m5(tc_family_t family) {
    return family == TC_FAMILY_APPLE10;
}

#endif
