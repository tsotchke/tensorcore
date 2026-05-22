/*
 * tensorcore - HIP/chipStar public ABI stubs.
 *
 * The HIP backend is staged behind the portable ABI. Until chipStar device
 * management lands, these symbols provide deterministic unsupported behavior.
 */

#include "tensorcore/hip.h"

#include <cstring>

extern "C" tc_status_t tc_hip_init(tc_context* ctx) {
    return ctx ? TC_ERR_UNSUPPORTED_FAMILY : TC_ERR_NOT_INITIALIZED;
}

extern "C" tc_status_t tc_hip_device_info_get(tc_context* ctx, tc_hip_device_info* out_info) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!out_info) return TC_ERR_INVALID_ARG;
    std::memset(out_info, 0, sizeof(*out_info));
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" int tc_hip_device_count(void) {
    return 0;
}

extern "C" tc_status_t tc_hip_device_at(int index, tc_hip_device_info* out_info) {
    if (index < 0 || !out_info) return TC_ERR_INVALID_ARG;
    std::memset(out_info, 0, sizeof(*out_info));
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_hip_select_device(tc_context* ctx, int index) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (index < 0) return TC_ERR_INVALID_ARG;
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" const char* tc_hip_last_kernel_name(void) {
    return "none";
}
