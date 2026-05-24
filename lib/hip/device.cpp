/*
 * tensorcore — HIP backend device init + enumeration.
 *
 * Implements tc_hip_init, tc_hip_device_info_get, and the device list.
 * The actual HIP runtime calls (hipInit, hipGetDeviceCount, etc.) are
 * gated on TC_ENABLE_HIP=ON; without it, this file compiles to a tiny
 * stub TU that returns TC_ERR_UNSUPPORTED_FAMILY. The full runtime
 * comes online when chipStar's libCHIP.so is linked in (Intel Level
 * Zero, NVIDIA OpenCL via POCL-CUDA, AMD OpenCL, ARM Mali OpenCL).
 *
 * Design: a tc_context can host either the Metal backend, the HIP
 * backend, or the CPU backend — but not multiple at once. When HIP
 * is selected, all subsequent tc_gemm / tc_attention_forward / etc.
 * dispatch into the HIP path. The selection is by tc_context init
 * options (TC_DEVICE_HIP) which currently routes to Metal-or-CPU only.
 *
 * Once this lands and `tc_init(ctx, TC_DEVICE_HIP)` is honored:
 *   - On Intel hosts: chipStar finds Level Zero, uses Intel GPU.
 *   - On NVIDIA hosts with POCL-CUDA: chipStar finds POCL, dispatches
 *     SPIR-V → PTX → CUDA runtime → NVIDIA GPU.
 *   - On AMD hosts: chipStar finds amdgpu OpenCL, dispatches direct.
 *   - On ARM Mali: same pattern via Mali OpenCL.
 */

#include "tensorcore/hip.h"
#include "tensorcore/tensorcore.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_HIP_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_HIP_INTERNAL
#endif

#if defined(TC_ENABLE_HIP)
#  include <hip/hip_runtime.h>
#endif

namespace {

#if defined(TC_ENABLE_HIP)

struct HipState {
    int  device_count = 0;
    int  selected_device = -1;
    tc_hip_device_info info = {};
    bool initialized = false;
};

std::mutex& state_mutex() { static std::mutex m; return m; }
HipState& state() { static HipState s; return s; }

tc_hip_vendor_t infer_vendor_from_name(const char* name) {
    if (!name) return TC_HIP_VENDOR_UNKNOWN;
    /* hipDeviceProp_t::name is the vendor-supplied string. */
    if (std::strstr(name, "NVIDIA") || std::strstr(name, "GeForce") ||
        std::strstr(name, "Tesla") || std::strstr(name, "RTX")) {
        return TC_HIP_VENDOR_NVIDIA;
    }
    if (std::strstr(name, "AMD") || std::strstr(name, "Radeon") ||
        std::strstr(name, "Vega") || std::strstr(name, "MI") || std::strstr(name, "RDNA")) {
        return TC_HIP_VENDOR_AMD;
    }
    if (std::strstr(name, "Intel") || std::strstr(name, "Iris") ||
        std::strstr(name, "Arc") || std::strstr(name, "Ponte Vecchio")) {
        return TC_HIP_VENDOR_INTEL;
    }
    if (std::strstr(name, "Mali") || std::strstr(name, "Carmel") || std::strstr(name, "Tegra")) {
        return TC_HIP_VENDOR_ARM_MALI;
    }
    return TC_HIP_VENDOR_UNKNOWN;
}

bool populate_info(int device_index, tc_hip_device_info* info) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_index) != hipSuccess) return false;
    std::memset(info, 0, sizeof(*info));
    size_t name_len = 0;
    while (name_len + 1 < sizeof(info->device_name) && prop.name[name_len] != '\0') {
        ++name_len;
    }
    std::memcpy(info->device_name, prop.name, name_len);
    info->device_name[name_len] = '\0';
    info->vendor = infer_vendor_from_name(prop.name);
    info->global_memory_bytes = prop.totalGlobalMem;
    info->local_memory_bytes = prop.sharedMemPerBlock;
    info->compute_units = prop.multiProcessorCount;
    info->max_workgroup_size = prop.maxThreadsPerBlock;
    info->preferred_subgroup_size = prop.warpSize;
    info->supports_fp16 = true;   /* All HIP-targetable devices do */
    /* fp64: chipStar's hipGetDeviceProperties may not set this; default true
     * for Intel/AMD/NVIDIA, false for ARM Mali (which is typically fp32 only). */
    info->supports_fp64 = info->vendor != TC_HIP_VENDOR_ARM_MALI;
    info->supports_int8_dot = info->vendor == TC_HIP_VENDOR_NVIDIA ||
                              info->vendor == TC_HIP_VENDOR_AMD;
    info->unified_memory = (prop.integrated != 0);
    /* Driver/OpenCL version strings: HIP doesn't expose directly; chipStar
     * environment hints would set these. */
    std::snprintf(info->driver_version, sizeof(info->driver_version),
                  "HIP-via-chipStar");
    std::snprintf(info->opencl_version, sizeof(info->opencl_version),
                  "SPIR-V");
    return true;
}

bool env_equals_ci(const char* value, const char* expected) {
    if (!value || !expected) return false;
    while (*value && *expected) {
        char a = *value++;
        char b = *expected++;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return false;
    }
    return *value == '\0' && *expected == '\0';
}

bool env_true(const char* value) {
    return value &&
           (env_equals_ci(value, "1") ||
            env_equals_ci(value, "true") ||
            env_equals_ci(value, "yes") ||
            env_equals_ci(value, "on") ||
            env_equals_ci(value, "enable") ||
            env_equals_ci(value, "enabled"));
}

bool env_false(const char* value) {
    return value &&
           (env_equals_ci(value, "0") ||
            env_equals_ci(value, "false") ||
            env_equals_ci(value, "no") ||
            env_equals_ci(value, "off") ||
            env_equals_ci(value, "disable") ||
            env_equals_ci(value, "disabled"));
}

bool hip_policy_disabled(void) {
    if (env_true(std::getenv("TC_DISABLE_HIP_GEMM"))) return true;
    if (env_false(std::getenv("TC_HIP_GEMM"))) return true;
    if (env_false(std::getenv("TC_USE_HIP_GEMM"))) return true;
    return false;
}

#endif  /* TC_ENABLE_HIP */

}  // namespace

extern "C" tc_status_t tc_hip_init(tc_context* ctx) {
#if !defined(TC_ENABLE_HIP)
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(state_mutex());
    auto& s = state();
    if (s.initialized) return TC_OK;

    if (hipInit(0) != hipSuccess) {
        (void)hipGetLastError();
        return TC_ERR_UNSUPPORTED_FAMILY;
    }
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        (void)hipGetLastError();
        return TC_ERR_UNSUPPORTED_FAMILY;   /* no HIP device available */
    }
    s.device_count = count;
    s.selected_device = 0;
    if (hipSetDevice(0) != hipSuccess) return TC_ERR_INTERNAL;
    if (!populate_info(0, &s.info)) return TC_ERR_INTERNAL;
    s.initialized = true;
    return TC_OK;
#endif
}

extern "C" tc_status_t tc_hip_device_info_get(tc_context* ctx, tc_hip_device_info* out_info) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!out_info) return TC_ERR_INVALID_ARG;
#if !defined(TC_ENABLE_HIP)
    std::memset(out_info, 0, sizeof(*out_info));
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    std::lock_guard<std::mutex> lk(state_mutex());
    auto& s = state();
    if (!s.initialized) {
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
            (void)hipGetLastError();
            std::memset(out_info, 0, sizeof(*out_info));
            return TC_ERR_UNSUPPORTED_FAMILY;
        }
        return TC_ERR_NOT_INITIALIZED;
    }
    *out_info = s.info;
    return TC_OK;
#endif
}

extern "C" int tc_hip_device_count(void) {
#if !defined(TC_ENABLE_HIP)
    return 0;
#else
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    return count;
#endif
}

extern "C" tc_status_t tc_hip_device_at(int index, tc_hip_device_info* out_info) {
    if (index < 0 || !out_info) return TC_ERR_INVALID_ARG;
#if !defined(TC_ENABLE_HIP)
    std::memset(out_info, 0, sizeof(*out_info));
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        (void)hipGetLastError();
        std::memset(out_info, 0, sizeof(*out_info));
        return TC_ERR_UNSUPPORTED_FAMILY;
    }
    if (index >= count) {
        std::memset(out_info, 0, sizeof(*out_info));
        return TC_ERR_INVALID_ARG;
    }
    if (!populate_info(index, out_info)) return TC_ERR_INTERNAL;
    return TC_OK;
#endif
}

extern "C" tc_status_t tc_hip_select_device(tc_context* ctx, int index) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (index < 0) return TC_ERR_INVALID_ARG;
#if !defined(TC_ENABLE_HIP)
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    std::lock_guard<std::mutex> lk(state_mutex());
    auto& s = state();
    if (!s.initialized || index >= s.device_count) return TC_ERR_INVALID_ARG;
    if (hipSetDevice(index) != hipSuccess) return TC_ERR_INTERNAL;
    s.selected_device = index;
    if (!populate_info(index, &s.info)) return TC_ERR_INTERNAL;
    return TC_OK;
#endif
}

namespace {
thread_local const char* g_last_kernel = "none";
}

extern "C" const char* tc_hip_last_kernel_name(void) {
    return g_last_kernel;
}

/* Internal symbol for sibling TUs (gemm.cpp, attention.cpp) to update. */
extern "C" TC_HIP_INTERNAL void tc_hip_set_last_kernel(const char* name) {
    g_last_kernel = name ? name : "unknown";
}

extern "C" TC_HIP_INTERNAL int tc_hip_runtime_initialized(void) {
#if !defined(TC_ENABLE_HIP)
    return 0;
#else
    std::lock_guard<std::mutex> lk(state_mutex());
    return state().initialized ? 1 : 0;
#endif
}

extern "C" TC_HIP_INTERNAL int tc_hip_is_active(void) {
#if !defined(TC_ENABLE_HIP)
    return 0;
#else
    if (hip_policy_disabled()) return 0;
    return tc_hip_runtime_initialized();
#endif
}
