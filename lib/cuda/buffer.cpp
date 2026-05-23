/*
 * tensorcore - CUDA managed-buffer storage hooks.
 *
 * Provides the small set of cross-TU symbols that lib/core/device_cpu.cpp's
 * tc_buffer_alloc/free queries to decide whether to back a buffer with
 * cudaMallocManaged (when a CUDA device is active) instead of plain
 * malloc. With managed memory, cuBLAS dereferences the user pointer
 * directly without host/device staging.
 *
 * Stubs to no-op on builds without TC_ENABLE_CUDA.
 */

#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

#if defined(TC_ENABLE_CUDA)
#  include <cuda_runtime.h>
#  include <cstdlib>
#endif

namespace {

#if defined(TC_ENABLE_CUDA)
extern "C" TC_CUDA_INTERNAL int tc_cuda_runtime_initialized(void);

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

bool cuda_policy_disabled(void) {
    if (env_true(std::getenv("TC_DISABLE_CUDA_GEMM"))) return true;
    if (env_false(std::getenv("TC_CUDA_GEMM"))) return true;
    if (env_false(std::getenv("TC_USE_CUDA_GEMM"))) return true;
    return false;
}
#endif

}  // namespace

/* CUDA-enabled builds auto-activate managed allocations once tc_cuda_init()
 * succeeds (tc_init attempts it on CUDA builds). TC_USE_CUDA_GEMM=1 remains
 * accepted for older scripts; TC_CUDA_GEMM=0 or TC_DISABLE_CUDA_GEMM=1 force
 * the host/CPU policy for debugging and A/B comparisons. */
extern "C" TC_CUDA_INTERNAL int tc_cuda_is_active(void) {
#if defined(TC_ENABLE_CUDA)
    if (cuda_policy_disabled()) return 0;
    return tc_cuda_runtime_initialized() ? 1 : 0;
#else
    return 0;
#endif
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_managed_alloc(size_t bytes, void** out_ptr) {
#if defined(TC_ENABLE_CUDA)
    if (!out_ptr) return -1;
    void* p = nullptr;
    if (cudaMallocManaged(&p, bytes, cudaMemAttachGlobal) != cudaSuccess) {
        return -1;
    }
    *out_ptr = p;
    return 0;
#else
    (void)bytes; (void)out_ptr;
    return -1;
#endif
}

extern "C" TC_CUDA_INTERNAL void tc_cuda_managed_free(void* ptr) {
#if defined(TC_ENABLE_CUDA)
    if (ptr) cudaFree(ptr);
#else
    (void)ptr;
#endif
}
