/*
 * tensorcore — direct CUDA backend, device init + enumeration.
 *
 * Gated on TC_ENABLE_CUDA=ON. When CUDA is unavailable (no toolkit on
 * host, or no NVIDIA device at runtime), this TU compiles to stubs that
 * return TC_ERR_UNSUPPORTED_FAMILY — the public dispatch then falls
 * through to whichever next backend can serve the call.
 */

#include "tensorcore/cuda.h"
#include "tensorcore/tensorcore.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

#if defined(TC_ENABLE_CUDA)
#  include <cuda_runtime.h>
#endif

namespace {

#if defined(TC_ENABLE_CUDA)

struct CudaState {
    int  device_count = 0;
    int  selected_device = -1;
    tc_cuda_device_info info = {};
    bool initialized = false;
};

std::mutex& state_mutex() { static std::mutex m; return m; }
CudaState& state() { static CudaState s; return s; }

bool populate_info(int index, tc_cuda_device_info* info) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) return false;
    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->device_name, prop.name, sizeof(info->device_name) - 1);
    std::snprintf(info->compute_capability, sizeof(info->compute_capability),
                  "%d.%d", prop.major, prop.minor);
    info->major = prop.major;
    info->minor = prop.minor;
    info->global_memory_bytes = prop.totalGlobalMem;
    info->shared_memory_per_block = prop.sharedMemPerBlock;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->warp_size = prop.warpSize;
    /* Feature gating by compute capability — these are stable defaults. */
    info->supports_fp16            = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);
    info->supports_bf16            = (prop.major >= 8);
    info->supports_int8_tensor_core = (prop.major > 7) || (prop.major == 7 && prop.minor >= 2);
    info->supports_tf32            = (prop.major >= 8);
    info->unified_memory           = (prop.integrated != 0);
    return true;
}

#endif  /* TC_ENABLE_CUDA */

thread_local const char* g_last_kernel = "none";

}  // namespace

extern "C" tc_status_t tc_cuda_init(tc_context* ctx) {
#if !defined(TC_ENABLE_CUDA)
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(state_mutex());
    auto& s = state();
    if (s.initialized) return TC_OK;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return TC_ERR_UNSUPPORTED_FAMILY;
    }
    if (cudaSetDevice(0) != cudaSuccess) return TC_ERR_INTERNAL;
    if (!populate_info(0, &s.info)) return TC_ERR_INTERNAL;
    s.device_count = count;
    s.selected_device = 0;
    s.initialized = true;
    return TC_OK;
#endif
}

extern "C" int tc_cuda_device_count(void) {
#if !defined(TC_ENABLE_CUDA)
    return 0;
#else
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#endif
}

extern "C" tc_status_t tc_cuda_device_at(int index, tc_cuda_device_info* out_info) {
    if (index < 0 || !out_info) return TC_ERR_INVALID_ARG;
#if !defined(TC_ENABLE_CUDA)
    std::memset(out_info, 0, sizeof(*out_info));
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!populate_info(index, out_info)) return TC_ERR_INVALID_ARG;
    return TC_OK;
#endif
}

extern "C" tc_status_t tc_cuda_select_device(tc_context* ctx, int index) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (index < 0) return TC_ERR_INVALID_ARG;
#if !defined(TC_ENABLE_CUDA)
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    std::lock_guard<std::mutex> lk(state_mutex());
    auto& s = state();
    if (!s.initialized || index >= s.device_count) return TC_ERR_INVALID_ARG;
    if (cudaSetDevice(index) != cudaSuccess) return TC_ERR_INTERNAL;
    s.selected_device = index;
    if (!populate_info(index, &s.info)) return TC_ERR_INTERNAL;
    return TC_OK;
#endif
}

extern "C" const char* tc_cuda_last_kernel_name(void) {
    return g_last_kernel;
}

extern "C" TC_CUDA_INTERNAL void tc_cuda_set_last_kernel(const char* name) {
    g_last_kernel = name ? name : "unknown";
}
