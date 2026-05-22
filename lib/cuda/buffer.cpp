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

/* Gate managed allocations on the same env var the user already sets to
 * opt into CUDA GEMM dispatch. Keeps the surface coherent: if the user
 * has chosen CUDA for compute, buffers go straight to managed memory. */
extern "C" TC_CUDA_INTERNAL int tc_cuda_is_active(void) {
#if defined(TC_ENABLE_CUDA)
    const char* env = std::getenv("TC_USE_CUDA_GEMM");
    return (env && env[0] == '1') ? 1 : 0;
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
