#ifndef TENSORCORE_CUDA_H
#define TENSORCORE_CUDA_H

/*
 * tensorcore — direct CUDA backend (NVIDIA-native fast path).
 *
 * Why this exists alongside the HIP/chipStar backend:
 *
 *   chipStar's HIP→SPIR-V→OpenCL/Level Zero pipeline gives vendor-neutral
 *   GPU compute on Intel, AMD, and ARM. NVIDIA's OpenCL driver does not
 *   ingest SPIR-V, so chipStar can't reach NVIDIA hardware. The clean
 *   answer is a direct CUDA backend: tensorcore's tc_gemm / tc_attention_*
 *   route into cuBLAS / cuDNN / native CUDA kernels when an NVIDIA device
 *   is present and the user opts in.
 *
 *   This is the *fast path* for NVIDIA — full hardware tensor-core access,
 *   PTX-level kernel control, no SPIR-V translation overhead. The
 *   substrate's architecture intentionally treats CUDA as one backend
 *   among several: required for production NVIDIA use, optional in the
 *   sense that nothing else depends on it.
 *
 * Build gate: TC_ENABLE_CUDA=ON in CMake. Requires CUDA Toolkit >= 11.0
 * and a recent driver. Off by default; auto-detected when nvcc is on
 * PATH at configure time.
 *
 * Runtime dispatch:
 *
 *   if   APPLE && Metal device              → Metal backend
 *   elif CUDA toolkit + NVIDIA device       → CUDA backend         (this header)
 *   elif chipStar HIP + SPIR-V GPU          → HIP backend
 *   else                                     → CPU SIMD backend
 *
 * The CUDA backend is therefore selected automatically when an NVIDIA
 * device is the best available match; the user doesn't have to pick.
 */

#include <stdint.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     device_name[128];
    char     compute_capability[16];   /* e.g. "8.6" for Ampere */
    int      major;                    /* compute capability major */
    int      minor;
    uint64_t global_memory_bytes;
    uint64_t shared_memory_per_block;
    uint32_t multiprocessor_count;
    uint32_t max_threads_per_block;
    uint32_t warp_size;                /* always 32 on current NVIDIA */
    bool     supports_fp16;            /* true on sm_53+ (Maxwell mobile / Pascal+) */
    bool     supports_bf16;            /* true on sm_80+ (A100, H100, RTX 30/40) */
    bool     supports_int8_tensor_core;/* true on sm_72+ */
    bool     supports_tf32;            /* true on sm_80+ */
    bool     unified_memory;           /* true on Tegra (Jetson) */
} tc_cuda_device_info;

/* Initialize the CUDA backend. Returns TC_ERR_UNSUPPORTED_FAMILY if no
 * CUDA toolkit or NVIDIA device is available. On success, attaches a
 * CUDA context, cuBLAS handle, and cuDNN handle (if linked) to ctx. */
tc_status_t tc_cuda_init(tc_context* ctx);

/* Number of CUDA devices visible to the process. 0 on non-CUDA hosts. */
int tc_cuda_device_count(void);

/* Query info about device `index`. Doesn't require tc_cuda_init. */
tc_status_t tc_cuda_device_at(int index, tc_cuda_device_info* out_info);

/* Switch the active CUDA device. cuBLAS handle is recreated. */
tc_status_t tc_cuda_select_device(tc_context* ctx, int index);

/* Diagnostic: which kernel name served the last call. */
const char* tc_cuda_last_kernel_name(void);

#ifdef __cplusplus
}
#endif
#endif
