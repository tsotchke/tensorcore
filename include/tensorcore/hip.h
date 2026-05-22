#ifndef TENSORCORE_HIP_H
#define TENSORCORE_HIP_H

/*
 * tensorcore — HIP backend (vendor-neutral GPU compute via chipStar).
 *
 * chipStar compiles HIP/CUDA source to SPIR-V and runs on:
 *   - Intel GPUs via Level Zero (Aurora supercomputer, Intel Arc, iGPUs)
 *   - NVIDIA GPUs via OpenCL (RTX 30/40-series, H100, A100, ...)
 *   - AMD GPUs via OpenCL (Vega, RDNA, MI100/250/300, ...)
 *   - ARM Mali via OpenCL (Jetson, mobile, embedded)
 *
 * Apple GPUs use the Metal backend; everything else uses HIP+chipStar.
 *
 * The runtime dispatch decision:
 *
 *   if   APPLE && Metal device available     → Metal backend
 *   elif chipStar HIP runtime + SPIR-V GPU   → HIP backend
 *   elif CUDA toolkit + NVIDIA device        → CUDA backend (optional fast path)
 *   else                                      → CPU SIMD backend
 *
 * This header declares the HIP-backend-specific entry points. The
 * portable C ABI (tc_gemm, tc_attention_forward, ...) doesn't change;
 * the dispatch picks the right backend per call.
 *
 * Build gate: TC_ENABLE_HIP=ON in CMake. Auto-detected when chipStar
 * (or AMD's standard HIP) is found on the host. Off on Apple by default.
 */

#include <stdint.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* HIP-side device characterization, parallel to tc_device_info but for the
 * SPIR-V/HIP target. The dispatch layer uses this to pick the right kernel
 * variant per device family. */
typedef enum {
    TC_HIP_VENDOR_UNKNOWN  = 0,
    TC_HIP_VENDOR_INTEL    = 1,   /* Level Zero: Arc, Ponte Vecchio, iGPU */
    TC_HIP_VENDOR_NVIDIA   = 2,   /* OpenCL via NVIDIA driver: 30/40/H100 */
    TC_HIP_VENDOR_AMD      = 3,   /* OpenCL via amdgpu: Vega/RDNA/CDNA */
    TC_HIP_VENDOR_ARM_MALI = 4,   /* OpenCL on Mali / Tegra integrated */
} tc_hip_vendor_t;

typedef struct {
    tc_hip_vendor_t vendor;
    char            device_name[128];   /* e.g. "NVIDIA GeForce RTX 3090" */
    char            driver_version[64];
    char            opencl_version[64]; /* "OpenCL 3.0 CUDA" etc. */
    uint64_t        global_memory_bytes;
    uint64_t        local_memory_bytes;   /* per-CU, like Apple's threadgroup memory */
    uint32_t        compute_units;
    uint32_t        max_workgroup_size;
    uint32_t        preferred_subgroup_size; /* equivalent of Apple's simdgroup width */
    bool            supports_fp16;
    bool            supports_fp64;
    bool            supports_int8_dot;     /* DP4A on NVIDIA, similar on others */
    bool            unified_memory;        /* true on iGPUs + Tegra; false on dGPUs */
} tc_hip_device_info;

/* Initialize the HIP backend. Returns TC_ERR_UNSUPPORTED_FAMILY if no
 * chipStar/HIP runtime is available, or if no SPIR-V GPU device is found.
 * On success the HIP context is owned by the global tensorcore context;
 * tc_init must have been called first. */
tc_status_t tc_hip_init(tc_context* ctx);

/* Query the HIP device info. Only valid after tc_hip_init has succeeded. */
tc_status_t tc_hip_device_info_get(tc_context* ctx, tc_hip_device_info* out_info);

/* List available HIP devices (for multi-GPU dispatch). 0..count-1 are
 * valid indexes for tc_hip_select_device. */
int  tc_hip_device_count(void);
tc_status_t tc_hip_device_at(int index, tc_hip_device_info* out_info);
tc_status_t tc_hip_select_device(tc_context* ctx, int index);

/* The standard tc_gemm / tc_attention_forward / etc. dispatch into this
 * backend when no Metal device is available. The HIP-specific entry points
 * below are reserved for diagnostics / explicit overrides; user code
 * shouldn't need them. */

/* Per-thread diagnostic: which HIP kernel name served the last call. */
const char* tc_hip_last_kernel_name(void);

#ifdef __cplusplus
}
#endif
#endif
