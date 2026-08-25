#include "tensorcore/capabilities.h"
#include "tensorcore/tensorcore.h"

#include <algorithm>
#include <cstring>

extern "C" int tc_cuda_is_active(void);
extern "C" int tc_hip_is_active(void);

static uint64_t compiled_backend_mask(void) {
    uint64_t mask = 0;

#if defined(TC_ENABLE_METAL)
    mask |= TC_BACKEND_MASK_SIMDGROUP_MATRIX |
            TC_BACKEND_MASK_MPS |
            TC_BACKEND_MASK_ACCELERATE_CPU |
            TC_BACKEND_MASK_METAL_COMPUTE;
#  if defined(TC_HAVE_METAL4_SDK)
    mask |= TC_BACKEND_MASK_TENSOROPS_M5;
#  endif
#else
    mask |= TC_BACKEND_MASK_PORTABLE_CPU;
#endif

#if defined(TC_ENABLE_CUDA)
    mask |= TC_BACKEND_MASK_CUDA;
#endif
#if defined(TC_ENABLE_HIPBLAS)
    mask |= TC_BACKEND_MASK_HIP;
#endif
    return mask;
}

static uint64_t available_backend_mask(const tc_device_info& info) {
    uint64_t mask = 0;

#if defined(TC_ENABLE_METAL)
    mask |= TC_BACKEND_MASK_MPS |
            TC_BACKEND_MASK_ACCELERATE_CPU |
            TC_BACKEND_MASK_METAL_COMPUTE;
    if (info.family >= TC_FAMILY_APPLE7) {
        mask |= TC_BACKEND_MASK_SIMDGROUP_MATRIX;
    }
#  if defined(TC_HAVE_METAL4_SDK)
    if (info.supports_tensorops_m5) {
        mask |= TC_BACKEND_MASK_TENSOROPS_M5;
    }
#  endif
#else
    (void)info;
    mask |= TC_BACKEND_MASK_PORTABLE_CPU;
#endif

#if defined(TC_ENABLE_CUDA)
    if (tc_cuda_is_active()) mask |= TC_BACKEND_MASK_CUDA;
#endif
#if defined(TC_ENABLE_HIPBLAS)
    if (tc_hip_is_active()) mask |= TC_BACKEND_MASK_HIP;
#endif
    return mask;
}

static uint64_t available_capability_mask(void) {
    uint64_t mask = TC_CAPABILITY_GEMM_F32 |
                    TC_CAPABILITY_GEMM_F16 |
                    TC_CAPABILITY_GEMM_BF16 |
                    TC_CAPABILITY_GEMM_I8 |
                    TC_CAPABILITY_DISTRIBUTED_SINGLE |
                    TC_CAPABILITY_DILOCO |
                    TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE |
                    TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME |
                    TC_CAPABILITY_REMOTE_TENSOR |
                    TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH |
                    TC_CAPABILITY_DILOCO_CAPABILITY_QUERY |
                    TC_CAPABILITY_DILOCO_STATE_ABI_V2;

    /* gloo_tcp.cpp provides its real socket implementation on these targets
     * and deterministic unsupported stubs elsewhere. */
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    mask |= TC_CAPABILITY_DISTRIBUTED_GLOO |
            TC_CAPABILITY_DILOCO_SPARSE_GLOO;
#endif

    /* Elastic membership and actual FP16 wire packing intentionally stay
     * absent until implemented and evidenced. */
    return mask;
}

extern "C" tc_status_t tc_runtime_capabilities_get(
    tc_context* ctx,
    uint32_t requested_abi_version,
    tc_runtime_capabilities* out,
    size_t out_size) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    if (requested_abi_version != TC_RUNTIME_CAPABILITIES_ABI_VERSION_1) {
        return TC_ERR_ABI_MISMATCH;
    }
    if (out_size < TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE) {
        return TC_ERR_INVALID_ARG;
    }

    tc_device_info info{};
    const tc_status_t info_status = tc_device_info_get(ctx, &info);
    if (info_status != TC_OK) return info_status;

    tc_runtime_capabilities capabilities{};
    capabilities.struct_size = (uint32_t)sizeof(capabilities);
    capabilities.abi_version = TC_RUNTIME_CAPABILITIES_ABI_VERSION_1;
    capabilities.runtime_version_major = TENSORCORE_VERSION_MAJOR;
    capabilities.runtime_version_minor = TENSORCORE_VERSION_MINOR;
    capabilities.runtime_version_patch = TENSORCORE_VERSION_PATCH;
    capabilities.known_capability_mask = TC_CAPABILITY_V1_KNOWN_MASK;
    capabilities.available_capability_mask = available_capability_mask();
    capabilities.compiled_backend_mask = compiled_backend_mask();
    capabilities.available_backend_mask = available_backend_mask(info);

    const size_t copy_size = std::min(out_size, sizeof(capabilities));
    std::memcpy(out, &capabilities, copy_size);
    return TC_OK;
}
