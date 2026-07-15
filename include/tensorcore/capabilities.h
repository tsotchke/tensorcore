#ifndef TENSORCORE_CAPABILITIES_H
#define TENSORCORE_CAPABILITIES_H

/*
 * Versioned runtime capability discovery.
 *
 * This is the stable feature-test surface for downstream adapters. Consumers
 * must not infer support from the package version, invent weak helper symbols,
 * or treat an unknown bit as available. A capability is usable only when it
 * appears in both known_capability_mask and available_capability_mask.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorcore/device.h"
#include "tensorcore/gemm.h"
#include "tensorcore/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TC_RUNTIME_CAPABILITIES_ABI_VERSION_1       UINT32_C(1)
#define TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT \
    TC_RUNTIME_CAPABILITIES_ABI_VERSION_1

/* Production capability bits. Bits are append-only within this ABI. */
#define TC_CAPABILITY_GEMM_F32 \
    (UINT64_C(1) << 0)
#define TC_CAPABILITY_DISTRIBUTED_SINGLE \
    (UINT64_C(1) << 1)
#define TC_CAPABILITY_DISTRIBUTED_GLOO \
    (UINT64_C(1) << 2)
#define TC_CAPABILITY_DILOCO \
    (UINT64_C(1) << 3)
#define TC_CAPABILITY_DILOCO_SPARSE_GLOO \
    (UINT64_C(1) << 4)
#define TC_CAPABILITY_REMOTE_TENSOR \
    (UINT64_C(1) << 5)

/* Known campaign capabilities. Availability is reported independently as
 * each safety and runtime-evidence gate passes; exposing all bits here lets
 * consumers distinguish "known but unavailable" from an unknown feature. */
#define TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE \
    (UINT64_C(1) << 6)
#define TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME \
    (UINT64_C(1) << 7)
#define TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP \
    (UINT64_C(1) << 8)
#define TC_CAPABILITY_DILOCO_FP16_WIRE \
    (UINT64_C(1) << 9)
#define TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH \
    (UINT64_C(1) << 10)

#define TC_CAPABILITY_V1_KNOWN_MASK ( \
    TC_CAPABILITY_GEMM_F32 | \
    TC_CAPABILITY_DISTRIBUTED_SINGLE | \
    TC_CAPABILITY_DISTRIBUTED_GLOO | \
    TC_CAPABILITY_DILOCO | \
    TC_CAPABILITY_DILOCO_SPARSE_GLOO | \
    TC_CAPABILITY_REMOTE_TENSOR | \
    TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE | \
    TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME | \
    TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP | \
    TC_CAPABILITY_DILOCO_FP16_WIRE | \
    TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH)

/* Backend masks use tc_backend_t values as stable bit positions. */
#define TC_BACKEND_MASK_SIMDGROUP_MATRIX \
    (UINT64_C(1) << TC_BACKEND_SIMDGROUP_MATRIX)
#define TC_BACKEND_MASK_TENSOROPS_M5 \
    (UINT64_C(1) << TC_BACKEND_TENSOROPS_M5)
#define TC_BACKEND_MASK_MPS \
    (UINT64_C(1) << TC_BACKEND_MPS)
#define TC_BACKEND_MASK_ACCELERATE_CPU \
    (UINT64_C(1) << TC_BACKEND_ACCELERATE_CPU)
#define TC_BACKEND_MASK_PORTABLE_CPU \
    (UINT64_C(1) << TC_BACKEND_PORTABLE_CPU)
#define TC_BACKEND_MASK_METAL_COMPUTE \
    (UINT64_C(1) << TC_BACKEND_METAL_COMPUTE)
#define TC_BACKEND_MASK_CUDA \
    (UINT64_C(1) << TC_BACKEND_CUDA)
#define TC_BACKEND_MASK_HIP \
    (UINT64_C(1) << TC_BACKEND_HIP)

typedef struct {
    /* Size of the runtime's structure, not the caller's buffer. */
    uint32_t struct_size;
    /* ABI version actually written. */
    uint32_t abi_version;

    uint32_t runtime_version_major;
    uint32_t runtime_version_minor;
    uint32_t runtime_version_patch;
    uint32_t reserved0;

    /* known & available is the only supported feature-test result. */
    uint64_t known_capability_mask;
    uint64_t available_capability_mask;

    /* Compiled means code is present. Available means this context can select
     * the backend now under current hardware and runtime policy. */
    uint64_t compiled_backend_mask;
    uint64_t available_backend_mask;

    /* Must be zero. Reserved for append-only ABI growth. */
    uint64_t reserved[4];
} tc_runtime_capabilities;

/* The v1 prefix through available_capability_mask. Older v1 callers may pass
 * exactly this size; newer runtimes copy only the prefix that fits. */
#define TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE \
    (offsetof(tc_runtime_capabilities, available_capability_mask) + \
     sizeof(((tc_runtime_capabilities*)0)->available_capability_mask))

/* Query runtime capabilities for ctx.
 *
 * requested_abi_version must be a version implemented by the runtime. A
 * newer/unknown version returns TC_ERR_ABI_MISMATCH without modifying out.
 * out_size may be between TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE and any larger
 * value. The runtime copies at most sizeof(tc_runtime_capabilities), leaving a
 * newer caller's tail untouched. This makes old/new struct-size behavior
 * deterministic in both directions.
 */
tc_status_t tc_runtime_capabilities_get(
    tc_context* ctx,
    uint32_t requested_abi_version,
    tc_runtime_capabilities* out,
    size_t out_size);

/* Header-only fail-closed helper; it adds no exported ABI symbol. */
static inline bool tc_runtime_capability_available(
    const tc_runtime_capabilities* capabilities,
    uint64_t capability) {
    return capabilities != NULL && capability != 0 &&
           (capabilities->known_capability_mask & capability) == capability &&
           (capabilities->available_capability_mask & capability) == capability;
}

#ifdef __cplusplus
}
#endif
#endif /* TENSORCORE_CAPABILITIES_H */
