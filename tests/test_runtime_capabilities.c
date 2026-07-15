#include "tensorcore/tensorcore.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(const char* message) {
    fprintf(stderr, "runtime capabilities: %s\n", message);
    return 1;
}

static int all_bytes_are(const void* data, size_t size, unsigned char value) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < size; ++i) {
        if (bytes[i] != value) return 0;
    }
    return 1;
}

int main(void) {
    tc_context* ctx = NULL;
    const tc_status_t init_status = tc_init(&ctx);
    if ((init_status != TC_OK && init_status != TC_ERR_ALREADY_INITIALIZED) ||
        !ctx) {
        return fail("tc_init failed");
    }

    int rc = 0;
    tc_runtime_capabilities capabilities;
    memset(&capabilities, 0xa5, sizeof(capabilities));
    if (tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT,
            &capabilities,
            sizeof(capabilities)) != TC_OK) {
        rc |= fail("v1 query failed");
        goto cleanup;
    }

    if (capabilities.struct_size != sizeof(tc_runtime_capabilities) ||
        capabilities.abi_version != TC_RUNTIME_CAPABILITIES_ABI_VERSION_1) {
        rc |= fail("runtime did not report the v1 size/version");
    }
    if (capabilities.runtime_version_major != TENSORCORE_VERSION_MAJOR ||
        capabilities.runtime_version_minor != TENSORCORE_VERSION_MINOR ||
        capabilities.runtime_version_patch != TENSORCORE_VERSION_PATCH) {
        rc |= fail("runtime package version does not match the public header");
    }
    if (capabilities.known_capability_mask != TC_CAPABILITY_V1_KNOWN_MASK) {
        rc |= fail("known capability mask drifted from the v1 contract");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_GEMM_F32)) {
        rc |= fail("the portable public fp32 GEMM path is unavailable");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DISTRIBUTED_SINGLE)) {
        rc |= fail("the single-rank distributed path is unavailable");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DILOCO)) {
        rc |= fail("the implemented DiLoCo baseline is unavailable");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE)) {
        rc |= fail("snapshot-safe async DiLoCo is unavailable");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME)) {
        rc |= fail("versioned DiLoCo checkpoint state is unavailable");
    }
    if (!tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH)) {
        rc |= fail("versioned transport identity authentication is unavailable");
    }
    if (tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP) ||
        tc_runtime_capability_available(
            &capabilities, TC_CAPABILITY_DILOCO_FP16_WIRE)) {
        rc |= fail("an unimplemented DiLoCo capability was advertised");
    }
    if (tc_runtime_capability_available(
            &capabilities, UINT64_C(1) << 63)) {
        rc |= fail("an unknown future capability did not fail closed");
    }
    if (capabilities.available_backend_mask == 0 ||
        (capabilities.available_backend_mask &
         ~capabilities.compiled_backend_mask) != 0) {
        rc |= fail("active backend mask is empty or not a compiled subset");
    }
    for (size_t i = 0; i < sizeof(capabilities.reserved) /
                                sizeof(capabilities.reserved[0]); ++i) {
        if (capabilities.reserved[i] != 0) {
            rc |= fail("reserved capability fields are not zero");
            break;
        }
    }

    /* Older v1 caller: the runtime must accept the documented prefix and
     * report its full size without writing beyond the supplied bytes. */
    union {
        uint64_t alignment;
        unsigned char bytes[TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE];
    } prefix;
    memset(&prefix, 0x5a, sizeof(prefix));
    if (tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_1,
            (tc_runtime_capabilities*)prefix.bytes,
            sizeof(prefix.bytes)) != TC_OK) {
        rc |= fail("minimum-size v1 caller was rejected");
    } else {
        tc_runtime_capabilities prefix_copy;
        memset(&prefix_copy, 0, sizeof(prefix_copy));
        memcpy(&prefix_copy, prefix.bytes, sizeof(prefix.bytes));
        if (prefix_copy.struct_size != sizeof(tc_runtime_capabilities) ||
            !tc_runtime_capability_available(
                &prefix_copy, TC_CAPABILITY_GEMM_F32)) {
            rc |= fail("minimum-size v1 prefix was not populated correctly");
        }
    }

    /* Newer caller with a larger v1 buffer: its unknown tail must survive. */
    struct {
        tc_runtime_capabilities v1;
        unsigned char future_tail[32];
    } future;
    memset(&future, 0x6b, sizeof(future));
    if (tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_1,
            &future.v1,
            sizeof(future)) != TC_OK) {
        rc |= fail("larger v1 caller buffer was rejected");
    } else if (!all_bytes_are(
                   future.future_tail, sizeof(future.future_tail), 0x6b)) {
        rc |= fail("runtime overwrote a newer caller's unknown tail");
    }

    /* Unknown ABI versions fail without partially rewriting the output. */
    memset(&capabilities, 0x7c, sizeof(capabilities));
    if (tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT + 1,
            &capabilities,
            sizeof(capabilities)) != TC_ERR_ABI_MISMATCH) {
        rc |= fail("future ABI version did not return TC_ERR_ABI_MISMATCH");
    } else if (!all_bytes_are(&capabilities, sizeof(capabilities), 0x7c)) {
        rc |= fail("ABI mismatch modified the caller's output");
    }

    memset(&capabilities, 0x3d, sizeof(capabilities));
    if (tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_1,
            &capabilities,
            TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE - 1) != TC_ERR_INVALID_ARG) {
        rc |= fail("undersized v1 buffer was not rejected");
    } else if (!all_bytes_are(&capabilities, sizeof(capabilities), 0x3d)) {
        rc |= fail("undersized query modified the caller's output");
    }

    if (tc_runtime_capabilities_get(
            NULL,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_1,
            &capabilities,
            sizeof(capabilities)) != TC_ERR_INVALID_ARG ||
        tc_runtime_capabilities_get(
            ctx,
            TC_RUNTIME_CAPABILITIES_ABI_VERSION_1,
            NULL,
            sizeof(capabilities)) != TC_ERR_INVALID_ARG) {
        rc |= fail("null arguments were not rejected");
    }
    if (strcmp(tc_status_string(TC_ERR_ABI_MISMATCH),
               "ABI version mismatch") != 0) {
        rc |= fail("ABI mismatch status text is missing");
    }
    if (strcmp(tc_status_string(TC_ERR_AUTH),
               "transport authentication failed") != 0) {
        rc |= fail("transport authentication status text is missing");
    }

cleanup:
    if (tc_shutdown(ctx) != TC_OK) rc |= fail("tc_shutdown failed");
    if (rc == 0) {
        puts("runtime capabilities: positive, unavailable, and mixed-version behavior OK");
    }
    return rc;
}
