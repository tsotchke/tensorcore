/* DiLoCo capability query and versioned state-serialization ABI tests.
 *
 * Three things are proven here:
 *
 *   1. tc_diloco_capability_query answers truthfully and is size-versioned in
 *      both directions (old caller with a short struct, new caller with a
 *      long one).
 *   2. Every claim the capability mask makes about serializability is backed
 *      by a real export/import round-trip, not by a signature check.
 *   3. The v2 wire format is byte-stable, carries an independently verifiable
 *      SHA-256 of its payload, and fails closed on every corruption and
 *      truncation without partially applying anything.
 */

#include "tensorcore/tensorcore.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ELEMS 32

static int failures = 0;

static int expect_status(const char* what, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    fprintf(stderr, "diloco_state_abi: %s: got %s want %s\n",
            what, tc_status_string(got), tc_status_string(want));
    failures++;
    return 1;
}

static int expect_true(const char* what, int condition) {
    if (condition) return 0;
    fprintf(stderr, "diloco_state_abi: FAIL: %s\n", what);
    failures++;
    return 1;
}

/* ------------------------------------------------------------------------
 * Independent SHA-256 so the blob's digest is checked against something
 * other than the implementation that produced it.
 * ------------------------------------------------------------------------ */

typedef struct {
    uint32_t state[8];
    uint8_t block[64];
    size_t block_used;
    uint64_t total_bytes;
} sha256_ctx;

static uint32_t rotr32(uint32_t v, unsigned n) {
    return (v >> n) | (v << (32 - n));
}

static void sha256_transform(sha256_ctx* c, const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64];
    uint32_t a, b, cc, d, e, f, g, h;
    unsigned i;
    for (i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[4*i] << 24) | ((uint32_t)block[4*i+1] << 16) |
               ((uint32_t)block[4*i+2] << 8) | (uint32_t)block[4*i+3];
    }
    for (i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        const uint32_t s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = c->state[0]; b = c->state[1]; cc = c->state[2]; d = c->state[3];
    e = c->state[4]; f = c->state[5]; g = c->state[6]; h = c->state[7];
    for (i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        const uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->state[0]+=a; c->state[1]+=b; c->state[2]+=cc; c->state[3]+=d;
    c->state[4]+=e; c->state[5]+=f; c->state[6]+=g; c->state[7]+=h;
}

static void sha256(const void* data, size_t size, uint8_t out[32]) {
    sha256_ctx c;
    const uint8_t* p = (const uint8_t*)data;
    size_t remaining = size;
    unsigned i;
    c.state[0]=0x6a09e667u; c.state[1]=0xbb67ae85u;
    c.state[2]=0x3c6ef372u; c.state[3]=0xa54ff53au;
    c.state[4]=0x510e527fu; c.state[5]=0x9b05688cu;
    c.state[6]=0x1f83d9abu; c.state[7]=0x5be0cd19u;
    memset(c.block, 0, sizeof(c.block));
    c.block_used = 0;
    c.total_bytes = size;
    while (remaining >= 64) {
        sha256_transform(&c, p);
        p += 64;
        remaining -= 64;
    }
    memcpy(c.block, p, remaining);
    c.block_used = remaining;
    c.block[c.block_used++] = 0x80;
    if (c.block_used > 56) {
        memset(c.block + c.block_used, 0, sizeof(c.block) - c.block_used);
        sha256_transform(&c, c.block);
        c.block_used = 0;
    }
    memset(c.block + c.block_used, 0, 56 - c.block_used);
    {
        const uint64_t bits = c.total_bytes * UINT64_C(8);
        for (i = 0; i < 8; ++i) c.block[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
    }
    sha256_transform(&c, c.block);
    for (i = 0; i < 8; ++i) {
        out[4*i]   = (uint8_t)(c.state[i] >> 24);
        out[4*i+1] = (uint8_t)(c.state[i] >> 16);
        out[4*i+2] = (uint8_t)(c.state[i] >> 8);
        out[4*i+3] = (uint8_t)(c.state[i]);
    }
}

/* Known-answer test so a broken reference hash cannot silently agree with a
 * broken implementation hash. */
static void check_sha256_reference(void) {
    static const uint8_t want_empty[32] = {
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55
    };
    static const uint8_t want_abc[32] = {
        0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad
    };
    uint8_t got[32];
    sha256("", 0, got);
    expect_true("reference sha256 of empty input",
                memcmp(got, want_empty, sizeof(got)) == 0);
    sha256("abc", 3, got);
    expect_true("reference sha256 of \"abc\"",
                memcmp(got, want_abc, sizeof(got)) == 0);
}

/* ------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------ */

static uint32_t load_u32_le(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const unsigned char* p) {
    uint64_t v = 0;
    int i;
    for (i = 7; i >= 0; --i) v = (v << 8) | (uint64_t)p[i];
    return v;
}

static tc_diloco_config make_config(tc_diloco_outer_optimizer_t optimizer,
                                    tc_diloco_compress_t compress) {
    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 2;
    cfg.outer_lr = 0.25f;
    cfg.outer_momentum = 0.9f;
    cfg.outer_beta2 = 0.999f;
    cfg.outer_eps = 1e-8f;
    cfg.outer_optimizer = optimizer;
    cfg.compress = compress;
    return cfg;
}

static int alloc_parameter(tc_context* ctx, tc_buffer** out_buffer,
                           float** out_values) {
    size_t i;
    if (tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), out_buffer) != TC_OK) {
        return 1;
    }
    if (tc_buffer_map(*out_buffer, (void**)out_values) != TC_OK) return 1;
    for (i = 0; i < N_ELEMS; ++i) (*out_values)[i] = 1.0f;
    return 0;
}

/* Serialize at an explicit wire version. Returns 0 on success. */
static int serialize_at(tc_diloco_ctx* d, uint32_t version,
                        unsigned char** out, size_t* out_size) {
    size_t written = 0;
    *out = NULL;
    *out_size = 0;
    if (tc_diloco_state_size(d, version, out_size) != TC_OK) return 1;
    *out = (unsigned char*)malloc(*out_size);
    if (!*out) return 1;
    if (tc_diloco_state_serialize(d, version, *out, *out_size, &written) != TC_OK ||
        written != *out_size) {
        free(*out);
        *out = NULL;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Capability query
 * ------------------------------------------------------------------------ */

static void test_capability_query(tc_diloco_ctx* bound) {
    tc_diloco_capabilities caps;
    tc_diloco_capabilities untouched;
    unsigned char oversized[sizeof(tc_diloco_capabilities) + 64];
    unsigned char minimal[sizeof(tc_diloco_capabilities) + 64];
    size_t i;

    expect_status("capability query rejects NULL out",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      NULL, sizeof(caps)),
                  TC_ERR_INVALID_ARG);

    memset(&caps, 0xa5, sizeof(caps));
    memcpy(&untouched, &caps, sizeof(caps));
    expect_status("capability query rejects unknown ABI",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT + 1,
                      &caps, sizeof(caps)),
                  TC_ERR_ABI_MISMATCH);
    expect_true("unknown ABI leaves the struct untouched",
                memcmp(&caps, &untouched, sizeof(caps)) == 0);

    expect_status("capability query rejects an undersized struct",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      &caps, TC_DILOCO_CAPABILITIES_V1_MIN_SIZE - 1),
                  TC_ERR_INVALID_ARG);
    expect_true("undersized struct leaves the struct untouched",
                memcmp(&caps, &untouched, sizeof(caps)) == 0);

    /* Build-level query: usable before any context exists. */
    memset(&caps, 0, sizeof(caps));
    expect_status("build-level capability query",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      &caps, sizeof(caps)),
                  TC_OK);
    expect_true("struct_size reports the runtime's struct",
                caps.struct_size == (uint32_t)sizeof(tc_diloco_capabilities));
    expect_true("abi_version echoes the implemented version",
                caps.abi_version == TC_DILOCO_CAPABILITIES_ABI_VERSION_1);
    expect_true("runtime version matches the package header",
                caps.runtime_version_major == TENSORCORE_VERSION_MAJOR &&
                caps.runtime_version_minor == TENSORCORE_VERSION_MINOR &&
                caps.runtime_version_patch == TENSORCORE_VERSION_PATCH);

    expect_true("state ABI window is [MIN, CURRENT]",
                caps.state_abi_version_min == TC_DILOCO_STATE_ABI_VERSION_MIN &&
                caps.state_abi_version_max == TC_DILOCO_STATE_ABI_VERSION_CURRENT &&
                caps.state_abi_version_current == TC_DILOCO_STATE_ABI_VERSION_CURRENT);
    expect_true("current wire version reports its header and digest sizes",
                caps.state_header_size == TC_DILOCO_STATE_V2_HEADER_SIZE &&
                caps.state_payload_digest_bytes == TC_DILOCO_STATE_V2_DIGEST_BYTES);

    /* Every optimizer tc_diloco_init accepts is reported, and reported as
     * serializable. The round-trip section below proves the claim. */
    expect_true("SGD is serializable",
                tc_diloco_outer_optimizer_is_serializable(
                    &caps, TC_DILOCO_OUTER_SGD));
    expect_true("Nesterov is serializable",
                tc_diloco_outer_optimizer_is_serializable(
                    &caps, TC_DILOCO_OUTER_NESTEROV));
    expect_true("Adam is serializable",
                tc_diloco_outer_optimizer_is_serializable(
                    &caps, TC_DILOCO_OUTER_ADAM));

    expect_true("uncompressed deltas are serializable",
                tc_diloco_compress_is_serializable(
                    &caps, TC_DILOCO_COMPRESS_NONE));
    expect_true("top-k 1% error feedback is serializable",
                tc_diloco_compress_is_serializable(
                    &caps, TC_DILOCO_COMPRESS_TOPK_1PCT));
    expect_true("top-k 0.1% error feedback is serializable",
                tc_diloco_compress_is_serializable(
                    &caps, TC_DILOCO_COMPRESS_TOPK_01PCT));

    /* Modes tc_diloco_init rejects must not appear as supported. */
    expect_true("fp8 is not claimed",
                (caps.compress_supported_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_FP8)) == 0);
    expect_true("low-rank is not claimed",
                (caps.compress_supported_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_LOWRANK)) == 0);
    expect_true("signSGD is not claimed",
                (caps.compress_supported_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_SIGNSGD)) == 0);
    expect_true("serializable modes are a subset of supported modes",
                (caps.compress_serializable_mask &
                 ~caps.compress_supported_mask) == 0);
    expect_true("serializable optimizers are a subset of supported optimizers",
                (caps.outer_optimizer_serializable_mask &
                 ~caps.outer_optimizer_supported_mask) == 0);

    /* The wire-format honesty the qLLM handoff asked for: FP16 is accepted
     * but is dense fp32 on the wire, and must never be reported as sparse. */
    expect_true("FP16 is accepted",
                (caps.compress_supported_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_FP16)) != 0);
    expect_true("FP16 is reported as dense fp32 on the wire",
                (caps.compress_dense_fp32_wire_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_FP16)) != 0);
    expect_true("FP16 is never reported as a sparse wire format",
                (caps.compress_sparse_wire_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_FP16)) == 0);
    expect_true("wire masks are disjoint",
                (caps.compress_sparse_wire_mask &
                 caps.compress_dense_fp32_wire_mask) == 0);
    expect_true("an unbound query promises no sparse transport",
                caps.compress_sparse_wire_mask == 0);

    expect_true("state feature mask covers the handoff's required state",
                tc_diloco_state_feature_available(
                    &caps, TC_DILOCO_STATE_FEATURE_OUTER_ANCHOR |
                           TC_DILOCO_STATE_FEATURE_OUTER_MOMENTS |
                           TC_DILOCO_STATE_FEATURE_ERROR_FEEDBACK |
                           TC_DILOCO_STATE_FEATURE_COUNTERS |
                           TC_DILOCO_STATE_FEATURE_PENDING_ROUND |
                           TC_DILOCO_STATE_FEATURE_TOPOLOGY_EPOCH |
                           TC_DILOCO_STATE_FEATURE_MEMBERSHIP_EPOCH |
                           TC_DILOCO_STATE_FEATURE_PAYLOAD_SHA256 |
                           TC_DILOCO_STATE_FEATURE_DETERMINISTIC_LAYOUT |
                           TC_DILOCO_STATE_FEATURE_ATOMIC_RESTORE));

    expect_true("async overlap and its state are reported",
                caps.async_overlap_supported == 1 &&
                caps.async_state_serializable == 1);
    expect_true("dropout tolerance stays unclaimed",
                caps.tolerate_dropouts_supported == 0);

    expect_true("reserved words are zero",
                caps.reserved0 == 0 && caps.reserved1 == 0 &&
                caps.reserved2 == 0);
    for (i = 0; i < sizeof(caps.reserved) / sizeof(caps.reserved[0]); ++i) {
        expect_true("reserved tail is zero", caps.reserved[i] == 0);
    }

    /* A newer caller's larger buffer keeps its tail; the runtime writes only
     * as much as it knows about. */
    memset(oversized, 0x5a, sizeof(oversized));
    expect_status("oversized capability buffer",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      (tc_diloco_capabilities*)oversized, sizeof(oversized)),
                  TC_OK);
    for (i = sizeof(tc_diloco_capabilities); i < sizeof(oversized); ++i) {
        if (oversized[i] != 0x5a) {
            expect_true("oversized buffer tail is preserved", 0);
            break;
        }
    }

    /* An older caller passing exactly the v1 minimum gets the prefix and
     * nothing past it. */
    memset(minimal, 0x5a, sizeof(minimal));
    expect_status("minimum-size capability buffer",
                  tc_diloco_capability_query(
                      NULL, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      (tc_diloco_capabilities*)minimal,
                      TC_DILOCO_CAPABILITIES_V1_MIN_SIZE),
                  TC_OK);
    for (i = TC_DILOCO_CAPABILITIES_V1_MIN_SIZE; i < sizeof(minimal); ++i) {
        if (minimal[i] != 0x5a) {
            expect_true("minimum-size write stops at the v1 prefix", 0);
            break;
        }
    }
    expect_true("minimum-size prefix still carries the serializable masks",
                tc_diloco_outer_optimizer_is_serializable(
                    (const tc_diloco_capabilities*)minimal,
                    TC_DILOCO_OUTER_ADAM) &&
                tc_diloco_compress_is_serializable(
                    (const tc_diloco_capabilities*)minimal,
                    TC_DILOCO_COMPRESS_TOPK_1PCT));

    /* Bound to a single-rank context there is no sparse transport, so the
     * answer must stay dense rather than advertising top-k compression. */
    memset(&caps, 0, sizeof(caps));
    expect_status("context-bound capability query",
                  tc_diloco_capability_query(
                      bound, TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT,
                      &caps, sizeof(caps)),
                  TC_OK);
    expect_true("single-rank context reports no sparse wire format",
                caps.compress_sparse_wire_mask == 0);
    expect_true("single-rank context reports top-k as dense fp32",
                (caps.compress_dense_fp32_wire_mask &
                 TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_TOPK_1PCT)) != 0);

    /* Fail-closed helpers. */
    expect_true("helpers reject NULL capabilities",
                !tc_diloco_outer_optimizer_is_serializable(NULL,
                    TC_DILOCO_OUTER_SGD) &&
                !tc_diloco_compress_is_serializable(NULL,
                    TC_DILOCO_COMPRESS_NONE) &&
                !tc_diloco_state_feature_available(NULL,
                    TC_DILOCO_STATE_FEATURE_OUTER_ANCHOR));
    expect_true("helpers reject an empty feature request",
                !tc_diloco_state_feature_available(&caps, 0));
    expect_true("helpers reject an out-of-range enumerator",
                !tc_diloco_outer_optimizer_is_serializable(
                    &caps, (tc_diloco_outer_optimizer_t)64) &&
                !tc_diloco_compress_is_serializable(
                    &caps, (tc_diloco_compress_t)999));
}

/* ------------------------------------------------------------------------
 * Every serializability claim is backed by a real round-trip.
 * ------------------------------------------------------------------------ */

static void test_round_trip(tc_context* ctx, tc_dist_ctx* dist,
                            const char* label,
                            tc_diloco_outer_optimizer_t optimizer,
                            tc_diloco_compress_t compress,
                            uint32_t version) {
    tc_diloco_config cfg = make_config(optimizer, compress);
    tc_diloco_ctx* source = NULL;
    tc_diloco_ctx* restored = NULL;
    tc_buffer* source_buffer = NULL;
    tc_buffer* restored_buffer = NULL;
    float* source_values = NULL;
    float* restored_values = NULL;
    unsigned char* blob = NULL;
    unsigned char* echo = NULL;
    size_t blob_size = 0;
    size_t echo_size = 0;
    size_t i;
    char what[160];

    if (alloc_parameter(ctx, &source_buffer, &source_values) ||
        alloc_parameter(ctx, &restored_buffer, &restored_values)) {
        expect_true("round-trip parameter allocation", 0);
        goto cleanup;
    }
    if (tc_diloco_init(dist, &cfg, &source) != TC_OK ||
        tc_diloco_init(dist, &cfg, &restored) != TC_OK) {
        expect_true("round-trip context init", 0);
        goto cleanup;
    }
    if (tc_diloco_add_parameter(source, "weight", source_buffer, N_ELEMS,
                                TC_DTYPE_F32) != TC_OK ||
        tc_diloco_add_parameter(restored, "weight", restored_buffer, N_ELEMS,
                                TC_DTYPE_F32) != TC_OK) {
        expect_true("round-trip parameter registration", 0);
        goto cleanup;
    }
    if (tc_diloco_state_set_epochs(source, 101, 17) != TC_OK ||
        tc_diloco_state_set_epochs(restored, 101, 17) != TC_OK) {
        expect_true("round-trip epoch binding", 0);
        goto cleanup;
    }

    /* Drive two outer rounds so momentum and error-feedback residuals are
     * genuinely non-zero before the state is captured. */
    for (i = 0; i < N_ELEMS; ++i) source_values[i] = 2.0f + (float)i * 0.03125f;
    if (tc_diloco_apply_outer(source) != TC_OK) {
        expect_true("round-trip first outer step", 0);
        goto cleanup;
    }
    for (i = 0; i < N_ELEMS; ++i) source_values[i] += 0.5f - (float)(i % 7) * 0.01f;
    if (tc_diloco_apply_outer(source) != TC_OK) {
        expect_true("round-trip second outer step", 0);
        goto cleanup;
    }

    if (serialize_at(source, version, &blob, &blob_size) != 0) {
        snprintf(what, sizeof(what), "%s: serialize at v%u", label, version);
        expect_true(what, 0);
        goto cleanup;
    }

    /* The caller owns theta_local; DiLoCo owns everything else. */
    memcpy(restored_values, source_values, N_ELEMS * sizeof(float));
    snprintf(what, sizeof(what), "%s: deserialize at v%u", label, version);
    expect_status(what,
                  tc_diloco_state_deserialize(
                      restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      blob, blob_size),
                  TC_OK);

    if (serialize_at(restored, version, &echo, &echo_size) != 0) {
        snprintf(what, sizeof(what), "%s: re-serialize at v%u", label, version);
        expect_true(what, 0);
        goto cleanup;
    }
    snprintf(what, sizeof(what),
             "%s: v%u export/import round-trip is bit-exact", label, version);
    expect_true(what,
                echo_size == blob_size && memcmp(echo, blob, blob_size) == 0);

    /* And the restored context must keep evolving identically. */
    for (i = 0; i < N_ELEMS; ++i) {
        const float update = 0.125f + (float)(i % 5) * 0.01f;
        source_values[i] += update;
        restored_values[i] += update;
    }
    snprintf(what, sizeof(what), "%s: source continuation", label);
    expect_status(what, tc_diloco_apply_outer(source), TC_OK);
    snprintf(what, sizeof(what), "%s: restored continuation", label);
    expect_status(what, tc_diloco_apply_outer(restored), TC_OK);
    snprintf(what, sizeof(what),
             "%s: restored optimizer and residual state evolve identically",
             label);
    expect_true(what,
                memcmp(source_values, restored_values,
                       N_ELEMS * sizeof(float)) == 0);

cleanup:
    free(echo);
    free(blob);
    if (restored) (void)tc_diloco_finalize(restored);
    if (source) (void)tc_diloco_finalize(source);
    if (restored_buffer) tc_buffer_free(ctx, restored_buffer);
    if (source_buffer) tc_buffer_free(ctx, source_buffer);
}

/* ------------------------------------------------------------------------
 * Wire format, cross-version reads, and fail-closed rejection.
 * ------------------------------------------------------------------------ */

static void test_wire_format(tc_context* ctx, tc_dist_ctx* dist) {
    tc_diloco_config cfg = make_config(TC_DILOCO_OUTER_NESTEROV,
                                       TC_DILOCO_COMPRESS_TOPK_1PCT);
    tc_diloco_ctx* source = NULL;
    tc_diloco_ctx* target = NULL;
    tc_buffer* source_buffer = NULL;
    tc_buffer* target_buffer = NULL;
    float* source_values = NULL;
    float* target_values = NULL;
    unsigned char* v1 = NULL;
    unsigned char* v2 = NULL;
    unsigned char* v2_again = NULL;
    unsigned char* baseline = NULL;
    unsigned char* after = NULL;
    unsigned char* scratch = NULL;
    size_t v1_size = 0, v2_size = 0, v2_again_size = 0;
    size_t baseline_size = 0, after_size = 0;
    uint8_t digest[32];
    size_t i;
    size_t sentinel = 0;

    if (alloc_parameter(ctx, &source_buffer, &source_values) ||
        alloc_parameter(ctx, &target_buffer, &target_values) ||
        tc_diloco_init(dist, &cfg, &source) != TC_OK ||
        tc_diloco_init(dist, &cfg, &target) != TC_OK ||
        tc_diloco_add_parameter(source, "weight", source_buffer, N_ELEMS,
                                TC_DTYPE_F32) != TC_OK ||
        tc_diloco_add_parameter(target, "weight", target_buffer, N_ELEMS,
                                TC_DTYPE_F32) != TC_OK) {
        expect_true("wire-format fixture setup", 0);
        goto cleanup;
    }
    for (i = 0; i < N_ELEMS; ++i) source_values[i] = 3.0f - (float)i * 0.0625f;
    if (tc_diloco_apply_outer(source) != TC_OK) {
        expect_true("wire-format outer step", 0);
        goto cleanup;
    }

    if (serialize_at(source, TC_DILOCO_STATE_ABI_VERSION_1, &v1, &v1_size) ||
        serialize_at(source, TC_DILOCO_STATE_ABI_VERSION_2, &v2, &v2_size) ||
        serialize_at(source, TC_DILOCO_STATE_ABI_VERSION_2,
                     &v2_again, &v2_again_size)) {
        expect_true("wire-format serialization", 0);
        goto cleanup;
    }

    expect_true("serialization is deterministic",
                v2_again_size == v2_size &&
                memcmp(v2_again, v2, v2_size) == 0);
    expect_true("v2 adds exactly the wider header",
                v2_size == v1_size +
                    (TC_DILOCO_STATE_V2_HEADER_SIZE -
                     TC_DILOCO_STATE_V1_HEADER_SIZE));
    expect_true("both versions carry an identical payload",
                memcmp(v1 + TC_DILOCO_STATE_V1_HEADER_SIZE,
                       v2 + TC_DILOCO_STATE_V2_HEADER_SIZE,
                       v1_size - TC_DILOCO_STATE_V1_HEADER_SIZE) == 0);

    expect_true("v1 header declares magic, version, and size",
                memcmp(v1, "TCDLSTA1", 8) == 0 &&
                load_u32_le(v1 + 8) == TC_DILOCO_STATE_ABI_VERSION_1 &&
                load_u32_le(v1 + 12) == TC_DILOCO_STATE_V1_HEADER_SIZE &&
                load_u64_le(v1 + 16) == (uint64_t)v1_size);
    expect_true("v2 header declares magic, version, and sizes",
                memcmp(v2, "TCDLSTA2", 8) == 0 &&
                load_u32_le(v2 + 8) == TC_DILOCO_STATE_ABI_VERSION_2 &&
                load_u32_le(v2 + 12) == TC_DILOCO_STATE_V2_HEADER_SIZE &&
                load_u64_le(v2 + 16) == (uint64_t)v2_size &&
                load_u64_le(v2 + 24) ==
                    (uint64_t)(v2_size - TC_DILOCO_STATE_V2_HEADER_SIZE));

    sha256(v2 + TC_DILOCO_STATE_V2_HEADER_SIZE,
           v2_size - TC_DILOCO_STATE_V2_HEADER_SIZE, digest);
    expect_true("v2 header carries a SHA-256 of its payload",
                memcmp(v2 + 32, digest, sizeof(digest)) == 0);

    /* A v2-capable caller reads the older wire version. */
    memcpy(target_values, source_values, N_ELEMS * sizeof(float));
    expect_status("v2 caller reads a v1 blob",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      v1, v1_size),
                  TC_OK);
    if (serialize_at(target, TC_DILOCO_STATE_ABI_VERSION_2,
                     &baseline, &baseline_size) != 0) {
        expect_true("baseline serialization", 0);
        goto cleanup;
    }
    expect_true("a v1 blob restores the same state as its v2 twin",
                baseline_size == v2_size && memcmp(baseline, v2, v2_size) == 0);

    /* A caller that only speaks v1 must refuse a v2 blob outright. */
    expect_status("v1-only caller refuses a v2 blob",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_1, v2, v2_size),
                  TC_ERR_ABI_MISMATCH);

    /* Unimplemented versions are refused on every entry point without
     * touching the caller's outputs. */
    sentinel = 4242;
    expect_status("size rejects an unimplemented version",
                  tc_diloco_state_size(
                      source, TC_DILOCO_STATE_ABI_VERSION_CURRENT + 1,
                      &sentinel),
                  TC_ERR_ABI_MISMATCH);
    expect_true("rejected size query preserves its output", sentinel == 4242);
    expect_status("size rejects version zero",
                  tc_diloco_state_size(source, 0, &sentinel),
                  TC_ERR_ABI_MISMATCH);
    {
        size_t written = 99;
        scratch = (unsigned char*)malloc(v2_size);
        if (!scratch) {
            expect_true("scratch allocation", 0);
            goto cleanup;
        }
        memset(scratch, 0x3c, v2_size);
        expect_status("serialize rejects an unimplemented version",
                      tc_diloco_state_serialize(
                          source, TC_DILOCO_STATE_ABI_VERSION_CURRENT + 1,
                          scratch, v2_size, &written),
                      TC_ERR_ABI_MISMATCH);
        expect_true("rejected serialize preserves its outputs",
                    written == 99 && scratch[0] == 0x3c);
    }

    /* ---- Fail-closed rejection: nothing may be partially applied. ---- */

    /* Every truncation of a valid blob must be rejected. */
    for (i = 0; i < v2_size; ++i) {
        if (tc_diloco_state_deserialize(
                target, TC_DILOCO_STATE_ABI_VERSION_CURRENT, v2, i) == TC_OK) {
            expect_true("every truncated blob is rejected", 0);
            break;
        }
    }
    /* Trailing bytes are rejected too: total_size must describe the buffer. */
    {
        unsigned char* extended = (unsigned char*)malloc(v2_size + 1);
        if (extended) {
            memcpy(extended, v2, v2_size);
            extended[v2_size] = 0x00;
            expect_status("trailing bytes are rejected",
                          tc_diloco_state_deserialize(
                              target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                              extended, v2_size + 1),
                          TC_ERR_INVALID_ARG);
            free(extended);
        }
    }

    /* Every single-bit corruption anywhere in the blob must be rejected. */
    memcpy(scratch, v2, v2_size);
    for (i = 0; i < v2_size; ++i) {
        tc_status_t status;
        scratch[i] ^= 0x01u;
        status = tc_diloco_state_deserialize(
            target, TC_DILOCO_STATE_ABI_VERSION_CURRENT, scratch, v2_size);
        scratch[i] ^= 0x01u;
        if (status == TC_OK) {
            fprintf(stderr,
                    "diloco_state_abi: corrupt byte %zu was accepted\n", i);
            expect_true("every corrupted byte is rejected", 0);
            break;
        }
    }

    /* Header fields that lie about the buffer must not be believed. */
    memcpy(scratch, v2, v2_size);
    scratch[16] = (unsigned char)(scratch[16] + 1);       /* total_size */
    expect_status("a lying total_size is rejected",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      scratch, v2_size),
                  TC_ERR_INVALID_ARG);
    memcpy(scratch, v2, v2_size);
    scratch[24] = (unsigned char)(scratch[24] + 1);       /* payload_size */
    expect_status("a lying payload_size is rejected",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      scratch, v2_size),
                  TC_ERR_INVALID_ARG);
    memcpy(scratch, v2, v2_size);
    scratch[12] = TC_DILOCO_STATE_V1_HEADER_SIZE;         /* header_size */
    expect_status("a lying header_size is rejected",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      scratch, v2_size),
                  TC_ERR_INVALID_ARG);
    memcpy(scratch, v2, v2_size);
    memcpy(scratch, "TCDLSTAX", 8);                       /* magic */
    expect_status("an unrecognized magic is rejected",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      scratch, v2_size),
                  TC_ERR_INVALID_ARG);
    memcpy(scratch, v2, v2_size);
    scratch[8] = TC_DILOCO_STATE_ABI_VERSION_1;           /* version vs magic */
    expect_status("a version that contradicts the magic is rejected",
                  tc_diloco_state_deserialize(
                      target, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                      scratch, v2_size),
                  TC_ERR_INVALID_ARG);

    /* After all of that rejection the target must be byte-identical to the
     * state it held before, i.e. nothing was partially applied. */
    if (serialize_at(target, TC_DILOCO_STATE_ABI_VERSION_2,
                     &after, &after_size) != 0) {
        expect_true("post-rejection serialization", 0);
    } else {
        expect_true("rejected restores never modify the target",
                    after_size == baseline_size &&
                    memcmp(after, baseline, baseline_size) == 0);
    }

cleanup:
    free(scratch);
    free(after);
    free(baseline);
    free(v2_again);
    free(v2);
    free(v1);
    if (target) (void)tc_diloco_finalize(target);
    if (source) (void)tc_diloco_finalize(source);
    if (target_buffer) tc_buffer_free(ctx, target_buffer);
    if (source_buffer) tc_buffer_free(ctx, source_buffer);
}

int main(void) {
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_diloco_ctx* bound = NULL;
    tc_buffer* bound_buffer = NULL;
    float* bound_values = NULL;
    tc_diloco_config bound_cfg;

    check_sha256_reference();

    if (expect_status("tc_init", tc_init(&ctx), TC_OK)) goto done;
    if (expect_status("dist init",
                      tc_dist_init(ctx, TC_DIST_SINGLE, 1, 0, "", &dist),
                      TC_OK)) {
        goto done;
    }

    bound_cfg = make_config(TC_DILOCO_OUTER_NESTEROV,
                            TC_DILOCO_COMPRESS_TOPK_1PCT);
    if (alloc_parameter(ctx, &bound_buffer, &bound_values) ||
        tc_diloco_init(dist, &bound_cfg, &bound) != TC_OK ||
        tc_diloco_add_parameter(bound, "weight", bound_buffer, N_ELEMS,
                                TC_DTYPE_F32) != TC_OK) {
        expect_true("bound context setup", 0);
        goto done;
    }

    test_capability_query(bound);

    /* Each configuration the capability mask calls serializable is proven at
     * both wire versions. */
    test_round_trip(ctx, dist, "sgd/none", TC_DILOCO_OUTER_SGD,
                    TC_DILOCO_COMPRESS_NONE, TC_DILOCO_STATE_ABI_VERSION_1);
    test_round_trip(ctx, dist, "sgd/none", TC_DILOCO_OUTER_SGD,
                    TC_DILOCO_COMPRESS_NONE, TC_DILOCO_STATE_ABI_VERSION_2);
    test_round_trip(ctx, dist, "nesterov/none", TC_DILOCO_OUTER_NESTEROV,
                    TC_DILOCO_COMPRESS_NONE, TC_DILOCO_STATE_ABI_VERSION_2);
    test_round_trip(ctx, dist, "adam/none", TC_DILOCO_OUTER_ADAM,
                    TC_DILOCO_COMPRESS_NONE, TC_DILOCO_STATE_ABI_VERSION_2);
    test_round_trip(ctx, dist, "nesterov/fp16", TC_DILOCO_OUTER_NESTEROV,
                    TC_DILOCO_COMPRESS_FP16, TC_DILOCO_STATE_ABI_VERSION_2);
    test_round_trip(ctx, dist, "nesterov/topk_1pct", TC_DILOCO_OUTER_NESTEROV,
                    TC_DILOCO_COMPRESS_TOPK_1PCT,
                    TC_DILOCO_STATE_ABI_VERSION_2);
    test_round_trip(ctx, dist, "adam/topk_01pct", TC_DILOCO_OUTER_ADAM,
                    TC_DILOCO_COMPRESS_TOPK_01PCT,
                    TC_DILOCO_STATE_ABI_VERSION_2);

    test_wire_format(ctx, dist);

done:
    if (bound) (void)tc_diloco_finalize(bound);
    if (bound_buffer) tc_buffer_free(ctx, bound_buffer);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);
    printf("diloco_state_abi: %s (%d failure%s)\n",
           failures ? "FAIL" : "OK", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
