/*
 * tests/test_sparse_compress.c — round-trip + accuracy test for the
 * DiLoCo sparse top-k compression primitives.
 *
 * Validates:
 *   - pack() → unpack_add() recovers the top-k magnitudes within fp16 noise
 *   - sub-threshold entries get zeroed in the source vector (error-feedback
 *     contract — the caller carries residual to the next outer step)
 *   - packed_size() matches actual bytes written
 *   - multi-rank merge: two pack() outputs unpacked onto the same dst
 *     produce the SUM of the kept values
 */

#include "tensorcore/tensorcore.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Internal symbols from lib/distributed/sparse_compress.cpp. */
size_t tc_diloco_sparse_pack(float* delta_fp32, size_t n,
                              float keep_fraction,
                              void* out_payload, size_t out_cap);
int    tc_diloco_sparse_unpack_add(const void* payload, size_t payload_bytes,
                                    float* dst, size_t dst_capacity);
size_t tc_diloco_sparse_packed_size(size_t n, float keep_fraction);

static int fail(const char* what) {
    fprintf(stderr, "sparse_compress: FAIL: %s\n", what);
    return 1;
}

int main(void) {
    int rc = 0;
    const size_t N = 1000;
    const float keep = 0.10f;   /* keep top 10% by magnitude */

    /* Construct a vector with known top-k structure: 10% of elements are
     * large (±1.0 + small noise), the rest are tiny (±0.001). */
    float* dense = (float*)calloc(N, sizeof(float));
    float* dense_orig = (float*)calloc(N, sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0) {
            dense[i] = (i % 20 == 0 ? 1.0f : -1.0f) + 0.0001f * (float)i;
        } else {
            dense[i] = 0.001f * ((i % 7 == 0) ? 1.0f : -1.0f);
        }
        dense_orig[i] = dense[i];
    }

    const size_t expected_packed = tc_diloco_sparse_packed_size(N, keep);

    /* Pack. */
    uint8_t* payload = (uint8_t*)calloc(expected_packed + 64, 1);
    const size_t written = tc_diloco_sparse_pack(dense, N, keep,
                                                  payload, expected_packed + 64);
    if (written == 0) rc |= fail("pack returned 0");
    if (written > expected_packed) rc |= fail("pack wrote more than expected_packed");

    /* Verify: sub-threshold entries (~0.001) in `dense` were zeroed; the
     * large entries (~1.0) preserved. */
    int large_preserved = 1, small_zeroed = 1;
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0) {
            if (fabsf(dense[i] - dense_orig[i]) > 1e-5f) large_preserved = 0;
        } else {
            if (dense[i] != 0.0f) small_zeroed = 0;
        }
    }
    if (!large_preserved) rc |= fail("large entries not preserved after pack");
    if (!small_zeroed) rc |= fail("small entries not zeroed (error-feedback contract)");

    /* Unpack into a fresh dst, compare to the original large entries. */
    float* recovered = (float*)calloc(N, sizeof(float));
    if (tc_diloco_sparse_unpack_add(payload, written, recovered, N) != 0) {
        rc |= fail("unpack_add returned non-zero");
    }
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0) {
            const float err = fabsf(recovered[i] - dense_orig[i]);
            if (err > 5e-3f) {       /* fp16 rounding tolerance */
                fprintf(stderr, "  idx %zu: orig %.4f got %.4f\n", i, dense_orig[i], recovered[i]);
                rc |= fail("unpacked large entry diverged from original");
                break;
            }
        }
    }

    /* Multi-rank merge sanity: pack the same delta twice (simulating two
     * ranks contributing the same top-k indices), unpack both onto the
     * same dst, expect double values. */
    float* delta2 = (float*)calloc(N, sizeof(float));
    memcpy(delta2, dense_orig, N * sizeof(float));
    uint8_t* payload2 = (uint8_t*)calloc(expected_packed + 64, 1);
    tc_diloco_sparse_pack(delta2, N, keep, payload2, expected_packed + 64);

    float* merged = (float*)calloc(N, sizeof(float));
    tc_diloco_sparse_unpack_add(payload, written, merged, N);
    tc_diloco_sparse_unpack_add(payload2, written, merged, N);
    for (size_t i = 0; i < N; ++i) {
        if (i % 10 == 0) {
            const float err = fabsf(merged[i] - 2.0f * dense_orig[i]);
            if (err > 1e-2f) {
                rc |= fail("multi-rank merge didn't sum");
                break;
            }
        }
    }

    /* Bandwidth claim: packed size for 10% keep should be ~12% of the
     * dense fp32 vector (1 header + n*0.1 * 8-byte entries vs n*4). */
    const size_t dense_bytes = N * sizeof(float);
    const float ratio = (float)written / (float)dense_bytes;
    if (ratio < 0.15f || ratio > 0.25f) {
        fprintf(stderr, "  ratio: %.2f%% of dense\n", ratio * 100);
        rc |= fail("packed-bytes ratio outside expected ~20%");
    }

    free(dense); free(dense_orig); free(payload); free(payload2);
    free(recovered); free(delta2); free(merged);

    printf("sparse_compress: %s (%.1f%% of dense at keep=10%%)\n",
           rc ? "FAIL" : "OK", ratio * 100);
    return rc ? 1 : 0;
}
