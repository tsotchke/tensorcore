/*
 * tests/test_checkpoint.c - activation checkpointing real impl validation.
 *
 * Exercises the full discard -> realize lifecycle:
 *   1. Allocate a tc_buffer, fill it with a recognizable pattern.
 *   2. Register it as a checkpoint with a recompute callback.
 *   3. Verify resident, bytes-discarded = 0.
 *   4. Discard. Verify the buffer's underlying storage is actually freed
 *      (tc_buffer_map fails, tc_checkpoint_is_resident returns 0,
 *      tc_checkpoint_total_bytes_discarded reports the bytes).
 *   5. Realize. Verify the user callback was invoked, the buffer is
 *      resident again, and the recomputed contents match the original.
 *
 * Validates: memory is actually reclaimed during discard (not just a
 * flag toggle), the same handle survives discard+realize, and the
 * recompute callback fills the buffer correctly.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/checkpoint.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_ELEMS 65536      /* 256 KB of fp32 */

static int recompute_count = 0;

static tc_status_t recompute_fill(void* user_data) {
    tc_buffer* buf = (tc_buffer*)user_data;
    void* p = NULL;
    tc_status_t s = tc_buffer_map(buf, &p);
    if (s != TC_OK) {
        fprintf(stderr, "recompute: map failed (status=%d)\n", s);
        return s;
    }
    float* data = (float*)p;
    /* Same pattern the test wrote originally. */
    for (int i = 0; i < N_ELEMS; ++i) data[i] = (float)i * 0.5f;
    recompute_count++;
    return TC_OK;
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "tc_init failed\n");
        return 1;
    }

    /* Allocate + fill. */
    tc_buffer* buf = NULL;
    if (tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), &buf) != TC_OK) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    void* p = NULL;
    if (tc_buffer_map(buf, &p) != TC_OK) {
        fprintf(stderr, "map failed\n");
        return 1;
    }
    float* data = (float*)p;
    for (int i = 0; i < N_ELEMS; ++i) data[i] = (float)i * 0.5f;
    printf("  alloc + fill OK\n");

    /* Register. */
    tc_checkpoint_id id = 0;
    if (tc_checkpoint_register(buf, recompute_fill, buf, &id) != TC_OK) {
        fprintf(stderr, "register failed\n");
        return 1;
    }
    if (!tc_checkpoint_is_resident(id)) {
        fprintf(stderr, "should be resident after register\n");
        return 1;
    }
    if (tc_checkpoint_total_bytes_discarded() != 0) {
        fprintf(stderr, "bytes_discarded=%llu, expected 0\n",
            (unsigned long long)tc_checkpoint_total_bytes_discarded());
        return 1;
    }
    printf("  register OK id=%llu\n", (unsigned long long)id);

    /* Discard. */
    tc_status_t s = tc_checkpoint_discard(id);
    if (s != TC_OK) {
        if (s == TC_ERR_UNSUPPORTED_FAMILY) {
            printf("[skip] tc_buffer_discard_storage not implemented on this backend\n");
            tc_checkpoint_unregister(id);
            tc_buffer_free(ctx, buf);
            tc_shutdown(ctx);
            return 77;
        }
        fprintf(stderr, "discard failed (status=%d)\n", s);
        return 1;
    }
    if (tc_checkpoint_is_resident(id)) {
        fprintf(stderr, "should NOT be resident after discard\n");
        return 1;
    }
    const uint64_t expected_bytes = N_ELEMS * sizeof(float);
    if (tc_checkpoint_total_bytes_discarded() != expected_bytes) {
        fprintf(stderr, "bytes_discarded=%llu, expected %llu\n",
            (unsigned long long)tc_checkpoint_total_bytes_discarded(),
            (unsigned long long)expected_bytes);
        return 1;
    }
    /* Verify the buffer's storage was actually freed: map should fail. */
    void* p2 = NULL;
    if (tc_buffer_map(buf, &p2) == TC_OK) {
        fprintf(stderr, "map should fail on discarded buffer; got p=%p\n", p2);
        return 1;
    }
    printf("  discard OK (%llu bytes reclaimed; map fails as expected)\n",
        (unsigned long long)expected_bytes);

    /* Realize. */
    recompute_count = 0;
    if (tc_checkpoint_realize(id) != TC_OK) {
        fprintf(stderr, "realize failed\n");
        return 1;
    }
    if (recompute_count != 1) {
        fprintf(stderr, "recompute called %d times, expected 1\n", recompute_count);
        return 1;
    }
    if (!tc_checkpoint_is_resident(id)) {
        fprintf(stderr, "should be resident after realize\n");
        return 1;
    }
    if (tc_checkpoint_total_bytes_discarded() != 0) {
        fprintf(stderr, "bytes_discarded=%llu, expected 0 after realize\n",
            (unsigned long long)tc_checkpoint_total_bytes_discarded());
        return 1;
    }
    /* Verify the realized contents match the original. */
    if (tc_buffer_map(buf, &p2) != TC_OK) {
        fprintf(stderr, "map failed after realize\n");
        return 1;
    }
    float* recomputed = (float*)p2;
    for (int i = 0; i < N_ELEMS; ++i) {
        const float expected = (float)i * 0.5f;
        if (recomputed[i] != expected) {
            fprintf(stderr, "mismatch at %d: got %.3f, expected %.3f\n",
                i, recomputed[i], expected);
            return 1;
        }
    }
    printf("  realize OK (recompute_count=%d, contents match)\n",
        recompute_count);

    /* Cleanup. */
    tc_checkpoint_unregister(id);
    tc_buffer_free(ctx, buf);
    tc_shutdown(ctx);
    printf("OK\n");
    return 0;
}
