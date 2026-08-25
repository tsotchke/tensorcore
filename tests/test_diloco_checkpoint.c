/* Versioned DiLoCo checkpoint round-trip, mismatch, and async-resume tests. */

#include "tensorcore/tensorcore.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ELEMS 32

static int expect_status(const char* what, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    fprintf(stderr, "diloco_checkpoint: %s: got %s want %s\n",
            what, tc_status_string(got), tc_status_string(want));
    return 1;
}

static int expect_true(const char* what, int condition) {
    if (condition) return 0;
    fprintf(stderr, "diloco_checkpoint: FAIL: %s\n", what);
    return 1;
}

static tc_diloco_config config(int async_overlap) {
    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 2;
    cfg.outer_lr = 0.25f;
    cfg.outer_momentum = 0.9f;
    cfg.outer_beta2 = 0.999f;
    cfg.outer_eps = 1e-8f;
    cfg.outer_optimizer = async_overlap
        ? TC_DILOCO_OUTER_SGD : TC_DILOCO_OUTER_NESTEROV;
    cfg.compress = async_overlap
        ? TC_DILOCO_COMPRESS_NONE : TC_DILOCO_COMPRESS_TOPK_1PCT;
    cfg.async_overlap = async_overlap != 0;
    return cfg;
}

static int alloc_parameter(tc_context* ctx, float initial,
                           tc_buffer** out_buffer, float** out_values) {
    if (tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), out_buffer) != TC_OK) {
        return 1;
    }
    if (tc_buffer_map(*out_buffer, (void**)out_values) != TC_OK) return 1;
    for (size_t i = 0; i < N_ELEMS; ++i) (*out_values)[i] = initial;
    return 0;
}

static int serialize_state(tc_diloco_ctx* d, unsigned char** out,
                           size_t* out_size) {
    if (tc_diloco_state_size(
            d, TC_DILOCO_STATE_ABI_VERSION_CURRENT, out_size) != TC_OK) {
        return 1;
    }
    *out = (unsigned char*)malloc(*out_size);
    if (!*out) return 1;
    size_t written = 0;
    if (tc_diloco_state_serialize(
            d, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
            *out, *out_size, &written) != TC_OK || written != *out_size) {
        free(*out);
        *out = NULL;
        return 1;
    }
    return 0;
}

static int values_equal(const float* a, const float* b) {
    for (size_t i = 0; i < N_ELEMS; ++i) {
        if (fabsf(a[i] - b[i]) > 1e-6f) return 0;
    }
    return 1;
}

int main(void) {
    int rc = 0;
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_diloco_ctx* source = NULL;
    tc_diloco_ctx* restored = NULL;
    tc_diloco_ctx* mismatch = NULL;
    tc_buffer* source_buffer = NULL;
    tc_buffer* restored_buffer = NULL;
    tc_buffer* mismatch_buffer = NULL;
    float* source_values = NULL;
    float* restored_values = NULL;
    float* mismatch_values = NULL;
    unsigned char* blob = NULL;
    size_t blob_size = 0;

    rc |= expect_status("tc_init", tc_init(&ctx), TC_OK);
    if (rc) goto cleanup;
    rc |= expect_status(
        "dist init", tc_dist_init(ctx, TC_DIST_SINGLE, 1, 0, "", &dist), TC_OK);
    if (rc) goto cleanup;
    rc |= alloc_parameter(ctx, 1.0f, &source_buffer, &source_values);
    rc |= alloc_parameter(ctx, 1.0f, &restored_buffer, &restored_values);
    rc |= alloc_parameter(ctx, 1.0f, &mismatch_buffer, &mismatch_values);
    if (rc) goto cleanup;

    tc_diloco_config cfg = config(0);
    rc |= expect_status("source init", tc_diloco_init(dist, &cfg, &source), TC_OK);
    rc |= expect_status("restored init", tc_diloco_init(dist, &cfg, &restored), TC_OK);
    rc |= expect_status(
        "source add", tc_diloco_add_parameter(
            source, "weight", source_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "restored add", tc_diloco_add_parameter(
            restored, "weight", restored_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "source epochs", tc_diloco_state_set_epochs(source, 41, 7), TC_OK);
    rc |= expect_status(
        "restored epochs", tc_diloco_state_set_epochs(restored, 41, 7), TC_OK);
    if (rc) goto cleanup;

    for (size_t i = 0; i < N_ELEMS; ++i) {
        source_values[i] = 2.0f + (float)i * 0.03125f;
    }
    bool pending = false;
    rc |= expect_status("step 1", tc_diloco_step(source, &pending), TC_OK);
    rc |= expect_status("step 2", tc_diloco_step(source, &pending), TC_OK);
    rc |= expect_true("outer boundary restored", pending);
    rc |= expect_status("first outer", tc_diloco_apply_outer(source), TC_OK);
    if (rc || serialize_state(source, &blob, &blob_size) != 0) {
        rc |= expect_true("serialize source", 0);
        goto cleanup;
    }

    memcpy(restored_values, source_values, N_ELEMS * sizeof(float));
    rc |= expect_status(
        "deserialize source",
        tc_diloco_state_deserialize(
            restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT, blob, blob_size),
        TC_OK);
    uint64_t topology_epoch = 0;
    uint64_t membership_epoch = 0;
    rc |= expect_status(
        "get restored epochs",
        tc_diloco_state_get_epochs(
            restored, &topology_epoch, &membership_epoch), TC_OK);
    rc |= expect_true(
        "epochs round trip", topology_epoch == 41 && membership_epoch == 7);

    unsigned char* restored_blob = NULL;
    size_t restored_blob_size = 0;
    if (serialize_state(restored, &restored_blob, &restored_blob_size) != 0) {
        rc |= expect_true("serialize restored", 0);
    } else {
        rc |= expect_true(
            "serialized state is deterministic",
            restored_blob_size == blob_size &&
            memcmp(restored_blob, blob, blob_size) == 0);
    }

    /* Both contexts must evolve identically after restore, including
     * Nesterov momentum and top-k error-feedback residuals. */
    for (size_t i = 0; i < N_ELEMS; ++i) {
        const float update = 0.125f + (float)(i % 5) * 0.01f;
        source_values[i] += update;
        restored_values[i] += update;
    }
    rc |= expect_status("source next outer", tc_diloco_apply_outer(source), TC_OK);
    rc |= expect_status("restored next outer", tc_diloco_apply_outer(restored), TC_OK);
    rc |= expect_true(
        "restored optimizer/error state evolves exactly",
        values_equal(source_values, restored_values));
    rc |= expect_true(
        "restored counters evolve exactly",
        tc_diloco_inner_steps_completed(source) ==
            tc_diloco_inner_steps_completed(restored) &&
        tc_diloco_outer_steps_completed(source) ==
            tc_diloco_outer_steps_completed(restored));

    /* An unknown ABI or undersized destination must not modify outputs. */
    size_t sentinel_size = 12345;
    rc |= expect_status(
        "unknown state ABI",
        tc_diloco_state_size(
            source, TC_DILOCO_STATE_ABI_VERSION_CURRENT + 1, &sentinel_size),
        TC_ERR_ABI_MISMATCH);
    rc |= expect_true("ABI mismatch preserves size output", sentinel_size == 12345);
    if (blob_size > 1) {
        unsigned char* short_output = (unsigned char*)malloc(blob_size - 1);
        if (!short_output) {
            rc |= expect_true("allocate short output", 0);
        } else {
            memset(short_output, 0x5a, blob_size - 1);
            size_t written = 6789;
            rc |= expect_status(
                "undersized serialization",
                tc_diloco_state_serialize(
                    source, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                    short_output, blob_size - 1, &written),
                TC_ERR_INVALID_ARG);
            rc |= expect_true("undersized write count preserved", written == 6789);
            rc |= expect_true("undersized output preserved", short_output[0] == 0x5a);
            free(short_output);
        }
    }

    /* Corruption and identity drift are rejected without changing the target. */
    free(restored_blob);
    restored_blob = NULL;
    restored_blob_size = 0;
    free(blob);
    blob = NULL;
    blob_size = 0;
    if (serialize_state(restored, &blob, &blob_size) != 0 ||
        serialize_state(restored, &restored_blob, &restored_blob_size) != 0) {
        rc |= expect_true("serialize corruption baseline", 0);
    } else {
        blob[blob_size - 1] ^= 0x80u;
        rc |= expect_status(
            "corrupt checkpoint",
            tc_diloco_state_deserialize(
                restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT, blob, blob_size),
            TC_ERR_INVALID_ARG);
        unsigned char* after_corruption = NULL;
        size_t after_corruption_size = 0;
        if (serialize_state(
                restored, &after_corruption, &after_corruption_size) != 0) {
            rc |= expect_true("serialize after corruption", 0);
        } else {
            rc |= expect_true(
                "corrupt restore is atomic",
                after_corruption_size == restored_blob_size &&
                memcmp(after_corruption, restored_blob, restored_blob_size) == 0);
        }
        free(after_corruption);
    }

    tc_diloco_config mismatch_cfg = cfg;
    mismatch_cfg.outer_lr = 0.5f;
    rc |= expect_status(
        "mismatch init", tc_diloco_init(dist, &mismatch_cfg, &mismatch), TC_OK);
    rc |= expect_status(
        "mismatch add", tc_diloco_add_parameter(
            mismatch, "weight", mismatch_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    if (restored_blob) {
        rc |= expect_status(
            "config mismatch",
            tc_diloco_state_deserialize(
                mismatch, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                restored_blob, restored_blob_size),
            TC_ERR_INVALID_ARG);
    }
    rc |= expect_status("mismatch finalize", tc_diloco_finalize(mismatch), TC_OK);
    mismatch = NULL;

    rc |= expect_status(
        "epoch mismatch init", tc_diloco_init(dist, &cfg, &mismatch), TC_OK);
    rc |= expect_status(
        "epoch mismatch add", tc_diloco_add_parameter(
            mismatch, "weight", mismatch_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "epoch mismatch set", tc_diloco_state_set_epochs(mismatch, 42, 7), TC_OK);
    if (restored_blob) {
        rc |= expect_status(
            "epoch mismatch restore",
            tc_diloco_state_deserialize(
                mismatch, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                restored_blob, restored_blob_size),
            TC_ERR_INVALID_ARG);
    }
    rc |= expect_status("epoch mismatch finalize", tc_diloco_finalize(mismatch), TC_OK);
    mismatch = NULL;

    free(blob);
    blob = NULL;
    free(restored_blob);
    restored_blob = NULL;
    blob_size = restored_blob_size = 0;
    rc |= expect_status("source finalize", tc_diloco_finalize(source), TC_OK);
    source = NULL;
    rc |= expect_status("restored finalize", tc_diloco_finalize(restored), TC_OK);
    restored = NULL;

    /* Adam's packed first/second moments use a 2*N internal vector. Prove the
     * larger optimizer state is also exact across restore. */
    cfg = config(0);
    cfg.outer_optimizer = TC_DILOCO_OUTER_ADAM;
    cfg.compress = TC_DILOCO_COMPRESS_NONE;
    for (size_t i = 0; i < N_ELEMS; ++i) {
        source_values[i] = 1.0f;
        restored_values[i] = 1.0f;
    }
    rc |= expect_status("Adam source init", tc_diloco_init(dist, &cfg, &source), TC_OK);
    rc |= expect_status("Adam restored init", tc_diloco_init(dist, &cfg, &restored), TC_OK);
    rc |= expect_status(
        "Adam source add", tc_diloco_add_parameter(
            source, "weight", source_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "Adam restored add", tc_diloco_add_parameter(
            restored, "weight", restored_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status("Adam source epochs", tc_diloco_state_set_epochs(source, 55, 13), TC_OK);
    rc |= expect_status("Adam restored epochs", tc_diloco_state_set_epochs(restored, 55, 13), TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        source_values[i] = 1.5f + (float)i * 0.015625f;
    }
    rc |= expect_status("Adam first outer", tc_diloco_apply_outer(source), TC_OK);
    if (serialize_state(source, &blob, &blob_size) != 0) {
        rc |= expect_true("serialize Adam state", 0);
    } else {
        memcpy(restored_values, source_values, N_ELEMS * sizeof(float));
        rc |= expect_status(
            "restore Adam state",
            tc_diloco_state_deserialize(
                restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                blob, blob_size),
            TC_OK);
        for (size_t i = 0; i < N_ELEMS; ++i) {
            const float update = 0.05f + (float)(i % 3) * 0.0078125f;
            source_values[i] += update;
            restored_values[i] += update;
        }
        rc |= expect_status("Adam source continuation", tc_diloco_apply_outer(source), TC_OK);
        rc |= expect_status("Adam restored continuation", tc_diloco_apply_outer(restored), TC_OK);
        rc |= expect_true(
            "restored Adam moments evolve exactly",
            values_equal(source_values, restored_values));
    }
    free(blob);
    blob = NULL;
    blob_size = 0;
    rc |= expect_status("Adam source finalize", tc_diloco_finalize(source), TC_OK);
    source = NULL;
    rc |= expect_status("Adam restored finalize", tc_diloco_finalize(restored), TC_OK);
    restored = NULL;

    /* READY async state is serializable and commits identically after a
     * process-style restore when the caller has restored its live model. */
    cfg = config(1);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        source_values[i] = 1.0f;
        restored_values[i] = 1.0f;
    }
    rc |= expect_status("async source init", tc_diloco_init(dist, &cfg, &source), TC_OK);
    rc |= expect_status("async restored init", tc_diloco_init(dist, &cfg, &restored), TC_OK);
    rc |= expect_status(
        "async source add", tc_diloco_add_parameter(
            source, "weight", source_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "async restored add", tc_diloco_add_parameter(
            restored, "weight", restored_buffer, N_ELEMS, TC_DTYPE_F32), TC_OK);
    rc |= expect_status(
        "parameter registration checks buffer capacity",
        tc_diloco_add_parameter(
            source, "too-large", source_buffer, N_ELEMS + 1, TC_DTYPE_F32),
        TC_ERR_INVALID_ARG);
    rc |= expect_status("async source epochs", tc_diloco_state_set_epochs(source, 9, 3), TC_OK);
    rc |= expect_status("async restored epochs", tc_diloco_state_set_epochs(restored, 9, 3), TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) source_values[i] = 3.0f;
    rc |= expect_status("async launch", tc_diloco_apply_outer(source), TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) source_values[i] = 4.0f;
    rc |= expect_status("async wait", tc_diloco_async_wait(source), TC_OK);
    if (serialize_state(source, &blob, &blob_size) != 0) {
        rc |= expect_true("serialize ready state", 0);
    } else {
        memcpy(restored_values, source_values, N_ELEMS * sizeof(float));
        rc |= expect_status(
            "restore ready state",
            tc_diloco_state_deserialize(
                restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT, blob, blob_size),
            TC_OK);
        tc_diloco_async_state_t state = TC_DILOCO_ASYNC_IDLE;
        tc_status_t worker_status = TC_ERR_INTERNAL;
        uint64_t round_id = 0;
        rc |= expect_status(
            "poll restored ready",
            tc_diloco_async_poll(
                restored, &state, &worker_status, &round_id), TC_OK);
        rc |= expect_true(
            "ready state/round restored",
            state == TC_DILOCO_ASYNC_READY && worker_status == TC_OK &&
            round_id == 1);
        rc |= expect_status("commit source ready", tc_diloco_async_commit(source), TC_OK);
        rc |= expect_status("commit restored ready", tc_diloco_async_commit(restored), TC_OK);
        rc |= expect_true(
            "restored ready commit matches",
            values_equal(source_values, restored_values));
        rc |= expect_true(
            "restored ready commit counted",
            tc_diloco_outer_steps_completed(source) == 1 &&
            tc_diloco_outer_steps_completed(restored) == 1);
    }

    /* Reserved recovery and unknown optimizer settings fail closed. */
    mismatch_cfg = config(0);
    mismatch_cfg.tolerate_dropouts = true;
    rc |= expect_status(
        "dropout recovery unsupported",
        tc_diloco_init(dist, &mismatch_cfg, &mismatch), TC_ERR_INVALID_ARG);
    rc |= expect_true("unsupported dropout leaves output null", mismatch == NULL);
    mismatch_cfg = config(0);
    mismatch_cfg.outer_optimizer = (tc_diloco_outer_optimizer_t)99;
    rc |= expect_status(
        "unknown optimizer unsupported",
        tc_diloco_init(dist, &mismatch_cfg, &mismatch), TC_ERR_INVALID_ARG);
    rc |= expect_true("unknown optimizer leaves output null", mismatch == NULL);

cleanup:
    free(blob);
    if (mismatch) (void)tc_diloco_finalize(mismatch);
    if (restored) (void)tc_diloco_finalize(restored);
    if (source) (void)tc_diloco_finalize(source);
    if (mismatch_buffer) tc_buffer_free(ctx, mismatch_buffer);
    if (restored_buffer) tc_buffer_free(ctx, restored_buffer);
    if (source_buffer) tc_buffer_free(ctx, source_buffer);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);
    printf("diloco_checkpoint: %s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}
