/* Snapshot-safe async DiLoCo state-machine and rebase regression test. */

#include "tensorcore/tensorcore.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_ELEMS 8

static int expect_status(const char* what, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    fprintf(stderr, "diloco_async: %s: got %s want %s\n",
            what, tc_status_string(got), tc_status_string(want));
    return 1;
}

static int expect_true(const char* what, int condition) {
    if (condition) return 0;
    fprintf(stderr, "diloco_async: FAIL: %s\n", what);
    return 1;
}

static int expect_values(const char* what, const float* values, float want) {
    for (size_t i = 0; i < N_ELEMS; ++i) {
        if (fabsf(values[i] - want) > 1e-6f) {
            fprintf(stderr,
                    "diloco_async: %s[%zu]: got %.8g want %.8g\n",
                    what, i, values[i], want);
            return 1;
        }
    }
    return 0;
}

static tc_diloco_config async_config(void) {
    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 1;
    cfg.outer_lr = 0.5f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = TC_DILOCO_COMPRESS_NONE;
    cfg.async_overlap = true;
    return cfg;
}

int main(void) {
    int rc = 0;
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_diloco_ctx* d = NULL;
    tc_buffer* theta = NULL;
    float* values = NULL;

    rc |= expect_status("tc_init", tc_init(&ctx), TC_OK);
    if (rc) goto done;
    rc |= expect_status(
        "dist init", tc_dist_init(ctx, TC_DIST_SINGLE, 1, 0, "", &dist), TC_OK);
    if (rc) goto done;
    rc |= expect_status(
        "buffer alloc", tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), &theta), TC_OK);
    if (rc) goto done;
    rc |= expect_status("buffer map", tc_buffer_map(theta, (void**)&values), TC_OK);
    if (rc) goto done;
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 1.0f;

    tc_diloco_config cfg = async_config();
    rc |= expect_status("diloco init", tc_diloco_init(dist, &cfg, &d), TC_OK);
    if (rc) goto done;
    rc |= expect_status(
        "add parameter",
        tc_diloco_add_parameter(d, "p", theta, N_ELEMS, TC_DTYPE_F32),
        TC_OK);
    if (rc) goto done;

    /* Round snapshot=3, old anchor=1, lr=.5 => new anchor=2. The caller then
     * advances live parameters to 4 while communication runs. Commit must
     * preserve that +1 post-snapshot drift: 2 + (4 - 3) = 3. */
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 3.0f;
    rc |= expect_status("launch round 1", tc_diloco_apply_outer(d), TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 4.0f;

    tc_diloco_async_state_t state = TC_DILOCO_ASYNC_IDLE;
    tc_status_t worker_status = TC_ERR_INTERNAL;
    uint64_t round_id = 0;
    rc |= expect_status("wait round 1", tc_diloco_async_wait(d), TC_OK);
    rc |= expect_status(
        "poll ready",
        tc_diloco_async_poll(d, &state, &worker_status, &round_id),
        TC_OK);
    rc |= expect_true("round 1 is ready", state == TC_DILOCO_ASYNC_READY);
    rc |= expect_true("round 1 worker status is OK", worker_status == TC_OK);
    rc |= expect_true("round 1 identity", round_id == 1);
    rc |= expect_true(
        "uncommitted round is not counted",
        tc_diloco_outer_steps_completed(d) == 0);
    rc |= expect_values("worker did not touch live values", values, 4.0f);
    rc |= expect_status(
        "second launch before commit", tc_diloco_apply_outer(d), TC_ERR_BUSY);
    rc |= expect_status(
        "registration before commit",
        tc_diloco_add_parameter(d, "late", theta, N_ELEMS, TC_DTYPE_F32),
        TC_ERR_BUSY);

    rc |= expect_status("commit round 1", tc_diloco_async_commit(d), TC_OK);
    rc |= expect_values("rebased commit", values, 3.0f);
    rc |= expect_true(
        "committed round is counted", tc_diloco_outer_steps_completed(d) == 1);
    rc |= expect_status(
        "poll idle",
        tc_diloco_async_poll(d, &state, &worker_status, &round_id),
        TC_OK);
    rc |= expect_true("state returns to idle", state == TC_DILOCO_ASYNC_IDLE);
    rc |= expect_true("last committed round identity remains visible", round_id == 1);

    /* A second round proves the previous worker is joined and round identity
     * advances. With no overlapping drift, snapshot=4, anchor=2 => anchor=3. */
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 4.0f;
    rc |= expect_status("launch round 2", tc_diloco_apply_outer(d), TC_OK);
    rc |= expect_status("wait round 2", tc_diloco_async_wait(d), TC_OK);
    rc |= expect_status(
        "poll round 2",
        tc_diloco_async_poll(d, &state, &worker_status, &round_id),
        TC_OK);
    rc |= expect_true("round 2 identity", round_id == 2);
    rc |= expect_values("round 2 remains private before commit", values, 4.0f);
    rc |= expect_status("commit round 2", tc_diloco_async_commit(d), TC_OK);
    rc |= expect_values("round 2 commit", values, 3.0f);
    rc |= expect_true("two committed rounds", tc_diloco_outer_steps_completed(d) == 2);

    rc |= expect_status(
        "poll null state",
        tc_diloco_async_poll(d, NULL, &worker_status, &round_id),
        TC_ERR_INVALID_ARG);
    rc |= expect_status("wait null", tc_diloco_async_wait(NULL), TC_ERR_INVALID_ARG);
    rc |= expect_status("commit null", tc_diloco_async_commit(NULL), TC_ERR_INVALID_ARG);
    rc |= expect_true(
        "busy status text",
        strcmp(tc_status_string(TC_ERR_BUSY),
               "operation busy or result pending") == 0);

    rc |= expect_status("finalize", tc_diloco_finalize(d), TC_OK);
    d = NULL;

    /* Finalize must preserve and report a successful but uncommitted result
     * instead of silently counting, applying, or destroying it. */
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 1.0f;
    rc |= expect_status("second diloco init", tc_diloco_init(dist, &cfg, &d), TC_OK);
    rc |= expect_status(
        "second add parameter",
        tc_diloco_add_parameter(d, "p", theta, N_ELEMS, TC_DTYPE_F32),
        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 2.0f;
    rc |= expect_status("launch uncommitted", tc_diloco_apply_outer(d), TC_OK);
    rc |= expect_status("wait uncommitted", tc_diloco_async_wait(d), TC_OK);
    rc |= expect_status("finalize uncommitted", tc_diloco_finalize(d), TC_ERR_BUSY);
    rc |= expect_values("finalize did not commit", values, 2.0f);
    rc |= expect_status("commit after refused finalize", tc_diloco_async_commit(d), TC_OK);
    rc |= expect_status("finalize after commit", tc_diloco_finalize(d), TC_OK);
    d = NULL;

done:
    if (d) (void)tc_diloco_finalize(d);
    if (theta) tc_buffer_free(ctx, theta);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);
    printf("diloco_async: %s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}
