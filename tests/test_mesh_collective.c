/*
 * test_mesh_collective.c — fork-based 2-peer mesh collective smoke.
 *
 * Parent process = rank 0, child = rank 1. Both bind tc_remote servers
 * on distinct localhost ports, connect to each other, and run a small
 * battery: AllReduce SUM, AllReduce AVG, Broadcast, AllGather. Then
 * byte-identical comparison of expected vs actual results.
 *
 * Bound to localhost, but the transport is the production
 * tc_remote_tensor_fetch — same wire protocol used across machines.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/mesh_collective.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <math.h>

static int compare_f32(const float* a, const float* b, size_t n, float tol) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "    differs at i=%zu: %g vs %g (delta %g)\n",
                    i, a[i], b[i], a[i] - b[i]);
            return 0;
        }
    }
    return 1;
}

static int run_peer(int my_rank,
                     const char* my_url, const char* peer_url) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "rank %d: tc_init failed\n", my_rank); return 1; }

    const char* urls[2];
    urls[0] = (my_rank == 0) ? my_url : peer_url;
    urls[1] = (my_rank == 1) ? my_url : peer_url;

    tc_mesh_group_t* g = NULL;
    if (tc_mesh_group_init(ctx, 2, my_rank, urls, &g) != TC_OK) {
        fprintf(stderr, "rank %d: group_init failed\n", my_rank);
        tc_shutdown(ctx);
        return 1;
    }

    int fails = 0;
    const size_t N = 32;

    /* ===== AllReduce SUM ===== */
    {
        float buf[32];
        for (size_t i = 0; i < N; ++i) buf[i] = (float)(my_rank + 1) * (float)(i + 1);  /* rank0: 1*(i+1), rank1: 2*(i+1) */
        tc_status_t s = tc_mesh_allreduce(g, buf, N, TC_COLL_DTYPE_F32, TC_REDUCE_SUM);
        if (s != TC_OK) { fprintf(stderr, "rank %d: AR SUM failed status=%d\n", my_rank, s); fails++; }
        /* Expected: sum = 1*(i+1) + 2*(i+1) = 3*(i+1) */
        float expect[32];
        for (size_t i = 0; i < N; ++i) expect[i] = 3.0f * (float)(i + 1);
        if (!compare_f32(buf, expect, N, 1e-5f)) {
            fprintf(stderr, "rank %d: AR SUM mismatch\n", my_rank); fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank %d AllReduce SUM\n", my_rank);
        }
    }

    /* ===== AllReduce AVG ===== */
    {
        float buf[32];
        for (size_t i = 0; i < N; ++i) buf[i] = (float)(my_rank + 1) * (float)(i + 1);
        tc_status_t s = tc_mesh_allreduce(g, buf, N, TC_COLL_DTYPE_F32, TC_REDUCE_AVG);
        if (s != TC_OK) { fprintf(stderr, "rank %d: AR AVG failed status=%d\n", my_rank, s); fails++; }
        /* Expected: avg of 1*(i+1) and 2*(i+1) = 1.5*(i+1) */
        float expect[32];
        for (size_t i = 0; i < N; ++i) expect[i] = 1.5f * (float)(i + 1);
        if (!compare_f32(buf, expect, N, 1e-5f)) {
            fprintf(stderr, "rank %d: AR AVG mismatch\n", my_rank); fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank %d AllReduce AVG\n", my_rank);
        }
    }

    /* ===== Broadcast from rank 0 ===== */
    {
        float buf[32];
        if (my_rank == 0) {
            for (size_t i = 0; i < N; ++i) buf[i] = (float)i * 0.5f - 8.0f;  /* known pattern */
        } else {
            for (size_t i = 0; i < N; ++i) buf[i] = -999.0f;  /* sentinel */
        }
        tc_status_t s = tc_mesh_broadcast(g, buf, N, TC_COLL_DTYPE_F32, 0);
        if (s != TC_OK) { fprintf(stderr, "rank %d: BC failed status=%d\n", my_rank, s); fails++; }
        float expect[32];
        for (size_t i = 0; i < N; ++i) expect[i] = (float)i * 0.5f - 8.0f;
        if (!compare_f32(buf, expect, N, 1e-6f)) {
            fprintf(stderr, "rank %d: BC mismatch\n", my_rank); fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank %d Broadcast\n", my_rank);
        }
    }

    /* ===== AllGather ===== */
    {
        float send[16], recv[32];  /* 16 per rank, 2 ranks → 32 total */
        for (size_t i = 0; i < 16; ++i) send[i] = (float)(my_rank * 100 + i);
        tc_status_t s = tc_mesh_allgather(g, send, 16, recv, TC_COLL_DTYPE_F32);
        if (s != TC_OK) { fprintf(stderr, "rank %d: AG failed status=%d\n", my_rank, s); fails++; }
        /* Expected: recv[0..16) = rank 0's send, recv[16..32) = rank 1's send */
        float expect[32];
        for (size_t i = 0; i < 16; ++i) expect[i]      = (float)i;          /* rank 0: 0..15 */
        for (size_t i = 0; i < 16; ++i) expect[16 + i] = (float)(100 + i);  /* rank 1: 100..115 */
        if (!compare_f32(recv, expect, 32, 1e-6f)) {
            fprintf(stderr, "rank %d: AG mismatch\n", my_rank); fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank %d AllGather\n", my_rank);
        }
    }

    if (my_rank == 0) {
        printf("  total mesh bytes shipped through rank %d: %llu\n",
               my_rank, (unsigned long long)tc_mesh_total_bytes(g));
    }

    tc_mesh_group_shutdown(g);
    tc_shutdown(ctx);
    return fails;
}

int main(void) {
    const char* rank0_url = "tcp://127.0.0.1:54310";
    const char* rank1_url = "tcp://127.0.0.1:54311";

    pid_t child = fork();
    if (child < 0) { perror("fork"); return 1; }

    if (child == 0) {
        /* Child = rank 1. Stagger startup so rank 0 binds first. */
        sleep(1);
        int rc = run_peer(1, rank1_url, rank0_url);
        _exit(rc == 0 ? 0 : 1);
    }

    /* Parent = rank 0. */
    int rc0 = run_peer(0, rank0_url, rank1_url);

    int status = 0;
    waitpid(child, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

    const int total = rc0 + rc1;
    printf("\n  rank0 fails=%d  rank1 fails=%d  → %s\n",
           rc0, rc1, total == 0 ? "ALL PASS" : "SOME FAIL");
    return total == 0 ? 0 : 1;
}
