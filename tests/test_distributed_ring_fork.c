/*
 * tensorcore — multi-PROCESS ring all-reduce via fork().
 *
 * Same algorithm as test_distributed_ring.c (which used threads) but each
 * rank is now a real OS process. This validates the algorithm at the same
 * boundary as multi-Mac TB5/RDMA: each rank only sees its own buffers + its
 * neighbor sockets, no shared memory.
 *
 * The transport-swap to multi-Mac is now a single change: replace `socketpair`
 * with `connect(AF_INET6 over Thunderbolt Bridge)` or RDMA verbs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include "tensorcore/tensorcore.h"

extern tc_status_t tc_dist_ring_pair_make(int world_size, int* out_socks);
typedef struct { int sock_left, sock_right; } tc_ring_state;
extern tc_status_t tc_dist_ring_local_allreduce_ex(tc_ring_state* st,
    int world_size, int rank, void* data, size_t elements, size_t elem_bytes,
    tc_reduce_op_t op);

#define WORLD 4
#define N_ELEMS 1024

int main(void) {
    int socks[2 * WORLD];
    tc_status_t s = tc_dist_ring_pair_make(WORLD, socks);
    if (s != TC_OK) {
        fprintf(stderr, "ring_pair_make: %s\n", tc_status_string(s)); return 1;
    }

    /* Build per-rank input buffers in this process (still in heap). After
     * fork(), each child has its own COW copy of these buffers. */
    float* bufs[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        bufs[r] = malloc(N_ELEMS * sizeof(float));
        for (int i = 0; i < N_ELEMS; ++i) bufs[r][i] = (float)(r + i);
    }
    /* Reference: simple sum. */
    float* ref = malloc(N_ELEMS * sizeof(float));
    for (int i = 0; i < N_ELEMS; ++i) {
        float v = 0;
        for (int r = 0; r < WORLD; ++r) v += bufs[r][i];
        ref[i] = v;
    }

    /* Fork N children, each runs the ring as its own process. */
    int pipes[WORLD][2];  /* child→parent result pipe */
    pid_t pids[WORLD];
    for (int r = 0; r < WORLD; ++r) pipe(pipes[r]);

    for (int r = 0; r < WORLD; ++r) {
        pids[r] = fork();
        if (pids[r] == 0) {
            /* Child r */
            close(pipes[r][0]);   /* close read end */
            /* Close all OTHER ranks' sockets that aren't ours, and all
             * OTHER ranks' result pipes. */
            for (int j = 0; j < WORLD; ++j) {
                if (j != r) {
                    close(socks[2*j + 0]);
                    close(socks[2*j + 1]);
                    close(pipes[j][1]);
                }
            }
            tc_ring_state st = { .sock_left = socks[2*r + 0],
                                 .sock_right = socks[2*r + 1] };
            tc_status_t rs = tc_dist_ring_local_allreduce_ex(
                &st, WORLD, r, bufs[r], N_ELEMS, sizeof(float), TC_REDUCE_SUM);
            /* Send result + the reduced buffer to parent via pipe. */
            int rc = (rs == TC_OK) ? 0 : (int)rs;
            write(pipes[r][1], &rc, sizeof(rc));
            write(pipes[r][1], bufs[r], N_ELEMS * sizeof(float));
            close(pipes[r][1]);
            close(st.sock_left); close(st.sock_right);
            _exit(0);
        }
    }
    /* Parent: close all child-only ends, collect results. */
    for (int r = 0; r < WORLD; ++r) {
        close(pipes[r][1]);
        close(socks[2*r + 0]);
        close(socks[2*r + 1]);
    }
    int all_ok = 1;
    double max_err = 0.0;
    float* result = malloc(N_ELEMS * sizeof(float));
    for (int r = 0; r < WORLD; ++r) {
        int rc = -1;
        read(pipes[r][0], &rc, sizeof(rc));
        read(pipes[r][0], result, N_ELEMS * sizeof(float));
        close(pipes[r][0]);
        if (rc != 0) { all_ok = 0; continue; }
        for (int i = 0; i < N_ELEMS; ++i) {
            double e = fabs((double)result[i] - (double)ref[i]);
            if (e > max_err) max_err = e;
        }
    }
    /* Reap. */
    for (int r = 0; r < WORLD; ++r) {
        int st_v = 0; waitpid(pids[r], &st_v, 0);
    }

    printf("ring_allreduce_fork WORLD=%d N=%d   max_abs_err=%.3e  %s\n",
           WORLD, N_ELEMS, max_err,
           (all_ok && max_err == 0.0) ? "OK" : "FAIL");

    for (int r = 0; r < WORLD; ++r) free(bufs[r]);
    free(ref); free(result);
    return (all_ok && max_err == 0.0) ? 0 : 5;
}
