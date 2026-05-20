/*
 * tensorcore — single-host ring all-reduce correctness.
 *
 * Spawns N pthreads, each acting as a ring rank, connects them via
 * socketpair-created UDS pairs, runs ring_local_allreduce_ex on a fp32
 * vector. Validates result equals the single-process sum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include "tensorcore/tensorcore.h"

extern tc_status_t tc_dist_ring_pair_make(int world_size, int* out_socks);
typedef struct { int sock_left, sock_right; } tc_ring_state;
extern tc_status_t tc_dist_ring_local_allreduce_ex(tc_ring_state* st,
    int world_size, int rank, void* data, size_t elements, size_t elem_bytes,
    tc_reduce_op_t op);

#define WORLD 4
#define N_ELEMS 1024

typedef struct {
    int rank;
    tc_ring_state st;
    float* buf;
    int result;
} arg_t;

static void* rank_thread(void* p) {
    arg_t* a = (arg_t*)p;
    tc_status_t s = tc_dist_ring_local_allreduce_ex(
        &a->st, WORLD, a->rank, a->buf, N_ELEMS, sizeof(float), TC_REDUCE_SUM);
    a->result = (s == TC_OK) ? 0 : (int)s;
    return NULL;
}

int main(void) {
    /* Build the ring sockets. */
    int socks[2 * WORLD];
    tc_status_t s = tc_dist_ring_pair_make(WORLD, socks);
    if (s != TC_OK) {
        fprintf(stderr, "ring_pair_make: %s\n", tc_status_string(s));
        return 1;
    }

    /* Per-rank input: rank r gets values [r, r+1, r+2, ...]. */
    float* bufs[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        bufs[r] = (float*)malloc(N_ELEMS * sizeof(float));
        for (int i = 0; i < N_ELEMS; ++i) bufs[r][i] = (float)(r + i);
    }

    /* Reference: simple sum across ranks. */
    float* ref = (float*)malloc(N_ELEMS * sizeof(float));
    for (int i = 0; i < N_ELEMS; ++i) {
        ref[i] = 0;
        for (int r = 0; r < WORLD; ++r) ref[i] += bufs[r][i];
    }

    /* Spawn threads. */
    pthread_t ths[WORLD];
    arg_t args[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        args[r].rank = r;
        args[r].st.sock_left  = socks[2*r + 0];
        args[r].st.sock_right = socks[2*r + 1];
        args[r].buf = bufs[r];
        args[r].result = -1;
        pthread_create(&ths[r], NULL, rank_thread, &args[r]);
    }
    for (int r = 0; r < WORLD; ++r) pthread_join(ths[r], NULL);

    /* Each rank should now hold the reduced sum. */
    int all_ok = 1;
    double max_err = 0.0;
    for (int r = 0; r < WORLD; ++r) {
        if (args[r].result != 0) { all_ok = 0; continue; }
        for (int i = 0; i < N_ELEMS; ++i) {
            double e = fabs((double)bufs[r][i] - (double)ref[i]);
            if (e > max_err) max_err = e;
        }
    }

    printf("ring_allreduce world=%d elements=%d   max_abs_err=%.3e  %s\n",
           WORLD, N_ELEMS, max_err,
           (all_ok && max_err == 0.0) ? "OK" : "FAIL");

    for (int r = 0; r < WORLD; ++r) {
        free(bufs[r]);
        close(socks[2*r + 0]);
        close(socks[2*r + 1]);
    }
    free(ref);
    return (all_ok && max_err == 0.0) ? 0 : 5;
}
