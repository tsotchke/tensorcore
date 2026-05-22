/*
 * tests/test_diloco_sparse_fork.c — multi-rank DiLoCo with TOPK_01PCT
 * compression over GLOO TCP.
 *
 * Validates the cross-continent bandwidth multiplier: the outer-step
 * allreduce ships (idx, fp16-val) sparse payloads instead of dense fp32.
 * For TOPK_01PCT on a 1024-element param the dense path would send 4096
 * bytes per rank; sparse sends 8 (header) + max(1, 1) * 8 = 16 bytes per
 * rank — ~256× less. Both ranks should still converge to within fp16
 * noise of the same θ.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/diloco.h"
#include "tensorcore/distributed.h"

#if defined(_WIN32)
int main(void) { return 77; }
#else

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define WORLD 2
#define N_ELEMS 1024

static int fail(const char* what, int rank) {
    fprintf(stderr, "[rank %d] diloco_sparse: FAIL: %s\n", rank, what);
    return 1;
}

static int reserve_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return -1; }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) != 0) { close(fd); return -1; }
    const int port = (int)ntohs(addr.sin_port);
    close(fd);
    return port;
}

static uint16_t f32_to_f16(float v) {
    union { float f; uint32_t u; } x = {v};
    const uint32_t bits = x.u;
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { float r; memcpy(&r, &sign, 4); return r; }
        int e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
        mant &= 0x03ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float r; memcpy(&r, &bits, 4); return r;
}

static int run_rank(int rank, const char* url) {
    int rc = 0;
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_diloco_ctx* d = NULL;
    tc_buffer* theta = NULL;
    uint16_t* t = NULL;

    if (tc_init(&ctx) != TC_OK) return fail("tc_init", rank);
    if (tc_dist_init(ctx, TC_DIST_GLOO, WORLD, rank, url, &dist) != TC_OK) {
        rc |= fail("dist_init", rank); goto done;
    }

    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 1;
    cfg.outer_lr = 1.0f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = TC_DILOCO_COMPRESS_TOPK_01PCT;     /* 0.1% — sparse path */

    if (tc_diloco_init(dist, &cfg, &d) != TC_OK) {
        rc |= fail("diloco_init", rank); goto done;
    }

    if (tc_buffer_alloc(ctx, N_ELEMS * sizeof(uint16_t), &theta) != TC_OK) {
        rc |= fail("buffer_alloc", rank); goto done;
    }
    if (tc_buffer_map(theta, (void**)&t) != TC_OK) {
        rc |= fail("buffer_map", rank); goto done;
    }
    /* Initial θ = 1.0 everywhere. */
    for (int i = 0; i < N_ELEMS; ++i) t[i] = f32_to_f16(1.0f);

    if (tc_diloco_add_parameter(d, "p", theta, N_ELEMS, TC_DTYPE_F16) != TC_OK) {
        rc |= fail("add_parameter", rank); goto done;
    }

    /* Each rank's one inner step: create a large spike at one location
     * (idx = rank * 10) so top-k 0.1% has a clear winner. Other entries
     * stay at zero gradient.
     *
     * After step:
     *   rank 0: θ[0] = 1.0 + 10.0 = 11.0; others = 1.0
     *   rank 1: θ[10] = 1.0 + 20.0 = 21.0; others = 1.0
     * Δθ_rank0 = [10, 0, ..., 0]            (top-1 keeps idx 0 with val 10)
     * Δθ_rank1 = [0,...0,20,0,...0]         (top-1 keeps idx 10 with val 20)
     * Sparse allreduce sum → dense [10, 0,...0, 20, 0,...0]
     * Outer SGD lr=1.0 AVG = sum/2 → [5, 0,...0, 10, 0,...0]
     * θ_anchor = 1.0 + this → [6, 1,...,1, 11, 1,...,1]
     * θ_local resynced from θ_anchor.
     */
    const int spike_idx = rank * 10;
    const float spike_val = (rank == 0) ? 10.0f : 20.0f;
    t[spike_idx] = f32_to_f16(1.0f + spike_val);

    bool outer_pending = false;
    if (tc_diloco_step(d, &outer_pending) != TC_OK) rc |= fail("step", rank);
    if (!outer_pending) rc |= fail("outer not pending", rank);
    if (tc_diloco_apply_outer(d) != TC_OK) rc |= fail("apply_outer", rank);

    /* Both ranks should now hold the merged anchor. */
    const float expected_0  = 1.0f + 10.0f / 2.0f;   /* 6.0 */
    const float expected_10 = 1.0f + 20.0f / 2.0f;   /* 11.0 */
    int wrong = 0;
    for (int i = 0; i < N_ELEMS; ++i) {
        float want = 1.0f;
        if (i == 0)  want = expected_0;
        if (i == 10) want = expected_10;
        const float got = f16_to_f32(t[i]);
        if (fabsf(got - want) > 0.1f) {
            fprintf(stderr, "[rank %d] idx %d: want %.3f got %.3f\n", rank, i, want, got);
            wrong = 1;
            break;
        }
    }
    if (wrong) rc |= fail("post-outer θ", rank);

    /* Verify the bandwidth claim: the bytes sent reported by DiLoCo
     * should be ~payload size for sparse, much less than N_ELEMS*4. */
    const double bytes_sent = tc_diloco_last_outer_bytes_sent(d);
    const double dense_bytes = N_ELEMS * sizeof(float);
    if (bytes_sent >= dense_bytes) {
        fprintf(stderr, "[rank %d] sparse bytes_sent=%.0f (expected < %.0f dense)\n",
                rank, bytes_sent, dense_bytes);
        rc |= fail("bandwidth reduction", rank);
    } else {
        printf("[rank %d] bandwidth: %.0f bytes (vs %.0f dense, %.1fx less)\n",
               rank, bytes_sent, dense_bytes, dense_bytes / bytes_sent);
    }

done:
    if (d) tc_diloco_finalize(d);
    if (theta) tc_buffer_free(ctx, theta);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);
    return rc ? 1 : 0;
}

int main(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) { fprintf(stderr, "diloco_sparse: SKIP\n"); return 77; }
    char url[96];
    snprintf(url, sizeof(url), "tcp://127.0.0.1:%d", port);
    alarm(45);

    pid_t child = fork();
    if (child < 0) return fail("fork", -1);
    if (child == 0) {
        const int rc = run_rank(1, url);
        printf("[rank 1] diloco_sparse: %s\n", rc ? "FAIL" : "OK");
        fflush(stdout);
        fflush(stderr);
        _exit(rc ? 1 : 0);
    }

    const int parent_rc = run_rank(0, url);
    int status = 0;
    if (waitpid(child, &status, 0) < 0) return fail("waitpid", 0);
    const int child_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    printf("[rank 0] diloco_sparse: %s   (child=%s)\n",
           parent_rc ? "FAIL" : "OK", child_ok ? "OK" : "FAIL");
    return (parent_rc == 0 && child_ok) ? 0 : 1;
}

#endif
