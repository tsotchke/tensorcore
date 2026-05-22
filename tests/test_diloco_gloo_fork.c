/*
 * tests/test_diloco_gloo_fork.c - multi-rank DiLoCo over GLOO via fork().
 *
 * Two local processes initialize TC_DIST_GLOO over TCP, register the same
 * fp16 parameter with DiLoCo, take different inner updates, and verify that
 * the outer step averages delta-theta across ranks before resynchronizing
 * both local parameters to the same anchor.
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
#define N_ELEMS 8

static int fail(const char* what, int rank) {
    fprintf(stderr, "[rank %d] diloco_gloo: FAIL: %s\n", rank, what);
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
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) != 0) {
        close(fd);
        return -1;
    }
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
    if (rounded & 0x800000u) {
        rounded = 0;
        ++half_exp;
        if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            float r;
            memcpy(&r, &sign, 4);
            return r;
        }
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &bits, 4);
    return r;
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
        rc |= fail("dist_init GLOO", rank);
        goto done;
    }

    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 3;
    cfg.outer_lr = 1.0f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = TC_DILOCO_COMPRESS_NONE;

    if (tc_diloco_init(dist, &cfg, &d) != TC_OK) {
        rc |= fail("diloco_init", rank);
        goto done;
    }

    if (tc_buffer_alloc(ctx, N_ELEMS * sizeof(uint16_t), &theta) != TC_OK) {
        rc |= fail("buffer_alloc", rank);
        goto done;
    }
    if (tc_buffer_map(theta, (void**)&t) != TC_OK) {
        rc |= fail("buffer_map", rank);
        goto done;
    }
    for (int i = 0; i < N_ELEMS; ++i) t[i] = f32_to_f16(1.0f);

    if (tc_diloco_add_parameter(d, "p", theta, N_ELEMS, TC_DTYPE_F16) != TC_OK) {
        rc |= fail("add_parameter", rank);
        goto done;
    }

    /* rank 0: delta = +0.6 after 3 inner steps.
     * rank 1: delta = +1.2 after 3 inner steps.
     * Average delta = +0.9, outer SGD lr=1.0, initial anchor=1.0,
     * so both ranks should resynchronize to theta=1.9. */
    const float per_step_delta = (rank == 0) ? 0.2f : 0.4f;
    bool outer_pending = false;
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < N_ELEMS; ++i) {
            t[i] = f32_to_f16(f16_to_f32(t[i]) + per_step_delta);
        }
        if (tc_diloco_step(d, &outer_pending) != TC_OK) rc |= fail("diloco_step", rank);
    }
    if (!outer_pending) rc |= fail("outer not pending", rank);
    if (tc_diloco_apply_outer(d) != TC_OK) rc |= fail("apply_outer", rank);

    for (int i = 0; i < N_ELEMS; ++i) {
        const float got = f16_to_f32(t[i]);
        if (fabsf(got - 1.9f) > 5e-2f) {
            fprintf(stderr, "[rank %d] idx %d: want %.3f got %.3f\n", rank, i, 1.9f, got);
            rc |= fail("post-outer theta mismatch", rank);
            break;
        }
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
    if (port <= 0) {
        fprintf(stderr, "diloco_gloo: SKIP: no loopback port\n");
        return 77;
    }

    char url[96];
    snprintf(url, sizeof(url), "gloo+tcp://127.0.0.1:%d", port);
    alarm(45);

    pid_t child = fork();
    if (child < 0) return fail("fork", -1);
    if (child == 0) {
        const int rc = run_rank(1, url);
        printf("[rank 1] diloco_gloo: %s\n", rc ? "FAIL" : "OK");
        _exit(rc ? 1 : 0);
    }

    const int parent_rc = run_rank(0, url);

    int status = 0;
    if (waitpid(child, &status, 0) < 0) return fail("waitpid", 0);
    const int child_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;

    printf("[rank 0] diloco_gloo: %s   (child=%s)\n",
           parent_rc ? "FAIL" : "OK", child_ok ? "OK" : "FAIL");
    return (parent_rc == 0 && child_ok) ? 0 : 1;
}

#endif
