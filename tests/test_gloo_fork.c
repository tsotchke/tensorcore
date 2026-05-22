/*
 * tests/test_gloo_fork.c - portable CPU TC_DIST_GLOO TCP smoke.
 *
 * Spawns world_size=2 processes via fork(). Rank 0 listens on
 * 127.0.0.1:<free port>; rank 1 connects. Each rank initializes the public
 * tc_dist_ctx with TC_DIST_GLOO, then validates fp32/fp16 allreduce,
 * any-root fp32 broadcast, allgather, and barrier.
 */

#include "tensorcore/tensorcore.h"

#if defined(_WIN32)
int main(void) { return 77; }
#else

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define WORLD 2
#define N_ELEMS 16

static int fail_rank(int rank, const char* what) {
    fprintf(stderr, "[rank %d] gloo_fork: FAIL: %s\n", rank, what);
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

static int expect_status(int rank, const char* name, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    fprintf(stderr, "[rank %d] %s: got %s want %s\n",
            rank, name, tc_status_string(got), tc_status_string(want));
    return 1;
}

static int expect_close(int rank, const char* name, float got, float want, float tol) {
    if (fabsf(got - want) <= tol) return 0;
    fprintf(stderr, "[rank %d] %s: got %.8g want %.8g\n", rank, name, got, want);
    return 1;
}

static int run_rank(int rank, const char* url) {
    int rc = 0;
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_buffer* buf32 = NULL;
    tc_buffer* buf16 = NULL;
    tc_buffer* gather_out = NULL;
    float* p32 = NULL;
    uint16_t* p16 = NULL;
    float* gathered = NULL;

    rc |= expect_status(rank, "tc_init", tc_init(&ctx), TC_OK);
    if (rc) goto done;

    rc |= expect_status(rank, "tc_dist_init GLOO",
                        tc_dist_init(ctx, TC_DIST_GLOO, WORLD, rank, url, &dist),
                        TC_OK);
    if (rc) goto done;
    if (tc_dist_world_size(dist) != WORLD || tc_dist_rank(dist) != rank) {
        rc |= fail_rank(rank, "dist rank metadata");
    }

    rc |= expect_status(rank, "alloc fp32",
                        tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), &buf32),
                        TC_OK);
    if (rc) goto done;
    rc |= expect_status(rank, "map fp32", tc_buffer_map(buf32, (void**)&p32), TC_OK);
    if (rc) goto done;

    for (size_t i = 0; i < N_ELEMS; ++i) {
        p32[i] = (rank == 0) ? (float)(i + 1) : (float)(100 + i);
    }
    rc |= expect_status(rank, "fp32 SUM",
                        tc_allreduce(dist, buf32, N_ELEMS, TC_DTYPE_F32, TC_REDUCE_SUM),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "fp32 SUM value", p32[i], (float)(101 + 2 * (int)i), 1e-6f);
    }

    rc |= expect_status(rank, "alloc fp16",
                        tc_buffer_alloc(ctx, N_ELEMS * sizeof(uint16_t), &buf16),
                        TC_OK);
    if (rc) goto done;
    rc |= expect_status(rank, "map fp16", tc_buffer_map(buf16, (void**)&p16), TC_OK);
    if (rc) goto done;
    for (size_t i = 0; i < N_ELEMS; ++i) {
        p16[i] = f32_to_f16(rank == 0 ? 0.5f * (float)(i + 1) : 1.5f * (float)(i + 1));
    }
    rc |= expect_status(rank, "fp16 SUM",
                        tc_allreduce(dist, buf16, N_ELEMS, TC_DTYPE_F16, TC_REDUCE_SUM),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "fp16 SUM value",
                           f16_to_f32(p16[i]), 2.0f * (float)(i + 1), 1e-2f);
    }

    for (size_t i = 0; i < N_ELEMS; ++i) {
        p32[i] = (rank == 0) ? (float)(i + 2) : (float)(10 + i);
    }
    rc |= expect_status(rank, "fp32 AVG",
                        tc_allreduce(dist, buf32, N_ELEMS, TC_DTYPE_F32, TC_REDUCE_AVG),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "fp32 AVG value", p32[i], (float)(i + 6), 1e-6f);
    }

    for (size_t i = 0; i < N_ELEMS; ++i) p32[i] = (rank == 0) ? (float)(10 + i) : (float)(5 + i);
    rc |= expect_status(rank, "fp32 MIN",
                        tc_allreduce(dist, buf32, N_ELEMS, TC_DTYPE_F32, TC_REDUCE_MIN),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "fp32 MIN value", p32[i], (float)(5 + i), 1e-6f);
    }

    for (size_t i = 0; i < N_ELEMS; ++i) p32[i] = (rank == 0) ? (float)(10 + i) : (float)(5 + i);
    rc |= expect_status(rank, "fp32 MAX",
                        tc_allreduce(dist, buf32, N_ELEMS, TC_DTYPE_F32, TC_REDUCE_MAX),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "fp32 MAX value", p32[i], (float)(10 + i), 1e-6f);
    }

    for (size_t i = 0; i < N_ELEMS; ++i) p32[i] = (rank == 0) ? (float)(42 + i) : -1.0f;
    rc |= expect_status(rank, "broadcast root 0",
                        tc_broadcast(dist, buf32, N_ELEMS, TC_DTYPE_F32, 0),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "broadcast value", p32[i], (float)(42 + i), 1e-6f);
    }

    for (size_t i = 0; i < N_ELEMS; ++i) p32[i] = (rank == 1) ? (float)(70 + i) : -1.0f;
    rc |= expect_status(rank, "broadcast root 1",
                        tc_broadcast(dist, buf32, N_ELEMS, TC_DTYPE_F32, 1),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "broadcast root 1 value", p32[i], (float)(70 + i), 1e-6f);
    }

    rc |= expect_status(rank, "alloc allgather out",
                        tc_buffer_alloc(ctx, WORLD * N_ELEMS * sizeof(float), &gather_out),
                        TC_OK);
    if (rc) goto done;
    rc |= expect_status(rank, "map allgather out",
                        tc_buffer_map(gather_out, (void**)&gathered),
                        TC_OK);
    if (rc) goto done;
    for (size_t i = 0; i < N_ELEMS; ++i) p32[i] = (float)(rank * 100 + (int)i);
    rc |= expect_status(rank, "allgather",
                        tc_allgather(dist, buf32, gather_out, N_ELEMS, TC_DTYPE_F32),
                        TC_OK);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        rc |= expect_close(rank, "allgather rank 0", gathered[i], (float)i, 1e-6f);
        rc |= expect_close(rank, "allgather rank 1",
                           gathered[N_ELEMS + i], (float)(100 + (int)i), 1e-6f);
    }

    rc |= expect_status(rank, "barrier", tc_barrier(dist), TC_OK);

done:
    if (gather_out) tc_buffer_free(ctx, gather_out);
    if (buf16) tc_buffer_free(ctx, buf16);
    if (buf32) tc_buffer_free(ctx, buf32);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);
    return rc ? 1 : 0;
}

int main(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) {
        fprintf(stderr, "gloo_fork: SKIP: no loopback port\n");
        return 77;
    }

    char url[96];
    snprintf(url, sizeof(url), "gloo+tcp://127.0.0.1:%d", port);

    int pipes[WORLD][2];
    pid_t pids[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        if (pipe(pipes[r]) != 0) return fail_rank(-1, "pipe");
    }

    for (int r = 0; r < WORLD; ++r) {
        pids[r] = fork();
        if (pids[r] < 0) return fail_rank(-1, "fork");
        if (pids[r] == 0) {
            alarm(30);
            close(pipes[r][0]);
            for (int j = 0; j < WORLD; ++j) {
                if (j != r) {
                    close(pipes[j][0]);
                    close(pipes[j][1]);
                }
            }
            const int rc = run_rank(r, url);
            (void)write(pipes[r][1], &rc, sizeof(rc));
            close(pipes[r][1]);
            _exit(rc ? 1 : 0);
        }
    }

    int ok = 1;
    for (int r = 0; r < WORLD; ++r) close(pipes[r][1]);
    for (int r = 0; r < WORLD; ++r) {
        int rc = 1;
        const ssize_t n = read(pipes[r][0], &rc, sizeof(rc));
        close(pipes[r][0]);
        if (n != (ssize_t)sizeof(rc) || rc != 0) ok = 0;
    }
    for (int r = 0; r < WORLD; ++r) {
        int status = 0;
        if (waitpid(pids[r], &status, 0) < 0 ||
            !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            ok = 0;
        }
    }

    printf("gloo_fork world=%d elements=%d %s\n", WORLD, N_ELEMS, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

#endif
