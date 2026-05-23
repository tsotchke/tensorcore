/*
 * tests/test_gloo_ring_fork.c - 4-rank fork-based test of the ring
 * all-reduce topology.
 *
 * Validates:
 *   1. Ring topology setup succeeds for world_size=4 on IPv4 and IPv6
 *      loopback (opt-in via TC_GLOO_RING=1).
 *   2. Ring reduce-scatter + all-gather produces correct sums.
 *   3. Per-rank result matches expected sum(1..world).
 *   4. If direct ring neighbors are unreachable, init stays alive and
 *      collectives transparently fall back to the rank-0 broker.
 *
 * The cross-continent ring case is exercised manually with
 * test_dist_remote plus TC_GLOO_RING=1 / TC_GLOO_TRACE=1; this fork test
 * keeps the same direct-ring and coordinated-fallback paths in local CI.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/distributed.h"

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

#define WORLD 4

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

static int reserve_loopback_port_ipv6(void) {
    int fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int v6only = 1;
    (void)setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));
    struct sockaddr_in6 addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_loopback;
    addr.sin6_port = 0;
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) != 0) {
        close(fd);
        return -1;
    }
    const int port = (int)ntohs(addr.sin6_port);
    close(fd);
    return port;
}

static int run_rank(int rank, const char* url) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return 1;

    tc_dist_ctx* dist = NULL;
    if (tc_dist_init(ctx, TC_DIST_GLOO, WORLD, rank, url, &dist) != TC_OK) {
        fprintf(stderr, "[rank %d] dist_init failed\n", rank);
        return 1;
    }

    /* Small + medium + large to exercise chunk-padding edge cases. */
    const size_t sizes[] = { 17, 256, 4096, 65536 };
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); ++si) {
        const size_t N = sizes[si];
        tc_buffer* buf = NULL;
        if (tc_buffer_alloc(ctx, N * sizeof(float), &buf) != TC_OK) return 1;
        void* p = NULL; tc_buffer_map(buf, &p);
        float* data = (float*)p;
        for (size_t i = 0; i < N; ++i) data[i] = (float)(rank + 1);

        if (tc_allreduce(dist, buf, N, TC_DTYPE_F32, TC_REDUCE_SUM) != TC_OK) {
            fprintf(stderr, "[rank %d] allreduce N=%zu failed\n", rank, N);
            return 1;
        }
        const float expected = (float)(WORLD * (WORLD + 1) / 2);
        for (size_t i = 0; i < N; ++i) {
            if (fabsf(data[i] - expected) > 1e-3f) {
                fprintf(stderr, "[rank %d] N=%zu idx=%zu got %.3f want %.3f\n",
                        rank, N, i, data[i], expected);
                return 1;
            }
        }
        tc_buffer_free(ctx, buf);
        if (rank == 0) printf("  N=%zu OK\n", N);
    }

    tc_barrier(dist);
    tc_dist_finalize(dist);
    tc_shutdown(ctx);
    return 0;
}

static int run_case(const char* label, const char* url, int force_broker_fallback) {
    setenv("TC_GLOO_RING", "1", 1);
    if (force_broker_fallback) {
        setenv("TC_GLOO_ADVERTISE_HOST", "203.0.113.1", 1);
        setenv("TC_GLOO_RING_CONNECT_TIMEOUT_MS", "100", 1);
    } else {
        unsetenv("TC_GLOO_ADVERTISE_HOST");
        unsetenv("TC_GLOO_RING_CONNECT_TIMEOUT_MS");
    }

    printf("%s:\n", label);
    fflush(stdout);

    pid_t children[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            /* Small delay to let rank 0 begin listening. */
            if (r > 0) usleep(r * 100 * 1000);   /* stagger 100 ms per rank */
            exit(run_rank(r, url));
        }
        children[r] = pid;
    }

    int rc = 0;
    for (int i = 0; i < WORLD; ++i) {
        int status = 0;
        waitpid(children[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            fprintf(stderr, "child rank %d failed (status=%d)\n",
                    i, code);
            rc = 1;
        }
    }

    printf("%s\n", rc ? "FAIL" : "OK");
    fflush(stdout);
    return rc;
}

int main(void) {
    int rc = 0;
    const int port4 = reserve_loopback_port();
    if (port4 <= 0 || port4 > 65000) {
        fprintf(stderr, "gloo_ring_fork: SKIP: no suitable IPv4 loopback port\n");
        return 77;
    }
    char url4[96];
    snprintf(url4, sizeof(url4), "tcp://127.0.0.1:%d", port4);
    rc |= run_case("ring 4-rank fork test ipv4", url4, 0);
#if defined(TC_TEST_METAL_BUILD)
    printf("ring 4-rank broker fallback test: SKIP in Metal build; covered by portable CPU build\n");
    fflush(stdout);
#else
    const int fallback_port4 = reserve_loopback_port();
    if (fallback_port4 <= 0 || fallback_port4 > 65000) {
        fprintf(stderr, "gloo_ring_fork: SKIP: no suitable IPv4 fallback port\n");
        return 77;
    }
    char fallback_url4[96];
    snprintf(fallback_url4, sizeof(fallback_url4), "tcp://127.0.0.1:%d", fallback_port4);
    rc |= run_case("ring 4-rank broker fallback test ipv4", fallback_url4, 1);
#endif

    const int port6 = reserve_loopback_port_ipv6();
    if (port6 <= 0 || port6 > 65000) {
        printf("ring 4-rank fork test ipv6: SKIP: no IPv6 loopback port\n");
    } else {
        char url6[96];
        snprintf(url6, sizeof(url6), "tcp://[::1]:%d", port6);
        rc |= run_case("ring 4-rank fork test ipv6", url6, 0);
#if defined(TC_TEST_METAL_BUILD)
        printf("ring 4-rank broker fallback test ipv6: SKIP in Metal build; covered by portable CPU build\n");
        fflush(stdout);
#else
        const int fallback_port6 = reserve_loopback_port_ipv6();
        if (fallback_port6 <= 0 || fallback_port6 > 65000) {
            printf("ring 4-rank broker fallback test ipv6: SKIP: no IPv6 loopback port\n");
        } else {
            char fallback_url6[96];
            snprintf(fallback_url6, sizeof(fallback_url6), "tcp://[::1]:%d", fallback_port6);
            rc |= run_case("ring 4-rank broker fallback test ipv6", fallback_url6, 1);
        }
#endif
    }
    unsetenv("TC_GLOO_RING");
    unsetenv("TC_GLOO_ADVERTISE_HOST");
    unsetenv("TC_GLOO_RING_CONNECT_TIMEOUT_MS");
    return rc;
}

#endif
