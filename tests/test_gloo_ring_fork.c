/*
 * tests/test_gloo_ring_fork.c - 4-rank fork-based test of the ring
 * all-reduce topology.
 *
 * Validates:
 *   1. Ring topology setup succeeds for world_size=4 (opt-in via
 *      TC_GLOO_RING=1).
 *   2. Ring reduce-scatter + all-gather produces correct sums.
 *   3. Per-rank result matches expected sum(1..world).
 *
 * The cross-continent ring case (test_dist_remote --world 4) hits NAT
 * issues with Tailscale's getpeername-based topology discovery; this
 * test exercises the same code path locally where loopback works.
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

int main(void) {
    /* Opt into ring topology. */
    setenv("TC_GLOO_RING", "1", 1);

    const int port = reserve_loopback_port();
    if (port <= 0 || port > 65000) {
        fprintf(stderr, "gloo_ring_fork: SKIP: no suitable loopback port\n");
        return 77;
    }
    char url[64];
    snprintf(url, sizeof(url), "tcp://127.0.0.1:%d", port);

    pid_t children[WORLD - 1];
    for (int r = 1; r < WORLD; ++r) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            /* Child: small delay to let rank 0 begin listening. */
            usleep(r * 100 * 1000);   /* stagger 100 ms per rank */
            exit(run_rank(r, url));
        }
        children[r - 1] = pid;
    }

    /* Parent runs rank 0. */
    printf("ring 4-rank fork test:\n");
    int rc = run_rank(0, url);

    for (int i = 0; i < WORLD - 1; ++i) {
        int status = 0;
        waitpid(children[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "child rank %d failed (status=%d)\n",
                    i + 1, WEXITSTATUS(status));
            rc = 1;
        }
    }

    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}

#endif
