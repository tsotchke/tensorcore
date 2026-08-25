/*
 * tests/test_remote_tensor_fork.c — validates the remote tensor-fetch
 * transport contract.
 *
 * Spawns a parent (weight server) and a child (compute client), then:
 *   1. Server registers a 23.6 MiB "expert" bank (Kimi K2.6 Q4 size).
 *   2. Client fetches it over loopback.
 *   3. Verifies the bytes match exactly (zero-copy server, byte-identical
 *      client copy).
 *   4. Reports throughput (MB/s) and per-fetch latency, plus a 10-fetch
 *      sustained run to expose any per-call overhead.
 *
 * This is the loopback baseline for cross-host WAN benchmarks: numbers
 * here set the upper bound for what the transport can deliver before
 * network goes in.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/remote_tensor.h"

#include <errno.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define EXPERT_BYTES (23u * 1024u * 1024u + 614400u)   /* 23.6 MiB Q4 bank */
#define N_FETCHES    10

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int run_server(uint16_t port, int ready_fd, int done_fd) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "server: tc_init failed\n"); return 1; }

    char url[64];
    snprintf(url, sizeof(url), "tcp://0.0.0.0:%u", port);
    tc_remote_ctx* srv = NULL;
    if (tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER, url, &srv) != TC_OK) {
        fprintf(stderr, "server: tc_remote_init failed\n");
        return 1;
    }

    uint8_t* expert = (uint8_t*)malloc(EXPERT_BYTES);
    if (!expert) { fprintf(stderr, "server: alloc failed\n"); return 1; }
    /* Fill with deterministic pattern the client can verify. */
    for (size_t i = 0; i < EXPERT_BYTES; ++i) expert[i] = (uint8_t)(i ^ (i >> 11));

    if (tc_remote_register_tensor(srv, "expert/L31/E5/W2", expert, EXPERT_BYTES) != TC_OK) {
        fprintf(stderr, "server: register failed\n");
        return 1;
    }
    const char ready = 1;
    if (write(ready_fd, &ready, 1) != 1) {
        fprintf(stderr, "server: ready signal failed\n");
        return 1;
    }

    /* Wait for the client to finish its correctness, throughput, and error
     * path checks. A byte counter alone races the final request: the warm-up
     * contributes bytes too, and shutdown can otherwise sever timed fetch 10. */
    struct pollfd completion = {.fd = done_fd, .events = POLLIN, .revents = 0};
    int poll_rc;
    do {
        poll_rc = poll(&completion, 1, 30000);
    } while (poll_rc < 0 && errno == EINTR);
    char done = 0;
    const int completion_ok = poll_rc == 1 && read(done_fd, &done, 1) == 1;
    const uint64_t expected = (uint64_t)EXPERT_BYTES * (N_FETCHES + 1);
    const uint64_t served = tc_remote_total_bytes_served(srv);
    const int bytes_ok = served == expected;

    tc_remote_shutdown(srv);
    free(expert);
    tc_shutdown(ctx);
    if (!completion_ok || !bytes_ok) {
        fprintf(stderr, "server: completion=%s bytes=%llu expected=%llu\n",
                completion_ok ? "yes" : "no",
                (unsigned long long)served,
                (unsigned long long)expected);
        return 1;
    }
    return 0;
}

static int run_client(uint16_t port, int ready_fd) {
    struct pollfd readiness = {.fd = ready_fd, .events = POLLIN, .revents = 0};
    int poll_rc;
    do {
        poll_rc = poll(&readiness, 1, 30000);
    } while (poll_rc < 0 && errno == EINTR);
    char ready = 0;
    if (poll_rc != 1 || read(ready_fd, &ready, 1) != 1 || ready != 1) {
        fprintf(stderr, "client: server readiness failed\n");
        return 1;
    }

    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "client: tc_init failed\n"); return 1; }
    tc_remote_ctx* cli = NULL;
    if (tc_remote_init(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT, NULL, &cli) != TC_OK) {
        fprintf(stderr, "client: tc_remote_init failed\n");
        return 1;
    }
    char url[64];
    snprintf(url, sizeof(url), "tcp://127.0.0.1:%u", port);
    const int peer = tc_remote_connect(cli, url);
    if (peer < 0) { fprintf(stderr, "client: connect failed\n"); return 1; }

    uint8_t* dst = (uint8_t*)malloc(EXPERT_BYTES);
    if (!dst) return 1;

    /* Warm fetch + correctness check. */
    if (tc_remote_tensor_fetch(cli, peer, "expert/L31/E5/W2", dst, EXPERT_BYTES) != TC_OK) {
        fprintf(stderr, "client: warm fetch failed\n");
        return 1;
    }
    int bad = 0;
    for (size_t i = 0; i < EXPERT_BYTES; ++i) {
        if (dst[i] != (uint8_t)(i ^ (i >> 11))) { bad = 1; break; }
    }
    printf("  byte-identical fetch          %s\n", bad ? "FAIL" : "OK");
    if (bad) return 1;

    /* Sustained-throughput run. */
    const double t0 = now();
    for (int i = 0; i < N_FETCHES; ++i) {
        if (tc_remote_tensor_fetch(cli, peer, "expert/L31/E5/W2", dst, EXPERT_BYTES) != TC_OK) {
            fprintf(stderr, "client: timed fetch %d failed\n", i);
            return 1;
        }
    }
    const double dt = now() - t0;
    const double mb = (double)EXPERT_BYTES * N_FETCHES / (1024.0 * 1024.0);
    printf("  %d × %u-byte fetches in %.3f s\n", N_FETCHES, EXPERT_BYTES, dt);
    printf("  throughput:                   %.1f MB/s\n", mb / dt);
    printf("  per-fetch latency:            %.2f ms\n", dt / N_FETCHES * 1000.0);

    /* Bad-name path: server should respond NOT_FOUND, client returns INVALID_ARG. */
    const tc_status_t miss = tc_remote_tensor_fetch(cli, peer, "expert/DOES_NOT_EXIST",
                                                     dst, EXPERT_BYTES);
    printf("  missing-name returns invalid: %s\n",
           miss == TC_ERR_INVALID_ARG ? "OK" : "FAIL");

    tc_remote_shutdown(cli);
    tc_shutdown(ctx);
    free(dst);
    return (miss == TC_ERR_INVALID_ARG) ? 0 : 1;
}

int main(void) {
    /* Loopback port the test owns for the duration of this run. */
    const uint16_t port = 49152 + (uint16_t)(getpid() & 0x3FF);

    int ready_pipe[2];
    if (pipe(ready_pipe) != 0) { perror("pipe"); return 1; }
    int done_pipe[2];
    if (pipe(done_pipe) != 0) { perror("pipe"); return 1; }

    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        close(ready_pipe[1]);
        close(done_pipe[0]);
        const int rc = run_client(port, ready_pipe[0]);
        close(ready_pipe[0]);
        const char done = 1;
        if (write(done_pipe[1], &done, 1) != 1) _exit(8);
        close(done_pipe[1]);
        return rc;
    }
    /* parent = server */
    close(ready_pipe[0]);
    close(done_pipe[1]);
    const int srv_rc = run_server(port, ready_pipe[1], done_pipe[0]);
    close(ready_pipe[1]);
    close(done_pipe[0]);
    int cstat = 0;
    waitpid(pid, &cstat, 0);
    const int cli_rc = WIFEXITED(cstat) ? WEXITSTATUS(cstat) : 1;

    const int rc = (srv_rc | cli_rc);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
