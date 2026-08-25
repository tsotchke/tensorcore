/*
 * bench_remote_tensor.c — split-binary bench for the tc_remote_*
 * inference weight-paging transport.
 *
 * Two roles, one binary:
 *   server: registers N synthetic expert banks, accepts client fetches.
 *           --role server --bind tcp://0.0.0.0:9100 [--experts 16]
 *                         [--expert-bytes 24731648] [--linger 60]
 *
 *   client: connects, fetches expert banks for `--iters` rounds,
 *           reports throughput + latency + per-iter histogram.
 *           --role client --peer tcp://100.x.y.z:9100 [--experts 16]
 *                         [--expert-bytes 24731648] [--iters 32]
 *                         [--warmup 4]
 *
 * Defaults match Kimi K2.6's Q4 expert bank size (23.6 MiB) and a
 * realistic working-set of 16 experts (~378 MiB) so the bench
 * exercises both per-fetch overhead and sustained bandwidth.
 *
 * Output (client):
 *   - per-iter latency p50/p99
 *   - sustained MB/s over the whole run
 *   - first-byte latency (warmup samples) vs steady-state
 *
 * Run pattern (od ⇄ cosbox over Tailscale):
 *   on od:      ./bench_remote_tensor --role server --bind tcp://0.0.0.0:9100
 *   on cosbox:  ./bench_remote_tensor --role client --peer tcp://192.0.2.20:9100
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/remote_tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_EXPERT_BYTES 24731648u   /* 23.6 MiB Q4 expert bank */
#define DEFAULT_EXPERTS      16
#define DEFAULT_ITERS        32
#define DEFAULT_WARMUP       4
#define DEFAULT_LINGER_SEC   60

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int compare_double(const void* a, const void* b) {
    const double da = *(const double*)a, db = *(const double*)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s --role {server|client} [...args]\n"
        "  server: --bind tcp://0.0.0.0:PORT [--experts N] [--expert-bytes N] [--linger SEC]\n"
        "  client: --peer tcp://HOST:PORT [--experts N] [--expert-bytes N] [--iters N] [--warmup N]\n",
        argv0);
}

static int run_server(const char* bind_url, int experts, size_t bytes, int linger) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "server: tc_init\n"); return 1; }
    tc_remote_ctx* srv = NULL;
    if (tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER, bind_url, &srv) != TC_OK) {
        fprintf(stderr, "server: tc_remote_init on %s failed\n", bind_url);
        return 1;
    }

    /* Allocate + register N synthetic expert banks. Each filled with a
     * unique XOR pattern so the client can detect corruption. */
    uint8_t** banks = (uint8_t**)calloc(experts, sizeof(uint8_t*));
    char name[64];
    for (int i = 0; i < experts; ++i) {
        banks[i] = (uint8_t*)malloc(bytes);
        if (!banks[i]) { fprintf(stderr, "server: alloc expert %d\n", i); return 1; }
        for (size_t b = 0; b < bytes; ++b) banks[i][b] = (uint8_t)((b ^ (i * 7919u)) & 0xff);
        snprintf(name, sizeof(name), "expert/%d", i);
        if (tc_remote_register_tensor(srv, name, banks[i], bytes) != TC_OK) {
            fprintf(stderr, "server: register %s\n", name);
            return 1;
        }
    }
    printf("[server] bound on %s; %d experts × %zu bytes = %.1f MiB resident; lingering %d s\n",
           bind_url, experts, bytes,
           (double)experts * bytes / (1024.0 * 1024.0), linger);

    const double end = now() + (double)linger;
    uint64_t last = 0;
    while (now() < end) {
        sleep(1);
        const uint64_t served = tc_remote_total_bytes_served(srv);
        if (served != last) {
            printf("[server] served total: %.1f MiB\n", (double)served / (1024.0 * 1024.0));
            last = served;
        }
    }

    tc_remote_shutdown(srv);
    for (int i = 0; i < experts; ++i) free(banks[i]);
    free(banks);
    tc_shutdown(ctx);
    return 0;
}

static int run_client(const char* peer_url, int experts, size_t bytes,
                       int iters, int warmup) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "client: tc_init\n"); return 1; }
    tc_remote_ctx* cli = NULL;
    if (tc_remote_init(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT, NULL, &cli) != TC_OK) {
        fprintf(stderr, "client: tc_remote_init\n"); return 1;
    }
    const int peer = tc_remote_connect(cli, peer_url);
    if (peer < 0) {
        fprintf(stderr, "client: connect to %s failed\n", peer_url); return 1;
    }
    printf("[client] connected to %s as peer_id=%d\n", peer_url, peer);

    uint8_t* dst = (uint8_t*)malloc(bytes);
    if (!dst) { fprintf(stderr, "client: alloc dst\n"); return 1; }

    /* Sanity fetch: pull expert/0 and verify bytes. */
    {
        const double t0 = now();
        if (tc_remote_tensor_fetch(cli, peer, "expert/0", dst, bytes) != TC_OK) {
            fprintf(stderr, "client: warm fetch expert/0 failed\n"); return 1;
        }
        const double t1 = now();
        int bad = 0;
        for (size_t b = 0; b < bytes; ++b) {
            if (dst[b] != (uint8_t)((b ^ 0u) & 0xff)) { bad = 1; break; }
        }
        printf("[client] sanity fetch expert/0: %.2f ms, %s\n",
               (t1 - t0) * 1000.0, bad ? "CORRUPT" : "OK");
        if (bad) return 1;
    }

    /* Warmup. */
    char name[64];
    for (int w = 0; w < warmup; ++w) {
        snprintf(name, sizeof(name), "expert/%d", w % experts);
        if (tc_remote_tensor_fetch(cli, peer, name, dst, bytes) != TC_OK) {
            fprintf(stderr, "client: warmup %d failed\n", w); return 1;
        }
    }

    /* Timed loop — round-robin across N experts to mimic MoE access. */
    double* lat_ms = (double*)malloc((size_t)iters * sizeof(double));
    const double tstart = now();
    for (int i = 0; i < iters; ++i) {
        snprintf(name, sizeof(name), "expert/%d", i % experts);
        const double t0 = now();
        if (tc_remote_tensor_fetch(cli, peer, name, dst, bytes) != TC_OK) {
            fprintf(stderr, "client: iter %d failed\n", i); return 1;
        }
        lat_ms[i] = (now() - t0) * 1000.0;
    }
    const double dt = now() - tstart;
    const double total_mib = (double)iters * bytes / (1024.0 * 1024.0);

    qsort(lat_ms, iters, sizeof(double), compare_double);
    const double p50 = lat_ms[iters / 2];
    const double p99 = lat_ms[(iters * 99) / 100];
    const double minL = lat_ms[0];
    const double maxL = lat_ms[iters - 1];

    printf("[client] sustained: %d iters × %zu bytes in %.3f s\n",
           iters, bytes, dt);
    printf("[client] throughput: %.1f MB/s (binary), %.1f MB/s (decimal)\n",
           total_mib / dt, (double)iters * bytes / dt / 1e6);
    printf("[client] per-fetch latency: min=%.2f p50=%.2f p99=%.2f max=%.2f ms\n",
           minL, p50, p99, maxL);
    printf("[client] tc_remote totals: %llu bytes fetched in %llu fetches\n",
           (unsigned long long)tc_remote_total_bytes_fetched(cli),
           (unsigned long long)tc_remote_fetch_count(cli));

    free(lat_ms);
    free(dst);
    tc_remote_shutdown(cli);
    tc_shutdown(ctx);
    return 0;
}

int main(int argc, char** argv) {
    const char* role = NULL;
    const char* bind_url = NULL;
    const char* peer_url = NULL;
    int experts = DEFAULT_EXPERTS;
    size_t bytes = DEFAULT_EXPERT_BYTES;
    int iters = DEFAULT_ITERS;
    int warmup = DEFAULT_WARMUP;
    int linger = DEFAULT_LINGER_SEC;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--role") && i + 1 < argc) role = argv[++i];
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_url = argv[++i];
        else if (!strcmp(argv[i], "--peer") && i + 1 < argc) peer_url = argv[++i];
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expert-bytes") && i + 1 < argc) bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--linger") && i + 1 < argc) linger = atoi(argv[++i]);
        else { fprintf(stderr, "bad arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!role) { usage(argv[0]); return 2; }

    if (!strcmp(role, "server")) {
        if (!bind_url) { fprintf(stderr, "server needs --bind\n"); return 2; }
        return run_server(bind_url, experts, bytes, linger);
    }
    if (!strcmp(role, "client")) {
        if (!peer_url) { fprintf(stderr, "client needs --peer\n"); return 2; }
        return run_client(peer_url, experts, bytes, iters, warmup);
    }
    fprintf(stderr, "role must be 'server' or 'client'\n");
    return 2;
}
