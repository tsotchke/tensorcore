#ifndef TENSORCORE_REMOTE_TENSOR_H
#define TENSORCORE_REMOTE_TENSOR_H

/*
 * tensorcore — remote tensor-fetch transport (inference weight paging).
 *
 * Sister to tc_dist_* (training collectives) for the INFERENCE side: a
 * point-to-point, name-addressable byte transport so a GPU node can
 * stream weights it doesn't have cached out of a peer's RAM.
 *
 * Why this exists separately from tc_dist:
 *   - tc_dist's collectives (allreduce / broadcast / allgather / barrier)
 *     are correct for training but the wrong primitive for inference
 *     weight paging — there's no reduction, no synchronized group, just
 *     "rank X needs these bytes from rank Y, fast, ideally before the
 *     next GEMM finishes."
 *   - The Kimi K2.6 MoE serving case (NOTE-kimi-distributed-inference-
 *     2026-06-24.md) is the canonical workload: 543 GB Q4 weights resident
 *     on the high-RAM peer (old-donkey, 499 GB), 24 GB VRAM on the GPU
 *     peer (cosbox, RTX 3090), per-token active set 11.3 GiB. We need to
 *     fetch the active experts (~24 MB each) faster than cosbox's local
 *     240 MB/s spinning disk can serve them.
 *
 * Architecture:
 *   - Two roles per process: WEIGHT_SERVER (registers tensors) and
 *     COMPUTE_CLIENT (fetches). One process may hold both (server +
 *     forward-pass).
 *   - One TCP listener per server (binds to `bind_url` from
 *     tc_remote_init). Clients connect via tc_remote_connect.
 *   - Protocol: 32-byte fixed-size request header { magic, op, name_len,
 *     bytes }, then `name_len` UTF-8 bytes naming the tensor (e.g.
 *     "expert/L31/E5/W2"). Server responds with { status, bytes } then
 *     the raw bytes.
 *   - Zero-copy on the server side (writev directly out of the registered
 *     ptr). Client copies into the user-supplied dst buffer.
 *
 * Threading:
 *   - The server runs one worker thread per connected client. Worker
 *     blocks on read, services requests, writes back, loops.
 *   - The client's fetch call is synchronous (blocking) on the network
 *     read — the caller orchestrates async overlap by spawning fetches
 *     ahead of the compute that needs the bytes (use cudaMemcpyAsync
 *     after the fetch returns to overlap with the GPU pipeline).
 *
 * Versioning: protocol magic = 'TCRT' v1. Mismatch → TC_ERR_INTERNAL.
 *
 * Build gate: POSIX sockets, same as gloo_tcp.cpp. Compiles to UNSUPPORTED
 * stubs on Windows; production Windows builds would use Winsock.
 */

#include <stddef.h>
#include <stdint.h>
#include "tensorcore/status.h"
#include "tensorcore/tensorcore.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tc_remote_ctx tc_remote_ctx;

typedef enum {
    TC_REMOTE_ROLE_WEIGHT_SERVER = 0,    /* registers tensors, serves fetches */
    TC_REMOTE_ROLE_COMPUTE_CLIENT = 1,   /* connects to a server, calls fetch  */
} tc_remote_role_t;

/* Initialize a remote-tensor endpoint.
 *
 *   ctx        : owning tc_context (for diagnostics / future buffer alloc).
 *   role       : SERVER or CLIENT.
 *   bind_url   : SERVER: "tcp://0.0.0.0:port" address to listen on.
 *                CLIENT: ignored (pass NULL); use tc_remote_connect.
 *   out        : returns the new tc_remote_ctx handle. */
tc_status_t tc_remote_init(tc_context* ctx,
                            tc_remote_role_t role,
                            const char* bind_url,
                            tc_remote_ctx** out);

/* Shut down the endpoint. Closes the listener (server) or all open
 * connections (client). Joins all worker threads. */
tc_status_t tc_remote_shutdown(tc_remote_ctx* h);

/* WEIGHT_SERVER: register a tensor blob by name. The pointer must remain
 * valid for the lifetime of the handle; the server reads bytes directly
 * out of it on each fetch (no copy).
 *
 *   h          : SERVER handle from tc_remote_init.
 *   name       : NUL-terminated UTF-8 identifier (e.g. "expert/L31/E5/W2").
 *                Max 255 bytes.
 *   ptr        : host pointer to the resident bytes (e.g. a Q4 expert bank).
 *   bytes      : tensor size in bytes. */
tc_status_t tc_remote_register_tensor(tc_remote_ctx* h,
                                       const char* name,
                                       const void* ptr,
                                       size_t bytes);

/* Forget a previously registered tensor (frees the registry slot; does
 * NOT free the underlying ptr — caller owns the storage). */
tc_status_t tc_remote_unregister_tensor(tc_remote_ctx* h, const char* name);

/* Number of tensors currently registered on this server handle. */
size_t tc_remote_registered_count(tc_remote_ctx* h);

/* COMPUTE_CLIENT: connect to a weight-server at `peer_url`
 * ("tcp://host:port"). Returns a peer id (int) used by subsequent
 * fetches. Negative on error. */
int tc_remote_connect(tc_remote_ctx* h, const char* peer_url);

/* COMPUTE_CLIENT: synchronous fetch of a named tensor's bytes into the
 * caller-supplied destination buffer.
 *
 *   h          : CLIENT handle from tc_remote_init.
 *   peer_id    : id returned by tc_remote_connect.
 *   name       : NUL-terminated tensor identifier (must match a name
 *                registered on the server via tc_remote_register_tensor).
 *   dst        : destination buffer (host memory; caller does cudaMemcpyAsync
 *                onto the device with their own stream after this returns).
 *   bytes      : number of bytes to read into dst. Must match the
 *                server's registered size for `name`.
 *
 * Returns TC_OK on full read; TC_ERR_INTERNAL on protocol / network
 * failure; TC_ERR_INVALID_ARG if the server reports a name miss or size
 * mismatch. */
tc_status_t tc_remote_tensor_fetch(tc_remote_ctx* h,
                                    int peer_id,
                                    const char* name,
                                    void* dst,
                                    size_t bytes);

/* Diagnostics: bytes / fetches counted since this handle was created.
 * Useful for the K2.6 perf benches (throughput = bytes / wall_time). */
uint64_t tc_remote_total_bytes_served(tc_remote_ctx* h);
uint64_t tc_remote_total_bytes_fetched(tc_remote_ctx* h);
uint64_t tc_remote_fetch_count(tc_remote_ctx* h);

#ifdef __cplusplus
}
#endif
#endif
