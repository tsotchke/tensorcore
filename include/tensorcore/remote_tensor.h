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
 *   - Calls that fetch concurrently from the same peer are safe and are
 *     serialized at the connection. Different peer connections can proceed
 *     concurrently.
 *
 * Trust boundary:
 *   - tc_remote_init retains the v1 unauthenticated compatibility path.
 *     tc_remote_init_authenticated adds mutual identity authentication but
 *     not payload encryption. Bind to a private/overlay address (for example
 *     a Tailscale address) when confidentiality is required. The bind host is
 *     honored exactly; use "0.0.0.0" or "*" only intentionally.
 *   - Active client reads and all socket writes time out after 30 seconds by
 *     default. An idle server connection remains valid across long compute
 *     phases and is interrupted explicitly at shutdown. Override active I/O
 *     timeouts with TC_REMOTE_IO_TIMEOUT_MS (100..600000).
 *     TC_REMOTE_MAX_CLIENTS bounds simultaneous server connections (default
 *     64, range 1..4096).
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
#include "tensorcore/transport_auth.h"

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
 *   bind_url   : SERVER: "tcp://host:port" address to listen on. The host is
 *                not widened; "tcp://127.0.0.1:port" is loopback-only.
 *                CLIENT: ignored (pass NULL); use tc_remote_connect.
 *   out        : returns the new tc_remote_ctx handle. */
tc_status_t tc_remote_init(tc_context* ctx,
                            tc_remote_role_t role,
                            const char* bind_url,
                            tc_remote_ctx** out);

/* Authenticated constructor. The config is validated and deep-copied before
 * this function returns. An authenticated server rejects legacy clients; an
 * authenticated client must use tc_remote_connect_authenticated. */
tc_status_t tc_remote_init_authenticated(
    tc_context* ctx,
    tc_remote_role_t role,
    const char* bind_url,
    const tc_transport_auth_config* auth,
    tc_remote_ctx** out);

/* Shut down the endpoint. Closes the listener and all accepted connections
 * (server) or all peer connections (client), then waits for workers to exit. */
tc_status_t tc_remote_shutdown(tc_remote_ctx* h);

/* WEIGHT_SERVER: register a tensor blob by name. The server reads bytes
 * directly out of it on each fetch (no copy). The pointer must remain valid
 * until tc_remote_unregister_tensor returns, or until endpoint shutdown if
 * it is never unregistered.
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
 * NOT free the underlying ptr — caller owns the storage). This is a lifetime
 * barrier: when it returns, in-flight server writes no longer access ptr. */
tc_status_t tc_remote_unregister_tensor(tc_remote_ctx* h, const char* name);

/* Number of tensors currently registered on this server handle. */
size_t tc_remote_registered_count(tc_remote_ctx* h);

/* COMPUTE_CLIENT: connect to a weight-server at `peer_url`
 * ("tcp://host:port"). Returns a peer id (int) used by subsequent
 * fetches. Negative on error. */
int tc_remote_connect(tc_remote_ctx* h, const char* peer_url);

/* Authenticated connect with explicit server identity binding. On success,
 * out_peer_id receives the normal fetch peer ID. On failure it remains -1.
 * Legacy/authenticated mixing fails closed with TC_ERR_AUTH. */
tc_status_t tc_remote_connect_authenticated(
    tc_remote_ctx* h,
    const char* peer_url,
    const char* expected_peer_identity,
    int* out_peer_id);

/* Replace the keyring used by future handshakes. The endpoint identity must
 * remain unchanged. Existing authenticated sessions are not interrupted.
 * Keeping old and new key IDs together provides the rotation overlap window. */
tc_status_t tc_remote_auth_rotate(
    tc_remote_ctx* h,
    const tc_transport_auth_config* auth);

/* Authenticated identity/key observability for an established client peer.
 * Unauthenticated peers return NULL/0. The identity pointer lives until
 * endpoint shutdown. */
const char* tc_remote_peer_identity(tc_remote_ctx* h, int peer_id);
uint64_t tc_remote_peer_key_id(tc_remote_ctx* h, int peer_id);
uint64_t tc_remote_auth_failure_count(tc_remote_ctx* h);

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
