#ifndef TENSORCORE_MESH_COLLECTIVE_H
#define TENSORCORE_MESH_COLLECTIVE_H

/*
 * tensorcore — Mesh collectives over tc_remote_tensor_fetch.
 *
 * Builds on the TCP byte-fetch transport in lib/distributed/remote_tensor.cpp
 * to give callers MPI/NCCL-style group operations: AllReduce, AllGather,
 * Broadcast. The first implementation uses a centralized rank-0
 * coordinator (every rank registers its buffer; rank 0 fetches, reduces,
 * publishes; all ranks fetch the result). It's correct but rank 0 is
 * a bottleneck. A ring AllReduce can replace the inner reduce-scatter +
 * allgather without changing the public API.
 *
 * Group lifecycle:
 *   tc_mesh_group_init(ctx, n_peers, my_rank, peer_urls[]) → group
 *   tc_mesh_allreduce(group, buf, count, dtype, op) → in-place
 *   tc_mesh_broadcast(group, buf, count, dtype, root)
 *   tc_mesh_allgather(group, send_buf, send_count, recv_buf, dtype)
 *   tc_mesh_group_shutdown(group)
 *
 * `peer_urls[i]` is the bind URL of rank i. `my_rank` says which entry
 * is us. The group spins up:
 *   - a tc_remote_init(WEIGHT_SERVER) bound to peer_urls[my_rank] (so
 *     other ranks can fetch from us)
 *   - tc_remote_connect to each other peer (so we can fetch from them)
 *
 * `op` ∈ {SUM, MAX, MIN, PROD}. `dtype` is currently F32 only.
 *
 * Replaces ad-hoc collectives in the tsotchke-chan swarm scheduler and
 * provides the substrate for distributed GeometricLM gradient sync
 * (DiLoCo etc.).
 */

#include <stddef.h>
#include <stdint.h>
#include "tensorcore/status.h"
#include "tensorcore/device.h"
/* Reuse tc_reduce_op_t from distributed.h (SUM=0, AVG=1, MAX=2, MIN=3). */
#include "tensorcore/distributed.h"
#include "tensorcore/transport_auth.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tc_mesh_group tc_mesh_group_t;

typedef enum {
    TC_COLL_DTYPE_F32 = 0,
} tc_coll_dtype_t;

/* Initialise a mesh group. peer_urls is an array of n_peers strings;
 * peer_urls[my_rank] is our own bind URL (used for the local server). */
tc_status_t tc_mesh_group_init(tc_context* ctx,
                                int32_t n_peers, int32_t my_rank,
                                const char* const* peer_urls,
                                tc_mesh_group_t** out);

/* Authenticated group constructor. peer_identities has n_peers entries and
 * binds each mesh rank to a keyring identity. auth->local_identity must equal
 * peer_identities[my_rank]. Legacy/authenticated peers cannot mix. */
tc_status_t tc_mesh_group_init_authenticated(
    tc_context* ctx,
    int32_t n_peers,
    int32_t my_rank,
    const char* const* peer_urls,
    const char* const* peer_identities,
    const tc_transport_auth_config* auth,
    tc_mesh_group_t** out);

/* Shutdown: synchronize with all ranks, close outbound clients, drain inbound
 * clients, close the server, and free state. Every live rank in the group must
 * call shutdown; a missing rank is reported as a transport error. */
tc_status_t tc_mesh_group_shutdown(tc_mesh_group_t* group);

/* In-place reduce across all peers. buf size = count * sizeof(dtype). */
tc_status_t tc_mesh_allreduce(tc_mesh_group_t* group,
                               void* buf, size_t count,
                               tc_coll_dtype_t dtype,
                               tc_reduce_op_t op);

/* Broadcast `buf` from rank `root` to all peers. */
tc_status_t tc_mesh_broadcast(tc_mesh_group_t* group,
                               void* buf, size_t count,
                               tc_coll_dtype_t dtype,
                               int32_t root_rank);

/* Concatenate each rank's send_buf into recv_buf. recv_buf must hold
 * n_peers * send_count elements; rank i's contribution is written to
 * recv_buf[i * send_count : (i+1) * send_count]. */
tc_status_t tc_mesh_allgather(tc_mesh_group_t* group,
                               const void* send_buf, size_t send_count,
                               void* recv_buf,
                               tc_coll_dtype_t dtype);

/* Total bytes shipped through this group (sum of all fetches). */
uint64_t tc_mesh_total_bytes(const tc_mesh_group_t* group);

#ifdef __cplusplus
}
#endif
#endif
