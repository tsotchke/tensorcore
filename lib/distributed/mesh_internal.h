#ifndef TC_LIB_DISTRIBUTED_MESH_INTERNAL_H
#define TC_LIB_DISTRIBUTED_MESH_INTERNAL_H

/*
 * tensorcore — private mesh-group accessors for sibling .cpp files
 * inside lib/distributed/ (currently remote_shard.cpp). Not exported.
 *
 * Public callers go through include/tensorcore/mesh_collective.h.
 * These accessors avoid duplicating the tc_mesh_group struct or
 * promoting transport details into a public header.
 */

#include <stdint.h>
#include "tensorcore/mesh_collective.h"
#include "tensorcore/remote_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tc_remote_ctx* tc_mesh_internal_server(tc_mesh_group_t* g);
tc_remote_ctx* tc_mesh_internal_client(tc_mesh_group_t* g);

/* peer_id for `rank` (the value returned by tc_remote_connect for that
 * peer at group init). Returns -1 for invalid rank or for my_rank
 * itself (no loopback connection). */
int tc_mesh_internal_peer_id(tc_mesh_group_t* g, int32_t rank);

int32_t tc_mesh_internal_my_rank(tc_mesh_group_t* g);
int32_t tc_mesh_internal_n_peers(tc_mesh_group_t* g);

/* Bump the cumulative bytes_shipped counter so tc_mesh_total_bytes
 * stays accurate after shard fetches. */
void tc_mesh_internal_add_bytes(tc_mesh_group_t* g, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif
