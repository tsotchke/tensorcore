/*
 * tensorcore — distributed primitives.
 *
 * v0.1: TC_DIST_SINGLE backend (world_size=1, all collectives are no-ops).
 * The API is the same as the planned RING/GLOO backends; user code written
 * against this header will work unchanged once those backends ship.
 *
 * The RING backend (Thunderbolt-5 ring all-reduce, JACCL-style RDMA) is
 * scoped for v0.5 — see ROADMAP.md.  When implementing, the entrypoints
 * to fill in are below; the SINGLE path's behavior is the semantic
 * specification.
 */

#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "tensorcore/distributed.h"
#include "../core/internal.h"

#include <cstdio>
#include <new>
#include <string>
#include <cstring>

struct tc_dist_ctx {
    tc_context*       tc;
    tc_dist_backend_t backend;
    int               world_size;
    int               rank;
    std::string       rendezvous;
};

extern "C" tc_status_t tc_dist_init(tc_context* tc, tc_dist_backend_t backend,
                                    int world_size, int rank,
                                    const char* rendezvous_url,
                                    tc_dist_ctx** out) {
    if (!tc || !out || world_size <= 0 || rank < 0 || rank >= world_size)
        return TC_ERR_INVALID_ARG;

    if (backend == TC_DIST_RING || backend == TC_DIST_GLOO) {
        if (world_size > 1) {
            fprintf(stderr,
                "[tensorcore] dist backend %d not yet available; v0.5 ships TB5 ring + Gloo.\n",
                (int)backend);
            return TC_ERR_UNSUPPORTED_FAMILY;
        }
    }
    if (world_size == 1) {
        /* All backends collapse to no-op when alone. */
        backend = TC_DIST_SINGLE;
    }

    tc_dist_ctx* d = new (std::nothrow) tc_dist_ctx{};
    if (!d) return TC_ERR_ALLOC;
    d->tc = tc;
    d->backend = backend;
    d->world_size = world_size;
    d->rank = rank;
    d->rendezvous = rendezvous_url ? rendezvous_url : "";
    *out = d;
    return TC_OK;
}

extern "C" tc_status_t tc_dist_finalize(tc_dist_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    delete d;
    return TC_OK;
}

extern "C" int tc_dist_world_size(const tc_dist_ctx* d) {
    return d ? d->world_size : 0;
}
extern "C" int tc_dist_rank(const tc_dist_ctx* d) {
    return d ? d->rank : 0;
}

/* Internal helper: parent tc_context accessor for sibling TUs that
 * need to allocate temporary buffers in the same arena (DiLoCo runtime). */
extern "C" TC_INTERNAL_SYMBOL tc_context* tc_dist_get_context(tc_dist_ctx* d) {
    return d ? d->tc : nullptr;
}

extern "C" tc_status_t tc_allreduce(tc_dist_ctx* d, tc_buffer* buf,
                                    size_t num_elements, tc_dtype_t dtype,
                                    tc_reduce_op_t op) {
    if (!d || !buf || num_elements == 0) return TC_ERR_INVALID_ARG;
    if (d->backend == TC_DIST_SINGLE) {
        /* world_size==1: reduce is identity. AVG also identity. */
        (void)dtype; (void)op;
        return TC_OK;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_broadcast(tc_dist_ctx* d, tc_buffer* buf,
                                    size_t num_elements, tc_dtype_t dtype,
                                    int root) {
    if (!d || !buf) return TC_ERR_INVALID_ARG;
    if (d->backend == TC_DIST_SINGLE) {
        (void)num_elements; (void)dtype; (void)root;
        return TC_OK;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_allgather(tc_dist_ctx* d,
                                    const tc_buffer* in, tc_buffer* out,
                                    size_t per_rank, tc_dtype_t dtype) {
    if (!d || !in || !out) return TC_ERR_INVALID_ARG;
    if (d->backend == TC_DIST_SINGLE) {
        /* Copy in → out (rank 0 of 1). */
        const size_t bytes = per_rank * tc_dtype_size(dtype);
        void* src = nullptr; void* dst = nullptr;
        tc_buffer_map((tc_buffer*)in,  &src);
        tc_buffer_map(out, &dst);
        if (!src || !dst) return TC_ERR_INTERNAL;
        std::memcpy(dst, src, bytes);
        return TC_OK;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_barrier(tc_dist_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    /* SINGLE: no-op. RING: synchronize all ranks. */
    return TC_OK;
}
