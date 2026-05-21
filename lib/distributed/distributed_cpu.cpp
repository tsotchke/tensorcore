/*
 * tensorcore - portable CPU distributed backend.
 *
 * The only active CPU backend today is SINGLE. It is enough for unit tests,
 * bindings, and single-worker execution, while keeping the ABI identical to
 * the future multi-worker backends.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstring>
#include <new>
#include <string>

struct tc_dist_ctx {
    tc_context* tc;
    tc_dist_backend_t backend;
    int world_size;
    int rank;
    std::string rendezvous;
};

extern "C" tc_status_t tc_dist_init(tc_context* tc,
                                    tc_dist_backend_t backend,
                                    int world_size,
                                    int rank,
                                    const char* rendezvous_url,
                                    tc_dist_ctx** out) {
    if (!tc || !out || world_size <= 0 || rank < 0 || rank >= world_size) {
        return TC_ERR_INVALID_ARG;
    }
    if (backend == TC_DIST_RING || backend == TC_DIST_GLOO) {
        if (world_size > 1) return TC_ERR_UNSUPPORTED_FAMILY;
    }
    if (world_size == 1) backend = TC_DIST_SINGLE;

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

extern "C" tc_status_t tc_allreduce(tc_dist_ctx* d,
                                    tc_buffer* buf,
                                    size_t num_elements,
                                    tc_dtype_t dtype,
                                    tc_reduce_op_t op) {
    if (!d || !buf || num_elements == 0 || tc_dtype_size(dtype) == 0) {
        return TC_ERR_INVALID_ARG;
    }
    if (d->backend != TC_DIST_SINGLE) return TC_ERR_UNSUPPORTED_FAMILY;
    (void)op;
    return tc_buffer_validate(d->tc, buf, num_elements * tc_dtype_size(dtype));
}

extern "C" tc_status_t tc_broadcast(tc_dist_ctx* d,
                                    tc_buffer* buf,
                                    size_t num_elements,
                                    tc_dtype_t dtype,
                                    int root) {
    if (!d || !buf || root < 0 || root >= d->world_size || tc_dtype_size(dtype) == 0) {
        return TC_ERR_INVALID_ARG;
    }
    if (d->backend != TC_DIST_SINGLE) return TC_ERR_UNSUPPORTED_FAMILY;
    return tc_buffer_validate(d->tc, buf, num_elements * tc_dtype_size(dtype));
}

extern "C" tc_status_t tc_allgather(tc_dist_ctx* d,
                                    const tc_buffer* in,
                                    tc_buffer* out,
                                    size_t num_elements_per_rank,
                                    tc_dtype_t dtype) {
    if (!d || !in || !out || tc_dtype_size(dtype) == 0) return TC_ERR_INVALID_ARG;
    if (d->backend != TC_DIST_SINGLE) return TC_ERR_UNSUPPORTED_FAMILY;

    const size_t bytes = num_elements_per_rank * tc_dtype_size(dtype);
    tc_status_t s = tc_buffer_validate(d->tc, in, bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(d->tc, out, bytes);
    if (s != TC_OK) return s;

    void* src = nullptr;
    void* dst = nullptr;
    s = tc_buffer_map((tc_buffer*)in, &src);
    if (s != TC_OK) return s;
    s = tc_buffer_map(out, &dst);
    if (s != TC_OK) return s;
    std::memcpy(dst, src, bytes);
    return TC_OK;
}

extern "C" tc_status_t tc_barrier(tc_dist_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    return TC_OK;
}
