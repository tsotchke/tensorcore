/*
 * tensorcore - distributed primitives (Apple build).
 *
 * Backends:
 *   - TC_DIST_SINGLE: in-process no-op.
 *   - TC_DIST_GLOO:   TCP sockets via lib/distributed/gloo_tcp.cpp.
 *                      Same transport the CPU build uses; works fine on
 *                      Darwin (POSIX sockets). Used for Mac-to-Mac DDP
 *                      over Thunderbolt-bridge IPs and for cross-site
 *                      DiLoCo over Tailscale.
 *   - TC_DIST_RING:   reserved for Thunderbolt-5 ring (JACCL-style RDMA);
 *                     today returns UNSUPPORTED for world>1 until the
 *                     ring transport ships.
 */

#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "tensorcore/distributed.h"
#include "../core/internal.h"

#include <cstdio>
#include <new>
#include <string>
#include <cstring>
#include <cstdint>

/* Forward declarations of the GLOO TCP transport primitives. */
struct GlooState;
extern "C" GlooState* tc_gloo_init(int world_size, int rank, const char* rendezvous_url);
extern "C" void       tc_gloo_destroy(GlooState* s);
extern "C" int        tc_gloo_allreduce_f32_sum(GlooState* s, int world_size, int rank,
                                                  float* data, size_t n);
extern "C" int        tc_gloo_allreduce_f16_sum(GlooState* s, int world_size, int rank,
                                                  uint16_t* data, size_t n);
extern "C" int        tc_gloo_allreduce_f32_min(GlooState* s, int world_size, int rank,
                                                  float* data, size_t n);
extern "C" int        tc_gloo_allreduce_f32_max(GlooState* s, int world_size, int rank,
                                                  float* data, size_t n);
extern "C" int        tc_gloo_broadcast_any_root(GlooState* s, int world_size, int rank,
                                                   int root, void* data, size_t bytes);
extern "C" int        tc_gloo_allgather(GlooState* s, int world_size, int rank,
                                          void* out, size_t bytes_per_rank);
extern "C" int        tc_gloo_sparse_allreduce(GlooState* s, int world_size, int rank,
                                                 const void* payload_in, size_t payload_in_bytes,
                                                 float* dense_out, size_t n_total);
extern "C" int        tc_gloo_barrier(GlooState* s, int world_size, int rank);

struct tc_dist_ctx {
    tc_context*       tc;
    tc_dist_backend_t backend;
    int               world_size;
    int               rank;
    std::string       rendezvous;
    GlooState*        gloo;
};

extern "C" tc_status_t tc_dist_init(tc_context* tc, tc_dist_backend_t backend,
                                    int world_size, int rank,
                                    const char* rendezvous_url,
                                    tc_dist_ctx** out) {
    if (!tc || !out || world_size <= 0 || rank < 0 || rank >= world_size)
        return TC_ERR_INVALID_ARG;
    if (backend != TC_DIST_SINGLE && backend != TC_DIST_RING && backend != TC_DIST_GLOO) {
        return TC_ERR_INVALID_ARG;
    }
    if (backend == TC_DIST_RING && world_size > 1) {
        /* TB5 ring still pending. */
        fprintf(stderr, "[tensorcore] TC_DIST_RING is reserved for the TB5 transport; v0.5\n");
        return TC_ERR_UNSUPPORTED_FAMILY;
    }
    if (world_size == 1) backend = TC_DIST_SINGLE;
    if (backend == TC_DIST_GLOO && (!rendezvous_url || !rendezvous_url[0])) {
        return TC_ERR_INVALID_ARG;
    }

    tc_dist_ctx* d = new (std::nothrow) tc_dist_ctx{};
    if (!d) return TC_ERR_ALLOC;
    d->tc = tc;
    d->backend = backend;
    d->world_size = world_size;
    d->rank = rank;
    d->rendezvous = rendezvous_url ? rendezvous_url : "";
    d->gloo = nullptr;

    if (backend == TC_DIST_GLOO && world_size > 1) {
        d->gloo = tc_gloo_init(world_size, rank, rendezvous_url);
        if (!d->gloo) { delete d; return TC_ERR_INTERNAL; }
    }

    *out = d;
    return TC_OK;
}

extern "C" tc_status_t tc_dist_finalize(tc_dist_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    if (d->gloo) {
        tc_gloo_destroy(d->gloo);
        d->gloo = nullptr;
    }
    delete d;
    return TC_OK;
}

extern "C" int tc_dist_world_size(const tc_dist_ctx* d) {
    return d ? d->world_size : 0;
}
extern "C" int tc_dist_rank(const tc_dist_ctx* d) {
    return d ? d->rank : 0;
}

extern "C" TC_INTERNAL_SYMBOL tc_context* tc_dist_get_context(tc_dist_ctx* d) {
    return d ? d->tc : nullptr;
}

extern "C" TC_INTERNAL_SYMBOL GlooState* tc_dist_get_gloo_state(tc_dist_ctx* d) {
    return (d && d->backend == TC_DIST_GLOO) ? d->gloo : nullptr;
}

namespace {

inline float mm_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { float r; std::memcpy(&r, &sign, 4); return r; }
        int e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
        mant &= 0x03ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &bits, 4); return r;
}

inline uint16_t mm_f32_to_f16(float v) {
    union { float f; uint32_t u; } x = {v};
    const uint32_t bits = x.u;
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

}  // namespace

extern "C" tc_status_t tc_allreduce(tc_dist_ctx* d, tc_buffer* buf,
                                    size_t num_elements, tc_dtype_t dtype,
                                    tc_reduce_op_t op) {
    if (!d || !buf || num_elements == 0 || tc_dtype_size(dtype) == 0) {
        return TC_ERR_INVALID_ARG;
    }
    const size_t bytes = num_elements * tc_dtype_size(dtype);
    tc_status_t s = tc_buffer_validate(d->tc, buf, bytes);
    if (s != TC_OK) return s;
    if (d->backend == TC_DIST_SINGLE) {
        (void)op;
        return TC_OK;
    }
    if (d->backend == TC_DIST_GLOO && d->gloo) {
        void* mp = nullptr;
        s = tc_buffer_map(buf, &mp);
        if (s != TC_OK) return s;
        int rc = 0;
        if (op == TC_REDUCE_SUM || op == TC_REDUCE_AVG) {
            if (dtype == TC_DTYPE_F32) {
                rc = tc_gloo_allreduce_f32_sum(d->gloo, d->world_size, d->rank,
                                                (float*)mp, num_elements);
            } else if (dtype == TC_DTYPE_F16) {
                rc = tc_gloo_allreduce_f16_sum(d->gloo, d->world_size, d->rank,
                                                (uint16_t*)mp, num_elements);
            } else {
                return TC_ERR_UNSUPPORTED_DTYPE;
            }
        } else if (op == TC_REDUCE_MIN) {
            if (dtype != TC_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
            rc = tc_gloo_allreduce_f32_min(d->gloo, d->world_size, d->rank,
                                            (float*)mp, num_elements);
        } else if (op == TC_REDUCE_MAX) {
            if (dtype != TC_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
            rc = tc_gloo_allreduce_f32_max(d->gloo, d->world_size, d->rank,
                                            (float*)mp, num_elements);
        } else {
            return TC_ERR_UNSUPPORTED_FAMILY;
        }
        if (rc != 0) return TC_ERR_INTERNAL;
        if (op == TC_REDUCE_AVG && d->world_size > 1) {
            const float inv = 1.0f / (float)d->world_size;
            if (dtype == TC_DTYPE_F32) {
                float* p = (float*)mp;
                for (size_t i = 0; i < num_elements; ++i) p[i] *= inv;
            } else {
                uint16_t* p = (uint16_t*)mp;
                for (size_t i = 0; i < num_elements; ++i) {
                    p[i] = mm_f32_to_f16(mm_f16_to_f32(p[i]) * inv);
                }
            }
        }
        return TC_OK;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_broadcast(tc_dist_ctx* d, tc_buffer* buf,
                                    size_t num_elements, tc_dtype_t dtype,
                                    int root) {
    if (!d || !buf || root < 0 || root >= d->world_size ||
        tc_dtype_size(dtype) == 0) {
        return TC_ERR_INVALID_ARG;
    }
    const size_t bytes = num_elements * tc_dtype_size(dtype);
    tc_status_t s = tc_buffer_validate(d->tc, buf, bytes);
    if (s != TC_OK) return s;
    if (d->backend == TC_DIST_SINGLE) {
        return TC_OK;
    }
    if (d->backend == TC_DIST_GLOO && d->gloo) {
        void* mp = nullptr;
        s = tc_buffer_map(buf, &mp);
        if (s != TC_OK) return s;
        const int rc = tc_gloo_broadcast_any_root(d->gloo, d->world_size, d->rank,
                                                    root, mp, bytes);
        return rc == 0 ? TC_OK : TC_ERR_INTERNAL;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_allgather(tc_dist_ctx* d,
                                    const tc_buffer* in, tc_buffer* out,
                                    size_t per_rank, tc_dtype_t dtype) {
    if (!d || !in || !out || per_rank == 0 || tc_dtype_size(dtype) == 0) {
        return TC_ERR_INVALID_ARG;
    }
    const size_t bytes = per_rank * tc_dtype_size(dtype);
    if (bytes == 0) return TC_ERR_INVALID_ARG;
    if (d->backend == TC_DIST_SINGLE) {
        void* src = nullptr; void* dst = nullptr;
        tc_status_t s = tc_buffer_validate(d->tc, in, bytes);
        if (s != TC_OK) return s;
        s = tc_buffer_validate(d->tc, out, bytes);
        if (s != TC_OK) return s;
        s = tc_buffer_map((tc_buffer*)in,  &src);
        if (s != TC_OK) return s;
        s = tc_buffer_map(out, &dst);
        if (s != TC_OK) return s;
        std::memcpy(dst, src, bytes);
        return TC_OK;
    }
    if (d->backend == TC_DIST_GLOO && d->gloo) {
        void* src = nullptr; void* dst = nullptr;
        tc_status_t s = tc_buffer_validate(d->tc, in, bytes);
        if (s != TC_OK) return s;
        const size_t total = (size_t)d->world_size * bytes;
        s = tc_buffer_validate(d->tc, out, total);
        if (s != TC_OK) return s;
        s = tc_buffer_map((tc_buffer*)in, &src);
        if (s != TC_OK) return s;
        s = tc_buffer_map(out, &dst);
        if (s != TC_OK) return s;
        std::memcpy((uint8_t*)dst + (size_t)d->rank * bytes, src, bytes);
        const int rc = tc_gloo_allgather(d->gloo, d->world_size, d->rank, dst, bytes);
        return rc == 0 ? TC_OK : TC_ERR_INTERNAL;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" tc_status_t tc_barrier(tc_dist_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    if (d->backend == TC_DIST_SINGLE) return TC_OK;
    if (d->backend == TC_DIST_GLOO && d->gloo) {
        return tc_gloo_barrier(d->gloo, d->world_size, d->rank) == 0
                 ? TC_OK : TC_ERR_INTERNAL;
    }
    return TC_ERR_UNSUPPORTED_FAMILY;
}
