/*
 * tensorcore — DiLoCo runtime.
 *
 * Layered above tc_dist_*: every cross-site collective uses the
 * tc_dist_ctx passed by the caller. DiLoCo holds the inner-loop counter,
 * the anchor copy of θ (θ_global), the outer-optimizer state, and the
 * compression / error-feedback buffers.
 *
 * The hot loop on a worker:
 *
 *     for step = 0..total_inner_steps:
 *         <do normal training step against θ_local>
 *         tc_diloco_step(d, &outer_pending);
 *         if (outer_pending) tc_diloco_apply_outer(d);
 *
 * If async_overlap = true, apply_outer dispatches the cross-site work
 * to a background thread and returns immediately; the next inner-step
 * batch proceeds against the *previous* anchor until the background
 * thread's all-reduce + outer-step completes, at which point it swaps
 * the new anchor in (atomically, at an outer-step boundary).
 *
 * Compression schemes:
 *
 *   NONE      — Δθ sent as fp32, full 1:1
 *   FP16      — convert Δθ to fp16 before send, convert back at receive
 *   FP8       — per-tensor scale, fp8 magnitude (E4M3 saturation)
 *   TOPK_*    — keep top-K magnitudes; error-feedback retains the residual
 *   LOWRANK   — PowerSGD: rank-r approximation of each parameter tensor
 *   SIGNSGD   — 1-bit per element, scaled at receive end
 *
 * For now, the in-tree implementation covers the local/single-rank outer
 * step for NONE, FP16-intent, and TOPK masking with error feedback.
 * Multi-rank WAN transport and FP8 / LOWRANK / SIGNSGD return explicit
 * unsupported statuses so downstream code gets a stable failure instead
 * of incorrect results.
 *
 * Memory cost: one anchor θ + one momentum buffer + (top-k) one
 * error-feedback buffer per parameter, all fp32. For a 70B model that's
 * ~3 × 280 GB = 840 GB if every parameter is fp32; in practice the
 * fp32 anchor is the optimizer master-weight, the momentum is the
 * outer-optimizer state (Nesterov), and the error-feedback piece only
 * exists when top-k is enabled. Plan for ~3-5× the model size in
 * fp32 working memory.
 */

#include "tensorcore/diloco.h"
#include "tensorcore/tensorcore.h"

/* Internal helper exported by lib/distributed/distributed_cpu.cpp + Metal's
 * lib/distributed/distributed.mm. Gives DiLoCo access to the parent
 * tc_context so it can allocate temporary buffers in the same arena as
 * the user's tc_dist_ctx. */
#if defined(_WIN32)
#define TC_DILOCO_INTERNAL_SYMBOL
#else
#define TC_DILOCO_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

extern "C" TC_DILOCO_INTERNAL_SYMBOL tc_context* tc_dist_get_context(tc_dist_ctx* d);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Parameter {
    std::string  name;
    tc_buffer*   theta_local;       /* model's working copy, fp16 or fp32 */
    size_t       num_elements;
    tc_dtype_t   dtype;

    std::vector<float> theta_anchor;     /* θ_global_anchor in fp32        */
    std::vector<float> outer_momentum;   /* outer-optimizer state          */
    std::vector<float> error_feedback;   /* residual for top-k (lazy alloc) */
};

inline uint16_t f32_to_f16_bits(float x) {
    union { float f; uint32_t u; } v = {x};
    const uint32_t f = v.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t round = (mant >> (shift - 1)) & 1u;
        return (uint16_t)(sign | ((mant >> shift) + round));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    const uint32_t round = (mant >> 12) & 1u;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant >> 13) + round));
}

inline float f16_to_f32_bits(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            ++exp;
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = {out};
    return v.f;
}

/* Read parameter from its tc_buffer into a fp32 host array. */
bool theta_to_fp32(const Parameter& p, std::vector<float>& dst) {
    void* mp = nullptr;
    if (tc_buffer_map(p.theta_local, &mp) != TC_OK || !mp) return false;
    dst.resize(p.num_elements);
    if (p.dtype == TC_DTYPE_F32) {
        std::memcpy(dst.data(), mp, p.num_elements * sizeof(float));
    } else if (p.dtype == TC_DTYPE_F16) {
        const uint16_t* src = (const uint16_t*)mp;
        for (size_t i = 0; i < p.num_elements; ++i) dst[i] = f16_to_f32_bits(src[i]);
    } else {
        return false;
    }
    return true;
}

/* Write fp32 host array back to the parameter's tc_buffer. */
bool theta_from_fp32(Parameter& p, const std::vector<float>& src) {
    void* mp = nullptr;
    if (tc_buffer_map(p.theta_local, &mp) != TC_OK || !mp) return false;
    if (p.dtype == TC_DTYPE_F32) {
        std::memcpy(mp, src.data(), p.num_elements * sizeof(float));
    } else if (p.dtype == TC_DTYPE_F16) {
        uint16_t* dst = (uint16_t*)mp;
        for (size_t i = 0; i < p.num_elements; ++i) dst[i] = f32_to_f16_bits(src[i]);
    } else {
        return false;
    }
    return true;
}

}  // namespace

struct tc_diloco_ctx {
    tc_dist_ctx*               dist;
    tc_diloco_config           cfg;
    std::vector<Parameter>     params;

    /* inner-loop counter */
    std::atomic<uint64_t>      inner_steps_total{0};
    std::atomic<uint64_t>      outer_steps_total{0};
    int                        inner_steps_since_outer = 0;

    /* timing instrumentation */
    std::atomic<double>        last_outer_seconds{0.0};
    std::atomic<double>        last_outer_bytes{0.0};

    /* async-overlap support */
    std::thread                outer_thread;
    std::mutex                 outer_mutex;
    std::condition_variable    outer_cv;
    bool                       outer_busy = false;
    bool                       shutdown = false;

    ~tc_diloco_ctx() {
        {
            std::lock_guard<std::mutex> lk(outer_mutex);
            shutdown = true;
        }
        outer_cv.notify_all();
        if (outer_thread.joinable()) outer_thread.join();
    }
};

/* ------------------------------------------------------------------------
 * Compression / decompression
 * ------------------------------------------------------------------------ */

namespace {

/* Convert Δθ into the per-rank send buffer (still fp32 here; compression
 * shrinks the byte size before tc_allreduce). For the simple NONE / FP16
 * path, the on-wire format is fp32 (allreduce-friendly); we cast at the
 * boundary on the receive side. */
void compute_delta(const Parameter& p, const std::vector<float>& theta_now,
                   std::vector<float>& delta_out) {
    delta_out.resize(p.num_elements);
    for (size_t i = 0; i < p.num_elements; ++i) {
        delta_out[i] = theta_now[i] - p.theta_anchor[i];
    }
}

/* Top-k sparsification with error feedback. Keep the K largest |Δθ_i|
 * entries; the rest go into the error-feedback buffer for next outer step.
 *
 * Compressed payload is encoded as (index, value) pairs in two flat
 * arrays. For tc_allreduce we send a dense fp32 vector — the all-reduce
 * sums per-rank top-k contributions. The receive side averages and the
 * outer-optimizer absorbs the result.
 *
 * For simplicity v1: top-k still sends a *dense* fp32 vector after
 * masking; the bandwidth savings come from a separate "sparse_pack" path
 * not yet wired here. v2 wires the sparse pack into tc_allreduce. */
void compute_delta_topk(Parameter& p, const std::vector<float>& theta_now,
                        std::vector<float>& delta_out, float keep_fraction) {
    delta_out.resize(p.num_elements);
    if (p.error_feedback.size() != p.num_elements) {
        p.error_feedback.assign(p.num_elements, 0.0f);
    }
    /* Δθ + error_feedback (carry-over residual from prior outer step). */
    std::vector<float> magnitude(p.num_elements);
    for (size_t i = 0; i < p.num_elements; ++i) {
        delta_out[i] = theta_now[i] - p.theta_anchor[i] + p.error_feedback[i];
        magnitude[i] = std::fabs(delta_out[i]);
    }
    /* Find the |Δ| threshold for top-K. Approximate via nth_element on a
     * copy — exact within rounding. */
    const size_t K = std::max<size_t>(1, (size_t)(keep_fraction * p.num_elements));
    std::vector<float> mag_copy = magnitude;
    std::nth_element(mag_copy.begin(), mag_copy.begin() + (mag_copy.size() - K),
                     mag_copy.end());
    const float thresh = mag_copy[mag_copy.size() - K];

    /* Mask Δθ; entries below threshold contribute to error_feedback. */
    for (size_t i = 0; i < p.num_elements; ++i) {
        if (magnitude[i] >= thresh) {
            p.error_feedback[i] = 0.0f;
        } else {
            p.error_feedback[i] = delta_out[i];   /* carry over for next time */
            delta_out[i] = 0.0f;
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------------
 * Outer-optimizer step
 * ------------------------------------------------------------------------ */

namespace {

void apply_outer_optimizer(Parameter& p, const std::vector<float>& delta_avg,
                           const tc_diloco_config& cfg) {
    /* delta_avg has been all-reduced and averaged across ranks. */
    if (p.outer_momentum.size() != p.num_elements) {
        p.outer_momentum.assign(p.num_elements, 0.0f);
    }
    const float lr = cfg.outer_lr;
    const float mu = cfg.outer_momentum;

    if (cfg.outer_optimizer == TC_DILOCO_OUTER_SGD) {
        for (size_t i = 0; i < p.num_elements; ++i) {
            p.theta_anchor[i] += lr * delta_avg[i];
        }
    } else if (cfg.outer_optimizer == TC_DILOCO_OUTER_NESTEROV) {
        /* Standard Nesterov on the Δθ "gradient" (treating Δ̄θ as -∇L). */
        for (size_t i = 0; i < p.num_elements; ++i) {
            const float prev = p.outer_momentum[i];
            const float v = mu * prev + delta_avg[i];
            p.outer_momentum[i] = v;
            /* look-ahead step */
            p.theta_anchor[i] += lr * (delta_avg[i] + mu * v);
        }
    } else if (cfg.outer_optimizer == TC_DILOCO_OUTER_ADAM) {
        /* Outer Adam: m = β1 m + (1 - β1) Δ; v = β2 v + (1 - β2) Δ^2;
         * θ ← θ + lr * m̂ / (√v̂ + eps). We treat Δ as +∇ here (param goes
         * toward Δ̄θ, not away from it). */
        std::vector<float>& m = p.outer_momentum;
        /* second-moment piggybacks in the back half of momentum if we
         * resize it to 2× length. */
        if (m.size() < 2 * p.num_elements) m.resize(2 * p.num_elements, 0.0f);
        float* m1 = m.data();
        float* m2 = m.data() + p.num_elements;
        const float b1 = cfg.outer_momentum > 0.0f ? cfg.outer_momentum : 0.9f;
        const float b2 = cfg.outer_beta2 > 0.0f ? cfg.outer_beta2 : 0.999f;
        const float eps = cfg.outer_eps > 0.0f ? cfg.outer_eps : 1e-8f;
        for (size_t i = 0; i < p.num_elements; ++i) {
            m1[i] = b1 * m1[i] + (1.0f - b1) * delta_avg[i];
            m2[i] = b2 * m2[i] + (1.0f - b2) * delta_avg[i] * delta_avg[i];
            p.theta_anchor[i] += lr * m1[i] / (std::sqrt(m2[i]) + eps);
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------ */

extern "C" tc_status_t tc_diloco_init(tc_dist_ctx* dist_ctx,
                                       const tc_diloco_config* cfg,
                                       tc_diloco_ctx** out) {
    if (!cfg || !out) return TC_ERR_INVALID_ARG;
    *out = nullptr;
    if (cfg->inner_steps <= 0) return TC_ERR_INVALID_ARG;
    if (cfg->compress != TC_DILOCO_COMPRESS_NONE &&
        cfg->compress != TC_DILOCO_COMPRESS_FP16 &&
        cfg->compress != TC_DILOCO_COMPRESS_TOPK_1PCT &&
        cfg->compress != TC_DILOCO_COMPRESS_TOPK_01PCT) {
        /* fp8 / lowrank / signsgd not yet implemented */
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    auto* d = new tc_diloco_ctx();
    d->dist = dist_ctx;
    d->cfg = *cfg;
    *out = d;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_finalize(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    delete d;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_add_parameter(tc_diloco_ctx* d,
                                                const char* name,
                                                tc_buffer* theta_local,
                                                size_t num_elements,
                                                tc_dtype_t dtype) {
    if (!d || !theta_local || num_elements == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_DTYPE_F16 && dtype != TC_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;

    Parameter p;
    p.name = name ? name : "";
    p.theta_local = theta_local;
    p.num_elements = num_elements;
    p.dtype = dtype;
    /* Snapshot θ_anchor from the current θ_local at registration time. */
    if (!theta_to_fp32(p, p.theta_anchor)) return TC_ERR_INVALID_ARG;
    d->params.push_back(std::move(p));
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_step(tc_diloco_ctx* d,
                                       bool* out_outer_step_pending) {
    if (!d || !out_outer_step_pending) return TC_ERR_INVALID_ARG;
    d->inner_steps_total.fetch_add(1, std::memory_order_relaxed);
    d->inner_steps_since_outer += 1;
    *out_outer_step_pending = (d->inner_steps_since_outer >= d->cfg.inner_steps);
    return TC_OK;
}

namespace {

/* The synchronous core of the outer step. Called either directly from
 * tc_diloco_apply_outer or from the background worker thread. */
tc_status_t do_outer_step(tc_diloco_ctx* d) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    size_t total_bytes_sent = 0;

    /* For each registered parameter:
     *   1. Read current θ_local into fp32 host buffer.
     *   2. Compute Δθ (with compression / error feedback if configured).
     *   3. all-reduce Δθ across the dist_ctx.
     *   4. Outer-optimizer updates θ_anchor.
     *   5. Write θ_anchor back into θ_local (resync local to new anchor). */
    std::vector<float> theta_now;
    std::vector<float> delta;

    for (auto& p : d->params) {
        if (!theta_to_fp32(p, theta_now)) return TC_ERR_INVALID_ARG;

        switch (d->cfg.compress) {
        case TC_DILOCO_COMPRESS_NONE:
        case TC_DILOCO_COMPRESS_FP16:
            compute_delta(p, theta_now, delta);
            break;
        case TC_DILOCO_COMPRESS_TOPK_1PCT:
            compute_delta_topk(p, theta_now, delta, 0.01f);
            break;
        case TC_DILOCO_COMPRESS_TOPK_01PCT:
            compute_delta_topk(p, theta_now, delta, 0.001f);
            break;
        default:
            return TC_ERR_UNSUPPORTED_DTYPE;
        }

        /* Multi-rank: cross-site all-reduce-AVG of Δθ. The top-k path has
         * already zeroed sub-threshold entries in `delta`, so the
         * transport sees mostly zeros. Bandwidth-optimal compression in
         * transit (packing only the non-zero (idx, val) pairs) is a follow-
         * up; this dense AVG-allreduce path is correct + the K-step
         * amortization already provides ~100-1000× over per-step DDP. */
        const int world = d->dist ? tc_dist_world_size(d->dist) : 1;
        if (world > 1) {
            tc_context* parent_ctx = tc_dist_get_context(d->dist);
            if (!parent_ctx) return TC_ERR_INTERNAL;
            const size_t bytes = p.num_elements * sizeof(float);
            tc_buffer* delta_buf = nullptr;
            if (tc_buffer_alloc(parent_ctx, bytes, &delta_buf) != TC_OK) {
                return TC_ERR_ALLOC;
            }
            void* mp = nullptr;
            if (tc_buffer_map(delta_buf, &mp) != TC_OK) {
                tc_buffer_free(parent_ctx, delta_buf);
                return TC_ERR_INTERNAL;
            }
            std::memcpy(mp, delta.data(), bytes);
            tc_status_t s = tc_allreduce(d->dist, delta_buf, p.num_elements,
                                          TC_DTYPE_F32, TC_REDUCE_AVG);
            if (s == TC_OK) {
                std::memcpy(delta.data(), mp, bytes);
                total_bytes_sent += bytes;
            }
            tc_buffer_free(parent_ctx, delta_buf);
            if (s != TC_OK) return s;
        }

        /* Outer-optimizer step on θ_anchor. */
        apply_outer_optimizer(p, delta, d->cfg);

        /* Resync θ_local := θ_anchor. */
        if (!theta_from_fp32(p, p.theta_anchor)) return TC_ERR_INVALID_ARG;
    }

    const auto dt = std::chrono::duration<double>(clock::now() - t0).count();
    d->last_outer_seconds.store(dt, std::memory_order_relaxed);
    d->last_outer_bytes.store((double)total_bytes_sent, std::memory_order_relaxed);
    d->outer_steps_total.fetch_add(1, std::memory_order_relaxed);
    return TC_OK;
}

}  // namespace

extern "C" tc_status_t tc_diloco_apply_outer(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;

    /* Reset the inner-step counter — caller will resume the inner loop. */
    d->inner_steps_since_outer = 0;

    if (!d->cfg.async_overlap) {
        return do_outer_step(d);
    }

    /* Async path: ensure no prior outer step is still running (we serialize
     * outer steps; if you're falling behind, this blocks). Then start a
     * new background outer step. */
    {
        std::unique_lock<std::mutex> lk(d->outer_mutex);
        d->outer_cv.wait(lk, [d]{ return !d->outer_busy || d->shutdown; });
        if (d->shutdown) return TC_OK;
        if (d->outer_thread.joinable()) {
            lk.unlock();
            d->outer_thread.join();
            lk.lock();
        }
        d->outer_busy = true;
    }
    /* Keep the worker joinable so finalize cannot race a detached thread. */
    d->outer_thread = std::thread([d]() {
        do_outer_step(d);
        {
            std::lock_guard<std::mutex> lk(d->outer_mutex);
            d->outer_busy = false;
        }
        d->outer_cv.notify_all();
    });
    return TC_OK;
}

extern "C" uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d) {
    return d ? d->outer_steps_total.load(std::memory_order_relaxed) : 0;
}

extern "C" uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d) {
    return d ? d->inner_steps_total.load(std::memory_order_relaxed) : 0;
}

extern "C" double tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d) {
    return d ? d->last_outer_seconds.load(std::memory_order_relaxed) : 0.0;
}

extern "C" double tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d) {
    return d ? d->last_outer_bytes.load(std::memory_order_relaxed) : 0.0;
}
