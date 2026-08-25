/*
 * tensorcore — RiemannianAdam CUDA kernels (fp32).
 *
 * Manifold-aware Adam on the GPU: tangent-space gradient projection,
 * exp-map retraction, and first-moment parallel transport, fused into one
 * kernel launch per manifold. Mirrors lib/ops/riemannian_adam_cpu.cpp
 * formula-for-formula — that file is the reference, this is the fast path,
 * and tests/test_riemannian_adam_parity.c pins them together.
 *
 * Why a dedicated kernel rather than reusing adamw_step_fp32_kernel:
 * the curved steps are *not* elementwise. Each row needs whole-row
 * reductions interleaved with the pointwise work — ‖x‖² for the conformal
 * factor, ⟨g,x⟩ for the sphere's tangent projection, ‖step‖ for the
 * retraction, ⟨x,y⟩ and ‖y‖² for the Möbius addition. adamw_step_fp32_kernel
 * is one thread per *element* and can express none of that.
 *
 * Layout: one block per parameter row, blockDim a power of two near D.
 * Reductions are warp-shuffle + shared, and every thread leaves the
 * reduction holding the sum, so no separate broadcast is needed.
 *
 * The step vector lives in *registers* across those reductions — a row is
 * read once and never round-trips through global memory. That is the whole
 * reason for the kMaxPerThread register array: with blockDim ≥ D (which
 * holds for every D ≤ 1024, i.e. every embedding factor we train) each
 * thread owns exactly one coordinate and the array collapses to a single
 * register. Rows wider than kMaxPerThread × 1024 report unsupported and
 * fall back to the CPU reference rather than silently allocating scratch.
 *
 * Reduction order differs from the CPU's sequential accumulation (tree vs
 * left-to-right), so the two agree to fp32 round-off, not bitwise. The
 * parity test's tolerances are set for that and nothing looser.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

extern "C" TC_CUDA_INTERNAL void tc_cuda_set_last_kernel(const char* name);

/* Off-manifold clamp tally, drained into the process-global counters owned by
 * lib/ops/riemannian_adam_cpu.cpp so CPU and CUDA report through one surface.
 * See the contract in include/tensorcore/riemannian_adam.h. */
extern "C" void tc_riemannian_offmanifold_add(unsigned long long entry,
                                              unsigned long long retract);

__device__ unsigned long long tc_d_offmanifold_entry   = 0ULL;
__device__ unsigned long long tc_d_offmanifold_retract = 0ULL;

namespace {

/* Numerical safety constants — must match riemannian_adam_cpu.cpp exactly,
 * or the CPU and CUDA paths diverge at the boundary instead of at round-off. */
constexpr float kEpsNorm  = 1e-15f;
constexpr float kEpsAtanh = 1e-7f;
constexpr float kMargin   = 1e-5f;

/* Coordinates per thread. 8 × 512 threads = rows up to D = 4096.
 * Every manifold factor qLLM trains is D ≤ 256, so in practice K = 1.
 *
 * The block cap is 512, not the hardware's 1024, and that is a measured
 * limit rather than a guess. These kernels hold two kMaxPerThread float
 * arrays live across the reductions (the step vector and the retracted
 * point), so at 1024 threads the block's register demand exceeds the 64K
 * budget and the launch is rejected outright with
 * cudaErrorLaunchOutOfResources -- D=1024 returned TC_ERR_INTERNAL before
 * this cap. 512 halves the per-block demand and costs nothing for D ≤ 512,
 * where each thread still owns exactly one coordinate. */
constexpr int kMaxPerThread = 8;
constexpr int kMaxBlock     = 512;

constexpr int kCudaOk          = 0;
constexpr int kCudaUnsupported = 1;
constexpr int kCudaError       = -1;

/* Block-wide sum that leaves the result in *every* thread.
 *
 * The usual block_reduce_sum leaves the total in thread 0 and needs a
 * shared-memory broadcast before the next pointwise pass can use it. Here
 * every thread re-reads the (at most 32) per-warp partials itself, which
 * costs a handful of shared loads and saves a sync plus a broadcast slot.
 * The trailing __syncthreads() is what makes the scratch reusable by the
 * next call in the same kernel.
 */
__device__ inline float row_sum(float v, float* ws) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) >> 5;
    float total = 0.0f;
    for (int i = 0; i < n_warps; ++i) total += ws[i];
    __syncthreads();
    return total;
}

/* Two independent sums in one pass — the curved steps always need reductions
 * in pairs (⟨x,y⟩ with ‖y‖², ⟨m,x_new⟩ with ⟨x,x_new⟩), and folding them
 * halves the syncs. */
__device__ inline void row_sum2(float a, float b, float* ws,
                                float* out_a, float* out_b) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xFFFFFFFFu, a, off);
        b += __shfl_xor_sync(0xFFFFFFFFu, b, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { ws[warp] = a; ws[32 + warp] = b; }
    __syncthreads();
    const int n_warps = (blockDim.x + 31) >> 5;
    float ta = 0.0f, tb = 0.0f;
    for (int i = 0; i < n_warps; ++i) { ta += ws[i]; tb += ws[32 + i]; }
    __syncthreads();
    *out_a = ta;
    *out_b = tb;
}

__device__ inline float safe_norm_from_sq(float sq) {
    return sq > kEpsNorm * kEpsNorm ? sqrtf(sq) : kEpsNorm;
}

/* ----------------------------------------------------------------------- *
 * Poincaré ball, curvature c.
 *
 *   g_t   = g / λ_x²,             λ_x = 2/(1 − c‖x‖²)
 *   Adam moments, decoupled decay
 *   x_new = x ⊕_c (tanh(√c·λ_x·‖s‖/2)·s/(√c‖s‖))
 *   m    ← (λ_x/λ_{x_new})·m
 * ----------------------------------------------------------------------- */
__global__ void riemannian_adam_poincare_kernel(
        float* __restrict__ params,
        const float* __restrict__ grads,
        float* __restrict__ m,
        float* __restrict__ v,
        int D, float c,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    __shared__ float ws[64];
    const size_t row = (size_t)blockIdx.x * (size_t)D;
    float* __restrict__ x        = params + row;
    const float* __restrict__ g  = grads + row;
    float* __restrict__ mr       = m + row;
    float* __restrict__ vr       = v + row;

    /* Conformal factor at x. */
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) { const float xd = x[d]; acc += xd * xd; }
    }
    const float x2 = row_sum(acc, ws);
    float lam_denom = 1.0f - c * x2;
    if (lam_denom < kEpsNorm) {
        /* x was already off the ball on entry. x2 comes from a block-wide
         * reduction, so this branch is block-uniform and one thread counts
         * the row exactly once. Without the count the clamp below silently
         * drives 1/λ_x² to ~2.5e-31 and the row stops learning. */
        if (threadIdx.x == 0) atomicAdd(&tc_d_offmanifold_entry, 1ULL);
        lam_denom = kEpsNorm;
    }
    const float lam_x      = 2.0f / lam_denom;
    const float inv_lam_sq = 1.0f / (lam_x * lam_x);

    /* Moments + step direction. step[] stays in registers from here to the
     * retraction; the row is never written back as an intermediate. */
    float step[kMaxPerThread];
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        step[k] = 0.0f;
        if (d < D) {
            const float g_t   = g[d] * inv_lam_sq;
            const float new_m = beta1 * mr[d] + (1.0f - beta1) * g_t;
            const float new_v = beta2 * vr[d] + (1.0f - beta2) * g_t * g_t;
            mr[d] = new_m;
            vr[d] = new_v;
            const float m_hat = new_m / bias_correction1;
            const float v_hat = new_v / bias_correction2;
            float update = m_hat / (sqrtf(v_hat) + eps);
            update += weight_decay * x[d];   /* AdamW decoupled decay */
            step[k] = -lr * update;
            ssq += step[k] * step[k];
        }
    }
    const float nv     = safe_norm_from_sq(row_sum(ssq, ws));
    const float sqrt_c = sqrtf(c);
    const float sc_n   = sqrt_c * nv;
    const float second = (sc_n > kEpsNorm) ? (tanhf(sc_n * lam_x * 0.5f) / sc_n)
                                           : (lam_x * 0.5f);

    /* Scale in place, and collect the two inner products Möbius needs. */
    float y2_acc = 0.0f, xy_acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        step[k] *= second;
        if (d < D) {
            y2_acc += step[k] * step[k];
            xy_acc += x[d] * step[k];
        }
    }
    float y2 = 0.0f, xy = 0.0f;
    row_sum2(y2_acc, xy_acc, ws, &y2, &xy);

    const float a_co = 1.0f + 2.0f * c * xy + c * y2;
    const float b_co = 1.0f - c * x2;
    float denom = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    if (denom < kEpsNorm) denom = kEpsNorm;
    const float inv = 1.0f / denom;

    float xn[kMaxPerThread];
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        xn[k] = (d < D) ? (a_co * x[d] + b_co * step[k]) * inv : 0.0f;
    }

    /* Boundary projection. c is a kernel argument and the reduced norm is
     * uniform, so both branches are block-uniform and the syncs inside
     * row_sum stay legal. */
    if (sqrt_c != 0.0f) {
        float na = 0.0f;
#pragma unroll
        for (int k = 0; k < kMaxPerThread; ++k) {
            const int d = threadIdx.x + k * blockDim.x;
            if (d < D) na += xn[k] * xn[k];
        }
        const float nn    = safe_norm_from_sq(row_sum(na, ws));
        const float max_n = (1.0f - kMargin) / sqrt_c;
        if (nn > max_n) {
            const float s = max_n / nn;
#pragma unroll
            for (int k = 0; k < kMaxPerThread; ++k) xn[k] *= s;
        }
    }

    /* λ at the new point, measured on the committed coordinates (the CPU
     * reference re-reads them after projection, so we re-reduce too rather
     * than scaling the old norm — same value, same rounding). */
    float na2 = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) na2 += xn[k] * xn[k];
    }
    const float xn2 = row_sum(na2, ws);
    float lam_new_denom = 1.0f - c * xn2;
    if (lam_new_denom < kEpsNorm) {
        /* Retraction landed off the ball despite the boundary projection --
         * the step was too large for the local geometry. The transport ratio
         * below collapses to ~0, wiping the moment instead of moving it. */
        if (threadIdx.x == 0) atomicAdd(&tc_d_offmanifold_retract, 1ULL);
        lam_new_denom = kEpsNorm;
    }
    const float lam_new = 2.0f / lam_new_denom;
    const float ratio   = lam_x / lam_new;

#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) {
            mr[d] *= ratio;      /* parallel transport of the first moment */
            x[d]   = xn[k];
        }
    }
}

/* ----------------------------------------------------------------------- *
 * Unit sphere.
 *
 *   g_t   = g − ⟨g,x⟩x
 *   Adam moments, decoupled decay, then re-project (decay adds a normal
 *   component that must come back off before the exp map)
 *   x_new = cos‖s‖·x + sin‖s‖·s/‖s‖, renormalised
 *   m    ← m − ⟨m,x_new⟩/(1+⟨x,x_new⟩)·(x + x_new)
 * ----------------------------------------------------------------------- */
__global__ void riemannian_adam_sphere_kernel(
        float* __restrict__ params,
        const float* __restrict__ grads,
        float* __restrict__ m,
        float* __restrict__ v,
        int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    __shared__ float ws[64];
    const size_t row = (size_t)blockIdx.x * (size_t)D;
    float* __restrict__ x       = params + row;
    const float* __restrict__ g = grads + row;
    float* __restrict__ mr      = m + row;
    float* __restrict__ vr      = v + row;

    /* Tangent projection needs ⟨g, x⟩. */
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) acc += g[d] * x[d];
    }
    const float gx = row_sum(acc, ws);

    float step[kMaxPerThread];
    float sx_acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        step[k] = 0.0f;
        if (d < D) {
            const float g_t   = g[d] - gx * x[d];
            const float new_m = beta1 * mr[d] + (1.0f - beta1) * g_t;
            const float new_v = beta2 * vr[d] + (1.0f - beta2) * g_t * g_t;
            mr[d] = new_m;
            vr[d] = new_v;
            const float m_hat = new_m / bias_correction1;
            const float v_hat = new_v / bias_correction2;
            float update = m_hat / (sqrtf(v_hat) + eps);
            update += weight_decay * x[d];
            step[k] = -lr * update;
            sx_acc += step[k] * x[d];
        }
    }
    const float step_x = row_sum(sx_acc, ws);

    /* Strip the normal component the decay reintroduced, then ‖step‖. */
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) {
            step[k] -= step_x * x[d];
            ssq += step[k] * step[k];
        }
    }
    const float n_s = safe_norm_from_sq(row_sum(ssq, ws));
    const float c_v = cosf(n_s);
    const float s_v = sinf(n_s);

    float xn[kMaxPerThread];
    float na = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        xn[k] = 0.0f;
        if (d < D) {
            xn[k] = c_v * x[d] + s_v * step[k] / n_s;
            na += xn[k] * xn[k];
        }
    }
    /* Renormalise: the exp map is exact in exact arithmetic, not in fp32. */
    const float nx = safe_norm_from_sq(row_sum(na, ws));
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) xn[k] /= nx;

    float mxn_acc = 0.0f, xxn_acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) {
            mxn_acc += mr[d] * xn[k];
            xxn_acc += x[d] * xn[k];
        }
    }
    float m_xn = 0.0f, x_xn = 0.0f;
    row_sum2(mxn_acc, xxn_acc, ws, &m_xn, &x_xn);
    x_xn = fminf(fmaxf(x_xn, -1.0f + kEpsAtanh), 1.0f - kEpsAtanh);
    const float coef = m_xn / (1.0f + x_xn);

#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        if (d < D) {
            mr[d] -= coef * (x[d] + xn[k]);   /* uses the *old* x, as on CPU */
            x[d]   = xn[k];
        }
    }
}

/* ----------------------------------------------------------------------- *
 * Euclidean (c → 0). Elementwise; identical formula to
 * adamw_step_fp32_kernel, kept here so the product-manifold optimizer has
 * one dispatch surface across all three factors.
 * ----------------------------------------------------------------------- */
__global__ void riemannian_adam_euclidean_kernel(
        float* __restrict__ params,
        const float* __restrict__ grads,
        float* __restrict__ m,
        float* __restrict__ v,
        int n_elements,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    const float g     = grads[i];
    const float new_m = beta1 * m[i] + (1.0f - beta1) * g;
    const float new_v = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = new_m;
    v[i] = new_v;
    const float m_hat  = new_m / bias_correction1;
    const float v_hat  = new_v / bias_correction2;
    const float update = m_hat / (sqrtf(v_hat) + eps);
    params[i] = (params[i] - lr * weight_decay * params[i]) - lr * update;
}

/* Pointer classification.
 *
 * Managed memory is reachable from host and device; plain device memory is
 * reachable only from the device. The kernels need nothing more than device
 * addressability, so both qualify -- but the CPU reference can only touch the
 * managed kind, and that distinction decides whether falling back is legal.
 *
 * Accepting device pointers is what lets a PyTorch CUDA tensor go straight to
 * the kernel via tc_buffer_from_ptr, with no allocation and no copy. Requiring
 * `managed` forced the binding through numpy, and that round trip -- not the
 * arithmetic -- was the entire cost of the optimizer step. */
enum PtrKind { kPtrHost = 0, kPtrManaged = 1, kPtrDevice = 2 };

PtrKind classify_ptr(const void* ptr) {
    if (!ptr) return kPtrHost;
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return kPtrHost;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
    const auto t = attr.type;
#else
    const auto t = attr.memoryType;
#endif
    if (t == cudaMemoryTypeManaged) return kPtrManaged;
    if (t == cudaMemoryTypeDevice)  return kPtrDevice;
    return kPtrHost;
}

bool device_addressable(const void* p) { return classify_ptr(p) != kPtrHost; }
bool host_addressable(const void* p)   { return classify_ptr(p) != kPtrDevice; }

bool all_device_addressable4(const void* a, const void* b,
                             const void* c, const void* d) {
    const bool ok = device_addressable(a) && device_addressable(b) &&
                    device_addressable(c) && device_addressable(d);
    if (!ok && std::getenv("TC_DEBUG_PTR_KIND")) {
        std::fprintf(stderr,
            "[tc] ptr kinds p=%d g=%d m=%d v=%d (0=host 1=managed 2=device)\n",
            (int)classify_ptr(a), (int)classify_ptr(b),
            (int)classify_ptr(c), (int)classify_ptr(d));
    }
    return ok;
}

/* True when at least one buffer is device-only, i.e. the CPU reference would
 * dereference device memory from the host and segfault. In that case an
 * unsupported shape must be reported as a hard error, never as "fall back". */
bool any_device_only4(const void* a, const void* b,
                      const void* c, const void* d) {
    return !host_addressable(a) || !host_addressable(b) ||
           !host_addressable(c) || !host_addressable(d);
}

/* Power-of-two block size near D, so each thread owns ⌈D/blockDim⌉ ≤
 * kMaxPerThread coordinates. */
int block_for_dim(int D) {
    int block = 32;
    while (block < D && block < kMaxBlock) block <<= 1;
    return block;
}

int finish(const char* name) {
    if (cudaGetLastError() != cudaSuccess) return kCudaError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaError;
    tc_cuda_set_last_kernel(name);
    return kCudaOk;
}

/* Zero the device tallies before a Poincaré launch. Only the Poincaré kernel
 * has a manifold domain to leave, so sphere/euclidean skip this entirely. */
void offmanifold_arm() {
    const unsigned long long z = 0ULL;
    cudaMemcpyToSymbol(tc_d_offmanifold_entry,   &z, sizeof z);
    cudaMemcpyToSymbol(tc_d_offmanifold_retract, &z, sizeof z);
}

/* Fold the device tallies into the process-global counters. Called after
 * finish(), so the kernel's cudaDeviceSynchronize has already retired the
 * atomics. Two 8-byte symbol reads per Poincaré step -- immaterial next to
 * the step itself, and only on the one or two curved parameters a model has. */
void offmanifold_drain() {
    unsigned long long entry = 0ULL, retract = 0ULL;
    if (cudaMemcpyFromSymbol(&entry,   tc_d_offmanifold_entry,   sizeof entry)
            != cudaSuccess) return;
    if (cudaMemcpyFromSymbol(&retract, tc_d_offmanifold_retract, sizeof retract)
            != cudaSuccess) return;
    tc_riemannian_offmanifold_add(entry, retract);
}

}  /* namespace */

/* C-linkage entry points, called from lib/ops/riemannian_adam_cpu.cpp when
 * the buffers are CUDA-managed. Return contract matches training.cu:
 *   0 -> done on GPU, 1 -> unsupported (caller runs the CPU reference),
 *  -1 -> CUDA was selected and failed. */

/* Is this pointer device memory the host cannot read?
 *
 * The public entry points call this before running the CPU reference. Without
 * it, any path that skips the CUDA branch — an inactive runtime, a disabled
 * policy, a shape the kernel declines — hands device pointers to an OpenMP
 * loop and segfaults instead of returning an error. That is exactly what
 * happened the first time the optimizer ran under the real trainer. */
extern "C" TC_CUDA_INTERNAL int tc_cuda_ptr_is_device_only(const void* p) {
    return classify_ptr(p) == kPtrDevice ? 1 : 0;
}

/* Can the GPU dereference this pointer (managed or device)? */
extern "C" TC_CUDA_INTERNAL int tc_cuda_ptr_is_device_addressable(const void* p) {
    return classify_ptr(p) != kPtrHost ? 1 : 0;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_riemannian_adam_step_poincare(
        void* params, const void* grads, void* m, void* v,
        int N, int D, float c,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    if (!params || !grads || !m || !v || N <= 0 || D <= 0) return kCudaError;
    if (!all_device_addressable4(params, grads, m, v)) return kCudaUnsupported;
    if (D > kMaxPerThread * kMaxBlock) {
        /* Too wide for the register layout. Falling back is only safe if the
         * host can actually read these buffers. */
        return any_device_only4(params, grads, m, v) ? kCudaError
                                                     : kCudaUnsupported;
    }
    offmanifold_arm();
    riemannian_adam_poincare_kernel<<<N, block_for_dim(D)>>>(
        (float*)params, (const float*)grads, (float*)m, (float*)v,
        D, c, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
    const int rc = finish("cuda_riemannian_adam_poincare");
    if (rc == kCudaOk) offmanifold_drain();
    return rc;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_riemannian_adam_step_sphere(
        void* params, const void* grads, void* m, void* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    if (!params || !grads || !m || !v || N <= 0 || D <= 0) return kCudaError;
    if (!all_device_addressable4(params, grads, m, v)) return kCudaUnsupported;
    if (D > kMaxPerThread * kMaxBlock) {
        return any_device_only4(params, grads, m, v) ? kCudaError
                                                     : kCudaUnsupported;
    }
    riemannian_adam_sphere_kernel<<<N, block_for_dim(D)>>>(
        (float*)params, (const float*)grads, (float*)m, (float*)v,
        D, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
    return finish("cuda_riemannian_adam_sphere");
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_riemannian_adam_step_euclidean(
        void* params, const void* grads, void* m, void* v,
        int N, int D,
        float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    if (!params || !grads || !m || !v || N <= 0 || D <= 0) return kCudaError;
    if (!all_device_addressable4(params, grads, m, v)) return kCudaUnsupported;
    const long long total = (long long)N * (long long)D;
    if (total > (long long)INT_MAX) {
        return any_device_only4(params, grads, m, v) ? kCudaError
                                                     : kCudaUnsupported;
    }
    const int n_elements = (int)total;
    const int block_size = 256;
    const int blocks     = (n_elements + block_size - 1) / block_size;
    riemannian_adam_euclidean_kernel<<<blocks, block_size>>>(
        (float*)params, (const float*)grads, (float*)m, (float*)v,
        n_elements, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
    return finish("cuda_riemannian_adam_euclidean");
}
