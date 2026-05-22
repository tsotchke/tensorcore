/*
 * tensorcore — fused training kernels.
 *
 * All the small ops that wrap a transformer training loop: RMSnorm,
 * LayerNorm, RoPE, SwiGLU, softmax, AdamW. Fwd + bwd for the ones that
 * need it. fp16 IO, fp32 accumulators throughout.
 */

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

/* Cross-simdgroup reduction: each simdgroup computes a partial, then the
 * first simdgroup combines partials, then we broadcast back to all threads
 * via threadgroup memory. Scratch needs >= max(n_simdgroups, 1) floats and
 * the final slot 0 is used for the broadcast. */
inline float tg_sum_broadcast(float local, threadgroup float* scratch,
                              uint tid, uint threads) {
    const float sg_partial = simd_sum(local);
    const uint sgid = tid >> 5;
    const uint lane = tid & 31;
    if (lane == 0) scratch[sgid] = sg_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_sg = (threads + 31) >> 5;
    float v = (tid < n_sg) ? scratch[tid] : 0.0f;
    v = simd_sum(v);
    if (tid == 0) scratch[0] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
inline float tg_max_broadcast(float local, threadgroup float* scratch,
                              uint tid, uint threads) {
    const float sg_partial = simd_max(local);
    const uint sgid = tid >> 5;
    const uint lane = tid & 31;
    if (lane == 0) scratch[sgid] = sg_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_sg = (threads + 31) >> 5;
    float v = (tid < n_sg) ? scratch[tid] : -INFINITY;
    v = simd_max(v);
    if (tid == 0) scratch[0] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

/* ====================================================================== *
 *  RMSnorm                                                                *
 *  y = (x / rms(x)) * gamma,  rms(x) = sqrt(mean(x^2) + eps)               *
 *  One TG per (batch_idx, seq_idx) row; reduce over hidden_dim.           *
 * ====================================================================== */

kernel void tc_rmsnorm_forward(
    device const half*  X       [[buffer(0)]],   /* [N, D] */
    device const half*  gamma   [[buffer(1)]],   /* [D]    */
    device       half*  Y       [[buffer(2)]],   /* [N, D] */
    device       float* rstd_o  [[buffer(3)]],   /* [N], saved for bwd */
    constant uint& N            [[buffer(4)]],
    constant uint& D            [[buffer(5)]],
    constant float& eps         [[buffer(6)]],
    uint  group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  threads               [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;

    threadgroup float scratch[32];
    float local = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        const float x = (float)X[row * D + i];
        local += x * x;
    }
    const float total = tg_sum_broadcast(local, scratch, tid, threads);

    const float rstd = rsqrt(total / (float)D + eps);
    if (tid == 0) rstd_o[row] = rstd;

    for (uint i = tid; i < D; i += threads) {
        const float g = (float)gamma[i];
        Y[row * D + i] = (half)((float)X[row * D + i] * rstd * g);
    }
}

kernel void tc_rmsnorm_backward(
    device const half*  X       [[buffer(0)]],
    device const half*  gamma   [[buffer(1)]],
    device const half*  dY      [[buffer(2)]],
    device const float* rstd    [[buffer(3)]],
    device       half*  dX      [[buffer(4)]],
    device       float* dgamma  [[buffer(5)]],   /* atomic accumulate */
    constant uint& N            [[buffer(6)]],
    constant uint& D            [[buffer(7)]],
    uint  group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  threads               [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;
    const float r = rstd[row];

    threadgroup float scratch[32];
    float local = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        local += (float)dY[row * D + i] * (float)gamma[i] * (float)X[row * D + i];
    }
    const float dot = tg_sum_broadcast(local, scratch, tid, threads);

    const float scale = r * r * r / (float)D;
    for (uint i = tid; i < D; i += threads) {
        const float x = (float)X[row * D + i];
        const float dy = (float)dY[row * D + i];
        const float g = (float)gamma[i];
        const float dx = dy * g * r - x * dot * scale;
        dX[row * D + i] = (half)dx;
        /* dgamma_i += dY_i * X_i * rstd  — atomic across rows */
        const float dg = dy * x * r;
        atomic_fetch_add_explicit(
            (device atomic_uint*)&dgamma[i], 0, memory_order_relaxed);  /* warmup */
        (void)dg;
        /* Actual accumulation: use atomic_float when available; fallback to
         * two-phase reduction in v0.2. For v0.1 we ship a separate reduction
         * kernel (tc_rmsnorm_reduce_dgamma) called after this. */
    }
}

/* Reduction kernel for dgamma: dgamma[d] = sum_n dY[n,d] * X[n,d] * rstd[n]. */
kernel void tc_rmsnorm_reduce_dgamma(
    device const half*  X       [[buffer(0)]],
    device const half*  dY      [[buffer(1)]],
    device const float* rstd    [[buffer(2)]],
    device       float* dgamma  [[buffer(3)]],
    constant uint& N            [[buffer(4)]],
    constant uint& D            [[buffer(5)]],
    uint  d                     [[thread_position_in_grid]])
{
    if (d >= D) return;
    float acc = 0.0f;
    for (uint n = 0; n < N; ++n) {
        acc += (float)dY[n * D + d] * (float)X[n * D + d] * rstd[n];
    }
    dgamma[d] = acc;
}

/* ====================================================================== *
 *  LayerNorm                                                              *
 *  y = ((x - mean) / std) * gamma + beta                                  *
 * ====================================================================== */

kernel void tc_layernorm_forward(
    device const half*  X       [[buffer(0)]],
    device const half*  gamma   [[buffer(1)]],
    device const half*  beta    [[buffer(2)]],
    device       half*  Y       [[buffer(3)]],
    device       float* mean_o  [[buffer(4)]],   /* [N] */
    device       float* rstd_o  [[buffer(5)]],   /* [N] */
    constant uint& N            [[buffer(6)]],
    constant uint& D            [[buffer(7)]],
    constant float& eps         [[buffer(8)]],
    uint  group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  threads               [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;

    threadgroup float scratch[32];
    float ls = 0.0f, lsq = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        const float x = (float)X[row * D + i];
        ls += x;
        lsq += x * x;
    }
    const float s = tg_sum_broadcast(ls, scratch, tid, threads);
    const float sq = tg_sum_broadcast(lsq, scratch, tid, threads);

    const float mean = s / (float)D;
    const float var  = sq / (float)D - mean * mean;
    const float rstd = rsqrt(var + eps);
    if (tid == 0) { mean_o[row] = mean; rstd_o[row] = rstd; }

    for (uint i = tid; i < D; i += threads) {
        const float x = (float)X[row * D + i];
        const float g = (float)gamma[i];
        const float b = (float)beta[i];
        Y[row * D + i] = (half)((x - mean) * rstd * g + b);
    }
}

kernel void tc_layernorm_backward(
    device const half*  X       [[buffer(0)]],
    device const half*  gamma   [[buffer(1)]],
    device const half*  dY      [[buffer(2)]],
    device const float* mean    [[buffer(3)]],
    device const float* rstd    [[buffer(4)]],
    device       half*  dX      [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& D            [[buffer(7)]],
    uint  group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  threads               [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;
    const float m = mean[row], r = rstd[row];

    threadgroup float scratch[32];
    float l_sum = 0.0f, l_dot = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        const float xh = ((float)X[row * D + i] - m) * r;
        const float dyg = (float)dY[row * D + i] * (float)gamma[i];
        l_sum += dyg;
        l_dot += dyg * xh;
    }
    const float sumg = tg_sum_broadcast(l_sum, scratch, tid, threads);
    const float dotg = tg_sum_broadcast(l_dot, scratch, tid, threads);

    const float inv_D = 1.0f / (float)D;
    for (uint i = tid; i < D; i += threads) {
        const float xh = ((float)X[row * D + i] - m) * r;
        const float dyg = (float)dY[row * D + i] * (float)gamma[i];
        const float dx = (dyg - sumg * inv_D - xh * dotg * inv_D) * r;
        dX[row * D + i] = (half)dx;
    }
}

/* ====================================================================== *
 *  RoPE (Rotary Position Embedding)                                       *
 *  Applies rotation pairs (cos, sin) per position, per head dim/2.        *
 *  Input: [B, H, S, D] half. Pairs (d, d+D/2) form (x, y) rotated by      *
 *  (cos_s, sin_s) where s is the position index.                          *
 * ====================================================================== */

kernel void tc_rope_forward(
    device       half*  X        [[buffer(0)]],   /* in-place: [B,H,S,D]    */
    device const float* cos_t    [[buffer(1)]],   /* [S, D/2]               */
    device const float* sin_t    [[buffer(2)]],   /* [S, D/2]               */
    constant uint& batch         [[buffer(3)]],
    constant uint& heads         [[buffer(4)]],
    constant uint& seq           [[buffer(5)]],
    constant uint& D             [[buffer(6)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    const uint d2 = gid.x;            /* 0..D/2-1 */
    const uint sh = gid.y;            /* seq + head packed */
    const uint b  = gid.z;
    const uint s  = sh / heads;
    const uint h  = sh % heads;
    if (b >= batch || h >= heads || s >= seq || d2 >= D / 2) return;

    const float c = cos_t[s * (D / 2) + d2];
    const float si = sin_t[s * (D / 2) + d2];

    const uint base = ((b * heads + h) * seq + s) * D;
    const float x = (float)X[base + d2];
    const float y = (float)X[base + d2 + D / 2];
    X[base + d2]         = (half)(x * c - y * si);
    X[base + d2 + D / 2] = (half)(x * si + y * c);
}

kernel void tc_rope_backward(
    device       half*  dX       [[buffer(0)]],   /* in-place: [B,H,S,D]    */
    device const float* cos_t    [[buffer(1)]],   /* [S, D/2]               */
    device const float* sin_t    [[buffer(2)]],   /* [S, D/2]               */
    constant uint& batch         [[buffer(3)]],
    constant uint& heads         [[buffer(4)]],
    constant uint& seq           [[buffer(5)]],
    constant uint& D             [[buffer(6)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    const uint d2 = gid.x;            /* 0..D/2-1 */
    const uint sh = gid.y;            /* seq + head packed */
    const uint b  = gid.z;
    const uint s  = sh / heads;
    const uint h  = sh % heads;
    if (b >= batch || h >= heads || s >= seq || d2 >= D / 2) return;

    const float c = cos_t[s * (D / 2) + d2];
    const float si = sin_t[s * (D / 2) + d2];

    const uint base = ((b * heads + h) * seq + s) * D;
    const float dy0 = (float)dX[base + d2];
    const float dy1 = (float)dX[base + d2 + D / 2];
    dX[base + d2]         = (half)(dy0 * c + dy1 * si);
    dX[base + d2 + D / 2] = (half)(-dy0 * si + dy1 * c);
}

/* ====================================================================== *
 *  SwiGLU activation                                                      *
 *  y = silu(x_gate) * x_up  where silu(z) = z * sigmoid(z)                *
 *  Two inputs side-by-side (gate, up).                                    *
 * ====================================================================== */

kernel void tc_swiglu_forward(
    device const half*  gate   [[buffer(0)]],
    device const half*  up     [[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    uint i                     [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float g = (float)gate[i];
    const float u = (float)up[i];
    const float silu_g = g / (1.0f + exp(-g));
    out[i] = (half)(silu_g * u);
}

kernel void tc_swiglu_backward(
    device const half*  gate   [[buffer(0)]],
    device const half*  up     [[buffer(1)]],
    device const half*  dout   [[buffer(2)]],
    device       half*  dgate  [[buffer(3)]],
    device       half*  dup    [[buffer(4)]],
    constant uint& n           [[buffer(5)]],
    uint i                     [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float g = (float)gate[i];
    const float u = (float)up[i];
    const float dz = (float)dout[i];
    const float sig = 1.0f / (1.0f + exp(-g));
    const float silu_g = g * sig;
    /* d(silu)/dg = sig * (1 + g * (1 - sig)) */
    const float d_silu = sig * (1.0f + g * (1.0f - sig));
    dgate[i] = (half)(dz * u * d_silu);
    dup[i]   = (half)(dz * silu_g);
}

/* ====================================================================== *
 *  Standalone softmax (numerically stable, fp32 acc)                      *
 *  Input/output: [N, D], row-wise softmax over D.                         *
 * ====================================================================== */

kernel void tc_softmax_forward(
    device const half*  X      [[buffer(0)]],
    device       half*  Y      [[buffer(1)]],
    constant uint& N           [[buffer(2)]],
    constant uint& D           [[buffer(3)]],
    uint  group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  threads              [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;

    threadgroup float scratch[32];
    float lm = -INFINITY;
    for (uint i = tid; i < D; i += threads) {
        lm = max(lm, (float)X[row * D + i]);
    }
    const float m = tg_max_broadcast(lm, scratch, tid, threads);

    float ls = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        ls += exp((float)X[row * D + i] - m);
    }
    const float s = tg_sum_broadcast(ls, scratch, tid, threads);
    const float inv_s = 1.0f / (s + 1e-30f);

    for (uint i = tid; i < D; i += threads) {
        Y[row * D + i] = (half)(exp((float)X[row * D + i] - m) * inv_s);
    }
}

kernel void tc_softmax_backward(
    device const half*  Y      [[buffer(0)]],   /* forward output */
    device const half*  dY     [[buffer(1)]],
    device       half*  dX     [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    uint  group_id             [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  threads              [[threads_per_threadgroup]])
{
    if (group_id >= N) return;
    const uint row = group_id;

    threadgroup float scratch[32];
    float ld = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        ld += (float)dY[row * D + i] * (float)Y[row * D + i];
    }
    const float dot = tg_sum_broadcast(ld, scratch, tid, threads);

    for (uint i = tid; i < D; i += threads) {
        const float y = (float)Y[row * D + i];
        const float dy = (float)dY[row * D + i];
        dX[row * D + i] = (half)(y * (dy - dot));
    }
}

/* ====================================================================== *
 *  Fused AdamW step                                                       *
 *    m_t = beta1 * m_{t-1} + (1-beta1) * g                                *
 *    v_t = beta2 * v_{t-1} + (1-beta2) * g^2                              *
 *    m_hat = m_t / (1 - beta1^t)                                          *
 *    v_hat = v_t / (1 - beta2^t)                                          *
 *    p_t = p_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p_{t-1})    *
 *  Params in fp32 master copy, gradient in fp16/fp32.                     *
 * ====================================================================== */

kernel void tc_adamw_step_f32(
    device       float* params  [[buffer(0)]],
    device       float* m       [[buffer(1)]],
    device       float* v       [[buffer(2)]],
    device const float* grads   [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    constant float& lr          [[buffer(5)]],
    constant float& beta1       [[buffer(6)]],
    constant float& beta2       [[buffer(7)]],
    constant float& eps         [[buffer(8)]],
    constant float& wd          [[buffer(9)]],
    constant float& bc1         [[buffer(10)]],   /* 1 - beta1^t          */
    constant float& bc2         [[buffer(11)]],   /* 1 - beta2^t          */
    uint i                      [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float g = grads[i];
    const float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    const float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;
    const float m_hat = m_new / bc1;
    const float v_hat = v_new / bc2;
    const float p = params[i];
    params[i] = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p);
}

/* fp16 grad variant (master weights stay fp32). */
kernel void tc_adamw_step_f16grad(
    device       float* params  [[buffer(0)]],
    device       float* m       [[buffer(1)]],
    device       float* v       [[buffer(2)]],
    device const half*  grads   [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    constant float& lr          [[buffer(5)]],
    constant float& beta1       [[buffer(6)]],
    constant float& beta2       [[buffer(7)]],
    constant float& eps         [[buffer(8)]],
    constant float& wd          [[buffer(9)]],
    constant float& bc1         [[buffer(10)]],
    constant float& bc2         [[buffer(11)]],
    uint i                      [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float g = (float)grads[i];
    const float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    const float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;
    const float m_hat = m_new / bc1;
    const float v_hat = v_new / bc2;
    const float p = params[i];
    params[i] = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p);
}
