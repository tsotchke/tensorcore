/*
 * tensorcore — fused RMSnorm + GEMV (inference path).
 *
 *   Y[m, n] = (sum_k X[m, k] * rstd(X[m]) * gamma[k] * W[k, n])
 *
 * Computes RMSnorm and matmul in one kernel, eliminating the round-trip of the
 * normalized intermediate. For LLM inference (batch=1 prefill/decode), this
 * is the workhorse — every Q/K/V projection, every MLP gate/up projection,
 * after every transformer block's pre-norm goes through this path.
 *
 * Layout:
 *   - One threadgroup per (output column n, row m). 64 threads, single
 *     simdgroup × 2.
 *   - Each thread accumulates into a partial; cooperative reduction at the
 *     end via simd_sum.
 *
 * Threadgroup memory: holds the row's rstd (1 fp32) — cooperative reduction
 * uses a tiny scratch. No persistent X_norm; computed on the fly.
 *
 * For batch sizes > 4, callers should use tc_rmsnorm_forward + tc_gemm
 * separately — the per-row cost of recomputing rstd dominates at larger M.
 */

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

inline float tg_sum32(float local, threadgroup float* scratch, uint tid, uint threads) {
    const float sg = simd_sum(local);
    const uint sgid = tid >> 5;
    const uint lane = tid & 31;
    if (lane == 0) scratch[sgid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_sg = (threads + 31) >> 5;
    float v = (tid < n_sg) ? scratch[tid] : 0.0f;
    v = simd_sum(v);
    if (tid == 0) scratch[0] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

kernel void tc_fused_rmsnorm_gemv_f16(
    device const half*  X       [[buffer(0)]],   /* [M, K]                  */
    device const half*  gamma   [[buffer(1)]],   /* [K]                     */
    device const half*  W       [[buffer(2)]],   /* [K, N]                  */
    device       half*  Y       [[buffer(3)]],   /* [M, N]                  */
    constant uint& M            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 tid_v                 [[thread_position_in_threadgroup]],
    uint3 tpg                   [[threads_per_threadgroup]])
{
    const uint n = tgid.x;
    const uint m = tgid.y;
    const uint tid = tid_v.x;
    const uint threads = tpg.x;
    if (n >= N || m >= M) return;

    /* Pass 1: compute rstd = rsqrt(mean(x^2) + eps). */
    threadgroup float scratch[8];
    float ss = 0.0f;
    for (uint k = tid; k < K; k += threads) {
        const float x = (float)X[m * K + k];
        ss += x * x;
    }
    const float total = tg_sum32(ss, scratch, tid, threads);
    const float rstd = rsqrt(total / (float)K + eps);

    /* Pass 2: y = sum_k (x[k] * rstd * gamma[k]) * W[k, n].
     * Each thread computes a partial sum over its slice of K. */
    float acc = 0.0f;
    for (uint k = tid; k < K; k += threads) {
        const float x_norm = (float)X[m * K + k] * rstd * (float)gamma[k];
        const float w      = (float)W[k * N + n];
        acc += x_norm * w;
    }
    const float total_y = tg_sum32(acc, scratch, tid, threads);

    if (tid == 0) {
        Y[m * N + n] = (half)total_y;
    }
}

kernel void tc_fused_layernorm_gemv_f16(
    device const half*  X       [[buffer(0)]],   /* [M, K]                  */
    device const half*  gamma   [[buffer(1)]],   /* [K]                     */
    device const half*  beta    [[buffer(2)]],   /* [K]                     */
    device const half*  W       [[buffer(3)]],   /* [K, N]                  */
    device       half*  Y       [[buffer(4)]],   /* [M, N]                  */
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    constant float& eps         [[buffer(8)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 tid_v                 [[thread_position_in_threadgroup]],
    uint3 tpg                   [[threads_per_threadgroup]])
{
    const uint n = tgid.x;
    const uint m = tgid.y;
    const uint tid = tid_v.x;
    const uint threads = tpg.x;
    if (n >= N || m >= M) return;

    threadgroup float scratch[8];

    float sum = 0.0f;
    float ss = 0.0f;
    for (uint k = tid; k < K; k += threads) {
        const float x = (float)X[m * K + k];
        sum += x;
        ss += x * x;
    }
    const float total = tg_sum32(sum, scratch, tid, threads);
    const float total_ss = tg_sum32(ss, scratch, tid, threads);
    const float mean = total / (float)K;
    const float var = max(total_ss / (float)K - mean * mean, 0.0f);
    const float rstd = rsqrt(var + eps);

    float acc = 0.0f;
    for (uint k = tid; k < K; k += threads) {
        const float x_norm = ((float)X[m * K + k] - mean) *
                             rstd * (float)gamma[k] + (float)beta[k];
        const float w = (float)W[k * N + n];
        acc += x_norm * w;
    }
    const float total_y = tg_sum32(acc, scratch, tid, threads);

    if (tid == 0) {
        Y[m * N + n] = (half)total_y;
    }
}
