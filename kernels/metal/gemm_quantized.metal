/*
 * tensorcore - quantized matmul kernels.
 *
 * Q4_0 weight quantization (ggml-style): 32 weights per block, one fp16 scale.
 * Each 4-bit weight w_i is dequantized as scale * (w_i - 8) where w_i is in [0, 15].
 * Range: scale*-8 to scale*7. Symmetric-ish around zero (slight asymmetry for
 * the unsigned offset; matches llama.cpp Q4_0).
 *
 * Layout for weight matrix W [OC, IC] stored as Q4_0:
 *   - IC must be divisible by QK4 = 32.
 *   - For each output channel oc and each block b in [0, IC/QK4):
 *       blocks[oc * (IC/QK4) + b] = { half scale; uint8 qs[QK4/2]; }
 *     Total bytes per block: 2 + 16 = 18.
 *   - The 32 weights are packed in GGML Q4_0 order: low nibble is weight i,
 *     high nibble is weight i+16.
 *
 * Kernel: y[m, n] = sum_k x[m, k] * dequant(w[n, k])
 *   - m = output row (batch element), n = output channel
 *   - One threadgroup per (block of m, single n). Cooperative dequant of W
 *     into threadgroup memory once per K-block, then accumulate.
 *
 * For inference (M=1, batched-size-1 generation step), the structure is
 * actually a GEMV; most weight reads dominate. The dequant kernel keeps
 * compute high enough that we're memory-bandwidth-bound on weight reads.
 *
 * Reported by llama.cpp on M2 Ultra: ~150 GB/s effective weight read rate
 * -> ~75 tok/s on 7B Q4_0. Our kernel targets the same regime.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint QK4 = 32;            /* weights per Q4_0 block */
constant constexpr uint Q4_BLOCK_BYTES = 18; /* scale (2) + 16 packed nibbles */

/* Q4_0 GEMV: M batches x K (fp16 activations) @ K x N (Q4_0 weights) -> M x N fp16.
 * For LLM inference M is typically 1 (next-token prediction); for prefill
 * M can be the prompt length.
 *
 * Threadgroup layout:
 *   - One simdgroup per (m_row, n_block_of_32).
 *   - 32 threads cooperatively decode and multiply one 32-wide K-chunk per step.
 *   - Reduction is via simd_sum.
 */
kernel void tc_q4_0_gemv_f16(
    device const half*     X        [[buffer(0)]],   /* [M, K]                    */
    device const uchar*    Wq       [[buffer(1)]],   /* packed Q4_0 blocks        */
    device       half*     Y        [[buffer(2)]],   /* [M, N]                    */
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& K               [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint  lane                      [[thread_index_in_simdgroup]])
{
    const uint n = tgid.x;                       /* output column           */
    const uint m = tgid.y;                       /* output row              */
    if (n >= N || m >= M) return;

    const uint nblocks = K / QK4;
    const uint w_row_bytes = nblocks * Q4_BLOCK_BYTES;
    /* Each output column n has its own row of weight blocks. */
    device const uchar* W_row = Wq + (size_t)n * w_row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < nblocks; ++b) {
        device const uchar* block = W_row + (size_t)b * Q4_BLOCK_BYTES;
        /* Read fp16 scale (2 bytes). */
        const half scale_h = *((device const half*)block);
        const float scale = (float)scale_h;
        /* Each lane handles one weight in the 32-wide block. */
        const uint i = lane;
        const uint k = b * QK4 + i;
        if (k < K) {
            const uchar packed = block[2 + (i % (QK4 / 2))];
            const uint nibble = (i < (QK4 / 2)) ? (packed & 0xF) : (packed >> 4);
            const float w = scale * ((float)nibble - 8.0f);
            const float x = (float)X[m * K + k];
            acc += x * w;
        }
    }
    /* Simdgroup reduction. */
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[m * N + n] = (half)acc;
    }
}

/* Q8_0: 32 int8 weights per block, one fp16 scale.
 * Dequant: scale * w (signed int8).
 * Higher fidelity than Q4_0; ~2x the size.
 *
 * Block layout: { half scale; int8 qs[32]; } = 34 bytes per block. */
constant constexpr uint Q8_BLOCK_BYTES = 34;

kernel void tc_q8_0_gemv_f16(
    device const half*     X        [[buffer(0)]],
    device const uchar*    Wq       [[buffer(1)]],
    device       half*     Y        [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& K               [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint  lane                      [[thread_index_in_simdgroup]])
{
    const uint n = tgid.x;
    const uint m = tgid.y;
    if (n >= N || m >= M) return;
    const uint nblocks = K / QK4;
    const uint w_row_bytes = nblocks * Q8_BLOCK_BYTES;
    device const uchar* W_row = Wq + (size_t)n * w_row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < nblocks; ++b) {
        device const uchar* block = W_row + (size_t)b * Q8_BLOCK_BYTES;
        const half scale_h = *((device const half*)block);
        const float scale = (float)scale_h;
        const uint i = lane;
        const uint k = b * QK4 + i;
        if (k < K) {
            const int8_t w_i8 = ((device const int8_t*)(block + 2))[i];
            const float w = scale * (float)w_i8;
            const float x = (float)X[m * K + k];
            acc += x * w;
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[m * N + n] = (half)acc;
    }
}

/* Q4_0 weight quantization (CPU-side; runs on GPU here for parallelism).
 * Input: [N, K] fp16 weights. Output: packed Q4_0 blocks.
 * One threadgroup per output channel. */
kernel void tc_quantize_q4_0(
    device const half*    W        [[buffer(0)]],   /* [N, K] fp16 input  */
    device       uchar*   Wq       [[buffer(1)]],   /* Q4_0 packed output */
    constant uint& N              [[buffer(2)]],
    constant uint& K              [[buffer(3)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]])
{
    const uint n = tgid.y;
    const uint b = tgid.x;       /* block index along K */
    if (n >= N) return;
    const uint nblocks = K / QK4;
    if (b >= nblocks) return;

    const uint k0 = b * QK4;
    /* Find max abs in this 32-wide block. */
    float my_abs = 0.0f;
    if (lane < QK4) {
        my_abs = fabs((float)W[n * K + k0 + lane]);
    }
    const float max_abs = simd_max(my_abs);
    /* Q4_0 scale = max_abs / -8 (the most-negative quantized value maps to
     * -max_abs). For ggml-exact compat use abs/8 but sign-positive scale. */
    const float scale = max_abs / 8.0f;
    const float inv_scale = (scale > 0.0f) ? (8.0f / max_abs) : 0.0f;

    const uint w_row_bytes = nblocks * Q4_BLOCK_BYTES;
    device uchar* dst = Wq + (size_t)n * w_row_bytes + (size_t)b * Q4_BLOCK_BYTES;
    if (lane == 0) {
        ((device half*)dst)[0] = (half)scale;
    }
    /* Quantize: q = round(w * inv_scale) + 8, clamped to [0, 15].
     * GGML Q4_0 packs low nibble = i, high nibble = i+16. */
    if (lane < QK4 / 2) {
        const uint i_lo = lane;
        const uint i_hi = lane + QK4 / 2;
        const float w_lo = (float)W[n * K + k0 + i_lo];
        const float w_hi = (float)W[n * K + k0 + i_hi];
        int q_lo = (int)round(w_lo * inv_scale) + 8;
        int q_hi = (int)round(w_hi * inv_scale) + 8;
        q_lo = clamp(q_lo, 0, 15);
        q_hi = clamp(q_hi, 0, 15);
        dst[2 + lane] = (uchar)((q_hi << 4) | q_lo);
    }
}

/* Q8_0 weight quantization.
 * Input: [N, K] fp16 weights. Output: packed Q8_0 blocks. */
kernel void tc_quantize_q8_0(
    device const half*    W        [[buffer(0)]],   /* [N, K] fp16 input  */
    device       uchar*   Wq       [[buffer(1)]],   /* Q8_0 packed output */
    constant uint& N              [[buffer(2)]],
    constant uint& K              [[buffer(3)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]])
{
    const uint n = tgid.y;
    const uint b = tgid.x;
    if (n >= N) return;
    const uint nblocks = K / QK4;
    if (b >= nblocks) return;

    const uint k0 = b * QK4;
    float my_abs = 0.0f;
    if (lane < QK4) {
        my_abs = fabs((float)W[n * K + k0 + lane]);
    }
    const float max_abs = simd_max(my_abs);
    const float scale = max_abs / 127.0f;
    const float inv_scale = (scale > 0.0f) ? (127.0f / max_abs) : 0.0f;

    const uint w_row_bytes = nblocks * Q8_BLOCK_BYTES;
    device uchar* dst = Wq + (size_t)n * w_row_bytes + (size_t)b * Q8_BLOCK_BYTES;
    if (lane == 0) {
        ((device half*)dst)[0] = (half)scale;
    }
    if (lane < QK4) {
        int q = (int)round((float)W[n * K + k0 + lane] * inv_scale);
        q = clamp(q, -128, 127);
        ((device int8_t*)(dst + 2))[lane] = (int8_t)q;
    }
}
