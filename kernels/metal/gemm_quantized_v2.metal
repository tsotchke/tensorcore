/*
 * tensorcore - Q4_0 GEMV v2 (llama.cpp-class perf).
 *
 * Pattern lifted from ggml/src/ggml-metal/ggml-metal.metal (master):
 *   - NR0 = 4 output rows per simdgroup
 *   - NSG = 2 simdgroups per threadgroup -> 8 outputs per TG
 *   - Each lane holds 16 y-values pre-scaled for the 4 nibble bit-fields
 *     (0x000F, 0x0F00, 0x00F0, 0xF000), so dequant becomes mask+FMA (no shift)
 *   - Q4_0 zero-point folded: `d * (sumy*-8 + sum(yl*nibble))`
 *   - row partials accumulated in registers (no threadgroup memory in hot loop)
 *   - Single `simd_sum` reduction per row
 *
 * Reported llama.cpp perf on M2 Ultra Q4_0 7B decode: ~55 tok/s. Our v1
 * (1 sg per output) got 13.7 tok/s; this v2 closes the 4x design gap.
 */

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant constexpr int QK4 = 32;
constant constexpr int NR0 = 4;          /* output rows per simdgroup */
constant constexpr int NSG_V2 = 2;       /* simdgroups per threadgroup */
constant constexpr int NQ = 16;          /* half-blocks per simd-lane pass */

struct block_q4_0 {
    half  d;
    uchar qs[QK4 / 2];
};

inline float block_q4_0_dot_y(device const block_q4_0* qb, float sumy,
                              thread float* yl, int il) {
    const float d = (float)qb->d;
    float2 acc = 0.f;
    /* il = 0 -> first half of block (qs[0..7]), il = 8 -> second half (qs[8..15]).
     * qs is uint16_t when reinterpreted, 8 uint16s total. */
    device const uint16_t* qs = ((device const uint16_t*)qb + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

kernel void tc_q4_0_gemv_v2_f16(
    device const half*       X         [[buffer(0)]],   /* [M, K] fp16 */
    device const uchar*      Wq_raw    [[buffer(1)]],   /* [N, K/32] Q4_0 */
    device       half*       Y         [[buffer(2)]],   /* [M, N] fp16 */
    constant uint& M                   [[buffer(3)]],
    constant uint& N                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    const int nb = K / QK4;
    const int r0 = (tgpig.x * NSG_V2 + sgitg) * NR0;   /* first output row this sg owns */
    const int m  = tgpig.y;
    if (m >= (int)M) return;
    if (r0 >= (int)N) return;
    const int active_rows = min(NR0, (int)N - r0);

    device const block_q4_0* Wq = (device const block_q4_0*)Wq_raw;

    /* NR0 row pointers: registers, no tg mem. */
    device const block_q4_0* ax[NR0];
    #pragma clang loop unroll(full)
    for (int row = 0; row < NR0; ++row) {
        const int gr = r0 + row;
        ax[row] = (row < active_rows) ? Wq + (size_t)gr * (size_t)nb : Wq;
    }

    float sumf[NR0] = {0.f, 0.f, 0.f, 0.f};

    /* Each lane covers half a block (16 y elements) at stride NQ=16 blocks. */
    const short ix = tiisg / 2;          /* 0..15: block index within stripe */
    const short il = (tiisg % 2) * 8;    /* 0 or 8: half-block offset */

    float yl[16];
    device const half* yb = X + (size_t)m * K + (size_t)(ix * QK4 + il);

    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0.f, sumy1 = 0.f;

        /* Load 16 y values; pre-scale for the four nibble masks so the
         * dot routine can mask+FMA with no shifts. */
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; i += 2) {
            const float a0 = (float)yb[i + 0];
            const float a1 = (float)yb[i + 1];
            const float b0 = (float)yb[i + 16];
            const float b1 = (float)yb[i + 17];
            sumy0 += a0 + a1;
            sumy1 += b0 + b1;
            yl[i + 0] = a0;
            yl[i + 1] = a1 / 256.f;     /* for mask 0x0F00 */
            yl[i + 8] = b0 / 16.f;      /* for mask 0x00F0 */
            yl[i + 9] = b1 / 4096.f;    /* for mask 0xF000 */
        }
        const float sumy = sumy0 + sumy1;

        if (active_rows == NR0) {
            #pragma clang loop unroll(full)
            for (int row = 0; row < NR0; ++row) {
                sumf[row] += block_q4_0_dot_y(ax[row] + ib, sumy, yl, il);
            }
        } else {
            #pragma clang loop unroll(full)
            for (int row = 0; row < NR0; ++row) {
                if (row < active_rows) {
                    sumf[row] += block_q4_0_dot_y(ax[row] + ib, sumy, yl, il);
                }
            }
        }
        yb += (size_t)QK4 * NQ;
    }

    /* Single simd_sum per row: no threadgroup memory, no barrier. */
    if (active_rows == NR0) {
        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row]);
            if (tiisg == 0) {
                Y[(size_t)m * N + (r0 + row)] = (half)tot;
            }
        }
    } else {
        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row]);
            if (tiisg == 0 && row < active_rows) {
                Y[(size_t)m * N + (r0 + row)] = (half)tot;
            }
        }
    }
}
