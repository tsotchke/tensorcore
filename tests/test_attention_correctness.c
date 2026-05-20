/*
 * Correctness test: tc_attention_forward (fp16, head_dim=64) vs naive CPU
 * reference computed in fp64.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round = (mant >> (shift - 1)) & 1;
        return (uint16_t)(sign | ((mant >> shift) + round));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    }
    uint32_t round = (mant >> 12) & 1;
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + round));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            ++exp; mant &= 0x3FF;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { out };
    return v.f;
}

static void ref_attention(int B, int H, int Sq, int Sk, int D, int causal,
                          float scale,
                          const float* Q, const float* K, const float* V,
                          float* O) {
    /* Naive O(B*H*Sq*Sk*D) reference in fp64. */
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int q = 0; q < Sq; ++q) {
                double* s = (double*)calloc(Sk, sizeof(double));
                double m = -INFINITY;
                for (int k = 0; k < Sk; ++k) {
                    if (causal && k > q) { s[k] = -INFINITY; continue; }
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) {
                        const float qv = Q[((b*H + h)*Sq + q)*D + d];
                        const float kv = K[((b*H + h)*Sk + k)*D + d];
                        dot += (double)qv * (double)kv;
                    }
                    s[k] = dot * scale;
                    if (s[k] > m) m = s[k];
                }
                double l = 0.0;
                for (int k = 0; k < Sk; ++k) {
                    s[k] = (s[k] > -1e30) ? exp(s[k] - m) : 0.0;
                    l += s[k];
                }
                for (int d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (int k = 0; k < Sk; ++k) {
                        const float vv = V[((b*H + h)*Sk + k)*D + d];
                        acc += s[k] * (double)vv;
                    }
                    O[((b*H + h)*Sq + q)*D + d] = (float)(acc / (l + 1e-30));
                }
                free(s);
            }
        }
    }
}

static int run_case(tc_context* ctx, int B, int H, int Sq, int Sk, int D);

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    int rc = 0;
    /* D=64 (GPT-2 style) */
    rc |= run_case(ctx, 1, 2,  64,  64,  64);
    rc |= run_case(ctx, 1, 2, 128, 128,  64);
    rc |= run_case(ctx, 1, 4, 256, 256,  64);
    /* D=128 (llama / mistral standard) */
    rc |= run_case(ctx, 1, 2,  64,  64, 128);
    rc |= run_case(ctx, 1, 2, 128, 128, 128);
    rc |= run_case(ctx, 1, 4, 256, 256, 128);

    tc_shutdown(ctx);
    return rc;
}

static int run_case(tc_context* ctx, int B, int H, int Sq, int Sk, int D) {
    const float scale = 1.0f / sqrtf((float)D);
    const int causal = 1;

    const size_t qkv_elems = (size_t)B * H * Sq * D;
    const size_t kv_elems  = (size_t)B * H * Sk * D;

    tc_buffer *Q = NULL, *K = NULL, *V = NULL, *O = NULL;
    tc_buffer_alloc(ctx, qkv_elems * sizeof(uint16_t), &Q);
    tc_buffer_alloc(ctx, kv_elems  * sizeof(uint16_t), &K);
    tc_buffer_alloc(ctx, kv_elems  * sizeof(uint16_t), &V);
    tc_buffer_alloc(ctx, qkv_elems * sizeof(uint16_t), &O);

    uint16_t *Qp, *Kp, *Vp, *Op;
    tc_buffer_map(Q, (void**)&Qp);
    tc_buffer_map(K, (void**)&Kp);
    tc_buffer_map(V, (void**)&Vp);
    tc_buffer_map(O, (void**)&Op);

    float* Qf = malloc(qkv_elems * sizeof(float));
    float* Kf = malloc(kv_elems  * sizeof(float));
    float* Vf = malloc(kv_elems  * sizeof(float));
    float* Or = malloc(qkv_elems * sizeof(float));

    srand(0xA77E);
    for (size_t i = 0; i < qkv_elems; ++i) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        Qf[i] = v; Qp[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < kv_elems; ++i) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        Kf[i] = v; Kp[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < kv_elems; ++i) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        Vf[i] = v; Vp[i] = f32_to_f16(v);
    }
    memset(Op, 0, qkv_elems * sizeof(uint16_t));

    ref_attention(B, H, Sq, Sk, D, causal, scale, Qf, Kf, Vf, Or);

    tc_attention_desc d = {0};
    d.batch = B; d.heads = H; d.seq_q = Sq; d.seq_kv = Sk; d.head_dim = D;
    d.io_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.softmax_scale = scale; d.causal = causal; d.return_lse = 0;
    tc_status_t s = tc_attention_forward(ctx, &d, Q, K, V, O, NULL);

    double max_abs = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    if (s == TC_OK) {
        for (size_t i = 0; i < qkv_elems; ++i) {
            const float got = f16_to_f32(Op[i]);
            const float ref = Or[i];
            const double e = fabs((double)got - (double)ref);
            if (e > max_abs) max_abs = e;
            sum_sq_err += e * e;
            sum_sq_ref += (double)ref * (double)ref;
        }
    }
    const double rms_err = sqrt(sum_sq_err / qkv_elems);
    const double rms_ref = sqrt(sum_sq_ref / qkv_elems);
    const double scaled  = rms_err / (rms_ref + 1e-9);

    printf("flash_attention B=%d H=%d Sq=%d Sk=%d D=%d causal=%d   backend=%-18s  "
           "max_abs=%.3e rms_err=%.3e rms_ref=%.3e scaled=%.3e  %s\n",
           B, H, Sq, Sk, D, causal,
           tc_backend_name(tc_last_backend()),
           max_abs, rms_err, rms_ref, scaled,
           (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Qf); free(Kf); free(Vf); free(Or);
    tc_buffer_free(ctx, Q); tc_buffer_free(ctx, K);
    tc_buffer_free(ctx, V); tc_buffer_free(ctx, O);

    /* fp16 attention with online softmax: ~5e-3 RMS-scaled is typical for D=64.
     * D=128 with Br=Bc=16 has more rounding per query block (×4 more updates
     * across the KV sequence), so we allow up to 2e-2. Phase-2 will widen Br
     * for D=128 on Apple9+ where TG memory is larger. */
    if (s != TC_OK) return (int)-s;
    return (scaled < 2e-2) ? 0 : 9;
}
