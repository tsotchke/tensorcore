/*
 * Correctness test: tc_attention_backward vs fp64 reference.
 *
 * Reference computes the analytic gradient in fp64 on CPU. fp16 GPU output
 * is compared via RMS-scaled metric (per-cell relative error is misleading
 * for near-zero gradient values).
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

/* fp64 reference: forward + backward attention for one (batch, head). */
static void ref_attention_fwd_bwd(int Sq, int Sk, int D, int causal,
                                  double scale,
                                  const float* Q, const float* K,
                                  const float* V, const float* dO,
                                  float* O,   /* output     */
                                  float* LSE, /* fp32 LSE  */
                                  float* dQ, float* dK, float* dV) {
    double* S = (double*)malloc(sizeof(double) * Sq * Sk);
    double* P = (double*)malloc(sizeof(double) * Sq * Sk);
    double* dP = (double*)malloc(sizeof(double) * Sq * Sk);
    double* dS = (double*)malloc(sizeof(double) * Sq * Sk);
    double* D_i = (double*)malloc(sizeof(double) * Sq);

    for (int q = 0; q < Sq; ++q) {
        for (int k = 0; k < Sk; ++k) {
            double s = 0.0;
            for (int d = 0; d < D; ++d) s += (double)Q[q*D+d] * (double)K[k*D+d];
            s *= scale;
            if (causal && k > q) s = -INFINITY;
            S[q*Sk + k] = s;
        }
        double m = -INFINITY;
        for (int k = 0; k < Sk; ++k) if (S[q*Sk+k] > m) m = S[q*Sk+k];
        double l = 0.0;
        for (int k = 0; k < Sk; ++k) {
            double p = (S[q*Sk+k] > -1e30) ? exp(S[q*Sk+k] - m) : 0.0;
            P[q*Sk + k] = p;
            l += p;
        }
        for (int k = 0; k < Sk; ++k) P[q*Sk+k] /= (l + 1e-30);
        LSE[q] = (float)(m + log(l + 1e-30));
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int k = 0; k < Sk; ++k) acc += P[q*Sk+k] * (double)V[k*D+d];
            O[q*D + d] = (float)acc;
        }
    }
    /* D_i = rowsum(dO * O). */
    for (int q = 0; q < Sq; ++q) {
        double d_i = 0.0;
        for (int d = 0; d < D; ++d) d_i += (double)dO[q*D+d] * (double)O[q*D+d];
        D_i[q] = d_i;
    }
    /* dV = P^T @ dO. */
    for (int k = 0; k < Sk; ++k) {
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int q = 0; q < Sq; ++q) acc += P[q*Sk+k] * (double)dO[q*D+d];
            dV[k*D + d] = (float)acc;
        }
    }
    /* dP = dO @ V^T. */
    for (int q = 0; q < Sq; ++q) {
        for (int k = 0; k < Sk; ++k) {
            double acc = 0.0;
            for (int d = 0; d < D; ++d) acc += (double)dO[q*D+d] * (double)V[k*D+d];
            dP[q*Sk + k] = acc;
        }
    }
    /* dS = P * (dP - D_i). */
    for (int q = 0; q < Sq; ++q)
        for (int k = 0; k < Sk; ++k)
            dS[q*Sk + k] = P[q*Sk+k] * (dP[q*Sk+k] - D_i[q]);
    /* dQ = dS @ K * scale. */
    for (int q = 0; q < Sq; ++q) {
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int k = 0; k < Sk; ++k) acc += dS[q*Sk+k] * (double)K[k*D+d];
            dQ[q*D + d] = (float)(acc * scale);
        }
    }
    /* dK = dS^T @ Q * scale. */
    for (int k = 0; k < Sk; ++k) {
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int q = 0; q < Sq; ++q) acc += dS[q*Sk+k] * (double)Q[q*D+d];
            dK[k*D + d] = (float)(acc * scale);
        }
    }
    free(S); free(P); free(dP); free(dS); free(D_i);
}

static double rms_scaled(const uint16_t* got, const float* ref, int n) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = (double)f16_to_f32(got[i]) - (double)ref[i];
        sum_sq_err += e * e;
        sum_sq_ref += (double)ref[i] * (double)ref[i];
    }
    return sqrt(sum_sq_err / n) / (sqrt(sum_sq_ref / n) + 1e-9);
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    const int B = 1, H = 1, Sq = 64, Sk = 64, D = 64;
    const float scale = 1.0f / sqrtf((float)D);
    const int causal = 1;
    const size_t qkv = (size_t)B * H * Sq * D;

    tc_buffer *Q, *K, *V, *O, *dO, *LSE, *dQ, *dK, *dV;
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &Q);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &K);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &V);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &O);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &dO);
    tc_buffer_alloc(ctx, B * H * Sq * sizeof(float), &LSE);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &dQ);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &dK);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &dV);

    uint16_t *Qp, *Kp, *Vp, *Op, *dOp, *dQp, *dKp, *dVp;
    float *LSEp;
    tc_buffer_map(Q, (void**)&Qp);
    tc_buffer_map(K, (void**)&Kp);
    tc_buffer_map(V, (void**)&Vp);
    tc_buffer_map(O, (void**)&Op);
    tc_buffer_map(dO, (void**)&dOp);
    tc_buffer_map(LSE, (void**)&LSEp);
    tc_buffer_map(dQ, (void**)&dQp);
    tc_buffer_map(dK, (void**)&dKp);
    tc_buffer_map(dV, (void**)&dVp);

    float *Qf = malloc(qkv*sizeof(float));
    float *Kf = malloc(qkv*sizeof(float));
    float *Vf = malloc(qkv*sizeof(float));
    float *dOf = malloc(qkv*sizeof(float));
    float *Or = malloc(qkv*sizeof(float));
    float *LSEr = malloc(Sq*sizeof(float));
    float *dQr = malloc(qkv*sizeof(float));
    float *dKr = malloc(qkv*sizeof(float));
    float *dVr = malloc(qkv*sizeof(float));

    srand(0xBACC);
    for (size_t i = 0; i < qkv; ++i) {
        float v = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        Qf[i] = v; Qp[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < qkv; ++i) {
        float v = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        Kf[i] = v; Kp[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < qkv; ++i) {
        float v = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        Vf[i] = v; Vp[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < qkv; ++i) {
        float v = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        dOf[i] = v; dOp[i] = f32_to_f16(v);
    }

    /* fp64 reference forward+backward */
    ref_attention_fwd_bwd(Sq, Sk, D, causal, (double)scale,
                          Qf, Kf, Vf, dOf, Or, LSEr, dQr, dKr, dVr);

    /* Forward (writes O, LSE). */
    tc_attention_desc fd = {0};
    fd.batch = B; fd.heads = H; fd.seq_q = Sq; fd.seq_kv = Sk; fd.head_dim = D;
    fd.io_dtype = TC_DTYPE_F16; fd.accum_dtype = TC_DTYPE_F32;
    fd.softmax_scale = scale; fd.causal = causal; fd.return_lse = 1;
    s = tc_attention_forward(ctx, &fd, Q, K, V, O, LSE);
    if (s != TC_OK) {
        fprintf(stderr, "forward failed: %s\n", tc_status_string(s));
        return 2;
    }
    /* Use fp64 LSE for backward (eliminates fwd numerical noise from bw test). */
    memcpy(LSEp, LSEr, Sq * sizeof(float));
    /* Also use fp64 O — same reason. Convert to fp16. */
    for (size_t i = 0; i < qkv; ++i) Op[i] = f32_to_f16(Or[i]);

    /* Backward. */
    tc_attention_desc bd = fd;
    s = tc_attention_backward(ctx, &bd, Q, K, V, O, dO, LSE, dQ, dK, dV);
    if (s != TC_OK) {
        fprintf(stderr, "backward failed: %s\n", tc_status_string(s));
        return 3;
    }

    const double dq_err = rms_scaled(dQp, dQr, qkv);
    const double dk_err = rms_scaled(dKp, dKr, qkv);
    const double dv_err = rms_scaled(dVp, dVr, qkv);

    printf("attention_backward Sq=%d Sk=%d D=%d causal=%d   "
           "dQ_rms_scaled=%.3e  dK_rms_scaled=%.3e  dV_rms_scaled=%.3e\n",
           Sq, Sk, D, causal, dq_err, dk_err, dv_err);

    free(Qf); free(Kf); free(Vf); free(dOf);
    free(Or); free(LSEr); free(dQr); free(dKr); free(dVr);
    tc_buffer_free(ctx, Q); tc_buffer_free(ctx, K); tc_buffer_free(ctx, V);
    tc_buffer_free(ctx, O); tc_buffer_free(ctx, dO); tc_buffer_free(ctx, LSE);
    tc_buffer_free(ctx, dQ); tc_buffer_free(ctx, dK); tc_buffer_free(ctx, dV);
    tc_shutdown(ctx);

    /* fp16 backward typically has 1-2% RMS error vs fp64 reference. */
    return (dq_err < 5e-2 && dk_err < 5e-2 && dv_err < 5e-2) ? 0 : 9;
}
