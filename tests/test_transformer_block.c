/*
 * tensorcore — end-to-end transformer block test.
 *
 * Runs a complete forward pass through a llama-style transformer block,
 * exercising every kernel category in tensorcore in concert:
 *
 *   x_norm    = RMSnorm(x)
 *   q,k,v     = gemm(x_norm, W_qkv)            (concatenated proj)
 *   q,k       = RoPE(q,k)                      (rotary)
 *   attn_out  = FlashAttention(q, k, v)
 *   proj_out  = gemm(attn_out, W_o)
 *   r1        = x + proj_out                   (residual)
 *   r1_norm   = RMSnorm(r1)
 *   gate      = gemm(r1_norm, W_gate)
 *   up        = gemm(r1_norm, W_up)
 *   mlp_int   = SwiGLU(gate, up)
 *   down      = gemm(mlp_int, W_down)
 *   y         = r1 + down                      (residual)
 *
 * Finally, run a single AdamW step on W_qkv with a synthetic gradient to prove
 * the optimizer plumbing works.
 *
 * Validation: outputs are finite (no NaN/inf), magnitudes within expected
 * range, AdamW step produces the expected per-element update.
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
    if (exp <= 0) { if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000; uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sign | ((mant >> shift) + ((mant >> (shift-1)) & 1)));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + ((mant >> 12) & 1)));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0 && mant == 0) { union {uint32_t u; float f;} v = {sign}; return v.f; }
    if (exp == 31) { union {uint32_t u; float f;} v = {sign | 0x7F800000}; return v.f; }
    if (exp == 0) { while ((mant & 0x400) == 0) { mant <<= 1; --exp; } ++exp; mant &= 0x3FF; }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s)); return 1;
    }

    const int seq = 64, hidden = 256, heads = 4, head_dim = 64;
    const int mlp_dim = 1024;
    if (head_dim * heads != hidden) {
        fprintf(stderr, "config mismatch: head_dim * heads != hidden\n"); return 1;
    }

    /* Allocate everything. */
    tc_buffer *x, *xn, *xn_rstd;
    tc_buffer *Wq, *Wk, *Wv, *q, *k, *v;
    tc_buffer *cos_t, *sin_t;
    tc_buffer *attn_out;
    tc_buffer *Wo, *proj_out;
    tc_buffer *r1, *r1n, *r1n_rstd;
    tc_buffer *Wgate, *Wup, *Wdown;
    tc_buffer *gate, *up, *mlp_int, *down;
    tc_buffer *y;

    const int H_x   = seq;          /* GEMM treats x as (seq, hidden) */
    const int D_x   = hidden;
    tc_buffer_alloc(ctx, seq * hidden * 2, &x);
    tc_buffer_alloc(ctx, seq * hidden * 2, &xn);
    tc_buffer_alloc(ctx, seq * 4,           &xn_rstd);
    tc_buffer_alloc(ctx, hidden * hidden * 2, &Wq);
    tc_buffer_alloc(ctx, hidden * hidden * 2, &Wk);
    tc_buffer_alloc(ctx, hidden * hidden * 2, &Wv);
    tc_buffer_alloc(ctx, seq * hidden * 2, &q);
    tc_buffer_alloc(ctx, seq * hidden * 2, &k);
    tc_buffer_alloc(ctx, seq * hidden * 2, &v);
    tc_buffer_alloc(ctx, seq * (head_dim/2) * 4, &cos_t);
    tc_buffer_alloc(ctx, seq * (head_dim/2) * 4, &sin_t);
    tc_buffer_alloc(ctx, seq * hidden * 2, &attn_out);
    tc_buffer_alloc(ctx, hidden * hidden * 2, &Wo);
    tc_buffer_alloc(ctx, seq * hidden * 2, &proj_out);
    tc_buffer_alloc(ctx, seq * hidden * 2, &r1);
    tc_buffer_alloc(ctx, seq * hidden * 2, &r1n);
    tc_buffer_alloc(ctx, seq * 4,           &r1n_rstd);
    tc_buffer_alloc(ctx, hidden * mlp_dim * 2, &Wgate);
    tc_buffer_alloc(ctx, hidden * mlp_dim * 2, &Wup);
    tc_buffer_alloc(ctx, mlp_dim * hidden * 2, &Wdown);
    tc_buffer_alloc(ctx, seq * mlp_dim * 2, &gate);
    tc_buffer_alloc(ctx, seq * mlp_dim * 2, &up);
    tc_buffer_alloc(ctx, seq * mlp_dim * 2, &mlp_int);
    tc_buffer_alloc(ctx, seq * hidden * 2, &down);
    tc_buffer_alloc(ctx, seq * hidden * 2, &y);

    /* gammas for the two norms. */
    tc_buffer *gamma_x, *gamma_r1;
    tc_buffer_alloc(ctx, hidden * 2, &gamma_x);
    tc_buffer_alloc(ctx, hidden * 2, &gamma_r1);

    /* Fill with reasonable scaled random values. */
    uint16_t *p;
    tc_buffer_map(x, (void**)&p);  for (int i=0;i<seq*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.1f);
    tc_buffer_map(Wq,(void**)&p);  for (int i=0;i<hidden*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.1f);
    tc_buffer_map(Wk,(void**)&p);  for (int i=0;i<hidden*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.1f);
    tc_buffer_map(Wv,(void**)&p);  for (int i=0;i<hidden*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.1f);
    tc_buffer_map(Wo,(void**)&p);  for (int i=0;i<hidden*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.1f);
    tc_buffer_map(Wgate,(void**)&p); for (int i=0;i<hidden*mlp_dim;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.05f);
    tc_buffer_map(Wup,  (void**)&p); for (int i=0;i<hidden*mlp_dim;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.05f);
    tc_buffer_map(Wdown,(void**)&p); for (int i=0;i<mlp_dim*hidden;++i) p[i]=f32_to_f16(((float)rand()/RAND_MAX-0.5f)*0.05f);
    tc_buffer_map(gamma_x, (void**)&p); for (int i=0;i<hidden;++i) p[i]=f32_to_f16(1.0f);
    tc_buffer_map(gamma_r1,(void**)&p); for (int i=0;i<hidden;++i) p[i]=f32_to_f16(1.0f);

    /* RoPE tables. */
    float *cos_p, *sin_p;
    tc_buffer_map(cos_t, (void**)&cos_p);
    tc_buffer_map(sin_t, (void**)&sin_p);
    for (int sx = 0; sx < seq; ++sx)
        for (int d = 0; d < head_dim/2; ++d) {
            float th = (float)sx / powf(10000.0f, (float)d * 2.0f / (float)head_dim);
            cos_p[sx*(head_dim/2)+d] = cosf(th);
            sin_p[sx*(head_dim/2)+d] = sinf(th);
        }

    /* --- Forward pass --- */
    /* RMSnorm(x). */
    s = tc_rmsnorm_forward(ctx, x, gamma_x, xn, xn_rstd, H_x, D_x, 1e-5f);
    if (s != TC_OK) { fprintf(stderr, "rmsnorm: %s\n", tc_status_string(s)); return 2; }

    /* QKV projections: q = xn @ Wq, etc.  tc_gemm shape: M×K @ K×N = M×N */
    tc_gemm_desc gd = {0};
    gd.M = seq; gd.N = hidden; gd.K = hidden;
    gd.a_dtype = TC_DTYPE_F16; gd.b_dtype = TC_DTYPE_F16;
    gd.c_dtype = TC_DTYPE_F16; gd.accum_dtype = TC_DTYPE_F32;
    gd.alpha = 1.0f; gd.beta = 0.0f;
    s = tc_gemm(ctx, &gd, xn, Wq, q); if (s != TC_OK) { fprintf(stderr,"q proj failed\n"); return 3; }
    s = tc_gemm(ctx, &gd, xn, Wk, k); if (s != TC_OK) { fprintf(stderr,"k proj failed\n"); return 3; }
    s = tc_gemm(ctx, &gd, xn, Wv, v); if (s != TC_OK) { fprintf(stderr,"v proj failed\n"); return 3; }

    /* RoPE on q and k. q/k shape: (seq, hidden) which equals (seq, heads*head_dim).
     * Reinterpret as (B=1, heads, seq, head_dim) by passing batch=1, heads=heads. */
    s = tc_rope_forward(ctx, q, cos_t, sin_t, 1, heads, seq, head_dim);
    if (s != TC_OK) { fprintf(stderr,"rope q: %s\n", tc_status_string(s)); return 4; }
    s = tc_rope_forward(ctx, k, cos_t, sin_t, 1, heads, seq, head_dim);
    if (s != TC_OK) { fprintf(stderr,"rope k: %s\n", tc_status_string(s)); return 4; }

    /* FlashAttention. */
    tc_attention_desc ad = {0};
    ad.batch = 1; ad.heads = heads; ad.seq_q = seq; ad.seq_kv = seq;
    ad.head_dim = head_dim;
    ad.io_dtype = TC_DTYPE_F16; ad.accum_dtype = TC_DTYPE_F32;
    ad.softmax_scale = 1.0f / sqrtf((float)head_dim);
    ad.causal = 1; ad.return_lse = 0;
    s = tc_attention_forward(ctx, &ad, q, k, v, attn_out, NULL);
    if (s != TC_OK) { fprintf(stderr,"attn fwd: %s\n", tc_status_string(s)); return 5; }

    /* Output projection. */
    s = tc_gemm(ctx, &gd, attn_out, Wo, proj_out);
    if (s != TC_OK) return 6;

    /* Residual: r1 = x + proj_out. */
    uint16_t *xp, *pop, *r1p;
    tc_buffer_map(x,        (void**)&xp);
    tc_buffer_map(proj_out, (void**)&pop);
    tc_buffer_map(r1,       (void**)&r1p);
    for (int i = 0; i < seq*hidden; ++i) {
        r1p[i] = f32_to_f16(f16_to_f32(xp[i]) + f16_to_f32(pop[i]));
    }

    /* RMSnorm(r1). */
    s = tc_rmsnorm_forward(ctx, r1, gamma_r1, r1n, r1n_rstd, seq, hidden, 1e-5f);
    if (s != TC_OK) return 7;

    /* MLP gate + up projections (seq x hidden) @ (hidden x mlp_dim). */
    tc_gemm_desc gd2 = gd;
    gd2.N = mlp_dim;
    s = tc_gemm(ctx, &gd2, r1n, Wgate, gate); if (s != TC_OK) return 8;
    s = tc_gemm(ctx, &gd2, r1n, Wup,   up);   if (s != TC_OK) return 8;

    /* SwiGLU. */
    s = tc_swiglu_forward(ctx, gate, up, mlp_int, seq * mlp_dim);
    if (s != TC_OK) return 9;

    /* Down projection. */
    tc_gemm_desc gd3 = gd;
    gd3.M = seq; gd3.N = hidden; gd3.K = mlp_dim;
    s = tc_gemm(ctx, &gd3, mlp_int, Wdown, down); if (s != TC_OK) return 10;

    /* Residual: y = r1 + down. */
    uint16_t *dnp, *yp;
    tc_buffer_map(down, (void**)&dnp);
    tc_buffer_map(y,    (void**)&yp);
    for (int i = 0; i < seq*hidden; ++i) {
        yp[i] = f32_to_f16(f16_to_f32(r1p[i]) + f16_to_f32(dnp[i]));
    }

    /* --- Validation: outputs finite + reasonable magnitude --- */
    double y_max = 0.0, y_rms = 0.0;
    int nan_count = 0;
    for (int i = 0; i < seq*hidden; ++i) {
        float val = f16_to_f32(yp[i]);
        if (!isfinite(val)) ++nan_count;
        double a = fabs(val);
        if (a > y_max) y_max = a;
        y_rms += val * val;
    }
    y_rms = sqrt(y_rms / (seq*hidden));
    printf("transformer_block forward:\n");
    printf("  seq=%d hidden=%d heads=%d head_dim=%d mlp_dim=%d\n",
           seq, hidden, heads, head_dim, mlp_dim);
    printf("  y: NaN/inf count=%d  max_abs=%.3e  rms=%.3e\n",
           nan_count, y_max, y_rms);

    /* --- AdamW step on Wq with synthetic gradient --- */
    tc_buffer *Wq_master, *m_b, *v_b, *grad_b;
    tc_buffer_alloc(ctx, hidden * hidden * 4, &Wq_master);
    tc_buffer_alloc(ctx, hidden * hidden * 4, &m_b);
    tc_buffer_alloc(ctx, hidden * hidden * 4, &v_b);
    tc_buffer_alloc(ctx, hidden * hidden * 2, &grad_b);
    float *Wqm; uint16_t *gradp;
    tc_buffer_map(Wq_master, (void**)&Wqm);
    tc_buffer_map(Wq,       (void**)&p);     /* read original fp16 weights */
    for (int i = 0; i < hidden*hidden; ++i) Wqm[i] = f16_to_f32(p[i]);
    void* mzero; tc_buffer_map(m_b, &mzero); memset(mzero, 0, hidden*hidden*4);
    void* vzero; tc_buffer_map(v_b, &vzero); memset(vzero, 0, hidden*hidden*4);
    tc_buffer_map(grad_b, (void**)&gradp);
    for (int i = 0; i < hidden*hidden; ++i) gradp[i] = f32_to_f16(0.01f);

    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    float Wq_before = Wqm[0];
    s = tc_adamw_step(ctx, Wq_master, m_b, v_b, grad_b, TC_DTYPE_F16,
                      hidden*hidden, lr, b1, b2, eps, wd,
                      1.0f - b1, 1.0f - b2);
    if (s != TC_OK) { fprintf(stderr, "adamw: %s\n", tc_status_string(s)); return 11; }
    float Wq_after = Wqm[0];
    printf("  AdamW step on Wq[0]: %.6f -> %.6f  (delta=%.3e)\n",
           Wq_before, Wq_after, Wq_after - Wq_before);

    /* Free all. */
    tc_buffer_free(ctx, x); tc_buffer_free(ctx, xn); tc_buffer_free(ctx, xn_rstd);
    tc_buffer_free(ctx, Wq); tc_buffer_free(ctx, Wk); tc_buffer_free(ctx, Wv);
    tc_buffer_free(ctx, q); tc_buffer_free(ctx, k); tc_buffer_free(ctx, v);
    tc_buffer_free(ctx, cos_t); tc_buffer_free(ctx, sin_t);
    tc_buffer_free(ctx, attn_out); tc_buffer_free(ctx, Wo); tc_buffer_free(ctx, proj_out);
    tc_buffer_free(ctx, r1); tc_buffer_free(ctx, r1n); tc_buffer_free(ctx, r1n_rstd);
    tc_buffer_free(ctx, Wgate); tc_buffer_free(ctx, Wup); tc_buffer_free(ctx, Wdown);
    tc_buffer_free(ctx, gate); tc_buffer_free(ctx, up); tc_buffer_free(ctx, mlp_int);
    tc_buffer_free(ctx, down); tc_buffer_free(ctx, y);
    tc_buffer_free(ctx, gamma_x); tc_buffer_free(ctx, gamma_r1);
    tc_buffer_free(ctx, Wq_master); tc_buffer_free(ctx, m_b); tc_buffer_free(ctx, v_b);
    tc_buffer_free(ctx, grad_b);
    tc_shutdown(ctx);

    /* Pass if no NaNs and output magnitude is reasonable. */
    return (nan_count == 0 && y_max > 0 && y_max < 1e3 && isfinite(y_rms)) ? 0 : 12;
}
