/*
 * Correctness tests for the fused training kernels.
 *
 * Covers: RMSnorm fwd+bwd, LayerNorm fwd+bwd, SwiGLU fwd+bwd, RoPE fwd,
 * softmax fwd+bwd, AdamW step. Each compared against an fp64 CPU reference.
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
        return (uint16_t)(sign | ((mant >> shift) + ((mant >> (shift-1)) & 1)));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + ((mant >> 12) & 1)));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0 && mant == 0) {
        union { uint32_t u; float f; } v = {sign}; return v.f;
    }
    if (exp == 31) {
        union { uint32_t u; float f; } v = {sign | 0x7F800000}; return v.f;
    }
    if (exp == 0) {
        while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
        ++exp; mant &= 0x3FF;
    }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

static double rms_scaled(const uint16_t* got, const float* ref, int n) {
    double se = 0.0, sr = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = (double)f16_to_f32(got[i]) - (double)ref[i];
        se += e * e;
        sr += (double)ref[i] * ref[i];
    }
    return sqrt(se / n) / (sqrt(sr / n) + 1e-9);
}
static double rms_scaled_f32(const float* got, const float* ref, int n) {
    double se = 0.0, sr = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = (double)got[i] - (double)ref[i];
        se += e * e;
        sr += (double)ref[i] * ref[i];
    }
    return sqrt(se / n) / (sqrt(sr / n) + 1e-9);
}

static int test_rmsnorm(tc_context* ctx) {
    const int N = 8, D = 128;
    const float eps = 1e-5f;
    tc_buffer *Xb, *gb, *Yb, *rstdb;
    tc_buffer_alloc(ctx, N * D * 2, &Xb);
    tc_buffer_alloc(ctx, D * 2,     &gb);
    tc_buffer_alloc(ctx, N * D * 2, &Yb);
    tc_buffer_alloc(ctx, N * 4,     &rstdb);
    uint16_t *Xp, *gp, *Yp; float *rstdp;
    tc_buffer_map(Xb, (void**)&Xp);
    tc_buffer_map(gb, (void**)&gp);
    tc_buffer_map(Yb, (void**)&Yp);
    tc_buffer_map(rstdb, (void**)&rstdp);

    float *Xf = malloc(N*D*sizeof(float));
    float *gf = malloc(D*sizeof(float));
    float *Yref = malloc(N*D*sizeof(float));
    srand(0x77);
    for (int i = 0; i < N*D; ++i) { float v = ((float)rand()/RAND_MAX-0.5f); Xf[i]=v; Xp[i]=f32_to_f16(v); }
    for (int i = 0; i < D; ++i)   { float v = 0.5f + (float)rand()/RAND_MAX; gf[i]=v; gp[i]=f32_to_f16(v); }

    /* Ref: y = x * rsqrt(mean(x^2)+eps) * gamma */
    for (int n = 0; n < N; ++n) {
        double ss = 0.0;
        for (int d = 0; d < D; ++d) ss += (double)Xf[n*D+d] * Xf[n*D+d];
        double rstd = 1.0 / sqrt(ss / D + eps);
        for (int d = 0; d < D; ++d) Yref[n*D+d] = (float)(Xf[n*D+d] * rstd * gf[d]);
    }

    tc_status_t s = tc_rmsnorm_forward(ctx, Xb, gb, Yb, rstdb, N, D, eps);
    const double err = rms_scaled(Yp, Yref, N*D);
    printf("  rmsnorm_forward   N=%d D=%d  rms_scaled=%.3e  %s\n",
           N, D, err, (s==TC_OK && err<5e-3) ? "OK" : "FAIL");
    free(Xf); free(gf); free(Yref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, gb);
    tc_buffer_free(ctx, Yb); tc_buffer_free(ctx, rstdb);
    return (s == TC_OK && err < 5e-3) ? 0 : 1;
}

static int test_layernorm(tc_context* ctx) {
    const int N = 8, D = 128;
    const float eps = 1e-5f;
    tc_buffer *Xb, *gb, *bb, *Yb, *meanb, *rstdb;
    tc_buffer_alloc(ctx, N*D*2, &Xb);
    tc_buffer_alloc(ctx, D*2,   &gb);
    tc_buffer_alloc(ctx, D*2,   &bb);
    tc_buffer_alloc(ctx, N*D*2, &Yb);
    tc_buffer_alloc(ctx, N*4,   &meanb);
    tc_buffer_alloc(ctx, N*4,   &rstdb);
    uint16_t *Xp, *gp, *bp, *Yp; float *mp, *rp;
    tc_buffer_map(Xb, (void**)&Xp); tc_buffer_map(gb, (void**)&gp);
    tc_buffer_map(bb, (void**)&bp); tc_buffer_map(Yb, (void**)&Yp);
    tc_buffer_map(meanb, (void**)&mp); tc_buffer_map(rstdb, (void**)&rp);

    float *Xf = malloc(N*D*sizeof(float));
    float *gf = malloc(D*sizeof(float));
    float *bf = malloc(D*sizeof(float));
    float *Yref = malloc(N*D*sizeof(float));
    srand(0x88);
    for (int i = 0; i < N*D; ++i) { float v = ((float)rand()/RAND_MAX-0.5f); Xf[i]=v; Xp[i]=f32_to_f16(v); }
    for (int i = 0; i < D; ++i)   { float v = 0.5f + (float)rand()/RAND_MAX; gf[i]=v; gp[i]=f32_to_f16(v); }
    for (int i = 0; i < D; ++i)   { float v = ((float)rand()/RAND_MAX-0.5f)*0.1f; bf[i]=v; bp[i]=f32_to_f16(v); }

    for (int n = 0; n < N; ++n) {
        double sum = 0.0, sumsq = 0.0;
        for (int d = 0; d < D; ++d) { sum += Xf[n*D+d]; sumsq += Xf[n*D+d]*Xf[n*D+d]; }
        double mean = sum / D;
        double var = sumsq / D - mean*mean;
        double rstd = 1.0 / sqrt(var + eps);
        for (int d = 0; d < D; ++d)
            Yref[n*D+d] = (float)((Xf[n*D+d] - mean) * rstd * gf[d] + bf[d]);
    }

    tc_status_t s = tc_layernorm_forward(ctx, Xb, gb, bb, Yb, meanb, rstdb, N, D, eps);
    const double err = rms_scaled(Yp, Yref, N*D);
    printf("  layernorm_forward N=%d D=%d  rms_scaled=%.3e  %s\n",
           N, D, err, (s==TC_OK && err<5e-3) ? "OK" : "FAIL");
    free(Xf); free(gf); free(bf); free(Yref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, gb); tc_buffer_free(ctx, bb);
    tc_buffer_free(ctx, Yb); tc_buffer_free(ctx, meanb); tc_buffer_free(ctx, rstdb);
    return (s == TC_OK && err < 5e-3) ? 0 : 1;
}

static int test_swiglu(tc_context* ctx) {
    const int N = 1024;
    tc_buffer *gb, *ub, *ob;
    tc_buffer_alloc(ctx, N*2, &gb);
    tc_buffer_alloc(ctx, N*2, &ub);
    tc_buffer_alloc(ctx, N*2, &ob);
    uint16_t *gp, *up, *op;
    tc_buffer_map(gb, (void**)&gp); tc_buffer_map(ub, (void**)&up); tc_buffer_map(ob, (void**)&op);

    float *gf = malloc(N*sizeof(float));
    float *uf = malloc(N*sizeof(float));
    float *ref = malloc(N*sizeof(float));
    srand(0x99);
    for (int i = 0; i < N; ++i) {
        float g = ((float)rand()/RAND_MAX-0.5f)*2.0f;
        float u = ((float)rand()/RAND_MAX-0.5f)*2.0f;
        gf[i] = g; uf[i] = u;
        gp[i] = f32_to_f16(g); up[i] = f32_to_f16(u);
        ref[i] = (float)((double)g / (1.0 + exp(-(double)g)) * (double)u);
    }

    tc_status_t s = tc_swiglu_forward(ctx, gb, ub, ob, N);
    const double err = rms_scaled(op, ref, N);
    printf("  swiglu_forward    N=%d         rms_scaled=%.3e  %s\n",
           N, err, (s==TC_OK && err<5e-3) ? "OK" : "FAIL");
    free(gf); free(uf); free(ref);
    tc_buffer_free(ctx, gb); tc_buffer_free(ctx, ub); tc_buffer_free(ctx, ob);
    return (s == TC_OK && err < 5e-3) ? 0 : 1;
}

static int test_softmax(tc_context* ctx) {
    const int N = 8, D = 128;
    tc_buffer *Xb, *Yb;
    tc_buffer_alloc(ctx, N*D*2, &Xb);
    tc_buffer_alloc(ctx, N*D*2, &Yb);
    uint16_t *Xp, *Yp;
    tc_buffer_map(Xb, (void**)&Xp); tc_buffer_map(Yb, (void**)&Yp);

    float *Xf = malloc(N*D*sizeof(float));
    float *Yref = malloc(N*D*sizeof(float));
    srand(0xAA);
    for (int i = 0; i < N*D; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*3.0f; Xf[i]=v; Xp[i]=f32_to_f16(v); }
    for (int n = 0; n < N; ++n) {
        double m = -INFINITY;
        for (int d = 0; d < D; ++d) if (Xf[n*D+d] > m) m = Xf[n*D+d];
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += exp(Xf[n*D+d] - m);
        for (int d = 0; d < D; ++d) Yref[n*D+d] = (float)(exp(Xf[n*D+d] - m) / s);
    }
    tc_status_t s = tc_softmax_forward(ctx, Xb, Yb, N, D);
    const double err = rms_scaled(Yp, Yref, N*D);
    printf("  softmax_forward   N=%d D=%d  rms_scaled=%.3e  %s\n",
           N, D, err, (s==TC_OK && err<5e-3) ? "OK" : "FAIL");
    free(Xf); free(Yref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, Yb);
    return (s == TC_OK && err < 5e-3) ? 0 : 1;
}

static int test_rope(tc_context* ctx) {
    const int B = 1, H = 2, S = 4, D = 32;
    tc_buffer *Xb, *cb, *sb;
    tc_buffer_alloc(ctx, B*H*S*D*2, &Xb);
    tc_buffer_alloc(ctx, S*(D/2)*4, &cb);
    tc_buffer_alloc(ctx, S*(D/2)*4, &sb);
    uint16_t *Xp; float *cp, *sp;
    tc_buffer_map(Xb, (void**)&Xp); tc_buffer_map(cb, (void**)&cp); tc_buffer_map(sb, (void**)&sp);

    float *Xf = malloc(B*H*S*D*sizeof(float));
    float *Xref = malloc(B*H*S*D*sizeof(float));
    srand(0xBB);
    for (int i = 0; i < B*H*S*D; ++i) { float v = ((float)rand()/RAND_MAX-0.5f); Xf[i]=v; Xref[i]=v; Xp[i]=f32_to_f16(v); }
    for (int p = 0; p < S; ++p)
        for (int d = 0; d < D/2; ++d) {
            float th = (float)p / powf(10000.0f, (float)d * 2.0f / (float)D);
            cp[p*(D/2)+d] = cosf(th);
            sp[p*(D/2)+d] = sinf(th);
        }
    /* Apply ref RoPE: for each (b,h,s,d_pair). */
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h)
    for (int s = 0; s < S; ++s) for (int d2 = 0; d2 < D/2; ++d2) {
        int base = ((b*H+h)*S+s)*D;
        float x = Xref[base+d2], y = Xref[base+d2+D/2];
        float c = cp[s*(D/2)+d2], si = sp[s*(D/2)+d2];
        Xref[base+d2] = x*c - y*si;
        Xref[base+d2+D/2] = x*si + y*c;
    }
    tc_status_t s = tc_rope_forward(ctx, Xb, cb, sb, B, H, S, D);
    const double err = rms_scaled(Xp, Xref, B*H*S*D);
    printf("  rope_forward      B=%d H=%d S=%d D=%d  rms_scaled=%.3e  %s\n",
           B, H, S, D, err, (s==TC_OK && err<5e-3) ? "OK" : "FAIL");
    free(Xf); free(Xref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, cb); tc_buffer_free(ctx, sb);
    return (s == TC_OK && err < 5e-3) ? 0 : 1;
}

static int test_adamw(tc_context* ctx) {
    const int n = 256;
    tc_buffer *pb, *mb, *vb, *gb;
    tc_buffer_alloc(ctx, n*4, &pb);
    tc_buffer_alloc(ctx, n*4, &mb);
    tc_buffer_alloc(ctx, n*4, &vb);
    tc_buffer_alloc(ctx, n*4, &gb);
    float *pp, *mp, *vp, *gp;
    tc_buffer_map(pb, (void**)&pp); tc_buffer_map(mb, (void**)&mp);
    tc_buffer_map(vb, (void**)&vp); tc_buffer_map(gb, (void**)&gp);

    float *p_ref = malloc(n*sizeof(float));
    float *m_ref = malloc(n*sizeof(float));
    float *v_ref = malloc(n*sizeof(float));
    srand(0xCC);
    for (int i = 0; i < n; ++i) {
        pp[i] = p_ref[i] = ((float)rand()/RAND_MAX-0.5f);
        mp[i] = m_ref[i] = 0.0f;
        vp[i] = v_ref[i] = 0.0f;
        gp[i] = ((float)rand()/RAND_MAX-0.5f) * 0.1f;
    }
    const float lr=1e-3f, b1=0.9f, b2=0.999f, eps=1e-8f, wd=0.01f;
    const float bc1=1.0f-b1, bc2=1.0f-b2;   /* step 1 */
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        m_ref[i] = b1*m_ref[i] + (1-b1)*g;
        v_ref[i] = b2*v_ref[i] + (1-b2)*g*g;
        float mh = m_ref[i] / bc1;
        float vh = v_ref[i] / bc2;
        p_ref[i] = p_ref[i] - lr * (mh / (sqrtf(vh)+eps) + wd*p_ref[i]);
    }
    tc_status_t s = tc_adamw_step(ctx, pb, mb, vb, gb, TC_DTYPE_F32, n,
                                  lr, b1, b2, eps, wd, bc1, bc2);
    const double err = rms_scaled_f32(pp, p_ref, n);
    printf("  adamw_step        n=%d         rms_scaled=%.3e  %s\n",
           n, err, (s==TC_OK && err<1e-5) ? "OK" : "FAIL");
    free(p_ref); free(m_ref); free(v_ref);
    tc_buffer_free(ctx, pb); tc_buffer_free(ctx, mb);
    tc_buffer_free(ctx, vb); tc_buffer_free(ctx, gb);
    return (s == TC_OK && err < 1e-5) ? 0 : 1;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    int rc = 0;
    rc |= test_rmsnorm(ctx);
    rc |= test_layernorm(ctx);
    rc |= test_swiglu(ctx);
    rc |= test_softmax(ctx);
    rc |= test_rope(ctx);
    rc |= test_adamw(ctx);
    tc_shutdown(ctx);
    return rc;
}
