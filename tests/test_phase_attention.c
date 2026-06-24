/*
 * tests/test_phase_attention.c — validates phase-attention combine and
 * Born-rule output C primitives against analytic checks:
 *
 *   phase-attention:
 *     - With γ=0, λ=0, w=1, distances=0, identity test:
 *       score(p) = Σ_m ip[p,m] · cos(0+0) · exp(0) = Σ_m ip[p,m]
 *     - With γ=π/2 the cos term flips to sin, easy to check by hand.
 *
 *   Born-rule:
 *     - Rows of the output sum to 1 (proper distribution).
 *     - Single-amplitude case: P = h² / Σh² (skip s, e).
 *     - All-zero amplitudes: output all zeros (Z floor avoids /0).
 *     - Three-amplitude case bit-correct against the formula.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/phase_attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_(const char* what) { fprintf(stderr, "FAIL: %s\n", what); return 1; }
static float maxabs(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return fail_("tc_init");

    const int N_pairs = 8;
    const int M = 3;   /* 3 manifolds (H, S, R) */

    tc_buffer *bIP, *bPD, *bDD, *bW, *bG, *bL, *bScores;
    tc_buffer_alloc(ctx, (size_t)N_pairs * M * sizeof(float), &bIP);
    tc_buffer_alloc(ctx, (size_t)N_pairs * M * sizeof(float), &bPD);
    tc_buffer_alloc(ctx, (size_t)N_pairs * M * sizeof(float), &bDD);
    tc_buffer_alloc(ctx, (size_t)M * sizeof(float), &bW);
    tc_buffer_alloc(ctx, (size_t)M * sizeof(float), &bG);
    tc_buffer_alloc(ctx, (size_t)M * sizeof(float), &bL);
    tc_buffer_alloc(ctx, (size_t)N_pairs * sizeof(float), &bScores);

    void *Ip, *Pp, *Dp, *Wp, *Gp, *Lp, *Sp;
    tc_buffer_map(bIP, &Ip); tc_buffer_map(bPD, &Pp); tc_buffer_map(bDD, &Dp);
    tc_buffer_map(bW, &Wp); tc_buffer_map(bG, &Gp); tc_buffer_map(bL, &Lp);
    tc_buffer_map(bScores, &Sp);

    /* Fill ip with arbitrary, pd with 0, dd with 0, w=1, g=0, l=0 → score = Σ_m ip */
    float* ip = (float*)Ip; float* pd = (float*)Pp; float* dd = (float*)Dp;
    float* w = (float*)Wp; float* g = (float*)Gp; float* l = (float*)Lp;
    float* scores = (float*)Sp;
    srand(0x42);
    for (int i = 0; i < N_pairs * M; ++i) ip[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    memset(pd, 0, (size_t)N_pairs * M * sizeof(float));
    memset(dd, 0, (size_t)N_pairs * M * sizeof(float));
    for (int m = 0; m < M; ++m) { w[m] = 1.0f; g[m] = 0.0f; l[m] = 0.0f; }

    int rc = 0;

    tc_phase_attention_combine(ctx, bIP, bPD, bDD, bW, bG, bL, bScores, N_pairs, M);
    {
        float expected[N_pairs];
        for (int n = 0; n < N_pairs; ++n) {
            float s = 0.0f;
            for (int m = 0; m < M; ++m) s += ip[n * M + m];
            expected[n] = s;
        }
        const float e = maxabs(scores, expected, N_pairs);
        printf("  phase combine sum identity   max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5) rc = 1;
    }

    /* γ=π/2: cos(0 + π/2) = 0 → all scores zero. */
    for (int m = 0; m < M; ++m) g[m] = (float)M_PI_2;
    tc_phase_attention_combine(ctx, bIP, bPD, bDD, bW, bG, bL, bScores, N_pairs, M);
    {
        float max_v = 0.0f;
        for (int n = 0; n < N_pairs; ++n) max_v = fmaxf(max_v, fabsf(scores[n]));
        printf("  phase combine γ=π/2 → 0      max_abs=%.3e %s\n", max_v, max_v < 1e-5 ? "OK" : "FAIL");
        if (max_v >= 1e-5) rc = 1;
    }

    /* Born-rule output: single amplitude. */
    const int N = 4, V = 8;
    tc_buffer *bH, *bS, *bE, *bProbs;
    tc_buffer_alloc(ctx, (size_t)N * V * sizeof(float), &bH);
    tc_buffer_alloc(ctx, (size_t)N * V * sizeof(float), &bS);
    tc_buffer_alloc(ctx, (size_t)N * V * sizeof(float), &bE);
    tc_buffer_alloc(ctx, (size_t)N * V * sizeof(float), &bProbs);
    void *Hp, *SSp, *EEp, *Pp2;
    tc_buffer_map(bH, &Hp); tc_buffer_map(bS, &SSp); tc_buffer_map(bE, &EEp);
    tc_buffer_map(bProbs, &Pp2);
    float* h = (float*)Hp; float* s_ = (float*)SSp; float* e_ = (float*)EEp;
    float* probs = (float*)Pp2;
    for (int i = 0; i < N * V; ++i) h[i] = ((float)rand()/RAND_MAX - 0.5f) * 1.0f;

    /* h-only: probs = h² / Σh² per row. */
    tc_born_rule_output(ctx, bH, NULL, NULL, bProbs, N, V);
    {
        float expected[N * V];
        float row_sum_err = 0.0f;
        for (int n = 0; n < N; ++n) {
            float Z = 0.0f;
            for (int v = 0; v < V; ++v) Z += h[n*V+v] * h[n*V+v];
            for (int v = 0; v < V; ++v) expected[n*V+v] = h[n*V+v]*h[n*V+v] / Z;
            float row = 0.0f;
            for (int v = 0; v < V; ++v) row += probs[n*V+v];
            row_sum_err = fmaxf(row_sum_err, fabsf(row - 1.0f));
        }
        const float e = maxabs(probs, expected, N * V);
        printf("  born-rule h-only formula     max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        printf("  born-rule rows sum to 1      max_err=%.3e %s\n", row_sum_err, row_sum_err < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5 || row_sum_err >= 1e-5) rc = 1;
    }

    /* Three-amplitude case. */
    for (int i = 0; i < N * V; ++i) s_[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < N * V; ++i) e_[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
    tc_born_rule_output(ctx, bH, bS, bE, bProbs, N, V);
    {
        float expected[N * V];
        for (int n = 0; n < N; ++n) {
            float Z = 0.0f;
            for (int v = 0; v < V; ++v) {
                const float a = h[n*V+v] + s_[n*V+v] + e_[n*V+v];
                Z += a * a;
            }
            for (int v = 0; v < V; ++v) {
                const float a = h[n*V+v] + s_[n*V+v] + e_[n*V+v];
                expected[n*V+v] = a * a / Z;
            }
        }
        const float err = maxabs(probs, expected, N * V);
        printf("  born-rule 3-amp formula      max_err=%.3e %s\n", err, err < 1e-5 ? "OK" : "FAIL");
        if (err >= 1e-5) rc = 1;
    }

    /* All-zero amplitudes: output zeros (no NaN from /0). */
    memset(h, 0, (size_t)N * V * sizeof(float));
    tc_born_rule_output(ctx, bH, NULL, NULL, bProbs, N, V);
    {
        float bad = 0.0f;
        for (int i = 0; i < N * V; ++i) {
            if (!isfinite(probs[i])) bad = 1.0f;
            if (probs[i] != 0.0f) bad = fmaxf(bad, fabsf(probs[i]));
        }
        printf("  born-rule all-zero safe      max_v=%.3e %s\n", bad, bad < 1e-5 ? "OK" : "FAIL");
        if (bad >= 1e-5) rc = 1;
    }

    tc_buffer_free(ctx, bIP); tc_buffer_free(ctx, bPD); tc_buffer_free(ctx, bDD);
    tc_buffer_free(ctx, bW);  tc_buffer_free(ctx, bG);  tc_buffer_free(ctx, bL);
    tc_buffer_free(ctx, bScores);
    tc_buffer_free(ctx, bH); tc_buffer_free(ctx, bS); tc_buffer_free(ctx, bE);
    tc_buffer_free(ctx, bProbs);
    tc_shutdown(ctx);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
