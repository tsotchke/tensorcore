/*
 * tests/test_sparse_24.c — validates 2:4 prune + check + (dense-fallback)
 * sparse GEMM on the CPU side. The cusparseLt tensor-core path is
 * exercised by the cosbox / A100 / H100 integration benches.
 *
 * What we verify here:
 *   1. tc_sparse_24_prune keeps exactly 2 non-zeros per 4-block.
 *   2. The kept entries are the 2 largest by magnitude.
 *   3. tc_sparse_24_check accepts pruned matrices and rejects unpruned.
 *   4. A linear-weight view [N, K] can be pruned along K and transposed
 *      back into GEMM B [K, N].
 *   5. tc_sparse_24_gemm produces the bit-correct dense product when B is
 *      2:4-pruned (zeros contribute zero; fallback path is correctness-
 *      preserving even without the tensor-core speedup).
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/sparse_gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_(const char* what) { fprintf(stderr, "FAIL: %s\n", what); return 1; }

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return fail_("tc_init");

    enum { rows = 8, cols = 16 };   /* cols is a multiple of 4 ✓ */
    tc_buffer* bW;
    if (tc_buffer_alloc(ctx, (size_t)rows * cols * sizeof(float), &bW) != TC_OK) return 1;
    void* Wp = NULL; tc_buffer_map(bW, &Wp);
    float* W = (float*)Wp;

    int rc = 0;

    /* Fill with random fp32; remember a copy for the top-2 check. */
    srand(0x57);
    float W_orig[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        W[i] = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
        W_orig[i] = W[i];
    }

    /* Pre-prune: tc_sparse_24_check must reject (every 4-block has 4 nz). */
    {
        tc_status_t s = tc_sparse_24_check(ctx, bW, TC_DTYPE_F32, rows, cols);
        printf("  check rejects unpruned       %s\n", s == TC_ERR_INVALID_ARG ? "OK" : "FAIL");
        if (s != TC_ERR_INVALID_ARG) rc = 1;
    }

    /* Prune to 2:4. */
    if (tc_sparse_24_prune(ctx, bW, TC_DTYPE_F32, rows, cols) != TC_OK) return fail_("prune");

    /* Each 4-block must have exactly 2 non-zeros, and they must be the
     * two with largest |·| from the original. */
    int blocks_ok = 1;
    int top2_ok = 1;
    for (int r = 0; r < rows; ++r) {
        for (int c0 = 0; c0 < cols; c0 += 4) {
            int nz = 0;
            for (int k = 0; k < 4; ++k) if (W[r*cols + c0 + k] != 0.0f) ++nz;
            if (nz != 2) blocks_ok = 0;

            /* Top-2 by magnitude from W_orig should be exactly the
             * surviving entries (modulo ties; our random inputs avoid
             * exact ties at fp32 precision). */
            int kept[4]; int nk = 0;
            for (int k = 0; k < 4; ++k) {
                if (W[r*cols + c0 + k] != 0.0f) kept[nk++] = k;
            }
            /* Sort |W_orig| descending; first 2 must equal kept. */
            int idx[4] = {0,1,2,3};
            float abs_v[4];
            for (int k = 0; k < 4; ++k) abs_v[k] = fabsf(W_orig[r*cols + c0 + k]);
            for (int a = 0; a < 4; ++a) for (int b = a+1; b < 4; ++b) {
                if (abs_v[idx[b]] > abs_v[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
            }
            const int want_a = idx[0], want_b = idx[1];
            const int got_a = kept[0], got_b = kept[1];
            const int match =
                ((got_a == want_a && got_b == want_b) ||
                 (got_a == want_b && got_b == want_a));
            if (!match) top2_ok = 0;
        }
    }
    printf("  prune: 2 nz per 4-block       %s\n", blocks_ok ? "OK" : "FAIL");
    printf("  prune: kept are top-2 |·|     %s\n", top2_ok ? "OK" : "FAIL");
    if (!blocks_ok || !top2_ok) rc = 1;

    /* check should now accept. */
    {
        tc_status_t s = tc_sparse_24_check(ctx, bW, TC_DTYPE_F32, rows, cols);
        printf("  check accepts pruned         %s\n", s == TC_OK ? "OK" : "FAIL");
        if (s != TC_OK) rc = 1;
    }

    /* Sparse GEMM fallback: C = A @ B_pruned should be the dense product
     * (zeros contribute zero). Verify against a hand-computed dense ref. */
    enum { M = 4, K_ = cols, N_ = 4 };
    tc_buffer *bA, *bB, *bC;
    tc_buffer_alloc(ctx, (size_t)M * K_ * sizeof(float), &bA);
    tc_buffer_alloc(ctx, (size_t)K_ * N_ * sizeof(float), &bB);
    tc_buffer_alloc(ctx, (size_t)M * N_ * sizeof(float), &bC);
    void *Ap = NULL, *Bp = NULL, *Cp = NULL;
    tc_buffer_map(bA, &Ap); tc_buffer_map(bB, &Bp); tc_buffer_map(bC, &Cp);
    float* A = (float*)Ap; float* B = (float*)Bp; float* C = (float*)Cp;
    for (int i = 0; i < M * K_; ++i) A[i] = ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < K_ * N_; ++i) B[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    /* Prune in canonical linear-weight layout: W_linear is [N, K] row-major,
     * so its contiguous 4-blocks are exactly along the K axis. Then transpose
     * it back to B [K, N] for tensorcore's GEMM convention. */
    tc_buffer* bWlin = NULL;
    tc_buffer_alloc(ctx, (size_t)N_ * K_ * sizeof(float), &bWlin);
    void* Wlinp = NULL;
    tc_buffer_map(bWlin, &Wlinp);
    float* Wlin = (float*)Wlinp;
    for (int n = 0; n < N_; ++n) {
        for (int k = 0; k < K_; ++k) {
            Wlin[n*K_ + k] = B[k*N_ + n];
        }
    }
    if (tc_sparse_24_prune(ctx, bWlin, TC_DTYPE_F32, N_, K_) != TC_OK) {
        return fail_("linear-weight prune");
    }
    {
        tc_status_t s = tc_sparse_24_check(ctx, bWlin, TC_DTYPE_F32, N_, K_);
        printf("  canonical K-axis 2:4 layout %s\n", s == TC_OK ? "OK" : "FAIL");
        if (s != TC_OK) rc = 1;
    }
    for (int n = 0; n < N_; ++n) {
        for (int k = 0; k < K_; ++k) {
            B[k*N_ + n] = Wlin[n*K_ + k];
        }
    }

    /* tc_sparse_24_gemm: dense-fallback computes A @ B as standard GEMM. */
    if (tc_sparse_24_gemm(ctx, bA, bB, bC, M, N_, K_,
                           TC_DTYPE_F32, TC_DTYPE_F32, TC_DTYPE_F32,
                           1.0f, 0.0f) != TC_OK) return fail_("sparse_24_gemm");

    /* Hand-compute reference. */
    float ref[M * N_];
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N_; ++n) {
            float s = 0.0f;
            for (int k = 0; k < K_; ++k) s += A[m*K_ + k] * B[k*N_ + n];
            ref[m*N_ + n] = s;
        }
    }
    float max_err = 0.0f;
    for (int i = 0; i < M * N_; ++i) {
        const float e = fabsf(C[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("  fallback sparse GEMM = dense max_err=%.3e %s\n",
           max_err, max_err < 1e-4 ? "OK" : "FAIL");
    if (max_err >= 1e-4f) rc = 1;

    /* tc_sparse_24_available: 0 on portable builds without cusparseLt. */
    const int avail = tc_sparse_24_available();
    printf("  tc_sparse_24_available = %d (cusparseLt: %s)\n",
           avail, avail ? "yes" : "no — dense fallback");

    tc_buffer_free(ctx, bW);
    tc_buffer_free(ctx, bA); tc_buffer_free(ctx, bB); tc_buffer_free(ctx, bC);
    tc_buffer_free(ctx, bWlin);
    tc_shutdown(ctx);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
