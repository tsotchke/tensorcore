/*
 * Correctness test: tc_gemm in fp32 vs cblas_sgemm reference.
 *
 * Build: handled by tests/CMakeLists.txt. Run: bin/test_gemm_f32.
 *
 * The kernel hits the simdgroup_matrix fp32 path on Apple7+ and the MPS
 * fallback elsewhere — both should match Accelerate to within ~1e-3 relative.
 */

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

static int run_case(tc_context* ctx, int M, int N, int K, int trans_a, int trans_b) {
    const size_t bytes_a = (size_t)M * K * sizeof(float);
    const size_t bytes_b = (size_t)K * N * sizeof(float);
    const size_t bytes_c = (size_t)M * N * sizeof(float);

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK) return 2;
    if (tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) return 3;

    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    /* Deterministic random fill */
    srand(0xC0DE);
    for (int i = 0; i < M * K; ++i) Ap[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; ++i) Bp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    memset(Cp, 0, bytes_c);

    /* Reference: Accelerate sgemm */
    float* Cref = (float*)calloc((size_t)M * N, sizeof(float));
    cblas_sgemm(CblasRowMajor,
                trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans,
                M, N, K,
                1.0f, Ap, trans_a ? M : K,
                      Bp, trans_b ? K : N,
                0.0f, Cref, N);

    /* tensorcore */
    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = !!trans_a;
    d.transpose_b = !!trans_b;
    d.alpha = 1.0f; d.beta = 0.0f;

    tc_status_t s = tc_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  tc_gemm failed: %s\n", tc_status_string(s));
        free(Cref);
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
        return 4;
    }

    /* Compare. fp32 simdgroup_matrix with fp32 accum should be ~bit-exact for
     * small sizes; up to ~1e-4 rel for any reordering. */
    double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const double a = Cp[i], r = Cref[i];
        const double e = fabs(a - r);
        const double re = e / (fabs(r) + 1e-9);
        if (e > max_abs) max_abs = e;
        if (re > max_rel) max_rel = re;
        sum_sq += e * e;
    }
    const double rmse = sqrt(sum_sq / (M * N));

    const char* backend = tc_backend_name(tc_last_backend());
    printf("  M=%d N=%d K=%d ta=%d tb=%d   backend=%-18s  max_abs=%.3e  max_rel=%.3e  rmse=%.3e\n",
           M, N, K, trans_a, trans_b, backend, max_abs, max_rel, rmse);

    free(Cref);
    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);

    /* Tolerance: 1e-3 relative for fp32 matmul of u[-1,1] uniform inputs. */
    return (max_rel < 1e-3) ? 0 : 5;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    tc_device_info info;
    tc_device_info_get(ctx, &info);
    printf("device=%s  family=Apple%d  unified=%s\n",
           info.name, (int)info.family, info.unified_memory ? "yes" : "no");

    int rc = 0;
    /* Aligned shapes (no boundary path) */
    rc |= run_case(ctx,  64,  64,  64, 0, 0);
    rc |= run_case(ctx, 128, 128, 128, 0, 0);
    rc |= run_case(ctx, 256, 256, 256, 0, 0);
    rc |= run_case(ctx, 512, 512, 512, 0, 0);
    /* Non-aligned (exercises bounds-check slow path) */
    rc |= run_case(ctx,  65,  63,  47, 0, 0);
    rc |= run_case(ctx, 100, 200, 300, 0, 0);

    tc_shutdown(ctx);
    return rc;
}
