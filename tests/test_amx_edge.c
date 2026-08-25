/*
 * tests/test_amx_edge.c - validates the AMX wrapper paths for edge tiles
 * (M, N not divisible by 16) and general alpha/beta sgemm contract.
 *
 * Each test:
 *   - Allocates A, B, C with the given shape
 *   - Computes reference C_ref = alpha*A*B + beta*C via a straight loop
 *   - Runs tc_amx_gemm_f32 directly so NEON/CBLAS fallback cannot mask an
 *     AMX wrapper bug
 *   - Compares to reference; expects max abs error < 1e-3
 *
 * Skips with exit 77 unless TC_RUN_AMX_GEMM_TEST=1 is set. Hosted macOS
 * runners can compile AMX but trap the reverse-engineered instructions.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int tc_amx_gemm_f32(int M, int N, int K,
                           float alpha,
                           const float* A, int lda, int transpose_a,
                           const float* B, int ldb, int transpose_b,
                           float beta,
                           float* C, int ldc);
extern int tc_amx_gemm_f32_available(void);

static int amx_test_enabled(void) {
    const char* run = getenv("TC_RUN_AMX_GEMM_TEST");
    if (!run || strcmp(run, "1") != 0) {
        printf("AMX edge regression skipped; set TC_RUN_AMX_GEMM_TEST=1 to run\n");
        return 0;
    }
    return 1;
}

static void gemm_ref(int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += (double)A[(size_t)i * (size_t)lda + (size_t)k] *
                     (double)B[(size_t)k * (size_t)ldb + (size_t)j];
            }
            C[(size_t)i * (size_t)ldc + (size_t)j] =
                (float)(alpha * s) +
                beta * C[(size_t)i * (size_t)ldc + (size_t)j];
        }
    }
}

/* Forward decl for the transpose variant below — see run_case_t. */
static int run_case_t(int M, int N, int K,
                      float alpha, float beta,
                      int transpose_a, int transpose_b,
                      const char* label);

static int run_case(int M, int N, int K,
                    float alpha, float beta, const char* label) {
    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;
    const size_t c_elems = (size_t)M * (size_t)N;
    float* A = (float*)malloc((a_elems ? a_elems : 1u) * sizeof(float));
    float* B = (float*)malloc((b_elems ? b_elems : 1u) * sizeof(float));
    float* C_actual = (float*)malloc((c_elems ? c_elems : 1u) * sizeof(float));
    float* C_ref = (float*)malloc((c_elems ? c_elems : 1u) * sizeof(float));
    if (!A || !B || !C_actual || !C_ref) {
        fprintf(stderr, "allocation failed for %s\n", label);
        free(A);
        free(B);
        free(C_actual);
        free(C_ref);
        return 1;
    }

    srand(0xDEAD);
    for (size_t i = 0; i < a_elems; ++i) {
        A[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    for (size_t i = 0; i < b_elems; ++i) {
        B[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    for (size_t i = 0; i < c_elems; ++i) {
        const float c0 = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        C_actual[i] = c0;
        C_ref[i] = c0;
    }

    gemm_ref(M, N, K, alpha, A, K, B, N, beta, C_ref, N);

    const int s = tc_amx_gemm_f32(M, N, K, alpha,
                                  A, K, 0,
                                  B, N, 0,
                                  beta,
                                  C_actual, N);
    if (s != 0) {
        fprintf(stderr, "  %s: tc_amx_gemm_f32 failed (status=%d)\n", label, s);
        free(A);
        free(B);
        free(C_actual);
        free(C_ref);
        return 1;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < c_elems; ++i) {
        const float e = fabsf(C_actual[i] - C_ref[i]);
        if (e > max_err) max_err = e;
    }
    const int ok = max_err < 1e-3f;
    printf("  %-28s M=%d N=%d K=%d alpha=%.2f beta=%.2f max_err=%.2e %s\n",
           label, M, N, K, alpha, beta, max_err, ok ? "OK" : "FAIL");

    free(A);
    free(B);
    free(C_actual);
    free(C_ref);
    return ok ? 0 : 1;
}

/* Transpose case: passes A as K×M (if transpose_a) and B as N×K (if transpose_b)
 * to tc_amx_gemm_f32; reference still computes C = alpha * A_logical * B_logical
 * where A_logical and B_logical are the *un-transposed* views (M×K and K×N). */
static int run_case_t(int M, int N, int K,
                      float alpha, float beta,
                      int transpose_a, int transpose_b,
                      const char* label) {
    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;
    const size_t c_elems = (size_t)M * (size_t)N;
    float* A_log = (float*)malloc(a_elems * sizeof(float));   /* M×K */
    float* B_log = (float*)malloc(b_elems * sizeof(float));   /* K×N */
    float* C_actual = (float*)malloc(c_elems * sizeof(float));
    float* C_ref = (float*)malloc(c_elems * sizeof(float));

    srand(0xBEEF);
    for (size_t i = 0; i < a_elems; ++i) A_log[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.2f;
    for (size_t i = 0; i < b_elems; ++i) B_log[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.2f;
    for (size_t i = 0; i < c_elems; ++i) {
        const float c0 = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        C_actual[i] = c0;
        C_ref[i] = c0;
    }
    gemm_ref(M, N, K, alpha, A_log, K, B_log, N, beta, C_ref, N);

    /* Build storage views matching transpose flags. */
    float* A_view = A_log;
    int lda = K;
    float* A_T = NULL;
    if (transpose_a) {
        A_T = (float*)malloc((size_t)K * (size_t)M * sizeof(float));
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                A_T[(size_t)k * M + i] = A_log[(size_t)i * K + k];
        A_view = A_T;
        lda = M;
    }
    float* B_view = B_log;
    int ldb = N;
    float* B_T = NULL;
    if (transpose_b) {
        B_T = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                B_T[(size_t)j * K + k] = B_log[(size_t)k * N + j];
        B_view = B_T;
        ldb = K;
    }

    const int s = tc_amx_gemm_f32(M, N, K, alpha,
                                  A_view, lda, transpose_a,
                                  B_view, ldb, transpose_b,
                                  beta, C_actual, N);
    if (s != 0) {
        fprintf(stderr, "  %s: tc_amx_gemm_f32 failed (status=%d)\n", label, s);
        free(A_log); free(B_log); free(C_actual); free(C_ref);
        free(A_T); free(B_T);
        return 1;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < c_elems; ++i) {
        const float e = fabsf(C_actual[i] - C_ref[i]);
        if (e > max_err) max_err = e;
    }
    const int ok = max_err < 1e-3f;
    printf("  %-28s M=%d N=%d K=%d alpha=%.2f beta=%.2f tA=%d tB=%d max_err=%.2e %s\n",
           label, M, N, K, alpha, beta, transpose_a, transpose_b, max_err, ok ? "OK" : "FAIL");

    free(A_log); free(B_log); free(C_actual); free(C_ref);
    free(A_T); free(B_T);
    return ok ? 0 : 1;
}

int main(void) {
    if (!amx_test_enabled()) return 77;
    if (!tc_amx_gemm_f32_available()) {
        printf("AMX unavailable on this build\n");
        return 77;
    }

    int rc = 0;
    rc |= run_case(32, 32, 32, 1.0f, 0.0f, "aligned plain");
    rc |= run_case(64, 64, 64, 1.0f, 0.0f, "aligned plain larger");

    rc |= run_case(13, 31, 17, 1.0f, 0.0f, "edge MxN unaligned");
    rc |= run_case(17, 16, 32, 1.0f, 0.0f, "edge M only");
    rc |= run_case(16, 23, 48, 1.0f, 0.0f, "edge N only");
    rc |= run_case(100, 71, 50, 1.0f, 0.0f, "edge tiles medium");

    rc |= run_case(32, 32, 32, 2.5f, 0.0f, "alpha 2.5 beta 0");
    rc |= run_case(32, 32, 32, -1.0f, 0.0f, "alpha -1 beta 0");
    rc |= run_case(32, 32, 32, 1.0f, 0.5f, "alpha 1 beta 0.5");
    rc |= run_case(32, 32, 32, 1.0f, -1.0f, "alpha 1 beta -1");

    rc |= run_case(47, 33, 19, 1.5f, 0.25f, "general edge alpha beta");
    rc |= run_case(100, 100, 50, 0.7f, -0.3f, "general aligned alpha beta");

    rc |= run_case(16, 16, 0, 1.0f, 2.0f, "K=0 beta=2");
    rc |= run_case(8, 12, 0, 1.0f, 0.0f, "K=0 beta=0");

    /* Transpose A only: A stored K×M (lda=M). */
    rc |= run_case_t(32, 32, 32, 1.0f, 0.0f, 1, 0, "transpose A aligned");
    rc |= run_case_t(47, 16, 19, 1.0f, 0.0f, 1, 0, "transpose A edge");

    /* Transpose B only: B stored N×K (ldb=K). */
    rc |= run_case_t(32, 32, 32, 1.0f, 0.0f, 0, 1, "transpose B aligned");
    rc |= run_case_t(16, 47, 19, 1.0f, 0.0f, 0, 1, "transpose B edge");

    /* Transpose both. */
    rc |= run_case_t(32, 32, 32, 1.0f, 0.0f, 1, 1, "transpose AB aligned");
    rc |= run_case_t(47, 31, 19, 2.0f, 0.5f, 1, 1, "transpose AB edge alpha beta");

    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
