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

static float value_a(int m, int k) {
    return (float)(((m * 17 + k * 13) % 29) - 14) * 0.03125f;
}

static float value_b(int k, int n) {
    return (float)(((k * 11 + n * 7) % 31) - 15) * 0.02734375f;
}

static int run_case(int M, int N, int K) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;
    const size_t c_elems = (size_t)M * (size_t)N;
    float* A = (float*)malloc(a_elems * sizeof(float));
    float* B = (float*)malloc(b_elems * sizeof(float));
    float* C = (float*)malloc(c_elems * sizeof(float));
    float* Cref = (float*)malloc(c_elems * sizeof(float));
    if (!A || !B || !C || !Cref) {
        fprintf(stderr, "allocation failed for M=%d N=%d K=%d\n", M, N, K);
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 1;
    }

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            A[(size_t)m * (size_t)lda + (size_t)k] = value_a(m, k);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B[(size_t)k * (size_t)ldb + (size_t)n] = value_b(k, n);
        }
    }
    for (size_t i = 0; i < c_elems; ++i) C[i] = -7.0f;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc = fmaf(A[(size_t)m * (size_t)lda + (size_t)k],
                           B[(size_t)k * (size_t)ldb + (size_t)n],
                           acc);
            }
            Cref[(size_t)m * (size_t)ldc + (size_t)n] = acc;
        }
    }

    const int rc = tc_amx_gemm_f32(M, N, K, 1.0f, A, lda, 0, B, ldb, 0, 0.0f, C, ldc);
    if (rc != 0) {
        fprintf(stderr, "tc_amx_gemm_f32 failed for M=%d N=%d K=%d\n", M, N, K);
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 2;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < c_elems; ++i) {
        const float err = fabsf(C[i] - Cref[i]);
        if (err > max_abs) max_abs = err;
    }
    printf("amx M=%d N=%d K=%d max_abs=%.3e\n", M, N, K, max_abs);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return max_abs < 1.0e-4f ? 0 : 3;
}

static int run_k_zero_case(void) {
    enum { M = 16, N = 16, K = 0 };
    float C[M * N];
    float dummy = 1.0f;
    for (int i = 0; i < M * N; ++i) C[i] = 3.25f;
    const int rc = tc_amx_gemm_f32(M, N, K, 1.0f, &dummy, K, 0,
                                   &dummy, N, 0, 0.0f, C, N);
    if (rc != 0) {
        fprintf(stderr, "tc_amx_gemm_f32 failed K==0 case\n");
        return 1;
    }
    for (int i = 0; i < M * N; ++i) {
        if (C[i] != 0.0f) {
            fprintf(stderr, "K==0 did not zero C at index %d: %.8f\n", i, C[i]);
            return 2;
        }
    }
    printf("amx M=%d N=%d K=%d zero-output OK\n", M, N, K);
    return 0;
}

int main(void) {
    if (!tc_amx_gemm_f32_available()) {
        printf("AMX unavailable on this build\n");
        return 77;
    }

    int rc = 0;
    rc |= run_k_zero_case();
    rc |= run_case(16, 16, 1);
    rc |= run_case(16, 16, 17);
    rc |= run_case(16, 16, 257);
    rc |= run_case(256, 16, 33);
    return rc ? 1 : 0;
}
