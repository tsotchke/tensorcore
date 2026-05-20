/*
 * Correctness test: tc_gemm in int8 vs int32 reference.
 *
 * Requires Apple10+ (M4) for simdgroup_matrix<char,8,8>. Skips on older.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static int run_case(tc_context* ctx, int M, int N, int K) {
    const size_t ba = (size_t)M * K;
    const size_t bb = (size_t)K * N;
    const size_t bc = (size_t)M * N * sizeof(int32_t);

    tc_buffer *A, *B, *C;
    tc_buffer_alloc(ctx, ba, &A);
    tc_buffer_alloc(ctx, bb, &B);
    tc_buffer_alloc(ctx, bc, &C);
    int8_t  *Ap, *Bp;
    int32_t *Cp;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    srand(0x1788);
    for (size_t i = 0; i < ba; ++i) Ap[i] = (int8_t)((rand() & 0x7F) - 64);
    for (size_t i = 0; i < bb; ++i) Bp[i] = (int8_t)((rand() & 0x7F) - 64);
    memset(Cp, 0, bc);

    /* Reference: int32 matmul. */
    int32_t* Cr = calloc((size_t)M * N, sizeof(int32_t));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int k = 0; k < K; ++k) s += (int32_t)Ap[m*K+k] * (int32_t)Bp[k*N+n];
            Cr[m*N+n] = s;
        }

    tc_gemm_desc d = {0};
    d.M=M; d.N=N; d.K=K;
    d.a_dtype=TC_DTYPE_I8; d.b_dtype=TC_DTYPE_I8;
    d.c_dtype=TC_DTYPE_I32; d.accum_dtype=TC_DTYPE_I32;
    d.alpha=1.0f; d.beta=0.0f;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    if (s == TC_ERR_UNSUPPORTED_FAMILY) {
        printf("  M=%d N=%d K=%d   SKIPPED (i8 simdgroup_matrix requires Apple10+/M4+)\n",
               M, N, K);
        free(Cr);
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
        return 0;
    }

    int errors = 0;
    int64_t max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        int64_t e = (int64_t)Cp[i] - (int64_t)Cr[i];
        if (e < 0) e = -e;
        if (e > max_abs) max_abs = e;
        if (e != 0) ++errors;
    }
    printf("  M=%d N=%d K=%d   backend=%-18s  errors=%d/%d  max_abs=%lld  %s\n",
           M, N, K, tc_backend_name(tc_last_backend()),
           errors, M*N, (long long)max_abs,
           (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Cr);
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    /* int8 matmul should be bit-exact (integer arithmetic). */
    return (s == TC_OK && errors == 0) ? 0 : 5;
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
    if (!info.supports_i8_simdgroup) {
        printf("[skip] device family=Apple%d does not expose i8 simdgroup_matrix\n",
               (int)info.family);
        tc_shutdown(ctx);
        return 0;
    }
    int rc = 0;
    rc |= run_case(ctx, 64, 64, 64);
    rc |= run_case(ctx, 128, 128, 128);
    rc |= run_case(ctx, 256, 256, 256);
    tc_shutdown(ctx);
    return rc;
}
