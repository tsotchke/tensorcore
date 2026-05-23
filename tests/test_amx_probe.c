/*
 * Safe AMX capability probe regression.
 *
 * This test intentionally does not execute raw AMX instructions. It validates
 * the non-trapping metadata/stub contract used to decide whether the direct
 * AMX regressions may be enabled on trusted Apple-Silicon hosts.
 */

#include <stdio.h>

extern int tc_amx_gemm_f32_available(void);
extern int tc_amx_isa_version(void);
extern int tc_amx_cluster_count(void);
extern int tc_amx_gemm_f16(int M, int N, int K, float alpha,
                           const void* A, int lda, int transpose_a,
                           const void* B, int ldb, int transpose_b,
                           float beta, void* C, int ldc);
extern int tc_amx_gemm_bf16(int M, int N, int K, float alpha,
                            const void* A, int lda, int transpose_a,
                            const void* B, int ldb, int transpose_b,
                            float beta, void* C, int ldc);

int main(void) {
    const int available = tc_amx_gemm_f32_available();
    const int isa = tc_amx_isa_version();
    const int clusters = tc_amx_cluster_count();

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
    if (available != 1) {
        fprintf(stderr, "expected AMX metadata availability on Apple arm64, got %d\n",
                available);
        return 1;
    }
    if (isa < 1 || isa > 3) {
        fprintf(stderr, "unexpected AMX ISA version: %d\n", isa);
        return 2;
    }
    if (clusters < 1 || clusters > 2) {
        fprintf(stderr, "unexpected AMX cluster count: %d\n", clusters);
        return 3;
    }
#else
    if (available != 0 || isa != 0 || clusters != 0) {
        fprintf(stderr,
                "non-Apple AMX probes should be unavailable, got available=%d isa=%d clusters=%d\n",
                available, isa, clusters);
        return 4;
    }
#endif

    if (tc_amx_gemm_f16(0, 0, 0, 1.0f, NULL, 0, 0, NULL, 0, 0, 0.0f, NULL, 0) != -1) {
        fprintf(stderr, "tc_amx_gemm_f16 should remain gated until hardware validation\n");
        return 5;
    }
    if (tc_amx_gemm_bf16(0, 0, 0, 1.0f, NULL, 0, 0, NULL, 0, 0, 0.0f, NULL, 0) != -1) {
        fprintf(stderr, "tc_amx_gemm_bf16 should remain gated until hardware validation\n");
        return 6;
    }

    printf("AMX probe OK: available=%d isa=%d clusters=%d\n",
           available, isa, clusters);
    return 0;
}
