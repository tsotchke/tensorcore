#include <stdio.h>
#include <string.h>

#include "tensorcore/tensorcore.h"
#include "../lib/tensorops/tensorops_select.h"

static int expect_kernel(tc_dtype_t dtype, const char* expected) {
    tc_gemm_desc d = {0};
    d.M = 16;
    d.N = 16;
    d.K = 16;
    d.a_dtype = dtype;
    d.b_dtype = dtype;
    d.c_dtype = dtype;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    tc_status_t err = TC_ERR_INTERNAL;
    const char* got = tc_tensorops_gemm_kernel_name(&d, &err);
    const int ok = (err == TC_OK && got && strcmp(got, expected) == 0);
    printf("  tensorops_select %-4s -> %-16s %s\n",
           tc_dtype_name(dtype), got ? got : "(null)", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_unsupported(void) {
    tc_gemm_desc d = {0};
    d.M = 16;
    d.N = 16;
    d.K = 16;
    d.a_dtype = TC_DTYPE_I8;
    d.b_dtype = TC_DTYPE_I8;
    d.c_dtype = TC_DTYPE_I32;
    d.accum_dtype = TC_DTYPE_I32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    tc_status_t err = TC_OK;
    const char* got = tc_tensorops_gemm_kernel_name(&d, &err);
    const int ok = (err == TC_ERR_UNSUPPORTED_DTYPE && !got);
    printf("  tensorops_select i8   -> %-16s %s\n",
           got ? got : "(unsupported)", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_invalid(void) {
    tc_status_t err = TC_OK;
    const char* got = tc_tensorops_gemm_kernel_name(NULL, &err);
    const int ok = (err == TC_ERR_INVALID_ARG && !got);
    printf("  tensorops_select null -> %-16s %s\n",
           got ? got : "(invalid)", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_shape(int32_t M, int32_t N, int32_t K, int expected) {
    tc_gemm_desc d = {0};
    d.M = M;
    d.N = N;
    d.K = K;
    d.a_dtype = TC_DTYPE_F16;
    d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    const int got = tc_tensorops_gemm_shape_supported(&d) ? 1 : 0;
    const int ok = (got == expected);
    printf("  tensorops_shape %dx%dx%d -> %-3s %s\n",
           M, N, K, got ? "yes" : "no", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static tc_attention_desc attention_desc(int32_t head_dim,
                                        int32_t seq_q,
                                        int32_t seq_kv) {
    tc_attention_desc d;
    memset(&d, 0, sizeof(d));
    d.batch = 1;
    d.heads = 2;
    d.kv_heads = 1;
    d.seq_q = seq_q;
    d.seq_kv = seq_kv;
    d.head_dim = head_dim;
    d.io_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.softmax_scale = 1.0f;
    return d;
}

static int expect_attention_kernel(int32_t head_dim, const char* expected) {
    tc_attention_desc d = attention_desc(head_dim, 64, 64);
    tc_status_t err = TC_ERR_INTERNAL;
    const char* got = tc_tensorops_attention_kernel_name(&d, &err);
    const int ok = (err == TC_OK && got && strcmp(got, expected) == 0);
    printf("  tensorops_attention_select D=%-3d -> %-28s %s\n",
           head_dim, got ? got : "(null)", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_attention_kernel_unsupported(void) {
    tc_attention_desc d = attention_desc(80, 64, 64);
    tc_status_t err = TC_OK;
    const char* got = tc_tensorops_attention_kernel_name(&d, &err);
    const int ok = (err == TC_ERR_UNSUPPORTED_DTYPE && !got);
    printf("  tensorops_attention_select D=80  -> %-28s %s\n",
           got ? got : "(unsupported)", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_attention_shape(int32_t head_dim,
                                  int32_t seq_q,
                                  int32_t seq_kv,
                                  int expected) {
    tc_attention_desc d = attention_desc(head_dim, seq_q, seq_kv);
    const int got = tc_tensorops_attention_shape_supported(&d) ? 1 : 0;
    const int ok = (got == expected);
    printf("  tensorops_attention_shape D=%d Sq=%d Sk=%d -> %-3s %s\n",
           head_dim, seq_q, seq_kv, got ? "yes" : "no", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int expect_attention_shape_variant_rejected(void) {
    int rc = 0;
    tc_attention_desc d = attention_desc(64, 64, 64);
    d.return_lse = true;
    const int lse = tc_tensorops_attention_shape_supported(&d) ? 1 : 0;
    printf("  tensorops_attention_shape return_lse -> %-3s %s\n",
           lse ? "yes" : "no", !lse ? "OK" : "FAIL");
    rc |= lse ? 1 : 0;

    d = attention_desc(64, 64, 64);
    d.window_size = 16;
    const int window = tc_tensorops_attention_shape_supported(&d) ? 1 : 0;
    printf("  tensorops_attention_shape window     -> %-3s %s\n",
           window ? "yes" : "no", !window ? "OK" : "FAIL");
    rc |= window ? 1 : 0;

    d = attention_desc(64, 64, 64);
    float slopes[2] = {0.0f, 0.0f};
    d.alibi_slopes = slopes;
    const int alibi = tc_tensorops_attention_shape_supported(&d) ? 1 : 0;
    printf("  tensorops_attention_shape alibi      -> %-3s %s\n",
           alibi ? "yes" : "no", !alibi ? "OK" : "FAIL");
    rc |= alibi ? 1 : 0;
    return rc;
}

int main(void) {
    int rc = 0;
    rc |= expect_kernel(TC_DTYPE_F16, "tc4_gemm_f16");
    rc |= expect_kernel(TC_DTYPE_BF16, "tc4_gemm_bf16");
    rc |= expect_kernel(TC_DTYPE_F32, "tc4_gemm_f32");
    rc |= expect_unsupported();
    rc |= expect_invalid();
    rc |= expect_shape(64, 64, 64, 1);
    rc |= expect_shape(63, 64, 64, 0);
    rc |= expect_shape(64, 31, 64, 0);
    rc |= expect_shape(64, 64, 31, 0);
    if (tc_tensorops_gemm_shape_supported(NULL)) {
        printf("  tensorops_shape null -> yes FAIL\n");
        rc |= 1;
    } else {
        printf("  tensorops_shape null -> no  OK\n");
    }
    rc |= expect_attention_kernel(64, "tc4_flash_attention_f16_d64");
    rc |= expect_attention_kernel(128, "tc4_flash_attention_f16_d128");
    rc |= expect_attention_kernel_unsupported();
    rc |= expect_attention_shape(64, 64, 64, 1);
    rc |= expect_attention_shape(128, 128, 128, 1);
    rc |= expect_attention_shape(64, 63, 64, 0);
    rc |= expect_attention_shape(64, 64, 63, 0);
    rc |= expect_attention_shape(80, 64, 64, 0);
    rc |= expect_attention_shape_variant_rejected();
    return rc;
}
