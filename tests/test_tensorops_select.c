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

int main(void) {
    int rc = 0;
    rc |= expect_kernel(TC_DTYPE_F16, "tc4_gemm_f16");
    rc |= expect_kernel(TC_DTYPE_BF16, "tc4_gemm_bf16");
    rc |= expect_kernel(TC_DTYPE_F32, "tc4_gemm_f32");
    rc |= expect_unsupported();
    rc |= expect_invalid();
    return rc;
}
