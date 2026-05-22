#include "tensorops/tensorops_select.h"

const char* tc_tensorops_gemm_kernel_name(const tc_gemm_desc* desc,
                                          tc_status_t* err) {
    if (err) *err = TC_OK;
    if (!desc) {
        if (err) *err = TC_ERR_INVALID_ARG;
        return 0;
    }
    if (desc->a_dtype == TC_DTYPE_F16 &&
        desc->b_dtype == TC_DTYPE_F16 &&
        desc->c_dtype == TC_DTYPE_F16 &&
        desc->accum_dtype == TC_DTYPE_F32) {
        return "tc4_gemm_f16";
    }
    if (desc->a_dtype == TC_DTYPE_BF16 &&
        desc->b_dtype == TC_DTYPE_BF16 &&
        desc->c_dtype == TC_DTYPE_BF16 &&
        desc->accum_dtype == TC_DTYPE_F32) {
        return "tc4_gemm_bf16";
    }
    if (desc->a_dtype == TC_DTYPE_F32 &&
        desc->b_dtype == TC_DTYPE_F32 &&
        desc->c_dtype == TC_DTYPE_F32 &&
        desc->accum_dtype == TC_DTYPE_F32) {
        return "tc4_gemm_f32";
    }
    if (err) *err = TC_ERR_UNSUPPORTED_DTYPE;
    return 0;
}

TC_INTERNAL_SYMBOL bool tc_tensorops_gemm_shape_supported(const tc_gemm_desc* desc) {
    if (!desc) return false;

    /* The SDK26 TensorOps kernel currently writes 64x32 output tiles and uses
     * the public dynamic-K matmul2d shape. Keep dispatch to fully covered
     * tiles until M5 runtime evidence proves ragged edges. */
    return (desc->M % 64) == 0 &&
           (desc->N % 32) == 0 &&
           (desc->K % 32) == 0;
}
