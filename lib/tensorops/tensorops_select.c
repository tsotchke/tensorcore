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
