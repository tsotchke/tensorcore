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

TC_INTERNAL_SYMBOL const char* tc_tensorops_attention_kernel_name(
    const tc_attention_desc* desc,
    tc_status_t* err) {
    if (err) *err = TC_OK;
    if (!desc) {
        if (err) *err = TC_ERR_INVALID_ARG;
        return 0;
    }
    if (desc->io_dtype != TC_DTYPE_F16 || desc->accum_dtype != TC_DTYPE_F32) {
        if (err) *err = TC_ERR_UNSUPPORTED_DTYPE;
        return 0;
    }
    if (desc->head_dim == 64) return "tc4_flash_attention_f16_d64";
    if (desc->head_dim == 128) return "tc4_flash_attention_f16_d128";
    if (err) *err = TC_ERR_UNSUPPORTED_DTYPE;
    return 0;
}

TC_INTERNAL_SYMBOL bool tc_tensorops_attention_shape_supported(
    const tc_attention_desc* desc) {
    if (!desc) return false;
    if (desc->batch <= 0 || desc->heads <= 0 ||
        desc->seq_q <= 0 || desc->seq_kv <= 0 ||
        desc->head_dim <= 0 || desc->window_size != 0 ||
        desc->alibi_slopes != 0 || desc->return_lse) {
        return false;
    }
    const int32_t kv_heads = desc->kv_heads > 0 ? desc->kv_heads : desc->heads;
    if (kv_heads <= 0 || kv_heads > desc->heads || (desc->heads % kv_heads) != 0) {
        return false;
    }
    if (desc->io_dtype != TC_DTYPE_F16 || desc->accum_dtype != TC_DTYPE_F32) {
        return false;
    }
    if (desc->head_dim != 64 && desc->head_dim != 128) {
        return false;
    }

    /* Match the first planned M5 TensorOps attention probe: fully tiled
     * FlashAttention forward, no LSE/window/ALiBi variants, and whole
     * 64-token query/key blocks until ragged edges are proven on M5. */
    return (desc->seq_q % 64) == 0 && (desc->seq_kv % 64) == 0;
}
