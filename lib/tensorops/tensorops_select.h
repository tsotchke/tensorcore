#ifndef TENSORCORE_TENSOROPS_SELECT_H
#define TENSORCORE_TENSOROPS_SELECT_H

#include "tensorcore/gemm.h"
#include "tensorcore/attention.h"

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

const char* tc_tensorops_gemm_kernel_name(const tc_gemm_desc* desc,
                                          tc_status_t* err);
TC_INTERNAL_SYMBOL bool tc_tensorops_gemm_shape_supported(const tc_gemm_desc* desc);

TC_INTERNAL_SYMBOL const char* tc_tensorops_attention_kernel_name(
    const tc_attention_desc* desc,
    tc_status_t* err);
TC_INTERNAL_SYMBOL bool tc_tensorops_attention_shape_supported(
    const tc_attention_desc* desc);

#ifdef __cplusplus
}
#endif

#endif
