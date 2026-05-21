#ifndef TENSORCORE_TENSOROPS_SELECT_H
#define TENSORCORE_TENSOROPS_SELECT_H

#include "tensorcore/gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* tc_tensorops_gemm_kernel_name(const tc_gemm_desc* desc,
                                          tc_status_t* err);

#ifdef __cplusplus
}
#endif

#endif
