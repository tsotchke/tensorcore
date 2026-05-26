#ifndef TENSORCORE_ESHKOL_BRIDGE_H
#define TENSORCORE_ESHKOL_BRIDGE_H

/*
 * Small C ABI helpers for Eshkol's `extern` form.
 *
 * The primary tensorcore ABI uses out-parameters and descriptor structs.
 * Eshkol can call C functions directly, but the Scheme bridge needs a flatter
 * ABI for context creation, buffer allocation, GEMM descriptors, and attention
 * descriptors. These helpers keep that adaptation in tensorcore instead of in
 * compiler-specific codegen.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void*       tc_eshkol_init(void);
int32_t     tc_eshkol_shutdown(void* ctx);

const char* tc_eshkol_device_name(void* ctx);
int32_t     tc_eshkol_device_family(void* ctx);
int32_t     tc_eshkol_device_unified_memory(void* ctx);
int32_t     tc_eshkol_device_supports_bf16(void* ctx);
int32_t     tc_eshkol_device_supports_i8(void* ctx);
int32_t     tc_eshkol_device_supports_tensorops_m5(void* ctx);

void*       tc_eshkol_buffer_alloc(void* ctx, int64_t bytes);
int32_t     tc_eshkol_buffer_free(void* ctx, void* buf);
void*       tc_eshkol_buffer_map(void* buf);

int32_t     tc_eshkol_gemm(void* ctx,
                           int32_t dtype,
                           void* A,
                           void* B,
                           void* C,
                           int32_t M,
                           int32_t N,
                           int32_t K,
                           double alpha,
                           double beta,
                           int32_t transpose_a,
                           int32_t transpose_b);

int32_t     tc_eshkol_attention_forward(void* ctx,
                                        void* Q,
                                        void* K,
                                        void* V,
                                        void* O,
                                        int32_t batch,
                                        int32_t heads,
                                        int32_t seq_q,
                                        int32_t seq_kv,
                                        int32_t head_dim,
                                        double softmax_scale,
                                        int32_t causal);

const char* tc_eshkol_last_backend(void);
int32_t     tc_eshkol_last_backend_code(void);
const char* tc_eshkol_version(void);
const char* tc_eshkol_status_string(int32_t status);

#ifdef __cplusplus
}
#endif

#endif
