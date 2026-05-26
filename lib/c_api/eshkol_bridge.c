/*
 * tensorcore Eshkol bridge helpers.
 *
 * Eshkol's `extern` form can call C symbols with scalar/pointer arguments, but
 * the canonical tensorcore ABI intentionally uses out-parameters and descriptor
 * structs. These helpers adapt that ABI to the flatter call shape used by
 * eshkol/tensorcore.esk.
 */

#include "tensorcore/eshkol_bridge.h"
#include "tensorcore/tensorcore.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

static int32_t bool_to_i32(bool value) {
    return value ? 1 : 0;
}

static int32_t normalize_status(tc_status_t status) {
    return (int32_t)status;
}

static tc_dtype_t dtype_from_eshkol(int32_t dtype) {
    switch (dtype) {
        case 0: return TC_DTYPE_F16;
        case 1: return TC_DTYPE_BF16;
        case 2: return TC_DTYPE_F32;
        case 3: return TC_DTYPE_I8;
        default: return (tc_dtype_t)-1;
    }
}

void* tc_eshkol_init(void) {
    tc_context* ctx = NULL;
    tc_status_t status = tc_init(&ctx);
    if (status == TC_OK || status == TC_ERR_ALREADY_INITIALIZED) {
        return ctx;
    }
    return NULL;
}

int32_t tc_eshkol_shutdown(void* ctx) {
    return normalize_status(tc_shutdown((tc_context*)ctx));
}

static int get_device_info(void* ctx, tc_device_info* out_info) {
    if (!ctx || !out_info) return 0;
    memset(out_info, 0, sizeof(*out_info));
    return tc_device_info_get((tc_context*)ctx, out_info) == TC_OK;
}

const char* tc_eshkol_device_name(void* ctx) {
    static char name[sizeof(((tc_device_info*)0)->name)];
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return "unknown";
    memcpy(name, info.name, sizeof(name));
    name[sizeof(name) - 1] = '\0';
    return name;
}

int32_t tc_eshkol_device_family(void* ctx) {
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return 0;
    return (int32_t)info.family;
}

int32_t tc_eshkol_device_unified_memory(void* ctx) {
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return 0;
    return bool_to_i32(info.unified_memory);
}

int32_t tc_eshkol_device_supports_bf16(void* ctx) {
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return 0;
    return bool_to_i32(info.supports_bf16_simdgroup);
}

int32_t tc_eshkol_device_supports_i8(void* ctx) {
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return 0;
    return bool_to_i32(info.supports_i8_simdgroup);
}

int32_t tc_eshkol_device_supports_tensorops_m5(void* ctx) {
    tc_device_info info;
    if (!get_device_info(ctx, &info)) return 0;
    return bool_to_i32(info.supports_tensorops_m5);
}

void* tc_eshkol_buffer_alloc(void* ctx, int64_t bytes) {
    if (!ctx || bytes <= 0) return NULL;
    tc_buffer* buf = NULL;
    if (tc_buffer_alloc((tc_context*)ctx, (size_t)bytes, &buf) != TC_OK) {
        return NULL;
    }
    return buf;
}

int32_t tc_eshkol_buffer_free(void* ctx, void* buf) {
    return normalize_status(tc_buffer_free((tc_context*)ctx, (tc_buffer*)buf));
}

void* tc_eshkol_buffer_map(void* buf) {
    if (!buf) return NULL;
    void* out_ptr = NULL;
    if (tc_buffer_map((tc_buffer*)buf, &out_ptr) != TC_OK) {
        return NULL;
    }
    return out_ptr;
}

int32_t tc_eshkol_gemm(void* ctx,
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
                       int32_t transpose_b) {
    tc_dtype_t tc_dtype = dtype_from_eshkol(dtype);
    if (tc_dtype == (tc_dtype_t)-1) {
        return normalize_status(TC_ERR_UNSUPPORTED_DTYPE);
    }

    tc_gemm_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.a_dtype = tc_dtype;
    desc.b_dtype = tc_dtype;
    desc.c_dtype = (tc_dtype == TC_DTYPE_I8) ? TC_DTYPE_I32 : tc_dtype;
    desc.accum_dtype = (tc_dtype == TC_DTYPE_I8) ? TC_DTYPE_I32 : TC_DTYPE_F32;
    desc.transpose_a = transpose_a != 0;
    desc.transpose_b = transpose_b != 0;
    desc.alpha = (float)alpha;
    desc.beta = (float)beta;

    return normalize_status(tc_gemm((tc_context*)ctx,
                                    &desc,
                                    (const tc_buffer*)A,
                                    (const tc_buffer*)B,
                                    (tc_buffer*)C));
}

int32_t tc_eshkol_attention_forward(void* ctx,
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
                                    int32_t causal) {
    tc_attention_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.batch = batch;
    desc.heads = heads;
    desc.seq_q = seq_q;
    desc.seq_kv = seq_kv;
    desc.head_dim = head_dim;
    desc.io_dtype = TC_DTYPE_F16;
    desc.accum_dtype = TC_DTYPE_F32;
    desc.softmax_scale = (float)softmax_scale;
    desc.causal = causal != 0;
    desc.return_lse = false;
    desc.kv_heads = heads;

    return normalize_status(tc_attention_forward((tc_context*)ctx,
                                                 &desc,
                                                 (const tc_buffer*)Q,
                                                 (const tc_buffer*)K,
                                                 (const tc_buffer*)V,
                                                 (tc_buffer*)O,
                                                 NULL));
}

const char* tc_eshkol_last_backend(void) {
    return tc_backend_name(tc_last_backend());
}

int32_t tc_eshkol_last_backend_code(void) {
    return (int32_t)tc_last_backend();
}

const char* tc_eshkol_version(void) {
    return tc_version();
}

const char* tc_eshkol_status_string(int32_t status) {
    return tc_status_string((tc_status_t)status);
}
