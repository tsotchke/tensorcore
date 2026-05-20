#ifndef TENSORCORE_GGUF_H
#define TENSORCORE_GGUF_H

/*
 * Minimal GGUF (v3) file format reader. Parses metadata + tensor info table
 * and gives memory-mapped access to tensor data. Sufficient for loading a
 * real Q4_0 quantized model (TinyLlama 1.1B, Llama-2 7B, etc.) end-to-end.
 *
 * Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */

#include <stddef.h>
#include <stdint.h>
#include "tensorcore/device.h"
#include "tensorcore/quantized.h"
#include "tensorcore/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF tensor types (subset; mirrors GGML enums). */
typedef enum {
    TC_GGUF_TYPE_F32   = 0,
    TC_GGUF_TYPE_F16   = 1,
    TC_GGUF_TYPE_Q4_0  = 2,
    TC_GGUF_TYPE_Q4_1  = 3,
    TC_GGUF_TYPE_Q8_0  = 8,
    TC_GGUF_TYPE_BF16  = 30,
    TC_GGUF_TYPE_UNSUPPORTED = -1,
} tc_gguf_type_t;

typedef struct tc_gguf_file tc_gguf_file;
typedef struct tc_gguf_loaded_model tc_gguf_loaded_model;

typedef struct {
    const char*       name;        /* NUL-terminated, owned by the gguf file handle */
    int32_t           n_dims;
    uint64_t          dims[4];     /* up to 4D */
    tc_gguf_type_t    type;
    uint64_t          offset;      /* offset from start of tensor data region */
    size_t            n_bytes;     /* total bytes of this tensor */
    const void*       data;        /* pointer into the mmap */
} tc_gguf_tensor_info;

typedef struct {
    const char*       name;        /* NUL-terminated, owned by the loaded model */
    int32_t           n_dims;
    uint64_t          dims[4];
    tc_gguf_type_t    type;
    uint64_t          offset;
    size_t            n_bytes;
    tc_buffer*        buffer;      /* owned by the loaded model */
} tc_gguf_loaded_tensor_info;

typedef struct {
    int64_t context_length;
    int64_t embedding_length;
    int64_t feed_forward_length;
    int64_t block_count;
    int64_t attention_head_count;
    int64_t attention_head_count_kv;
    int64_t rope_dimension_count;
    int64_t vocab_size;
    double  rms_norm_epsilon;
    double  rope_freq_base;
    double  rope_freq_scale;
} tc_gguf_llama_config;

typedef struct {
    int              N;          /* output rows; GGUF dim[1] */
    int              K;          /* input columns; GGUF dim[0] */
    tc_gguf_type_t   gguf_type;
    tc_quant_t       quant_type;
    size_t           n_bytes;
    tc_buffer*       buffer;     /* set for loaded tensors, NULL for mmap tensors */
} tc_gguf_quantized_matrix_info;

/* Open a GGUF file: mmap, parse header + metadata + tensor info. */
tc_status_t tc_gguf_open(const char* path, tc_gguf_file** out);
void        tc_gguf_close(tc_gguf_file* f);

/* Stats / iteration. */
uint64_t    tc_gguf_tensor_count(const tc_gguf_file* f);
uint64_t    tc_gguf_metadata_count(const tc_gguf_file* f);

/* Look up a tensor by name. Returns TC_ERR_INVALID_ARG if not found. */
tc_status_t tc_gguf_get_tensor(const tc_gguf_file* f, const char* name,
                               tc_gguf_tensor_info* out_info);

/* Iterate: i in [0, tensor_count). */
tc_status_t tc_gguf_tensor_at(const tc_gguf_file* f, uint64_t i,
                              tc_gguf_tensor_info* out_info);

/* Lookup metadata by key. Returns C string for string values, NULL for
 * non-string or missing. Caller does not own the pointer. */
const char* tc_gguf_meta_get_str(const tc_gguf_file* f, const char* key);
int64_t     tc_gguf_meta_get_i64(const tc_gguf_file* f, const char* key, int64_t default_val);
double      tc_gguf_meta_get_f64(const tc_gguf_file* f, const char* key, double default_val);
uint64_t    tc_gguf_meta_array_count(const tc_gguf_file* f, const char* key);
tc_status_t tc_gguf_meta_array_get_str(const tc_gguf_file* f,
                                       const char* key,
                                       uint64_t index,
                                       const char** out_ptr,
                                       size_t* out_len);
int64_t     tc_gguf_meta_array_get_i64(const tc_gguf_file* f,
                                       const char* key,
                                       uint64_t index,
                                       int64_t default_val);
double      tc_gguf_meta_array_get_f64(const tc_gguf_file* f,
                                       const char* key,
                                       uint64_t index,
                                       double default_val);

/* Extract common LLaMA-family model config fields from GGUF metadata. */
tc_status_t tc_gguf_get_llama_config(const tc_gguf_file* f,
                                     tc_gguf_llama_config* out_config);

/* Allocate a tensorcore buffer and copy the named tensor bytes into it. This is
 * the bridge from mmap-backed GGUF tensor data to Metal kernels. Caller owns
 * the returned buffer and frees it with tc_buffer_free(ctx, buffer). */
tc_status_t tc_gguf_tensor_to_buffer(tc_context* ctx,
                                     const tc_gguf_file* f,
                                     const char* name,
                                     tc_buffer** out_buffer);

/* Interpret a 2D GGUF quantized tensor as the [N, K] matrix expected by
 * tc_gemv_quantized. GGUF stores matrix tensors as dim[0]=K, dim[1]=N. */
tc_status_t tc_gguf_tensor_quantized_matrix_info(
    const tc_gguf_tensor_info* tensor,
    tc_gguf_quantized_matrix_info* out_info);
tc_status_t tc_gguf_loaded_tensor_quantized_matrix_info(
    const tc_gguf_loaded_tensor_info* tensor,
    tc_gguf_quantized_matrix_info* out_info);

/* Copy all tensors with supported storage types into tensorcore buffers.
 * Unsupported GGUF tensor encodings are skipped, not treated as fatal; inspect
 * tc_gguf_loaded_skipped_tensor_count to detect partial loads. */
tc_status_t tc_gguf_load_supported_tensors(tc_context* ctx,
                                           const tc_gguf_file* f,
                                           tc_gguf_loaded_model** out_model);
void        tc_gguf_loaded_model_free(tc_context* ctx,
                                      tc_gguf_loaded_model* model);
uint64_t    tc_gguf_loaded_tensor_count(const tc_gguf_loaded_model* model);
uint64_t    tc_gguf_loaded_skipped_tensor_count(const tc_gguf_loaded_model* model);
tc_status_t tc_gguf_loaded_tensor_at(const tc_gguf_loaded_model* model,
                                     uint64_t i,
                                     tc_gguf_loaded_tensor_info* out_info);
tc_status_t tc_gguf_loaded_get_tensor(const tc_gguf_loaded_model* model,
                                      const char* name,
                                      tc_gguf_loaded_tensor_info* out_info);

#ifdef __cplusplus
}
#endif
#endif
