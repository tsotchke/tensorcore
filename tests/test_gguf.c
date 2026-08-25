/*
 * GGUF reader correctness - round-trip a small synthetic GGUF file.
 *
 * Writes a minimal valid GGUF v3 with one Q4_0 matrix tensor, one unsupported
 * tensor, and a few metadata KVs,
 * then opens it via tc_gguf_open and verifies header / metadata / tensor
 * info parses back to the same values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include "tensorcore/tensorcore.h"

#define GGUF_MAGIC 0x46554747u

static void w_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE* f, const char* s) {
    uint64_t n = strlen(s);
    w_u64(f, n);
    fwrite(s, 1, n, f);
}
static void w_kv_str(FILE* f, const char* key, const char* val) {
    w_str(f, key);
    w_u32(f, 8);   /* GGUF_TYPE_STRING */
    w_str(f, val);
}
static void w_kv_u32(FILE* f, const char* key, uint32_t val) {
    w_str(f, key);
    w_u32(f, 4);   /* GGUF_TYPE_UINT32 */
    w_u32(f, val);
}
static void w_kv_f32(FILE* f, const char* key, float val) {
    w_str(f, key);
    w_u32(f, 6);   /* GGUF_TYPE_FLOAT32 */
    fwrite(&val, 4, 1, f);
}
static void w_kv_f64(FILE* f, const char* key, double val) {
    w_str(f, key);
    w_u32(f, 12);  /* GGUF_TYPE_FLOAT64 */
    fwrite(&val, 8, 1, f);
}
static void w_kv_str_array2(FILE* f, const char* key, const char* a, const char* b) {
    w_str(f, key);
    w_u32(f, 9);   /* GGUF_TYPE_ARRAY */
    w_u32(f, 8);   /* GGUF_TYPE_STRING */
    w_u64(f, 2);
    w_str(f, a);
    w_str(f, b);
}
static void w_kv_f32_array2(FILE* f, const char* key, float a, float b) {
    w_str(f, key);
    w_u32(f, 9);   /* GGUF_TYPE_ARRAY */
    w_u32(f, 6);   /* GGUF_TYPE_FLOAT32 */
    w_u64(f, 2);
    fwrite(&a, 4, 1, f);
    fwrite(&b, 4, 1, f);
}
static void w_kv_i32_array2(FILE* f, const char* key, int32_t a, int32_t b) {
    w_str(f, key);
    w_u32(f, 9);   /* GGUF_TYPE_ARRAY */
    w_u32(f, 5);   /* GGUF_TYPE_INT32 */
    w_u64(f, 2);
    fwrite(&a, 4, 1, f);
    fwrite(&b, 4, 1, f);
}

int main(void) {
    const char* path = "/tmp/tc_test.gguf";
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* Header */
    w_u32(f, GGUF_MAGIC);
    w_u32(f, 3);             /* version 3 */
    w_u64(f, 2);             /* tensor_count = 2 */
    w_u64(f, 14);            /* metadata_kv_count = 14 */

    /* Metadata */
    w_kv_str(f, "general.architecture", "llama");
    w_kv_str(f, "general.name", "test-1m");
    w_kv_u32(f, "llama.context_length", 2048);
    w_kv_u32(f, "llama.embedding_length", 4096);
    w_kv_u32(f, "llama.feed_forward_length", 11008);
    w_kv_u32(f, "llama.block_count", 32);
    w_kv_u32(f, "llama.attention.head_count", 32);
    w_kv_u32(f, "llama.attention.head_count_kv", 8);
    w_kv_u32(f, "llama.rope.dimension_count", 128);
    w_kv_f32(f, "llama.attention.layer_norm_rms_epsilon", 0.125f);
    w_kv_f64(f, "llama.rope.freq_base", 10000.0);
    w_kv_str_array2(f, "tokenizer.ggml.tokens", "<unk>", "hello");
    w_kv_f32_array2(f, "tokenizer.ggml.scores", -1000.0f, 0.25f);
    w_kv_i32_array2(f, "tokenizer.ggml.token_type", 2, 1);

    /* Tensor info: tensor 0, Q4_0, shape [K=32, N=1] */
    w_str(f, "weight.test");
    w_u32(f, 2);                  /* n_dims */
    w_u64(f, 32);                 /* dim[0] = K (one Q4_0 block) */
    w_u64(f, 1);                  /* dim[1] = N */
    w_u32(f, 2);                  /* ggml_type Q4_0 */
    w_u64(f, 0);                  /* offset = 0 */

    /* Tensor info: tensor 1, unsupported storage. It has no copied payload. */
    w_str(f, "unsupported.test");
    w_u32(f, 1);                  /* n_dims */
    w_u64(f, 32);
    w_u32(f, 999);                /* unsupported ggml_type */
    w_u64(f, 18);                 /* offset = end of tensor 0 */

    /* Pad to alignment 32 */
    long pos = ftell(f);
    long pad = (32 - (pos % 32)) % 32;
    for (long i = 0; i < pad; ++i) fputc(0, f);

    /* Write one Q4_0 block: 2-byte half scale + 16-byte GGML nibble bytes. */
    uint16_t scale_h = 0x3800;    /* half(0.5) */
    fwrite(&scale_h, 2, 1, f);
    for (int i = 0; i < 16; ++i) fputc((uint8_t)(0xab), f);

    fclose(f);

    /* Now open + verify. */
    tc_gguf_file* g = NULL;
    tc_status_t s = tc_gguf_open(path, &g);
    if (s != TC_OK) { fprintf(stderr, "tc_gguf_open: %s\n", tc_status_string(s)); return 2; }

    const uint64_t tc = tc_gguf_tensor_count(g);
    const uint64_t mc = tc_gguf_metadata_count(g);
    printf("gguf: tensors=%llu, metadata=%llu\n",
           (unsigned long long)tc, (unsigned long long)mc);

    const char* arch = tc_gguf_meta_get_str(g, "general.architecture");
    const char* name = tc_gguf_meta_get_str(g, "general.name");
    int64_t ctxlen = tc_gguf_meta_get_i64(g, "llama.context_length", -1);
    double rms_eps = tc_gguf_meta_get_f64(g, "llama.attention.layer_norm_rms_epsilon", -1.0);
    double rope_base = tc_gguf_meta_get_f64(g, "llama.rope.freq_base", -1.0);
    double missing_f = tc_gguf_meta_get_f64(g, "missing.float", 3.5);
    const char* token1 = NULL;
    size_t token1_len = 0;
    tc_status_t token_s = tc_gguf_meta_array_get_str(g, "tokenizer.ggml.tokens", 1, &token1, &token1_len);
    uint64_t token_count = tc_gguf_meta_array_count(g, "tokenizer.ggml.tokens");
    double score1 = tc_gguf_meta_array_get_f64(g, "tokenizer.ggml.scores", 1, -1.0);
    int64_t type0 = tc_gguf_meta_array_get_i64(g, "tokenizer.ggml.token_type", 0, -1);
    tc_gguf_llama_config cfg;
    tc_status_t cfg_s = tc_gguf_get_llama_config(g, &cfg);
    printf("  arch         : %s\n", arch ? arch : "(null)");
    printf("  name         : %s\n", name ? name : "(null)");
    printf("  context_len  : %lld\n", (long long)ctxlen);
    printf("  rms_eps      : %.6f\n", rms_eps);
    printf("  rope_base    : %.1f\n", rope_base);
    printf("  token[1]     : %.*s\n", (int)token1_len, token1 ? token1 : "");

    tc_gguf_tensor_info t;
    if (tc_gguf_get_tensor(g, "weight.test", &t) != TC_OK) {
        fprintf(stderr, "tensor not found\n"); return 3;
    }
    tc_gguf_quantized_matrix_info qinfo;
    tc_status_t qinfo_s = tc_gguf_tensor_quantized_matrix_info(&t, &qinfo);
    printf("  tensor       : name=%s, dtype=%d, dims=[%llu,%llu], n_bytes=%zu\n",
           t.name, (int)t.type,
           (unsigned long long)t.dims[0], (unsigned long long)t.dims[1],
           t.n_bytes);

    /* Verify the Q4_0 block bytes round-tripped. */
    const uint8_t* p = (const uint8_t*)t.data;
    const uint16_t got_scale = *(const uint16_t*)p;
    int qs_ok = 1;
    for (int i = 0; i < 16; ++i) if (p[2 + i] != 0xab) { qs_ok = 0; break; }

    const int ok =
        (tc == 2) && (mc == 14) &&
        arch && strcmp(arch, "llama") == 0 &&
        name && strcmp(name, "test-1m") == 0 &&
        ctxlen == 2048 &&
        fabs(rms_eps - 0.125) < 1e-12 &&
        fabs(rope_base - 10000.0) < 1e-12 &&
        fabs(missing_f - 3.5) < 1e-12 &&
        token_count == 2 &&
        token_s == TC_OK &&
        token1_len == 5 &&
        token1 && memcmp(token1, "hello", 5) == 0 &&
        fabs(score1 - 0.25) < 1e-12 &&
        type0 == 2 &&
        cfg_s == TC_OK &&
        cfg.context_length == 2048 &&
        cfg.embedding_length == 4096 &&
        cfg.feed_forward_length == 11008 &&
        cfg.block_count == 32 &&
        cfg.attention_head_count == 32 &&
        cfg.attention_head_count_kv == 8 &&
        cfg.rope_dimension_count == 128 &&
        cfg.vocab_size == 2 &&
        fabs(cfg.rms_norm_epsilon - 0.125) < 1e-12 &&
        fabs(cfg.rope_freq_base - 10000.0) < 1e-12 &&
        fabs(cfg.rope_freq_scale - 1.0) < 1e-12 &&
        t.type == TC_GGUF_TYPE_Q4_0 &&
        t.n_dims == 2 &&
        t.dims[0] == 32 &&
        t.dims[1] == 1 &&
        t.n_bytes == 18 &&
        qinfo_s == TC_OK &&
        qinfo.N == 1 &&
        qinfo.K == 32 &&
        qinfo.gguf_type == TC_GGUF_TYPE_Q4_0 &&
        qinfo.quant_type == TC_QUANT_Q4_0 &&
        qinfo.n_bytes == 18 &&
        qinfo.buffer == NULL &&
        got_scale == scale_h && qs_ok;

    printf("  q4_0 data    : scale=0x%04x %s, qs all 0xab %s\n",
           got_scale, (got_scale == scale_h) ? "OK" : "FAIL",
           qs_ok ? "OK" : "FAIL");

    tc_context* ctx = NULL;
    tc_buffer* gb = NULL;
    tc_buffer* xb = NULL;
    tc_buffer* yb = NULL;
    tc_gguf_loaded_model* loaded = NULL;
    int gpu_copy_ok = 0;
    int bulk_load_ok = 0;
    int gguf_gemv_ok = 0;
    s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init: %s\n", tc_status_string(s));
    } else if (tc_gguf_tensor_to_buffer(ctx, g, "weight.test", &gb) != TC_OK) {
        fprintf(stderr, "tc_gguf_tensor_to_buffer failed\n");
    } else {
        uint8_t* gp = NULL;
        if (tc_buffer_map(gb, (void**)&gp) == TC_OK) {
            const uint16_t gpu_scale = *(const uint16_t*)gp;
            int gpu_qs_ok = 1;
            for (int i = 0; i < 16; ++i) {
                if (gp[2 + i] != 0xab) { gpu_qs_ok = 0; break; }
            }
            gpu_copy_ok = (tc_buffer_size(gb) == 18 && gpu_scale == scale_h && gpu_qs_ok);
        }
    }
    if (ctx && tc_gguf_load_supported_tensors(ctx, g, &loaded) == TC_OK) {
        tc_gguf_loaded_tensor_info li;
        tc_gguf_quantized_matrix_info lqinfo;
        uint8_t* lp = NULL;
        if (tc_gguf_loaded_tensor_count(loaded) == 1 &&
            tc_gguf_loaded_skipped_tensor_count(loaded) == 1 &&
            tc_gguf_loaded_get_tensor(loaded, "weight.test", &li) == TC_OK &&
            tc_gguf_loaded_tensor_quantized_matrix_info(&li, &lqinfo) == TC_OK &&
            li.buffer &&
            li.type == TC_GGUF_TYPE_Q4_0 &&
            li.n_bytes == 18 &&
            tc_buffer_map(li.buffer, (void**)&lp) == TC_OK) {
            const uint16_t bulk_scale = *(const uint16_t*)lp;
            int bulk_qs_ok = 1;
            for (int i = 0; i < 16; ++i) {
                if (lp[2 + i] != 0xab) { bulk_qs_ok = 0; break; }
            }
            bulk_load_ok = (bulk_scale == scale_h && bulk_qs_ok &&
                            lqinfo.N == 1 &&
                            lqinfo.K == 32 &&
                            lqinfo.quant_type == TC_QUANT_Q4_0 &&
                            lqinfo.n_bytes == 18 &&
                            lqinfo.buffer == li.buffer);
        }
    }
    if (gpu_copy_ok &&
        tc_buffer_alloc(ctx, 32 * sizeof(uint16_t), &xb) == TC_OK &&
        tc_buffer_alloc(ctx, sizeof(uint16_t), &yb) == TC_OK) {
        uint16_t* xp = NULL;
        uint16_t* yp = NULL;
        if (tc_buffer_map(xb, (void**)&xp) == TC_OK &&
            tc_buffer_map(yb, (void**)&yp) == TC_OK) {
            for (int i = 0; i < 32; ++i) xp[i] = 0x3c00;  /* half(1.0) */
            yp[0] = 0;
            if (tc_gemv_quantized(ctx, xb, gb, yb, TC_QUANT_Q4_0, 1, 1, 32) == TC_OK) {
                /* 0xab -> low q=11, high q=10. Scale=0.5 gives
                 * 16*1.5 + 16*1.0 = 40.0, half bits 0x5100. */
                gguf_gemv_ok = (yp[0] == 0x5100);
            }
        }
    }
    printf("  gpu copy     : %s\n", gpu_copy_ok ? "OK" : "FAIL");
    printf("  bulk load    : %s\n", bulk_load_ok ? "OK" : "FAIL");
    printf("  gguf gemv    : %s\n", gguf_gemv_ok ? "OK" : "FAIL");
    printf("test_gguf: %s\n", (ok && gpu_copy_ok && bulk_load_ok && gguf_gemv_ok) ? "OK" : "FAIL");

    if (yb) tc_buffer_free(ctx, yb);
    if (xb) tc_buffer_free(ctx, xb);
    if (gb) tc_buffer_free(ctx, gb);
    if (loaded) tc_gguf_loaded_model_free(ctx, loaded);
    if (ctx) tc_shutdown(ctx);
    tc_gguf_close(g);
    unlink(path);
    return (ok && gpu_copy_ok && bulk_load_ok && gguf_gemv_ok) ? 0 : 5;
}
