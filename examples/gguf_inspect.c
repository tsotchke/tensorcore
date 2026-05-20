/*
 * Inspect a GGUF file with tensorcore's minimal reader.
 *
 * Usage:
 *   gguf_inspect model.gguf [tensor-name-to-copy]
 *   gguf_inspect model.gguf --load-supported
 *
 * The optional tensor argument copies that tensor into a tensorcore buffer.
 * --load-supported copies every tensor with a supported storage type.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorcore/tensorcore.h"

static const char* gguf_type_name(tc_gguf_type_t type) {
    switch (type) {
        case TC_GGUF_TYPE_F32:  return "F32";
        case TC_GGUF_TYPE_F16:  return "F16";
        case TC_GGUF_TYPE_Q4_0: return "Q4_0";
        case TC_GGUF_TYPE_Q4_1: return "Q4_1";
        case TC_GGUF_TYPE_Q8_0: return "Q8_0";
        case TC_GGUF_TYPE_BF16: return "BF16";
        default:                return "unsupported";
    }
}

static void print_i64_meta(const tc_gguf_file* gguf, const char* key) {
    const int64_t missing = INT64_MIN;
    const int64_t value = tc_gguf_meta_get_i64(gguf, key, missing);
    if (value != missing) {
        printf("  %-28s %lld\n", key, (long long)value);
    }
}

static void print_f64_meta(const tc_gguf_file* gguf, const char* key) {
    const double missing = -1.23456789012345e300;
    const double value = tc_gguf_meta_get_f64(gguf, key, missing);
    if (value != missing) {
        printf("  %-28s %.9g\n", key, value);
    }
}

static void print_str_meta(const tc_gguf_file* gguf, const char* key) {
    const char* value = tc_gguf_meta_get_str(gguf, key);
    if (value) {
        printf("  %-28s %s\n", key, value);
    }
}

static void print_dims(const tc_gguf_tensor_info* tensor) {
    printf("[");
    for (int32_t i = 0; i < tensor->n_dims; ++i) {
        if (i) printf(" x ");
        printf("%llu", (unsigned long long)tensor->dims[i]);
    }
    printf("]");
}

static void print_tensor(const tc_gguf_tensor_info* tensor, uint64_t index) {
    printf("  %4llu  %-8s  %12zu  off=%12llu  ",
           (unsigned long long)index,
           gguf_type_name(tensor->type),
           tensor->n_bytes,
           (unsigned long long)tensor->offset);
    print_dims(tensor);
    printf("  %s\n", tensor->name);
}

static int copy_tensor_to_buffer(const char* name, const tc_gguf_file* gguf) {
    tc_context* ctx = NULL;
    tc_buffer* buffer = NULL;

    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    s = tc_gguf_tensor_to_buffer(ctx, gguf, name, &buffer);
    if (s != TC_OK) {
        fprintf(stderr, "copy tensor '%s' failed: %s\n", name, tc_status_string(s));
        if (ctx) tc_shutdown(ctx);
        return 1;
    }

    printf("\nCopied '%s' to tensorcore buffer: %zu bytes\n",
           name, tc_buffer_size(buffer));

    tc_buffer_free(ctx, buffer);
    if (ctx) tc_shutdown(ctx);
    return 0;
}

static int load_supported_tensors(const tc_gguf_file* gguf) {
    tc_context* ctx = NULL;
    tc_gguf_loaded_model* model = NULL;

    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    s = tc_gguf_load_supported_tensors(ctx, gguf, &model);
    if (s != TC_OK) {
        fprintf(stderr, "load supported tensors failed: %s\n", tc_status_string(s));
        if (ctx) tc_shutdown(ctx);
        return 1;
    }

    const uint64_t count = tc_gguf_loaded_tensor_count(model);
    const uint64_t skipped = tc_gguf_loaded_skipped_tensor_count(model);
    uint64_t loaded_bytes = 0;
    for (uint64_t i = 0; i < count; ++i) {
        tc_gguf_loaded_tensor_info info;
        if (tc_gguf_loaded_tensor_at(model, i, &info) == TC_OK) {
            loaded_bytes += (uint64_t)info.n_bytes;
        }
    }

    printf("\nLoaded supported tensors: %llu tensors, %llu bytes, %llu skipped\n",
           (unsigned long long)count,
           (unsigned long long)loaded_bytes,
           (unsigned long long)skipped);

    tc_gguf_loaded_model_free(ctx, model);
    if (ctx) tc_shutdown(ctx);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s model.gguf [tensor-name-to-copy]\n", argv[0]);
        return 2;
    }

    const char* path = argv[1];
    const int load_all = (argc == 3 && strcmp(argv[2], "--load-supported") == 0);
    const char* copy_name = (argc == 3 && !load_all) ? argv[2] : NULL;

    tc_gguf_file* gguf = NULL;
    tc_status_t s = tc_gguf_open(path, &gguf);
    if (s != TC_OK) {
        fprintf(stderr, "tc_gguf_open('%s') failed: %s\n", path, tc_status_string(s));
        return 1;
    }

    const uint64_t tensor_count = tc_gguf_tensor_count(gguf);
    const uint64_t metadata_count = tc_gguf_metadata_count(gguf);
    printf("GGUF: %s\n", path);
    printf("  tensors                    %llu\n", (unsigned long long)tensor_count);
    printf("  metadata                   %llu\n", (unsigned long long)metadata_count);

    printf("\nMetadata\n");
    print_str_meta(gguf, "general.architecture");
    print_str_meta(gguf, "general.name");
    print_i64_meta(gguf, "general.file_type");
    print_i64_meta(gguf, "general.quantization_version");
    print_i64_meta(gguf, "llama.context_length");
    print_i64_meta(gguf, "llama.embedding_length");
    print_i64_meta(gguf, "llama.block_count");
    print_i64_meta(gguf, "llama.attention.head_count");
    print_i64_meta(gguf, "llama.attention.head_count_kv");
    print_i64_meta(gguf, "llama.rope.dimension_count");
    print_f64_meta(gguf, "llama.attention.layer_norm_rms_epsilon");
    print_f64_meta(gguf, "llama.rope.freq_base");
    print_f64_meta(gguf, "llama.rope.freq_scale");
    print_str_meta(gguf, "tokenizer.ggml.model");
    const uint64_t token_count = tc_gguf_meta_array_count(gguf, "tokenizer.ggml.tokens");
    if (token_count) {
        printf("  %-28s %llu\n", "tokenizer.ggml.tokens",
               (unsigned long long)token_count);
        const char* token0 = NULL;
        size_t token0_len = 0;
        if (tc_gguf_meta_array_get_str(gguf, "tokenizer.ggml.tokens", 0,
                                       &token0, &token0_len) == TC_OK) {
            printf("  %-28s %.*s\n", "tokenizer.ggml.tokens[0]",
                   (int)token0_len, token0 ? token0 : "");
        }
    }

    const uint64_t limit = tensor_count < 16 ? tensor_count : 16;
    uint64_t total_bytes = 0;

    printf("\nTensors");
    if (tensor_count > limit) {
        printf(" (first %llu)", (unsigned long long)limit);
    }
    printf("\n");

    for (uint64_t i = 0; i < tensor_count; ++i) {
        tc_gguf_tensor_info tensor;
        s = tc_gguf_tensor_at(gguf, i, &tensor);
        if (s != TC_OK) {
            fprintf(stderr, "tc_gguf_tensor_at(%llu) failed: %s\n",
                    (unsigned long long)i, tc_status_string(s));
            tc_gguf_close(gguf);
            return 1;
        }

        total_bytes += (uint64_t)tensor.n_bytes;
        if (i < limit) {
            print_tensor(&tensor, i);
        }
    }

    if (tensor_count > limit) {
        printf("  ... %llu more tensors\n",
               (unsigned long long)(tensor_count - limit));
    }
    printf("\nTotal tensor bytes: %llu\n", (unsigned long long)total_bytes);

    int rc = 0;
    if (load_all) {
        rc = load_supported_tensors(gguf);
    } else if (copy_name) {
        rc = copy_tensor_to_buffer(copy_name, gguf);
    }

    tc_gguf_close(gguf);
    return rc;
}
