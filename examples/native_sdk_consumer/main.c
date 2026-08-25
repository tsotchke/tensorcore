#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

#define TC_STR2(x) #x
#define TC_STR(x) TC_STR2(x)

static int check_public_helpers(void) {
    const char* expected =
        "tensorcore " TC_STR(TENSORCORE_VERSION_MAJOR)
        "." TC_STR(TENSORCORE_VERSION_MINOR)
        "." TC_STR(TENSORCORE_VERSION_PATCH);
    if (strcmp(tc_version(), expected) != 0) {
        fprintf(stderr, "unexpected version: %s\n", tc_version());
        return 1;
    }
    if (tc_dtype_size(TC_DTYPE_BF16) != 2 ||
        strcmp(tc_dtype_name(TC_DTYPE_FP53), "fp53") != 0 ||
        strcmp(tc_backend_name(TC_BACKEND_ACCELERATE_CPU), "accelerate_cpu") != 0) {
        fprintf(stderr, "public helper check failed\n");
        return 1;
    }

    tc_gguf_tensor_info t = {0};
    t.n_dims = 2;
    t.dims[0] = 32;
    t.dims[1] = 1;
    t.type = TC_GGUF_TYPE_Q4_0;
    t.n_bytes = tc_quantized_size(TC_QUANT_Q4_0, 1, 32);

    tc_gguf_quantized_matrix_info q = {0};
    tc_status_t s = tc_gguf_tensor_quantized_matrix_info(&t, &q);
    if (s != TC_OK || q.N != 1 || q.K != 32 || q.quant_type != TC_QUANT_Q4_0) {
        fprintf(stderr, "GGUF quantized descriptor check failed: %s\n", tc_status_string(s));
        return 1;
    }
    return 0;
}

static int maybe_run_init_smoke(void) {
    const char* run = getenv("TC_CONSUMER_RUN_INIT");
    if (!run || strcmp(run, "1") != 0) return 0;

    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    tc_device_info info;
    memset(&info, 0, sizeof(info));
    tc_device_info_get(ctx, &info);
    printf("init=%s device=%s family=Apple%d\n",
           tc_status_string(s), info.name, (int)info.family);
    tc_shutdown(ctx);
    return 0;
}

int main(void) {
    if (check_public_helpers() != 0) return 1;
    if (maybe_run_init_smoke() != 0) return 1;
    printf("%s\n", tc_version());
    return 0;
}
