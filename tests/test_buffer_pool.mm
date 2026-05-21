#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tensorcore/tensorcore.h"
#include "internal.h"

static size_t expected_bucket_bytes(size_t bytes) {
    if (bytes < 256) bytes = 256;
    size_t bucket = 256;
    while (bucket < bytes) bucket <<= 1;
    return bucket;
}

static int check_alloc(tc_context* ctx, size_t requested) {
    tc_buffer* b = NULL;
    tc_status_t s = tc_buffer_alloc(ctx, requested, &b);
    if (s != TC_OK || !b) {
        fprintf(stderr, "tc_buffer_alloc(%zu): %s\n", requested, tc_status_string(s));
        return 1;
    }

    const size_t public_size = tc_buffer_size(b);
    const size_t actual_size = (size_t)[b->mtl length];
    const size_t expected_size = expected_bucket_bytes(requested);
    if (public_size != requested || b->bytes != requested) {
        fprintf(stderr, "requested size mismatch: requested=%zu public=%zu internal=%zu\n",
                requested, public_size, b->bytes);
        return 2;
    }
    if (b->bucket_bytes != expected_size || actual_size != expected_size) {
        fprintf(stderr,
                "bucket size mismatch: requested=%zu bucket=%zu actual=%zu expected=%zu\n",
                requested, b->bucket_bytes, actual_size, expected_size);
        return 3;
    }

    void* p = NULL;
    s = tc_buffer_map(b, &p);
    if (s != TC_OK || !p) {
        fprintf(stderr, "tc_buffer_map(%zu): %s\n", requested, tc_status_string(s));
        return 4;
    }
    memset(p, 0xA5, requested);

    s = tc_buffer_free(ctx, b);
    if (s != TC_OK) {
        fprintf(stderr, "tc_buffer_free(%zu): %s\n", requested, tc_status_string(s));
        return 5;
    }
    return 0;
}

static int check_reuse(tc_context* ctx, size_t requested) {
    tc_buffer* a = NULL;
    tc_buffer* b = NULL;
    tc_status_t s = tc_buffer_alloc(ctx, requested, &a);
    if (s != TC_OK || !a) return 1;
    id<MTLBuffer> first = a->mtl;

    s = tc_buffer_free(ctx, a);
    if (s != TC_OK) return 2;

    s = tc_buffer_alloc(ctx, requested, &b);
    if (s != TC_OK || !b) return 3;
    id<MTLBuffer> second = b->mtl;
    s = tc_buffer_free(ctx, b);
    if (s != TC_OK) return 4;

    if (first != second) {
        fprintf(stderr, "buffer pool did not reuse requested=%zu bucket=%zu\n",
                requested, expected_bucket_bytes(requested));
        return 5;
    }
    return 0;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    const size_t sizes[] = {
        1, 2, 255, 256, 257, 511, 512, 513, 1024, 4097, 65536
    };
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int rc = check_alloc(ctx, sizes[i]);
        if (rc) return 10 + rc;
    }
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int rc = check_reuse(ctx, sizes[i]);
        if (rc) return 20 + rc;
    }

    s = tc_shutdown(ctx);
    if (s != TC_OK) {
        fprintf(stderr, "tc_shutdown failed: %s\n", tc_status_string(s));
        return 30;
    }
    return 0;
}
