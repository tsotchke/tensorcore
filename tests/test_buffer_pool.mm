#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

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

static int check_from_ptr(tc_context* ctx) {
    const long page_l = sysconf(_SC_PAGESIZE);
    if (page_l <= 0) {
        fprintf(stderr, "sysconf(_SC_PAGESIZE) failed\n");
        return 1;
    }
    const size_t page = (size_t)page_l;

    void* external = NULL;
    if (posix_memalign(&external, page, page * 2) != 0 || !external) {
        fprintf(stderr, "posix_memalign failed\n");
        return 2;
    }
    memset(external, 0x3C, page * 2);

    tc_buffer* b = NULL;
    tc_status_t s = tc_buffer_from_ptr(ctx, external, page * 2, &b);
    if (s != TC_OK || !b) {
        fprintf(stderr, "tc_buffer_from_ptr: %s\n", tc_status_string(s));
        free(external);
        return 3;
    }
    if (tc_buffer_size(b) != page * 2 || b->bytes != page * 2 ||
        b->bucket_bytes != 0 || b->owner != ctx || b->owns_buffer) {
        fprintf(stderr, "wrapped buffer metadata mismatch\n");
        tc_buffer_free(ctx, b);
        free(external);
        return 4;
    }
    if ((size_t)[b->mtl length] != page * 2) {
        fprintf(stderr, "wrapped buffer length mismatch\n");
        tc_buffer_free(ctx, b);
        free(external);
        return 5;
    }

    void* mapped = NULL;
    s = tc_buffer_map(b, &mapped);
    if (s != TC_OK || mapped != external) {
        fprintf(stderr, "wrapped buffer map mismatch: %s\n", tc_status_string(s));
        tc_buffer_free(ctx, b);
        free(external);
        return 6;
    }
    ((uint8_t*)mapped)[17] = 0xA7;
    if (((uint8_t*)external)[17] != 0xA7) {
        fprintf(stderr, "wrapped buffer did not alias external storage\n");
        tc_buffer_free(ctx, b);
        free(external);
        return 7;
    }

    s = tc_buffer_validate(ctx, b, page * 2);
    if (s != TC_OK) {
        fprintf(stderr, "wrapped buffer validate returned %s\n", tc_status_string(s));
        tc_buffer_free(ctx, b);
        free(external);
        return 8;
    }

    s = tc_buffer_free(ctx, b);
    if (s != TC_OK) {
        fprintf(stderr, "wrapped buffer free: %s\n", tc_status_string(s));
        free(external);
        return 9;
    }
    if (((uint8_t*)external)[17] != 0xA7) {
        fprintf(stderr, "wrapped buffer free mutated external storage\n");
        free(external);
        return 10;
    }

    tc_buffer* bad = (tc_buffer*)0x1;
    s = tc_buffer_from_ptr(ctx, (void*)((uintptr_t)external + 1), page, &bad);
    if (s != TC_ERR_INVALID_ARG || bad != NULL) {
        fprintf(stderr, "unaligned from_ptr: status=%s bad=%p\n",
                tc_status_string(s), (void*)bad);
        free(external);
        return 11;
    }
    bad = (tc_buffer*)0x1;
    s = tc_buffer_from_ptr(ctx, external, page - 1, &bad);
    if (s != TC_ERR_INVALID_ARG || bad != NULL) {
        fprintf(stderr, "non-page-multiple from_ptr: status=%s bad=%p\n",
                tc_status_string(s), (void*)bad);
        free(external);
        return 12;
    }

    ((uint8_t*)external)[0] = 0x5A;  /* wrapper must not own external memory */
    free(external);
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
    {
        int rc = check_from_ptr(ctx);
        if (rc) return 40 + rc;
    }

    s = tc_shutdown(ctx);
    if (s != TC_OK) {
        fprintf(stderr, "tc_shutdown failed: %s\n", tc_status_string(s));
        return 30;
    }
    return 0;
}
