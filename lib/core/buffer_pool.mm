/*
 * Buffer pool — power-of-2 bucketed MTLBuffer recycling.
 *
 * Apple Silicon is unified memory; MTLResourceStorageModeShared makes the
 * buffer CPU-mappable with no copy. We bucket by ceil_pow2(bytes) and recycle
 * LIFO. Excess buckets (>8 cached) are released back to the device on free.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "internal.h"

#include <cstdio>
#include <new>

static constexpr int  TC_BUCKETS         = 32;   /* 256B through 512GB buckets */
static constexpr int  TC_BUCKET_CAPACITY = 8;
static constexpr size_t TC_MIN_BUCKET    = 256;  /* avoid pool churn on tiny allocs */

static int bucket_for(size_t bytes) {
    if (bytes < TC_MIN_BUCKET) bytes = TC_MIN_BUCKET;
    int b = 0;
    size_t s = TC_MIN_BUCKET;
    while (s < bytes && b < TC_BUCKETS - 1) { s <<= 1; ++b; }
    return b;
}

/* bucket i = TC_MIN_BUCKET * 2^i */
static size_t bytes_for_bucket(int b) {
    return TC_MIN_BUCKET << b;
}

@interface TCBufferPool : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) NSLock* lock;
- (id<MTLBuffer>)allocBucket:(int)b;
- (void)freeBucket:(int)b buffer:(id<MTLBuffer>)buf;
@end

@implementation TCBufferPool {
    NSMutableArray<id<MTLBuffer>>* _free_lists[TC_BUCKETS];
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if ((self = [super init])) {
        _device = device;
        _lock = [[NSLock alloc] init];
        for (int i = 0; i < TC_BUCKETS; ++i) {
            _free_lists[i] = [NSMutableArray arrayWithCapacity:TC_BUCKET_CAPACITY];
        }
    }
    return self;
}

- (id<MTLBuffer>)allocBucket:(int)b {
    [_lock lock];
    id<MTLBuffer> buf = nil;
    NSMutableArray* list = _free_lists[b];
    if ([list count] > 0) {
        buf = [list lastObject];
        [list removeLastObject];
    }
    [_lock unlock];
    if (!buf) {
        size_t bytes = bytes_for_bucket(b);
        buf = [_device newBufferWithLength:bytes
                                   options:MTLResourceStorageModeShared];
    }
    return buf;
}

- (void)freeBucket:(int)b buffer:(id<MTLBuffer>)buf {
    [_lock lock];
    if ([_free_lists[b] count] < TC_BUCKET_CAPACITY) {
        [_free_lists[b] addObject:buf];
    }
    /* If the bucket is full, drop the ref and let ARC release. */
    [_lock unlock];
}

- (void)dealloc {
    for (int i = 0; i < TC_BUCKETS; ++i) _free_lists[i] = nil;
}
@end

extern "C" TCBufferPool* tc_buffer_pool_create(id<MTLDevice> dev) {
    return [[TCBufferPool alloc] initWithDevice:dev];
}

extern "C" void tc_buffer_pool_destroy(TCBufferPool* p) {
    (void)p;  /* ARC */
}

extern "C" tc_status_t tc_buffer_pool_alloc(TCBufferPool* p, size_t bytes,
                                            struct tc_buffer** out) {
    if (!p || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    int b = bucket_for(bytes);
    id<MTLBuffer> mtl = [p allocBucket:b];
    if (!mtl) return TC_ERR_ALLOC;

    struct tc_buffer* tb = new (std::nothrow) tc_buffer();
    if (!tb) return TC_ERR_ALLOC;
    tb->mtl          = mtl;
    tb->bytes        = bytes;
    tb->bucket_bytes = bytes_for_bucket(b);
    tb->owner        = nullptr;   /* set by tc_buffer_alloc wrapper */
    *out = tb;
    return TC_OK;
}

extern "C" void tc_buffer_pool_free(TCBufferPool* p, struct tc_buffer* buf) {
    if (!p || !buf) return;
    int b = bucket_for(buf->bucket_bytes);
    [p freeBucket:b buffer:buf->mtl];
    buf->mtl = nil;
    delete buf;
}
