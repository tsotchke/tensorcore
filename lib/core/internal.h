/* Internal-only header — do not install. */
#ifndef TC_INTERNAL_H
#define TC_INTERNAL_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
#include <atomic>
#endif
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"
#include "tensorcore/gemm.h"

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- *
 * Opaque struct definitions visible only inside the library.    *
 * ------------------------------------------------------------- */

#ifdef __OBJC__

@interface TCPipelineCache : NSObject
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
@property (nonatomic, strong) NSLock* lock;
- (id<MTLComputePipelineState>)pipelineForName:(NSString*)name error:(NSError**)outErr;
@end

@class TCBufferPool;

struct tc_context {
    id<MTLDevice>          device;
    id<MTLCommandQueue>    queue;
    id<MTLLibrary>         library;
    TCPipelineCache*       pipelines;
    TCBufferPool*          buffer_pool;
    tc_device_info         info;
    std::atomic<int>       ref;
};

struct tc_buffer {
    id<MTLBuffer>          mtl;
    size_t                 bytes;
    size_t                 bucket_bytes;   /* 0 if not pool-owned       */
    struct tc_context*     owner;
};

struct tc_stream {
    /* Streams are encoded as per-stream MTLCommandQueue at first.
     * The default stream (NULL) routes to ctx->queue.                  */
    id<MTLCommandQueue>    queue;
    id<MTLCommandBuffer>   pending_cmd;
    struct tc_context*     owner;
};

#else
struct tc_context;
struct tc_buffer;
struct tc_stream;
#endif

/* ------------------------------------------------------------- *
 * Backend-selection helpers used by ops/.                       *
 * ------------------------------------------------------------- */
typedef enum {
    TC_KIND_SIMDGROUP = 0,
    TC_KIND_MPS       = 1,
    TC_KIND_TENSOROPS = 2,
    TC_KIND_ACCELERATE_CPU = 3,
} tc_kernel_kind_t;

/* Sets the diagnostic per-thread `tc_last_backend`. */
TC_INTERNAL_SYMBOL void tc_set_last_backend(tc_backend_t b);

/* Records a completed public dispatch, updates `tc_last_backend` on success,
 * and emits a one-line stderr trace when TC_TRACE=1 is set. */
TC_INTERNAL_SYMBOL tc_status_t tc_record_dispatch(const char* op,
                                                  tc_backend_t backend,
                                                  tc_status_t status);

/* Validates that a public buffer handle belongs to ctx and has at least
 * min_bytes requested bytes available. */
TC_INTERNAL_SYMBOL tc_status_t tc_buffer_validate(struct tc_context* ctx,
                                                  const struct tc_buffer* buf,
                                                  size_t min_bytes);

/* Activation checkpointing storage primitives. tc_buffer_discard_storage
 * frees the buffer's underlying memory but keeps the handle valid (size,
 * owner, storage class are remembered). tc_buffer_reallocate_storage
 * allocates fresh storage of the original size + class so the handle is
 * usable again. Returns TC_OK iff successful.
 *
 * Between discard and realloc, tc_buffer_map returns TC_ERR_INVALID_ARG.
 *
 * For the Metal build (lib/core/buffer_pool.mm), these are wired in a
 * follow-up; the CPU build implements them directly in device_cpu.cpp.
 *
 * Used by lib/core/checkpoint_stub.cpp to actually reclaim memory on
 * tc_checkpoint_discard rather than just toggling a flag. */
TC_INTERNAL_SYMBOL tc_status_t tc_buffer_discard_storage(struct tc_buffer* buf);
TC_INTERNAL_SYMBOL tc_status_t tc_buffer_reallocate_storage(struct tc_buffer* buf);
TC_INTERNAL_SYMBOL int tc_buffer_is_discarded(const struct tc_buffer* buf);

#ifdef __OBJC__
/* Returns a cached MTLComputePipelineState for `name`. NULL on failure. */
id<MTLComputePipelineState> tc_pipeline_get(struct tc_context* ctx,
                                            NSString* name,
                                            tc_status_t* out_err);
/* Returns a pending command buffer for batched async stream encoding. */
id<MTLCommandBuffer> tc_stream_command_buffer(struct tc_stream* s);
/* Returns the device family classifier this build saw at init. */
tc_family_t tc_device_family_from_mtl(id<MTLDevice> dev);
#endif

#ifdef __cplusplus
}
#endif
#endif /* TC_INTERNAL_H */
