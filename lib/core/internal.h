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
void tc_set_last_backend(tc_backend_t b);

#ifdef __OBJC__
/* Returns a cached MTLComputePipelineState for `name`. NULL on failure. */
id<MTLComputePipelineState> tc_pipeline_get(struct tc_context* ctx,
                                            NSString* name,
                                            tc_status_t* out_err);
/* Returns the device family classifier this build saw at init. */
tc_family_t tc_device_family_from_mtl(id<MTLDevice> dev);
#endif

#ifdef __cplusplus
}
#endif
#endif /* TC_INTERNAL_H */
