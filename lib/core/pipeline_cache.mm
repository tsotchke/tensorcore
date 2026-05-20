/*
 * Pipeline cache — kernel name -> MTLComputePipelineState, computed lazily.
 *
 * Threadsafe via NSLock; first miss compiles, subsequent calls hit O(1).
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "internal.h"

#include <cstdio>

@implementation TCPipelineCache
- (instancetype)initWithLibrary:(id<MTLLibrary>)library {
    if ((self = [super init])) {
        _library    = library;
        _pipelines  = [NSMutableDictionary dictionaryWithCapacity:32];
        _lock       = [[NSLock alloc] init];
    }
    return self;
}

- (id<MTLComputePipelineState>)pipelineForName:(NSString*)name error:(NSError**)outErr {
    [_lock lock];
    id<MTLComputePipelineState> pso = _pipelines[name];
    [_lock unlock];
    if (pso) return pso;

    id<MTLFunction> fn = [_library newFunctionWithName:name];
    if (!fn) {
        if (outErr) {
            *outErr = [NSError errorWithDomain:@"tensorcore"
                                          code:1
                                      userInfo:@{ NSLocalizedDescriptionKey:
                                          [NSString stringWithFormat:@"kernel '%@' not in metallib", name] }];
        }
        return nil;
    }

    NSError* err = nil;
    pso = [[_library device] newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        if (outErr) *outErr = err;
        return nil;
    }

    [_lock lock];
    if (!_pipelines[name]) _pipelines[name] = pso;
    else                   pso = _pipelines[name];   /* race: keep first writer's */
    [_lock unlock];
    return pso;
}
@end

extern "C" TCPipelineCache* tc_pipeline_cache_create(id<MTLLibrary> lib) {
    return [[TCPipelineCache alloc] initWithLibrary:lib];
}

extern "C" void tc_pipeline_cache_destroy(TCPipelineCache* c) {
    (void)c;  /* ARC handles release when context drops the ref */
}

extern "C" id<MTLComputePipelineState> tc_pipeline_get(struct tc_context* ctx,
                                                       NSString* name,
                                                       tc_status_t* out_err) {
    if (!ctx || !ctx->pipelines) {
        if (out_err) *out_err = TC_ERR_NOT_INITIALIZED;
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [(TCPipelineCache*)ctx->pipelines pipelineForName:name error:&err];
    if (!pso) {
        if (err) {
            fprintf(stderr, "[tensorcore] pipeline '%s' failed: %s\n",
                    [name UTF8String], [[err localizedDescription] UTF8String]);
        }
        if (out_err) *out_err = TC_ERR_PIPELINE;
    } else if (out_err) {
        *out_err = TC_OK;
    }
    return pso;
}
