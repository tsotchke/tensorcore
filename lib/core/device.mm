/*
 * tensorcore — device init, info, and global plumbing.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "internal.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <new>

#ifndef MTLGPUFamilyMetal4
#define MTLGPUFamilyMetal4 ((MTLGPUFamily)5002)
#endif

/* Forward — implemented in pipeline_cache.mm / buffer_pool.mm. */
@class TCPipelineCache;
@class TCBufferPool;
extern "C" TCPipelineCache* tc_pipeline_cache_create(id<MTLLibrary> lib);
extern "C" void              tc_pipeline_cache_destroy(TCPipelineCache* c);
extern "C" TCBufferPool*     tc_buffer_pool_create(id<MTLDevice> dev);
extern "C" void              tc_buffer_pool_destroy(TCBufferPool* p);
extern "C" tc_status_t       tc_buffer_pool_alloc(TCBufferPool* p, size_t bytes, struct tc_buffer** out);
extern "C" void              tc_buffer_pool_free (TCBufferPool* p, struct tc_buffer* buf);

/* Autotune (lib/core/autotune.cpp). Called from tc_init when TC_AUTOTUNE=1. */
extern "C" tc_status_t tc_autotune_load_cache(const char* device_name, char* out_json, size_t cap);
extern "C" tc_status_t tc_autotune_save_cache(const char* device_name, const char* json);
extern "C" tc_status_t tc_autotune_run_sweep(tc_context* ctx, char* out_json, size_t cap);

/* ----------------------------------------------------------------- */
/* Singleton context — only one MTLDevice per process for now.        */
/* ----------------------------------------------------------------- */
static struct tc_context* g_ctx = nullptr;
static std::mutex         g_ctx_mutex;

/* Thread-local diagnostic for tc_last_backend(). */
static thread_local tc_backend_t t_last_backend = TC_BACKEND_NONE;

extern "C" void tc_set_last_backend(tc_backend_t b) { t_last_backend = b; }
extern "C" tc_backend_t tc_last_backend(void)       { return t_last_backend; }

extern "C" const char* tc_backend_name(tc_backend_t b) {
    switch (b) {
        case TC_BACKEND_NONE:             return "none";
        case TC_BACKEND_SIMDGROUP_MATRIX: return "simdgroup_matrix";
        case TC_BACKEND_TENSOROPS_M5:     return "tensorops_m5";
        case TC_BACKEND_MPS:              return "mps";
        case TC_BACKEND_ACCELERATE_CPU:   return "accelerate_cpu";
        case TC_BACKEND_SF64_EMULATED:    return "sf64_emulated";
        case TC_BACKEND_OZAKI_II:         return "ozaki_ii";
        case TC_BACKEND_PORTABLE_CPU:     return "portable_cpu";
        case TC_BACKEND_METAL_COMPUTE:    return "metal_compute";
        case TC_BACKEND_CUDA:             return "cuda";
    }
    return "?";
}

#define TC_STRINGIFY_VERSION2(x) #x
#define TC_STRINGIFY_VERSION(x) TC_STRINGIFY_VERSION2(x)

extern "C" const char* tc_version(void) {
    return "tensorcore "
        TC_STRINGIFY_VERSION(TENSORCORE_VERSION_MAJOR) "."
        TC_STRINGIFY_VERSION(TENSORCORE_VERSION_MINOR) "."
        TC_STRINGIFY_VERSION(TENSORCORE_VERSION_PATCH);
}

#undef TC_STRINGIFY_VERSION
#undef TC_STRINGIFY_VERSION2

/* ----------------------------------------------------------------- */
/* Family detection — runtime probe of MTLGPUFamilyApple7..Apple11.   */
/* ----------------------------------------------------------------- */
extern "C" tc_family_t tc_device_family_from_mtl(id<MTLDevice> dev) {
    tc_family_t fam = TC_FAMILY_UNKNOWN;

    /* Probe high-to-low so we report the highest supported family.
     * Apple11 / MTLGPUFamilyMetal4 are macOS 26+ ; weak-link by branch on the
     * symbol availability. */
    if (@available(macOS 12.0, *)) {
        if ([dev supportsFamily:MTLGPUFamilyApple7]) fam = TC_FAMILY_APPLE7;
    }
    if (@available(macOS 13.0, *)) {
        if ([dev supportsFamily:MTLGPUFamilyApple8]) fam = TC_FAMILY_APPLE8;
    }
    if (@available(macOS 14.0, *)) {
        if ([dev supportsFamily:MTLGPUFamilyApple9]) fam = TC_FAMILY_APPLE9;
    }
#ifdef MTLGPUFamilyApple10
    if (@available(macOS 15.0, *)) {
        if ([dev supportsFamily:(MTLGPUFamily)MTLGPUFamilyApple10]) fam = TC_FAMILY_APPLE10;
    }
#endif
#ifdef MTLGPUFamilyApple11
    if (@available(macOS 26.0, *)) {
        if ([dev supportsFamily:(MTLGPUFamily)MTLGPUFamilyApple11]) fam = TC_FAMILY_APPLE11;
    }
#endif
    return fam;
}

static bool tc_device_supports_tensorops_m5(id<MTLDevice> dev) {
    if (!dev) return false;

    BOOL metal4 = NO;
    @try {
        metal4 = [dev supportsFamily:MTLGPUFamilyMetal4];
    } @catch (...) {
        metal4 = NO;
    }
    if (!metal4) return false;

    NSString* name = [dev name];
    if (!name) return false;
    return [name containsString:@"M5"] ||
           [name containsString:@"M6"] ||
           [name containsString:@"M7"];
}

/* ----------------------------------------------------------------- */
/* MTLLibrary load: env override, installed library/executable-relative paths,
 * build-tree path baked by CMake, then a local default.metallib fallback. */
/* ----------------------------------------------------------------- */
static id<MTLLibrary> load_metallib(id<MTLDevice> dev) {
    NSError* err = nil;

    const char* env = std::getenv("TC_METALLIB");
    NSString* candidates[8] = { nil, nil, nil, nil, nil, nil, nil, nil };
    int n = 0;
    if (env && *env) {
        candidates[n++] = [NSString stringWithUTF8String:env];
    }

    Dl_info dylib_info;
    if (dladdr((const void*)&tc_init, &dylib_info) && dylib_info.dli_fname) {
        NSString* dir = [[NSString stringWithUTF8String:dylib_info.dli_fname]
                            stringByDeletingLastPathComponent];
        candidates[n++] = [dir stringByAppendingPathComponent:@"tensorcore.metallib"];
        candidates[n++] = [dir stringByAppendingPathComponent:@"default.metallib"];
        candidates[n++] = [[dir stringByAppendingPathComponent:@"../lib/tensorcore.metallib"]
                            stringByStandardizingPath];
    }
#ifdef TC_METALLIB_PATH
    candidates[n++] = [NSString stringWithUTF8String:TC_METALLIB_PATH];
#endif
    candidates[n++] = @"default.metallib";

    for (int i = 0; i < n; ++i) {
        if (!candidates[i]) continue;
        NSURL* url = [NSURL fileURLWithPath:candidates[i]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (lib) {
            fprintf(stderr, "[tensorcore] loaded metallib: %s\n",
                    [candidates[i] UTF8String]);
            return lib;
        }
    }
    fprintf(stderr,
        "[tensorcore] failed to load metallib (TC_METALLIB_PATH=%s): %s\n",
#ifdef TC_METALLIB_PATH
        TC_METALLIB_PATH,
#else
        "(unset)",
#endif
        err ? [[err localizedDescription] UTF8String] : "(no error)");
    return nil;
}

/* ----------------------------------------------------------------- */
/* tc_init / tc_shutdown                                              */
/* ----------------------------------------------------------------- */
extern "C" tc_status_t tc_init(tc_context** out_ctx) {
    if (!out_ctx) return TC_ERR_INVALID_ARG;

    bool created = false;
    {
        std::lock_guard<std::mutex> lock(g_ctx_mutex);
        if (g_ctx) {
            g_ctx->ref.fetch_add(1);
            *out_ctx = g_ctx;
            return TC_ERR_ALREADY_INITIALIZED;
        }

        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (!dev) return TC_ERR_NO_DEVICE;

            id<MTLCommandQueue> q = [dev newCommandQueue];
            if (!q) return TC_ERR_NO_DEVICE;

            id<MTLLibrary> lib = load_metallib(dev);
            if (!lib) return TC_ERR_KERNEL_NOT_FOUND;

            TCPipelineCache* pcache = tc_pipeline_cache_create(lib);
            TCBufferPool*    bpool  = tc_buffer_pool_create(dev);
            if (!pcache || !bpool) {
                if (pcache) tc_pipeline_cache_destroy(pcache);
                if (bpool) tc_buffer_pool_destroy(bpool);
                return TC_ERR_ALLOC;
            }

            tc_context* ctx = new (std::nothrow) tc_context();
            if (!ctx) {
                tc_pipeline_cache_destroy(pcache);
                tc_buffer_pool_destroy(bpool);
                return TC_ERR_ALLOC;
            }

            ctx->device      = dev;
            ctx->queue       = q;
            ctx->library     = lib;
            ctx->pipelines   = pcache;
            ctx->buffer_pool = bpool;
            ctx->ref.store(1);

            tc_family_t fam = tc_device_family_from_mtl(dev);
            ctx->info.family                = fam;
            const char* dev_name = [[dev name] UTF8String];
            std::strncpy(ctx->info.name, dev_name ? dev_name : "", sizeof(ctx->info.name) - 1);
            ctx->info.max_buffer_bytes              = (uint64_t)[dev maxBufferLength];
            ctx->info.recommended_working_set_bytes = (uint64_t)[dev recommendedMaxWorkingSetSize];
            ctx->info.max_threadgroup_memory        = (uint32_t)[dev maxThreadgroupMemoryLength];
            ctx->info.max_threads_per_threadgroup   = 1024;
            ctx->info.thread_execution_width        = 32;  /* All Apple Silicon */
            ctx->info.unified_memory                = [dev hasUnifiedMemory];
            ctx->info.supports_bf16_simdgroup       = (fam >= TC_FAMILY_APPLE9);
            ctx->info.supports_i8_simdgroup         = (fam >= TC_FAMILY_APPLE10);
            ctx->info.supports_tensorops_m5         = tc_device_supports_tensorops_m5(dev);
            ctx->info.supports_fp64_native          = false;

            fprintf(stderr,
                "[tensorcore] device=\"%s\" family=Apple%d unified=%s vram=%lluMB "
                "bf16_sg=%s i8_sg=%s tensorops_m5=%s\n",
                ctx->info.name, (int)fam,
                ctx->info.unified_memory ? "yes" : "no",
                (unsigned long long)(ctx->info.recommended_working_set_bytes / (1024 * 1024)),
                ctx->info.supports_bf16_simdgroup ? "yes" : "no",
                ctx->info.supports_i8_simdgroup   ? "yes" : "no",
                ctx->info.supports_tensorops_m5   ? "yes" : "no");

            g_ctx = ctx;
            *out_ctx = g_ctx;
            created = true;
        }
    }

    /* Bench-driven autotune: TC_AUTOTUNE=1 to trigger sweep; otherwise load
     * cached config if present. The cache key is device name. */
    if (created) {
        if (const char* at = std::getenv("TC_AUTOTUNE")) {
            if (at[0] == '1') {
                char json[1024] = {0};
                tc_status_t s = tc_autotune_load_cache((*out_ctx)->info.name, json, sizeof(json));
                if (s == TC_OK && json[0]) {
                    fprintf(stderr, "[tensorcore] autotune: loaded cached config\n");
                } else {
                    fprintf(stderr, "[tensorcore] autotune: running sweep (one-time)\n");
                    s = tc_autotune_run_sweep(*out_ctx, json, sizeof(json));
                    if (s == TC_OK && json[0]) {
                        tc_autotune_save_cache((*out_ctx)->info.name, json);
                        fprintf(stderr, "[tensorcore] autotune: cached → %s\n", json);
                    }
                }
            }
        }
    }
    return TC_OK;
}

extern "C" tc_status_t tc_shutdown(tc_context* ctx) {
    if (!ctx) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(g_ctx_mutex);

    if (ctx != g_ctx) return TC_ERR_INVALID_ARG;
    if (ctx->ref.fetch_sub(1) > 1) return TC_OK;

    tc_pipeline_cache_destroy(ctx->pipelines);
    tc_buffer_pool_destroy(ctx->buffer_pool);
    /* ARC releases device/queue/library when ctx is destroyed. */
    ctx->device = nil;
    ctx->queue = nil;
    ctx->library = nil;
    g_ctx = nullptr;
    delete ctx;
    return TC_OK;
}

extern "C" tc_status_t tc_device_info_get(tc_context* ctx, tc_device_info* out_info) {
    if (!ctx || !out_info) return TC_ERR_INVALID_ARG;
    *out_info = ctx->info;
    return TC_OK;
}

/* ----------------------------------------------------------------- */
/* Buffer surface — thin wrapper over buffer_pool.                    */
/* ----------------------------------------------------------------- */
extern "C" tc_status_t tc_buffer_alloc(tc_context* ctx, size_t bytes, tc_buffer** out) {
    if (!ctx || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    tc_status_t s = tc_buffer_pool_alloc(ctx->buffer_pool, bytes, out);
    if (s == TC_OK && *out) (*out)->owner = ctx;
    return s;
}

extern "C" tc_status_t tc_buffer_from_ptr(tc_context* ctx, void* ptr,
                                          size_t bytes, tc_buffer** out) {
    if (!ctx || !ptr || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    *out = nullptr;

    const size_t page = (size_t)NSPageSize();
    const uintptr_t addr = (uintptr_t)ptr;
    if (page == 0 || (addr % page) != 0 || (bytes % page) != 0) {
        return TC_ERR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> mtl =
            [ctx->device newBufferWithBytesNoCopy:ptr
                                           length:bytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        if (!mtl) return TC_ERR_ALLOC;

        tc_buffer* buf = new (std::nothrow) tc_buffer();
        if (!buf) return TC_ERR_ALLOC;
        buf->mtl = mtl;
        buf->bytes = bytes;
        buf->bucket_bytes = 0;
        buf->owner = ctx;
        buf->owns_buffer = false;
        *out = buf;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_buffer_validate(tc_context* ctx,
                                          const tc_buffer* buf,
                                          size_t min_bytes) {
    if (!ctx || !buf || !buf->mtl) return TC_ERR_INVALID_ARG;
    if (buf->owner != ctx) return TC_ERR_INVALID_ARG;
    if (min_bytes > buf->bytes) return TC_ERR_INVALID_SHAPE;
    return TC_OK;
}

extern "C" tc_status_t tc_buffer_free(tc_context* ctx, tc_buffer* buf) {
    if (!ctx || !buf || buf->owner != ctx) return TC_ERR_INVALID_ARG;
    if (buf->mtl && buf->owns_buffer) {
        tc_buffer_pool_free(ctx->buffer_pool, buf);
    } else {
        buf->mtl = nil;
        delete buf;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_buffer_map(tc_buffer* buf, void** out_ptr) {
    if (!buf || !out_ptr || !buf->mtl) return TC_ERR_INVALID_ARG;
    *out_ptr = [buf->mtl contents];
    if (!*out_ptr) return TC_ERR_INTERNAL;
    return TC_OK;
}

extern "C" size_t tc_buffer_size(const tc_buffer* buf) {
    return buf ? buf->bytes : 0;
}

/* Activation-checkpointing storage primitives for the Metal build.
 *
 * The CPU backend can discard storage while keeping the tc_buffer handle
 * alive. The current Metal buffer-pool API frees the handle together with
 * its MTLBuffer, so a true Metal discard needs a handle-preserving detach
 * path first. Until then, fail without mutating buf; callers can treat this
 * as unsupported and keep the original buffer valid. */
extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_buffer_discard_storage(tc_buffer* buf) {
    if (!buf || !buf->owner) return TC_ERR_INVALID_ARG;
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_buffer_reallocate_storage(tc_buffer* buf) {
    if (!buf || !buf->owner || buf->bytes == 0) return TC_ERR_INVALID_ARG;
    if (buf->mtl) return TC_OK;
    return TC_ERR_UNSUPPORTED_FAMILY;
}

extern "C" TC_INTERNAL_SYMBOL int tc_buffer_is_discarded(const tc_buffer* buf) {
    return (buf && !buf->mtl && buf->bytes > 0) ? 1 : 0;
}

/* ----------------------------------------------------------------- */
/* Streams                                                            */
/* ----------------------------------------------------------------- */
extern "C" tc_status_t tc_stream_create(tc_context* ctx, tc_stream** out) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    @autoreleasepool {
        id<MTLCommandQueue> q = [ctx->device newCommandQueue];
        if (!q) return TC_ERR_ALLOC;
        tc_stream* s = new (std::nothrow) tc_stream();
        if (!s) return TC_ERR_ALLOC;
        s->queue = q;
        s->pending_cmd = nil;
        s->owner = ctx;
        *out = s;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_stream_destroy(tc_context* ctx, tc_stream* s) {
    (void)ctx;
    if (!s) return TC_ERR_INVALID_ARG;
    if (s->pending_cmd) (void)tc_stream_sync(s);
    s->queue = nil;
    s->pending_cmd = nil;
    delete s;
    return TC_OK;
}

extern "C" id<MTLCommandBuffer> tc_stream_command_buffer(tc_stream* s) {
    if (!s) return nil;
    if (!s->pending_cmd) {
        s->pending_cmd = [s->queue commandBuffer];
    }
    return s->pending_cmd;
}

extern "C" tc_status_t tc_stream_sync(tc_stream* s) {
    if (!s) return TC_ERR_INVALID_ARG;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = s->pending_cmd;
        if (cmd) {
            s->pending_cmd = nil;
        } else {
            /* Barrier for async work that was committed outside the pending
             * stream buffer path. */
            cmd = [s->queue commandBuffer];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] stream sync error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    return TC_OK;
}
