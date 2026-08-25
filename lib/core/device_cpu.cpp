/*
 * tensorcore - portable CPU runtime.
 *
 * This backend keeps the public C ABI usable on non-Apple mesh workers. It
 * deliberately implements host allocations and no-op streams only; GPU/Metal
 * acceleration remains in device.mm when TC_ENABLE_METAL is enabled.
 */

#include "tensorcore/tensorcore.h"
#include "internal.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

struct tc_context {
    tc_device_info   info;
    std::atomic<int> ref;
};

/* tc_buffer storage kind. When TC_ENABLE_CUDA is compiled in and
 * tc_cuda_init has been called on the context, new allocations land in
 * CUDA managed memory (cudaMallocManaged): the same pointer works from
 * host and device, the driver handles migration. This eliminates the
 * per-call cudaMemcpy in tc_cuda_gemm for buffers allocated while CUDA GEMM
 * is active. Falls back to plain malloc otherwise. */
enum tc_buffer_storage_t {
    TC_STORAGE_HOST          = 0,   /* plain malloc                       */
    TC_STORAGE_CUDA_MANAGED  = 1,   /* cudaMallocManaged                  */
};

struct tc_buffer {
    void*               ptr;
    size_t              bytes;
    tc_context*         owner;
    bool                owns_ptr;   /* false when wrapped via tc_buffer_from_ptr */
    tc_buffer_storage_t storage;
};

struct tc_stream {
    tc_context* owner;
};

static tc_context* g_ctx = nullptr;
static std::mutex  g_ctx_mutex;
static thread_local tc_backend_t t_last_backend = TC_BACKEND_NONE;

extern "C" TC_INTERNAL_SYMBOL void tc_set_last_backend(tc_backend_t b) { t_last_backend = b; }
extern "C" tc_backend_t tc_last_backend(void) { return t_last_backend; }

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
        case TC_BACKEND_HIP:              return "hip";
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

static uint64_t host_memory_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return (uint64_t)st.ullTotalPhys;
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

extern "C" tc_status_t tc_init(tc_context** out_ctx) {
    if (!out_ctx) return TC_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (g_ctx) {
        g_ctx->ref.fetch_add(1);
        *out_ctx = g_ctx;
        return TC_ERR_ALREADY_INITIALIZED;
    }

    tc_context* ctx = new (std::nothrow) tc_context();
    if (!ctx) return TC_ERR_ALLOC;
    std::memset(&ctx->info, 0, sizeof(ctx->info));

    ctx->info.family = TC_FAMILY_UNKNOWN;
    std::strncpy(ctx->info.name, "portable-cpu", sizeof(ctx->info.name) - 1);
    const uint64_t mem = host_memory_bytes();
    ctx->info.max_buffer_bytes = mem ? mem : (uint64_t)SIZE_MAX / 2u;
    ctx->info.recommended_working_set_bytes = mem;
    ctx->info.max_threadgroup_memory = 0;
    ctx->info.max_threads_per_threadgroup =
        (uint32_t)(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1u);
    ctx->info.thread_execution_width = 1;
    ctx->info.unified_memory = false;
    ctx->info.supports_bf16_simdgroup = false;
    ctx->info.supports_i8_simdgroup = false;
    ctx->info.supports_tensorops_m5 = false;
    ctx->info.supports_fp64_native = true;
    ctx->ref.store(1);

    g_ctx = ctx;
    *out_ctx = g_ctx;

    /* Auto-attempt the strongest GPU backend available. The user can
     * always call tc_cuda_init / tc_hip_init explicitly later, but
     * common-case usage benefits from automatic detection. We attempt
     * CUDA first (clean fast path on NVIDIA hardware), then HIP, then
     * leave the context at portable-CPU. Failures are silent: the
     * context stays valid regardless. */
#if defined(TC_ENABLE_CUDA)
    extern tc_status_t tc_cuda_init(tc_context*);
    tc_status_t cuda_s = tc_cuda_init(g_ctx);
    if (cuda_s == TC_OK) {
        std::strncpy(g_ctx->info.name, "cuda", sizeof(g_ctx->info.name) - 1);
    }
#endif
#if defined(TC_ENABLE_HIP)
    extern tc_status_t tc_hip_init(tc_context*);
    tc_status_t hip_s = tc_hip_init(g_ctx);
    if (hip_s == TC_OK) {
        /* Don't override the name if CUDA succeeded. */
        if (std::strcmp(g_ctx->info.name, "cuda") != 0) {
            std::strncpy(g_ctx->info.name, "hip", sizeof(g_ctx->info.name) - 1);
        }
    }
#endif

    return TC_OK;
}

extern "C" tc_status_t tc_shutdown(tc_context* ctx) {
    if (!ctx) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (ctx != g_ctx) return TC_ERR_INVALID_ARG;
    if (ctx->ref.fetch_sub(1) > 1) return TC_OK;
    g_ctx = nullptr;
    delete ctx;
    return TC_OK;
}

extern "C" tc_status_t tc_device_info_get(tc_context* ctx, tc_device_info* out_info) {
    if (!ctx || !out_info) return TC_ERR_INVALID_ARG;
    *out_info = ctx->info;
    return TC_OK;
}

/* Forward-declared in lib/cuda/device.cpp when TC_ENABLE_CUDA. Returns
 * 1 if a CUDA context has been initialized on this process AND the
 * current backend should use it for allocations. */
#if defined(TC_ENABLE_CUDA)
extern "C" int tc_cuda_managed_alloc(size_t bytes, void** out_ptr);
extern "C" void tc_cuda_managed_free(void* ptr);
extern "C" int tc_cuda_is_active(void);
#endif

extern "C" tc_status_t tc_buffer_alloc(tc_context* ctx, size_t bytes, tc_buffer** out) {
    if (!ctx || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    tc_buffer* buf = new (std::nothrow) tc_buffer();
    if (!buf) return TC_ERR_ALLOC;
    buf->storage = TC_STORAGE_HOST;
    buf->ptr = nullptr;

#if defined(TC_ENABLE_CUDA)
    /* When a CUDA device is active, prefer managed memory so subsequent
     * tc_cuda_gemm calls skip the host/device copy. Falls back to host
     * malloc on any error. */
    if (tc_cuda_is_active()) {
        if (tc_cuda_managed_alloc(bytes, &buf->ptr) == 0 && buf->ptr) {
            buf->storage = TC_STORAGE_CUDA_MANAGED;
        }
    }
#endif

    if (!buf->ptr) {
        buf->ptr = std::malloc(bytes);
        if (!buf->ptr) {
            delete buf;
            return TC_ERR_ALLOC;
        }
        buf->storage = TC_STORAGE_HOST;
    }
    buf->bytes = bytes;
    buf->owner = ctx;
    buf->owns_ptr = true;
    *out = buf;
    return TC_OK;
}

extern "C" tc_status_t tc_buffer_from_ptr(tc_context* ctx, void* ptr,
                                          size_t bytes, tc_buffer** out) {
    if (!ctx || !ptr || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    tc_buffer* buf = new (std::nothrow) tc_buffer();
    if (!buf) return TC_ERR_ALLOC;
    buf->ptr = ptr;
    buf->bytes = bytes;
    buf->owner = ctx;
    buf->owns_ptr = false;
    *out = buf;
    return TC_OK;
}

extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_buffer_validate(tc_context* ctx,
                                                             const tc_buffer* buf,
                                                             size_t min_bytes) {
    if (!ctx || !buf || !buf->ptr) return TC_ERR_INVALID_ARG;
    if (buf->owner != ctx) return TC_ERR_INVALID_ARG;
    if (min_bytes > buf->bytes) return TC_ERR_INVALID_SHAPE;
    return TC_OK;
}

extern "C" tc_status_t tc_buffer_free(tc_context* ctx, tc_buffer* buf) {
    /* Free must succeed even for a discarded buffer (ptr=NULL but valid
     * handle). Don't go through tc_buffer_validate because that requires
     * ptr != NULL. */
    if (!ctx || !buf || buf->owner != ctx) return TC_ERR_INVALID_ARG;
    if (buf->owns_ptr && buf->ptr) {
#if defined(TC_ENABLE_CUDA)
        if (buf->storage == TC_STORAGE_CUDA_MANAGED) {
            tc_cuda_managed_free(buf->ptr);
        } else
#endif
        {
            std::free(buf->ptr);
        }
    }
    buf->ptr = nullptr;
    delete buf;
    return TC_OK;
}

/* Activation-checkpointing storage primitives.
 *
 * Discard frees the buffer's underlying memory while keeping the handle
 * valid (size, owner, storage class remembered). Reallocate restores
 * storage of the same size + class.
 *
 * Between discard and reallocate, tc_buffer_map and tc_buffer_validate
 * fail — the buffer cannot back compute until realized. */
extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_buffer_discard_storage(tc_buffer* buf) {
    if (!buf) return TC_ERR_INVALID_ARG;
    if (!buf->owns_ptr) return TC_ERR_INVALID_ARG;  /* wrapped buffers untouched */
    if (!buf->ptr) return TC_OK;                    /* already discarded — idempotent */
#if defined(TC_ENABLE_CUDA)
    if (buf->storage == TC_STORAGE_CUDA_MANAGED) {
        tc_cuda_managed_free(buf->ptr);
    } else
#endif
    {
        std::free(buf->ptr);
    }
    buf->ptr = nullptr;
    return TC_OK;
}

extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_buffer_reallocate_storage(tc_buffer* buf) {
    if (!buf || buf->bytes == 0) return TC_ERR_INVALID_ARG;
    if (buf->ptr) return TC_OK;                     /* already resident — idempotent */
#if defined(TC_ENABLE_CUDA)
    /* Honor original storage class. CUDA-managed retries managed; if the
     * device isn't available anymore (e.g. driver reload), fall back to
     * host malloc — caller can still read/write, GEMM uses staged path. */
    if (buf->storage == TC_STORAGE_CUDA_MANAGED) {
        if (tc_cuda_managed_alloc(buf->bytes, &buf->ptr) != 0 || !buf->ptr) {
            buf->ptr = std::malloc(buf->bytes);
            if (!buf->ptr) return TC_ERR_ALLOC;
            buf->storage = TC_STORAGE_HOST;
        }
        return TC_OK;
    }
#endif
    buf->ptr = std::malloc(buf->bytes);
    if (!buf->ptr) return TC_ERR_ALLOC;
    return TC_OK;
}

extern "C" TC_INTERNAL_SYMBOL int tc_buffer_is_discarded(const tc_buffer* buf) {
    return (buf && buf->ptr == nullptr && buf->bytes > 0) ? 1 : 0;
}

extern "C" tc_status_t tc_buffer_map(tc_buffer* buf, void** out_ptr) {
    if (!buf || !buf->ptr || !out_ptr) return TC_ERR_INVALID_ARG;
    *out_ptr = buf->ptr;
    return TC_OK;
}

extern "C" size_t tc_buffer_size(const tc_buffer* buf) {
    return buf ? buf->bytes : 0;
}

extern "C" tc_status_t tc_stream_create(tc_context* ctx, tc_stream** out) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    tc_stream* s = new (std::nothrow) tc_stream();
    if (!s) return TC_ERR_ALLOC;
    s->owner = ctx;
    *out = s;
    return TC_OK;
}

extern "C" tc_status_t tc_stream_destroy(tc_context* ctx, tc_stream* s) {
    if (!ctx || !s || s->owner != ctx) return TC_ERR_INVALID_ARG;
    delete s;
    return TC_OK;
}

extern "C" tc_status_t tc_stream_sync(tc_stream* s) {
    if (!s) return TC_ERR_INVALID_ARG;
    return TC_OK;
}
