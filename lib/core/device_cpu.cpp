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

struct tc_buffer {
    void*       ptr;
    size_t      bytes;
    tc_context* owner;
    bool        owns_ptr;   /* false when wrapped via tc_buffer_from_ptr */
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

extern "C" tc_status_t tc_buffer_alloc(tc_context* ctx, size_t bytes, tc_buffer** out) {
    if (!ctx || !out || bytes == 0) return TC_ERR_INVALID_ARG;
    tc_buffer* buf = new (std::nothrow) tc_buffer();
    if (!buf) return TC_ERR_ALLOC;
    buf->ptr = std::malloc(bytes);
    if (!buf->ptr) {
        delete buf;
        return TC_ERR_ALLOC;
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
    tc_status_t s = tc_buffer_validate(ctx, buf, 0);
    if (s != TC_OK) return s;
    if (buf->owns_ptr) {
        std::free(buf->ptr);
    }
    buf->ptr = nullptr;
    delete buf;
    return TC_OK;
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
