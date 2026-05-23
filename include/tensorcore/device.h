#ifndef TENSORCORE_DEVICE_H
#define TENSORCORE_DEVICE_H

#include <stdint.h>
#include <stdbool.h>
#include "tensorcore/status.h"
#include "tensorcore/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. On Apple builds these wrap MTLDevice / MTLBuffer /
 * MTLCommandQueue. On portable CPU builds they wrap host allocations and
 * no-op streams so the same ABI can run on Linux mesh workers. */
typedef struct tc_context  tc_context;
typedef struct tc_buffer   tc_buffer;
typedef struct tc_stream   tc_stream;

/* Apple GPU family classification used for kernel selection. */
typedef enum {
    TC_FAMILY_UNKNOWN   = 0,
    TC_FAMILY_APPLE7    = 7,   /* M1                                  */
    TC_FAMILY_APPLE8    = 8,   /* M2                                  */
    TC_FAMILY_APPLE9    = 9,   /* M3, A17 Pro (+ bf16 simdgroup_matrix) */
    TC_FAMILY_APPLE10   = 10,  /* M4 (+ i8 simdgroup_matrix, + SME on CPU) */
    TC_FAMILY_APPLE11   = 11,  /* M5  (+ Metal 4 mpp::tensor_ops)          */
} tc_family_t;

typedef struct {
    tc_family_t family;
    char        name[128];           /* MTLDevice.name                   */
    uint64_t    max_buffer_bytes;
    uint64_t    recommended_working_set_bytes;
    uint32_t    max_threadgroup_memory;
    uint32_t    max_threads_per_threadgroup;
    uint32_t    thread_execution_width;   /* SIMD width, 32 on Apple7+   */
    bool        unified_memory;
    bool        supports_bf16_simdgroup;  /* Apple9+                     */
    bool        supports_i8_simdgroup;    /* Apple10+                    */
    bool        supports_tensorops_m5;    /* Metal4 runtime gate for M5+ */
    bool        supports_fp64_native;     /* false on Apple GPU          */
} tc_device_info;

/* Initialize the global context. Idempotent (returns TC_ERR_ALREADY_INITIALIZED
 * if called twice without tc_shutdown). */
tc_status_t tc_init(tc_context** out_ctx);
tc_status_t tc_shutdown(tc_context* ctx);

/* Query the underlying device. */
tc_status_t tc_device_info_get(tc_context* ctx, tc_device_info* out_info);

/* Buffer management.
 *
 * On Apple Silicon, buffers are unified-memory by default — `tc_buffer_map`
 * returns a CPU-addressable pointer with no copy. The buffer pool uses
 * power-of-2 size buckets and LIFO recycling.
 */
tc_status_t tc_buffer_alloc(tc_context* ctx, size_t bytes, tc_buffer** out);
tc_status_t tc_buffer_free (tc_context* ctx, tc_buffer* buf);
tc_status_t tc_buffer_map  (tc_buffer* buf, void** out_ptr);
size_t      tc_buffer_size (const tc_buffer* buf);

/* Zero-copy wrap of externally-owned memory. The returned tc_buffer
 * references `ptr` without taking ownership: tc_buffer_free will release
 * the wrapper but not the underlying allocation. Bridges (e.g.
 * bindings/pytorch) use this to pass framework-allocated tensor data to
 * the dispatcher without an alloc-and-memcpy.
 *
 * Constraints:
 *   * `ptr` must remain valid for the lifetime of the returned tc_buffer.
 *   * On portable CPU backends this is unconditionally zero-copy.
 *   * On Metal builds this wraps the memory with
 *     newBufferWithBytesNoCopy; `ptr` must be page-aligned and `bytes`
 *     must be a multiple of the page size.
 *
 * Returns TC_ERR_INVALID_ARG if any pointer is null. */
tc_status_t tc_buffer_from_ptr(tc_context* ctx, void* ptr, size_t bytes,
                               tc_buffer** out);

/* Streams ≈ MTLCommandQueue lanes; for now, NULL stream means default. */
tc_status_t tc_stream_create (tc_context* ctx, tc_stream** out);
tc_status_t tc_stream_destroy(tc_context* ctx, tc_stream* s);
tc_status_t tc_stream_sync   (tc_stream* s);

#ifdef __cplusplus
}
#endif
#endif
