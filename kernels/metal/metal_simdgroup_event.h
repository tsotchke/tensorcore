/*
 * metal_simdgroup_event.h — private AIR intrinsics for simdgroup async DMA.
 *
 * Source: Philip Turner's metal-flash-attention (MFA) reverse-engineered shim,
 * matched against Apple's leaked Xcode 14.2 headers (metal_simdgroup_future,
 * metal_simdgroup_async) that were removed in Xcode 14.3+.
 *
 * Reference impl:
 *   github.com/philipturner/metal-flash-attention/blob/main/
 *     Documentation/CppReference/GEMM/GEMMHeaders.cpp
 *   github.com/liuliu/ccv/blob/unstable/lib/nnc/mfa/kernels/GEMMHeaders.cpp
 *
 * Compatibility:
 *   - Xcode 16.x / macOS 15.x: __asm("air.simdgroup_async_copy_2d.…") accepted.
 *   - Xcode 17+ / macOS 26+:   compiler rejects the __asm form (regression).
 *     Workaround for v0.2: emit AIR LLVM IR directly, or precompile metallib
 *     on macOS 15 and ship. Tracked as a known limitation.
 *
 * The kernel here defines a single-simd-group async DMA primitive that runs
 * on a separate copy-engine queue inside the GPU core — not a TMA (Hopper-
 * class), but a true Ampere-class cp.async equivalent.
 */

#ifndef TC_METAL_SIMDGROUP_EVENT_H
#define TC_METAL_SIMDGROUP_EVENT_H

#include <metal_stdlib>
using namespace metal;

/* Opaque event type — the AIR runtime keeps the completion handle here. */
struct _simdgroup_event_t;

/* AIR intrinsics. The mangled symbol names mirror MFA's GEMMHeaders.cpp. */
thread _simdgroup_event_t* __metal_simdgroup_async_copy_1d(
    ulong size, ulong align,
    threadgroup void *dst,
    const device void *src,
    ulong n_elements)
    __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(
    ulong size, ulong align,
    threadgroup void *dst,
    ulong dst_elements_per_row,
    ulong dst_row_stride,
    ulong2 dst_tile_dims,
    const device void *src,
    ulong src_elements_per_row,
    ulong src_row_stride,
    ulong2 src_tile_dims,
    long2 offset_clip,
    int clamp_mode)
    __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d_to_device(
    ulong size, ulong align,
    device void *dst,
    ulong dst_elements_per_row,
    ulong dst_row_stride,
    ulong2 dst_tile_dims,
    const threadgroup void *src,
    ulong src_elements_per_row,
    ulong src_row_stride,
    ulong2 src_tile_dims,
    long2 offset_clip)
    __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

void __metal_wait_simdgroup_events(
    int count,
    thread _simdgroup_event_t **events)
    __asm("air.wait_simdgroup_events");

/* User-facing C++ wrapper — issued by ONE simdgroup (sidx==0), waited via
 * the static class method, then a threadgroup_barrier publishes data to
 * all simdgroups. */
namespace tc {
enum class async_copy_clamp_mode { clamp_to_zero = 0, clamp_to_edge = 1 };

struct simdgroup_event {
private:
    thread _simdgroup_event_t* event;
public:
    template <typename T>
    void async_copy(threadgroup T*       dst, ushort dst_elements_per_row, ushort2 dst_tile_dims,
                    const device T*     src, uint   src_elements_per_row, ushort2 src_tile_dims,
                    bool transpose = false,
                    async_copy_clamp_mode clamp = async_copy_clamp_mode::clamp_to_zero) {
        /* The 2D variant of MFA's wrapper — ignores transpose for v0.1 of this
         * path (set when needed by the kernel side). */
        (void)transpose;
        event = __metal_simdgroup_async_copy_2d(
            sizeof(T), alignof(T),
            (threadgroup void*)dst,
            (ulong)dst_elements_per_row, 1,
            ulong2((ulong)dst_tile_dims.x, (ulong)dst_tile_dims.y),
            (const device void*)src,
            (ulong)src_elements_per_row, 1,
            ulong2((ulong)src_tile_dims.x, (ulong)src_tile_dims.y),
            long2(0, 0),
            (int)clamp);
    }

    template <typename T>
    void async_copy(device T*           dst, uint  dst_elements_per_row, ushort2 dst_tile_dims,
                    const threadgroup T* src, ushort src_elements_per_row, ushort2 src_tile_dims) {
        event = __metal_simdgroup_async_copy_2d_to_device(
            sizeof(T), alignof(T),
            (device void*)dst,
            (ulong)dst_elements_per_row, 1,
            ulong2((ulong)dst_tile_dims.x, (ulong)dst_tile_dims.y),
            (const threadgroup void*)src,
            (ulong)src_elements_per_row, 1,
            ulong2((ulong)src_tile_dims.x, (ulong)src_tile_dims.y),
            long2(0, 0));
    }

    static void wait(int count, thread simdgroup_event *events) {
        __metal_wait_simdgroup_events(count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }
};
}  // namespace tc

#endif /* TC_METAL_SIMDGROUP_EVENT_H */
