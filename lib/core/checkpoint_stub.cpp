/*
 * tensorcore — activation-checkpointing runtime.
 *
 * Implements the tc_checkpoint_* public ABI on top of the internal
 * tc_buffer_discard_storage / tc_buffer_reallocate_storage primitives
 * in lib/core/device_cpu.cpp (CPU build). On Metal the buffer pool
 * defers to the underlying MTLBuffer realloc story.
 *
 * Semantics:
 *   register(buf, fn, ud) → returns an id, buffer is resident.
 *   discard(id)           → buffer storage is freed; bytes accounted to
 *                            discarded_bytes; user data + fn retained.
 *   realize(id)           → storage is re-allocated to its original size,
 *                            user's recompute_fn is invoked to refill it.
 *   unregister(id)        → drop from registry (buffer is NOT freed here;
 *                            caller still owns the handle).
 *
 * The realize path holds NO mutex during the user callback to avoid
 * deadlock if the callback calls back into the checkpoint API.
 */

#include "tensorcore/checkpoint.h"
#include "internal.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

/* This file is the real checkpoint ABI implementation. Keep the exported
 * entry points strong; weak public functions can fail to resolve internal
 * storage hooks under Darwin exported-symbols-list linking. */
#define TC_CHECKPOINT_API

namespace {

struct CheckpointEntry {
    tc_buffer*                     buf;
    tc_checkpoint_recompute_fn     recompute_fn;
    void*                          user_data;
    bool                           resident;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<tc_checkpoint_id, CheckpointEntry>& registry() {
    static std::unordered_map<tc_checkpoint_id, CheckpointEntry> r;
    return r;
}

std::atomic<tc_checkpoint_id>& next_id() {
    static std::atomic<tc_checkpoint_id> n{1};
    return n;
}

std::atomic<uint64_t>& discarded_bytes() {
    static std::atomic<uint64_t> d{0};
    return d;
}

}  // namespace

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_register(tc_buffer* buf,
                                                                 tc_checkpoint_recompute_fn recompute_fn,
                                                                 void* user_data,
                                                                 tc_checkpoint_id* out_id) {
    if (!buf || !recompute_fn || !out_id) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lk(registry_mutex());
    for (const auto& kv : registry()) {
        if (kv.second.buf == buf) return TC_ERR_INVALID_ARG;
    }
    const tc_checkpoint_id id = next_id().fetch_add(1);
    registry()[id] = CheckpointEntry{buf, recompute_fn, user_data, true};
    *out_id = id;
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_discard(tc_checkpoint_id id) {
    tc_buffer* buf = nullptr;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end()) return TC_ERR_INVALID_ARG;
        if (!it->second.resident) return TC_OK;
        buf = it->second.buf;
        bytes = tc_buffer_size(buf);
        it->second.resident = false;
    }
    /* Actually free the buffer's underlying storage. The handle stays
     * valid; tc_checkpoint_realize will re-allocate. Drop the registry
     * lock first so the discard call can grab the buffer pool's lock
     * without inversion. */
    tc_status_t s = tc_buffer_discard_storage(buf);
    if (s != TC_OK) {
        /* Roll back the resident flag — we didn't actually free anything. */
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it != registry().end()) it->second.resident = true;
        return s;
    }
    discarded_bytes().fetch_add(bytes);
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_realize(tc_checkpoint_id id) {
    tc_checkpoint_recompute_fn fn = nullptr;
    void* user_data = nullptr;
    tc_buffer* buf = nullptr;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end()) return TC_ERR_INVALID_ARG;
        if (it->second.resident) return TC_OK;
        fn = it->second.recompute_fn;
        user_data = it->second.user_data;
        buf = it->second.buf;
        bytes = tc_buffer_size(buf);
    }
    /* Re-allocate buffer storage BEFORE the callback so the user's
     * recompute_fn can write into a valid buffer. */
    tc_status_t s = tc_buffer_reallocate_storage(buf);
    if (s != TC_OK) return s;
    s = fn(user_data);
    if (s != TC_OK) {
        /* Recompute failed — leave the buffer allocated but not marked
         * resident. Caller can retry realize or unregister. */
        return s;
    }
    {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end()) return TC_ERR_INVALID_ARG;
        if (!it->second.resident) {
            it->second.resident = true;
            discarded_bytes().fetch_sub(bytes);
        }
    }
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API int tc_checkpoint_is_resident(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return 0;
    return it->second.resident ? 1 : 0;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_unregister(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return TC_ERR_INVALID_ARG;
    if (!it->second.resident) {
        discarded_bytes().fetch_sub(tc_buffer_size(it->second.buf));
    }
    registry().erase(it);
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_total_bytes_discarded(void) {
    return discarded_bytes().load();
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_count_resident(void) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    uint64_t n = 0;
    for (const auto& [id, entry] : registry()) {
        if (entry.resident) ++n;
    }
    return n;
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_count_discarded(void) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    uint64_t n = 0;
    for (const auto& [id, entry] : registry()) {
        if (!entry.resident) ++n;
    }
    return n;
}
