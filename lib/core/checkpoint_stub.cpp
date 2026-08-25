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
 * The realize path holds only the entry's per-checkpoint mutex while the
 * user callback runs. Same-id reentrancy is not supported; callbacks may
 * still realize other checkpoint ids without contending on the registry.
 */

#include "tensorcore/checkpoint.h"
#include "internal.h"

#include <atomic>
#include <cstdint>
#include <memory>
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
    size_t                         bytes;
    bool                           resident;
    std::mutex                     mutex;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

using CheckpointEntryPtr = std::shared_ptr<CheckpointEntry>;

std::unordered_map<tc_checkpoint_id, CheckpointEntryPtr>& registry() {
    static std::unordered_map<tc_checkpoint_id, CheckpointEntryPtr> r;
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

std::atomic<uint64_t>& resident_count() {
    static std::atomic<uint64_t> n{0};
    return n;
}

std::atomic<uint64_t>& discarded_count() {
    static std::atomic<uint64_t> n{0};
    return n;
}

CheckpointEntryPtr lookup_entry(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return nullptr;
    return it->second;
}

}  // namespace

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_register(tc_buffer* buf,
                                                                 tc_checkpoint_recompute_fn recompute_fn,
                                                                 void* user_data,
                                                                 tc_checkpoint_id* out_id) {
    if (!buf || !recompute_fn || !out_id) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lk(registry_mutex());
    for (const auto& kv : registry()) {
        if (kv.second->buf == buf) return TC_ERR_INVALID_ARG;
    }
    const tc_checkpoint_id id = next_id().fetch_add(1);
    auto entry = std::make_shared<CheckpointEntry>();
    entry->buf = buf;
    entry->recompute_fn = recompute_fn;
    entry->user_data = user_data;
    entry->bytes = tc_buffer_size(buf);
    entry->resident = true;
    registry()[id] = entry;
    resident_count().fetch_add(1);
    *out_id = id;
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_discard(tc_checkpoint_id id) {
    CheckpointEntryPtr entry = lookup_entry(id);
    if (!entry) return TC_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> lk(entry->mutex);
    if (!entry->resident) return TC_OK;

    /* Actually free the buffer's underlying storage. The handle stays valid;
     * tc_checkpoint_realize will re-allocate. Only this checkpoint's mutex is
     * held, so other checkpoints can still progress and callback dependency
     * chains do not deadlock on the global registry. */
    tc_status_t s = tc_buffer_discard_storage(entry->buf);
    if (s != TC_OK) return s;

    entry->resident = false;
    discarded_bytes().fetch_add(entry->bytes);
    resident_count().fetch_sub(1);
    discarded_count().fetch_add(1);
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_realize(tc_checkpoint_id id) {
    CheckpointEntryPtr entry = lookup_entry(id);
    if (!entry) return TC_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> lk(entry->mutex);
    if (entry->resident) return TC_OK;

    /* Re-allocate buffer storage BEFORE the callback so the user's
     * recompute_fn can write into a valid buffer. */
    tc_status_t s = tc_buffer_reallocate_storage(entry->buf);
    if (s != TC_OK) return s;
    s = entry->recompute_fn(entry->user_data);
    if (s != TC_OK) {
        /* Recompute failed — leave the buffer allocated but not marked
         * resident. Caller can retry realize or unregister. */
        return s;
    }
    entry->resident = true;
    discarded_bytes().fetch_sub(entry->bytes);
    discarded_count().fetch_sub(1);
    resident_count().fetch_add(1);
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API int tc_checkpoint_is_resident(tc_checkpoint_id id) {
    CheckpointEntryPtr entry = lookup_entry(id);
    if (!entry) return 0;
    std::lock_guard<std::mutex> lk(entry->mutex);
    return entry->resident ? 1 : 0;
}

extern "C" TC_CHECKPOINT_API tc_status_t tc_checkpoint_unregister(tc_checkpoint_id id) {
    CheckpointEntryPtr entry = lookup_entry(id);
    if (!entry) return TC_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> entry_lk(entry->mutex);
    {
        std::lock_guard<std::mutex> registry_lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end() || it->second.get() != entry.get()) {
            return TC_ERR_INVALID_ARG;
        }
        registry().erase(it);
    }

    if (entry->resident) {
        resident_count().fetch_sub(1);
    } else {
        discarded_bytes().fetch_sub(entry->bytes);
        discarded_count().fetch_sub(1);
    }
    return TC_OK;
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_total_bytes_discarded(void) {
    return discarded_bytes().load();
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_count_resident(void) {
    return resident_count().load();
}

extern "C" TC_CHECKPOINT_API uint64_t tc_checkpoint_count_discarded(void) {
    return discarded_count().load();
}
