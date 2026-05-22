/*
 * tensorcore — activation-checkpointing weak stubs.
 *
 * The full implementation lives in lib/core/checkpoint.cpp (TBD) once the
 * tc_buffer free/realloc plumbing for "discard then re-create with the
 * same handle" lands. These weak stubs satisfy the public ABI so client
 * code can call the checkpoint API unconditionally; until the runtime
 * grows the discard/realize behavior, the stubs treat every realize as
 * a no-op (the buffer is always resident).
 */

#include "tensorcore/checkpoint.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#ifdef __GNUC__
#  define TC_WEAK __attribute__((weak))
#else
#  define TC_WEAK
#endif

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

extern "C" TC_WEAK tc_status_t tc_checkpoint_register(tc_buffer* buf,
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

extern "C" TC_WEAK tc_status_t tc_checkpoint_discard(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return TC_ERR_INVALID_ARG;
    /* The runtime would free the buffer's underlying memory here. The
     * stub just marks it as discarded for the observability counters. */
    if (it->second.resident) {
        it->second.resident = false;
        const size_t bytes = tc_buffer_size(it->second.buf);
        discarded_bytes().fetch_add(bytes);
    }
    return TC_OK;
}

extern "C" TC_WEAK tc_status_t tc_checkpoint_realize(tc_checkpoint_id id) {
    /* Acquire entry pointer under the lock, then drop the lock before
     * invoking the user callback (which may take a long time). */
    tc_checkpoint_recompute_fn fn = nullptr;
    void* user_data = nullptr;
    size_t bytes = 0;
    bool was_discarded = false;
    {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end()) return TC_ERR_INVALID_ARG;
        if (it->second.resident) return TC_OK;
        fn = it->second.recompute_fn;
        user_data = it->second.user_data;
        was_discarded = true;
        bytes = tc_buffer_size(it->second.buf);
    }
    tc_status_t s = fn(user_data);
    if (s != TC_OK) return s;
    if (was_discarded) {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = registry().find(id);
        if (it == registry().end()) return TC_ERR_INVALID_ARG;
        if (!it->second.resident) {
            it->second.resident = true;
            /* Subtract back from discarded_bytes since the buffer is resident again. */
            discarded_bytes().fetch_sub(bytes);
        }
    }
    return s;
}

extern "C" TC_WEAK int tc_checkpoint_is_resident(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return 0;
    return it->second.resident ? 1 : 0;
}

extern "C" TC_WEAK tc_status_t tc_checkpoint_unregister(tc_checkpoint_id id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(id);
    if (it == registry().end()) return TC_ERR_INVALID_ARG;
    if (!it->second.resident) {
        discarded_bytes().fetch_sub(tc_buffer_size(it->second.buf));
    }
    registry().erase(it);
    return TC_OK;
}

extern "C" TC_WEAK uint64_t tc_checkpoint_total_bytes_discarded(void) {
    return discarded_bytes().load();
}

extern "C" TC_WEAK uint64_t tc_checkpoint_count_resident(void) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    uint64_t n = 0;
    for (const auto& [id, entry] : registry()) {
        if (entry.resident) ++n;
    }
    return n;
}

extern "C" TC_WEAK uint64_t tc_checkpoint_count_discarded(void) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    uint64_t n = 0;
    for (const auto& [id, entry] : registry()) {
        if (!entry.resident) ++n;
    }
    return n;
}
