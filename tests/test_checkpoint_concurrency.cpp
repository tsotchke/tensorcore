/*
 * Concurrent activation-checkpoint lifecycle test.
 *
 * Proves the checkpoint registry serializes same-id discard/realize calls:
 * a fan-out of realize calls after one discard must invoke the user
 * recompute callback exactly once, and a fan-out of discard calls after
 * realization must account the discarded bytes exactly once.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/checkpoint.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr int kElems = 4096;
constexpr int kThreads = 8;

struct RecomputeState {
    tc_buffer* buf;
    std::atomic<int> calls{0};
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
};

tc_status_t recompute_fill(void* user_data) {
    auto* st = static_cast<RecomputeState*>(user_data);
    const int now_active = st->active.fetch_add(1) + 1;
    int old_max = st->max_active.load();
    while (now_active > old_max &&
           !st->max_active.compare_exchange_weak(old_max, now_active)) {
    }

    void* p = nullptr;
    tc_status_t s = tc_buffer_map(st->buf, &p);
    if (s != TC_OK) {
        st->active.fetch_sub(1);
        return s;
    }
    auto* data = static_cast<float*>(p);
    for (int i = 0; i < kElems; ++i) data[i] = static_cast<float>(i) * 0.25f;
    st->calls.fetch_add(1);
    st->active.fetch_sub(1);
    return TC_OK;
}

int expect_status(const char* label, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    std::fprintf(stderr, "%s: got %d expected %d\n", label, got, want);
    return 1;
}

}  // namespace

int main() {
    tc_context* ctx = nullptr;
    if (tc_init(&ctx) != TC_OK) {
        std::fprintf(stderr, "tc_init failed\n");
        return 1;
    }

    tc_buffer* buf = nullptr;
    const size_t bytes = static_cast<size_t>(kElems) * sizeof(float);
    if (tc_buffer_alloc(ctx, bytes, &buf) != TC_OK) {
        std::fprintf(stderr, "buffer alloc failed\n");
        return 1;
    }

    RecomputeState st;
    st.buf = buf;
    tc_checkpoint_id id = 0;
    int rc = expect_status("checkpoint register",
                           tc_checkpoint_register(buf, recompute_fill, &st, &id),
                           TC_OK);
    if (rc || id == 0 || !tc_checkpoint_is_resident(id)) rc = 1;

    tc_status_t s = tc_checkpoint_discard(id);
    if (s == TC_ERR_UNSUPPORTED_FAMILY) {
        tc_checkpoint_unregister(id);
        tc_buffer_free(ctx, buf);
        tc_shutdown(ctx);
        return 77;
    }
    rc |= expect_status("initial discard", s, TC_OK);
    if (tc_checkpoint_count_discarded() != 1 ||
        tc_checkpoint_total_bytes_discarded() != bytes) {
        std::fprintf(stderr, "initial discard counters mismatch\n");
        rc = 1;
    }

    std::vector<tc_status_t> realize_status(kThreads, TC_ERR_INTERNAL);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            realize_status[t] = tc_checkpoint_realize(id);
        });
    }
    for (auto& thread : threads) thread.join();
    threads.clear();

    for (int t = 0; t < kThreads; ++t) {
        rc |= expect_status("concurrent realize", realize_status[t], TC_OK);
    }
    if (st.calls.load() != 1 || st.max_active.load() != 1 ||
        !tc_checkpoint_is_resident(id) ||
        tc_checkpoint_count_resident() != 1 ||
        tc_checkpoint_count_discarded() != 0 ||
        tc_checkpoint_total_bytes_discarded() != 0) {
        std::fprintf(stderr,
                     "concurrent realize mismatch calls=%d max_active=%d resident=%d discarded=%llu bytes=%llu\n",
                     st.calls.load(), st.max_active.load(),
                     (int)tc_checkpoint_count_resident(),
                     (unsigned long long)tc_checkpoint_count_discarded(),
                     (unsigned long long)tc_checkpoint_total_bytes_discarded());
        rc = 1;
    }

    std::vector<tc_status_t> discard_status(kThreads, TC_ERR_INTERNAL);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            discard_status[t] = tc_checkpoint_discard(id);
        });
    }
    for (auto& thread : threads) thread.join();

    for (int t = 0; t < kThreads; ++t) {
        rc |= expect_status("concurrent discard", discard_status[t], TC_OK);
    }
    if (tc_checkpoint_is_resident(id) ||
        tc_checkpoint_count_resident() != 0 ||
        tc_checkpoint_count_discarded() != 1 ||
        tc_checkpoint_total_bytes_discarded() != bytes) {
        std::fprintf(stderr, "concurrent discard counters mismatch\n");
        rc = 1;
    }

    rc |= expect_status("final realize", tc_checkpoint_realize(id), TC_OK);
    rc |= expect_status("unregister", tc_checkpoint_unregister(id), TC_OK);
    if (tc_checkpoint_count_resident() != 0 ||
        tc_checkpoint_count_discarded() != 0 ||
        tc_checkpoint_total_bytes_discarded() != 0) {
        std::fprintf(stderr, "final counters leaked\n");
        rc = 1;
    }

    tc_buffer_free(ctx, buf);
    tc_shutdown(ctx);
    if (!rc) std::printf("checkpoint concurrency OK\n");
    return rc;
}
