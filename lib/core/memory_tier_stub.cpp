/*
 * tensorcore — memory tier C-ABI weak stubs.
 *
 * Until the heterogeneous-mesh runtime ships, every buffer is L0 (local).
 * These stubs satisfy the public ABI so client code can call the tier API
 * unconditionally; the runtime will supersede these once tiering lands.
 */

#include "tensorcore/memory_tier.h"

#ifdef __GNUC__
#  define TC_WEAK __attribute__((weak))
#else
#  define TC_WEAK
#endif

extern "C" TC_WEAK tc_status_t tc_buffer_set_tier_hint(tc_buffer* b,
                                                        tc_tier_hint_t hint) {
    (void)hint;
    return b ? TC_OK : TC_ERR_INVALID_ARG;
}

extern "C" TC_WEAK tc_status_t tc_buffer_get_tier(const tc_buffer* b,
                                                   tc_memory_tier_t* out_tier) {
    if (!b || !out_tier) return TC_ERR_INVALID_ARG;
    *out_tier = TC_TIER_L0_DEVICE;
    return TC_OK;
}

extern "C" TC_WEAK tc_status_t tc_buffer_promote_async(tc_buffer* b,
                                                        tc_memory_tier_t target_tier,
                                                        tc_stream* stream) {
    (void)target_tier; (void)stream;
    return b ? TC_OK : TC_ERR_INVALID_ARG;
}

extern "C" TC_WEAK tc_status_t tc_buffer_demote_async(tc_buffer* b,
                                                       tc_memory_tier_t target_tier,
                                                       tc_stream* stream) {
    (void)target_tier; (void)stream;
    return b ? TC_OK : TC_ERR_INVALID_ARG;
}

extern "C" TC_WEAK tc_status_t tc_buffer_tier_sync(tc_buffer* b) {
    return b ? TC_OK : TC_ERR_INVALID_ARG;
}

extern "C" TC_WEAK tc_status_t tc_memory_tier_usage(tc_context* ctx,
                                                     tc_memory_tier_t tier,
                                                     uint64_t* out_bytes_resident,
                                                     uint64_t* out_bytes_capacity) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    (void)tier;
    if (out_bytes_resident) *out_bytes_resident = 0;
    if (out_bytes_capacity) *out_bytes_capacity = 0;
    return TC_OK;
}
