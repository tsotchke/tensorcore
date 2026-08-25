/*
 * tensorcore - internal dispatch tracing.
 */

#include "internal.h"

#include <cstdio>
#include <cstdlib>

namespace {

bool trace_enabled(void) {
    static const bool enabled = [] {
        const char* value = std::getenv("TC_TRACE");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

}  // namespace

extern "C" TC_INTERNAL_SYMBOL tc_status_t tc_record_dispatch(const char* op,
                                                             tc_backend_t backend,
                                                             tc_status_t status) {
    if (status == TC_OK) {
        tc_set_last_backend(backend);
    }
    if (trace_enabled()) {
        const tc_backend_t reported = (status == TC_OK) ? backend : tc_last_backend();
        std::fprintf(stderr,
                     "[tensorcore] trace op=%s status=%s backend=%s\n",
                     op ? op : "?",
                     tc_status_string(status),
                     tc_backend_name(reported));
    }
    return status;
}
