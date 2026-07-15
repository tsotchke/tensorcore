#ifndef TC_LIB_DISTRIBUTED_REMOTE_TENSOR_INTERNAL_H
#define TC_LIB_DISTRIBUTED_REMOTE_TENSOR_INTERNAL_H

/* Private lifecycle hooks shared by distributed transport consumers. */

#include <stdint.h>
#include "tensorcore/remote_tensor.h"

tc_status_t tc_remote_internal_wait_for_clients_closed(tc_remote_ctx* h,
                                                        uint32_t timeout_ms);

#endif
