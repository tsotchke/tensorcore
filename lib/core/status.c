#include "tensorcore/status.h"

const char* tc_status_string(tc_status_t s) {
    switch (s) {
        case TC_OK:                      return "ok";
        case TC_ERR_NOT_INITIALIZED:     return "context not initialized";
        case TC_ERR_ALREADY_INITIALIZED: return "context already initialized";
        case TC_ERR_NO_DEVICE:           return "no Metal device available";
        case TC_ERR_UNSUPPORTED_FAMILY:  return "operation unsupported on this GPU family";
        case TC_ERR_UNSUPPORTED_DTYPE:   return "dtype unsupported on this GPU family";
        case TC_ERR_INVALID_SHAPE:       return "invalid tensor shape";
        case TC_ERR_INVALID_ARG:         return "invalid argument";
        case TC_ERR_ALLOC:               return "allocation failure";
        case TC_ERR_KERNEL_NOT_FOUND:    return "kernel not in metallib";
        case TC_ERR_PIPELINE:            return "MTLComputePipelineState creation failed";
        case TC_ERR_DISPATCH:            return "command-buffer dispatch failed";
        case TC_ERR_INTERNAL:            return "internal error";
    }
    return "unknown status";
}
