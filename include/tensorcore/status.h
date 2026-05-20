#ifndef TENSORCORE_STATUS_H
#define TENSORCORE_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TC_OK                       = 0,
    TC_ERR_NOT_INITIALIZED      = -1,
    TC_ERR_ALREADY_INITIALIZED  = -2,
    TC_ERR_NO_DEVICE            = -3,
    TC_ERR_UNSUPPORTED_FAMILY   = -4,
    TC_ERR_UNSUPPORTED_DTYPE    = -5,
    TC_ERR_INVALID_SHAPE        = -6,
    TC_ERR_INVALID_ARG          = -7,
    TC_ERR_ALLOC                = -8,
    TC_ERR_KERNEL_NOT_FOUND     = -9,
    TC_ERR_PIPELINE             = -10,
    TC_ERR_DISPATCH             = -11,
    TC_ERR_INTERNAL             = -99,
} tc_status_t;

const char* tc_status_string(tc_status_t s);

#ifdef __cplusplus
}
#endif
#endif
