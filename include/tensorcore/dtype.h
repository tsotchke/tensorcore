#ifndef TENSORCORE_DTYPE_H
#define TENSORCORE_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dtype enum.
 *
 * Ordering is chosen so dispatch tables can index by (uint8_t)dtype.
 * Add new dtypes at the end; do not renumber. */
typedef enum {
    TC_DTYPE_F16  = 0,   /* IEEE 754 binary16, simdgroup_matrix on Apple7+        */
    TC_DTYPE_BF16 = 1,   /* bfloat16, simdgroup_matrix on Apple9+ (M3+)           */
    TC_DTYPE_F32  = 2,   /* IEEE 754 binary32, simdgroup_matrix on Apple7+        */
    TC_DTYPE_I8   = 3,   /* int8, simdgroup_matrix on Apple10+ (M4+), i32 accum   */
    TC_DTYPE_I32  = 4,   /* int32, used for i8 accumulators / indices             */
    TC_DTYPE_F64  = 5,   /* IEEE 754 binary64 — emulated (SF64) on GPU            */
    TC_DTYPE_SF64 = 6,   /* SoftFloat-64 storage (uint2)                          */
    TC_DTYPE_DF64 = 7,   /* Double-float (f32+f32 unevaluated sum)                */
    TC_DTYPE_FP24 = 8,   /* Custom 24-bit ML format from eshkol-platform          */
    TC_DTYPE_FP53 = 9,   /* Custom 53-bit format from eshkol-platform             */
} tc_dtype_t;

static inline size_t tc_dtype_size(tc_dtype_t d) {
    switch (d) {
        case TC_DTYPE_F16:
        case TC_DTYPE_BF16: return 2;
        case TC_DTYPE_F32:
        case TC_DTYPE_I32:
        case TC_DTYPE_FP24: return 4;
        case TC_DTYPE_I8:   return 1;
        case TC_DTYPE_F64:
        case TC_DTYPE_SF64:
        case TC_DTYPE_DF64:
        case TC_DTYPE_FP53: return 8;
    }
    return 0;
}

const char* tc_dtype_name(tc_dtype_t d);

#ifdef __cplusplus
}
#endif
#endif
