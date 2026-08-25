#include "tensorcore/dtype.h"

const char* tc_dtype_name(tc_dtype_t d) {
    switch (d) {
        case TC_DTYPE_F16:  return "f16";
        case TC_DTYPE_BF16: return "bf16";
        case TC_DTYPE_F32:  return "f32";
        case TC_DTYPE_I8:   return "i8";
        case TC_DTYPE_I32:  return "i32";
        case TC_DTYPE_F64:  return "f64";
        case TC_DTYPE_SF64: return "sf64";
        case TC_DTYPE_DF64: return "df64";
        case TC_DTYPE_FP24: return "fp24";
        case TC_DTYPE_FP53: return "fp53";
    }
    return "?";
}
