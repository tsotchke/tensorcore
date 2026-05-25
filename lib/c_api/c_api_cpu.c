/*
 * tensorcore - portable CPU C-ABI marker.
 */

#include "tensorcore/tensorcore.h"

#if defined(_MSC_VER)
#  define TC_ABI_MARKER_ATTR
#elif defined(__GNUC__) || defined(__clang__)
#  define TC_ABI_MARKER_ATTR __attribute__((visibility("default"), used))
#else
#  define TC_ABI_MARKER_ATTR
#endif

TC_ABI_MARKER_ATTR
const int tensorcore_abi_version = 1;

#undef TC_ABI_MARKER_ATTR
