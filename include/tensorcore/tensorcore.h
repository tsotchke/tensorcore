#ifndef TENSORCORE_TENSORCORE_H
#define TENSORCORE_TENSORCORE_H

/* Umbrella header. C ABI; safe from Eshkol FFI, Swift, Python ctypes, etc. */

#include "tensorcore/status.h"
#include "tensorcore/dtype.h"
#include "tensorcore/device.h"
#include "tensorcore/gemm.h"
#include "tensorcore/attention.h"
#include "tensorcore/training.h"
#include "tensorcore/conv.h"
#include "tensorcore/distributed.h"
#include "tensorcore/diloco.h"
#include "tensorcore/hip.h"
#include "tensorcore/checkpoint.h"
#include "tensorcore/memory_tier.h"
#include "tensorcore/quantized.h"
#include "tensorcore/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TENSORCORE_VERSION_MAJOR 0
#define TENSORCORE_VERSION_MINOR 1
#define TENSORCORE_VERSION_PATCH 22

const char* tc_version(void);

#ifdef __cplusplus
}
#endif
#endif
