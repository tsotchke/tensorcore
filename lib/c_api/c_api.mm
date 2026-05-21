/*
 * tensorcore — C-ABI shim and version.
 *
 * All public entry points already live in device.mm / gemm.mm / attention.mm.
 * This file exists to anchor any cross-cutting C ABI helpers that don't fit
 * elsewhere. Keep it small.
 */

#include "tensorcore/tensorcore.h"

/* Marker symbol so callers can link against -ltensorcore and feature-test. */
extern "C" __attribute__((visibility("default"), used))
const int tensorcore_abi_version = 1;
