/*
 * tensorcore - portable CPU unsupported-op stubs.
 *
 * The CPU backend intentionally starts with buffers, streams, GGUF loading,
 * distributed-single, and GEMM. Other public ABI entry points return a stable
 * unsupported status so Python/ctypes and downstream FFI imports can bind the
 * full surface without requiring Metal symbols.
 */

#include "tensorcore/tensorcore.h"

/* Attention (forward, forward_async, backward) provided by lib/ops/attention_cpu.cpp. */

/* Training kernels (RMSnorm, LayerNorm, RoPE, SwiGLU, softmax, AdamW,
 * fused_rmsnorm_gemv) are provided by lib/ops/training_cpu.cpp. */

/* Conv2D forward + backward provided by lib/ops/conv2d_cpu.cpp.
 * (backward_input + backward_weight return UNSUPPORTED there for now;
 * tc_conv2d_forward is fully implemented). */
