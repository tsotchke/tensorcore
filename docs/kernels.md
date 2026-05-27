# Kernels — per-file walkthrough

`kernels/metal/` contains every Metal kernel the library compiles into
`tensorcore.metallib`. Seventeen `.metal` files plus one shared header.
This page is the per-file reference: what each kernel does, its tile
layout, function constants, and which host op uses it.

Many of these files are heavily commented at the top with design
rationale; this page is the bird's-eye view that tells you which file to
open.

## GEMM family

### `gemm_simdgroup.metal` — default fp16 / bf16 / fp32 / i8

The workhorse. One kernel source, four dtype variants.

- **Tile:** BM = BN = 64, BK = 32
- **Threads:** 4 simdgroups × 32 = 128 threads / TG (WM=2, WN=2)
- **Per simdgroup:** 32 × 32 of the output, TM=4 × TN=4 of 8×8 MMA fragments
- **Loads:** vec4 cooperative (half4 / float4 / char4)
- **Accumulator:** fp32 unconditionally (i32 for int8); accumulation runs
  in-register, output is cast to the requested c_dtype on write
- **Function constants:** dtype, transpose_a, transpose_b, alpha=0/1 shortcut

Used by `tc_gemm` whenever the input shape and dtype combination is
covered (which is nearly always on Apple7+).

Reference impl path: `eshkol-platform/lib/backend/gpu/metal_softfloat.h`
`matmul_f32_simd_pure` extended with dtype templating + alpha/beta
+ compile-time transpose flags.

### `gemm_simdgroup_128.metal` — 128×128 opt-in tile

Same algorithm, larger tile.

- **Tile:** BM = BN = 128, BK = 8 (one inner MMA step per K-block)
- **Threads:** 16 simdgroups × 32 = 512 threads / TG (WM=4, WN=4)
- **Per simdgroup:** 32 × 32 of the output, TM=4, TN=4
- **Loads:** half4 / float4 cooperative
- **Accumulator:** fp32 (i32 for int8)

Activate via `TC_USE_128_TILE=1`. Today regresses on M2 (~10 vs ~18
TFLOPS at 4096³) due to register pressure; v0.2 retunes.

The theoretical win is real: each loaded element is reused across 128
output rows/cols instead of 64, so memory traffic per FLOP halves at
large shapes. Register-pressure-aware sg layout (WM=4×WN=2, TM=2×TN=4)
is the v0.2 target.

### `gemm_async.metal` — `simdgroup_async_copy` 64×64 (Xcode 16 only)

MFA-style async DMA pattern. One simdgroup (sidx==0) issues async DMAs
for A and B blocks, all simdgroups wait on a `threadgroup_barrier`,
compute proceeds. Compiler reorders compute instructions ahead of wait
when register pressure allows, giving latency hiding.

- **Tile:** BM = BN = 64, BK = 32
- **Dtypes:** fp16 and bf16 (bf16 requires Apple9+ and the async path)
- **Single-buffered** (matches MFA's production kernel)
- **TG memory:** 64×32×2 + 32×64×2 = 8 KB
- **Compatibility:** Xcode 16.x accepts the `__asm("air.simdgroup_async_copy_2d…")`
  intrinsics; Xcode 17+ rejects them. CMake gates this file behind
  `TC_SDK_VERSION < 26.0`.

MFA reports 10-30% end-to-end win on attention; raw GEMM win in the
8-15% range vs sync vec4 loads. The v0.2 fallback for newer SDKs is an
AIR-IR emit; tracked as a known limitation.

### `gemm_async_128.metal` — async DMA + 128×128

Same async DMA pattern, large tile.

- **Tile:** BM = BN = 128, BK = 8
- **Threads:** 16 simdgroups × 32 = 512 threads / TG
- **TG memory:** ~8 KB

Built when SDK < 26.0; activated by `TC_USE_128_TILE=1` plus async opt-in.

### `gemm_quantized.metal` — Q4_0 / Q8_0 GEMV (v1)

The original quantized GEMV kernel. One simdgroup per output cell; the
simdgroup walks K in 32-weight blocks, unpacks Q4_0 nibbles inline,
multiplies by fp16 activations, accumulates. Final `simd_sum` reduction.

- **Pattern:** 1 simdgroup per output cell
- **Throughput on 7B Q4_0 decode:** 13.7 tok/s (M2 Ultra, async batched)

Used when `TC_Q4_USE_V1=1` is set, for comparison.

### `gemm_quantized_v2.metal` — Q4_0 GEMV (llama.cpp-class, default)

Rewrite pattern lifted from `ggml/src/ggml-metal/ggml-metal.metal`:

- **NR0 = 4** output rows per simdgroup
- **NSG = 2** simdgroups per threadgroup (8 outputs per TG)
- **Pre-scaled y values:** each lane holds 16 y-values pre-scaled for the
  4 nibble bit-fields (`0x000F`, `0x0F00`, `0x00F0`, `0xF000`) so dequant
  becomes mask + FMA (no shift)
- **Q4_0 zero-point folded:** `d * (sumy * -8 + sum(yl * nibble))`
- **Row partials in registers** (no threadgroup memory in hot loop)
- **Single `simd_sum`** per row

This is the default path since v0.1.6, and the reason the 7B Q4_0 decode
harness now lands at **186 tok/s @ 632 GB/s on M2 Ultra** (~79% of
LPDDR5 peak; ~3× llama.cpp's published number on the same chip — see
[benchmarks.md](benchmarks.md)).

## Attention family

### `flash_attention.metal` — FlashAttention-2 forward, D=64

Fused QK·softmax·PV with online softmax. Single kernel; supports causal,
GQA, sliding window, and ALiBi via function constants.

- **Tile:** Br = Bc = 32
- **Threads:** 4 simdgroups × 32 = 128 (WM=2, WN=2)
- **Per simdgroup:** owns 16×16 of S (TM_S=TN_S=2 of 8×8) and 16×32 of O
  (TM_O=2, TN_O=4 at D=64)
- **TG memory:** ≈ 22 KB
  - sQ: 32×64×2 = 4 KB
  - sK: 32×64×2 = 4 KB (reused for sP after K is consumed)
  - sV: 32×64×2 = 4 KB
  - sS: 32×32×4 fp32 = 4 KB (in sV region pre-V-load)
  - sP: 32×32×2 = 2 KB (overlaps sK)
  - per-row m/l/alpha scratch + sO spill ≈ 1.5 KB
- **Accumulators:** fp32. IO: fp16 or bf16.
- **Dispatch grid:** (num_q_blocks, heads, batch)

Function constants: `g_causal`, `g_use_lse`, `g_use_window`, `g_use_alibi`.

Used by `tc_attention_forward` when `head_dim == 64`.

### `flash_attention_d128.metal` — D=128 forward

Same algorithm, larger D, smaller tile to fit threadgroup memory.

- **Tile:** Br = Bc = 16
- **Threads:** 4 simdgroups × 32 = 128 (WM=2, WN=2)
- **Per simdgroup:** owns 8×8 of S (single 8×8 fragment) and 8×64 of O
  (TM_O=1, TN_O=8)
- **TG memory:** ≈ 15 KB

The v0.2 plan recovers Br=Bc=32 on Apple9+ via aliased TG memory regions
(sK region recycled as sP after consumption). The 32 KB budget stays
constant; we just stop wasting it.

### `flash_attention_backward.metal` — D=64 backward

LSE-saved scheme, Dao FA-2 backward. Two kernels to avoid cross-block
atomic accumulation:

- `tc_flash_attention_backward_dq`: one TG per query block, iterates all
  KV blocks, writes dQ block.
- `tc_flash_attention_backward_dk_dv`: one TG per KV block, iterates all
  Q blocks, writes dK / dV block.

- **Tile:** Br = Bc = 32, D = 64
- **Threads:** 4 simdgroups × 32 = 128
- **Accumulators:** fp32; IO: fp16; LSE: fp32

Math (per Dao 2023):
```
S_ij  = Q_i @ K_j^T * scale
P_ij  = exp(S_ij - LSE_i)
D_i   = sum(dO_i * O_i, dim=-1)
dV_j += P_ij^T @ dO_i
dP_ij = dO_i @ V_j^T
dS_ij = P_ij * (dP_ij - D_i)
dQ_i += dS_ij @ K_j * scale
dK_j += dS_ij^T @ Q_i * scale
```

### `flash_attention_backward_d128.metal` — D=128 backward

Same split-kernel design as the D=64 backward, Br = Bc = 16 to fit
threadgroup memory at D=128.

- **Per simdgroup:** 8 of the 8×8 fragments of the 16×128 output
  (TM_O=1, TN_O=8 for dQ; same for dK / dV)
- **Threads:** 4 simdgroups × 32 = 128

### `tensorops_flash_attention.metal` — M5 path (SDK 26+)

Planned FlashAttention-2 path on top of two `mpp::tensor_ops::matmul2d`
invocations (QK^T then SV) with an online softmax pass between them. The
host selector now records the first supported envelope, but public dispatch
still stays on the validated simdgroup-matrix implementation until M5
runtime evidence proves this backend numerically.

- **Tile:** Br = Bc = 64 (the M5 tensor units can sustain this without
  threadgroup memory pressure on the GEMM piece)
- **Two entry points:** D=64 and D=128
- **Initial promotion scope:** fully tiled sequence lengths, no LSE/window/
  ALiBi variants, fp16 IO + fp32 accum
- **v0.2 scope:** backward, GQA, ALiBi

Requires Xcode 26.0+ SDK (gated at CMake time via `TC_HAVE_METAL4`).

## Training kernels

### `training_kernels.metal` — RMSnorm / LayerNorm / RoPE / SwiGLU / softmax / AdamW

All the small ops that wrap a transformer training loop. fp16 IO, fp32
accumulators throughout.

Helpers:
- `tg_sum_broadcast(...)`: cross-simdgroup row reduction with broadcast
  back via threadgroup memory; the workhorse for normalization passes.

Kernels in this file (forward + backward where they exist):
- `tc_rmsnorm_forward`, `tc_rmsnorm_backward`
- `tc_layernorm_forward`, `tc_layernorm_backward`
- `tc_rope_forward` / `tc_rope_backward` (in-place on X/dX = [B, H, S, D])
- `tc_swiglu_forward`, `tc_swiglu_backward`
- `tc_softmax_forward`, `tc_softmax_backward`
- `tc_adamw_step` (fused; fp32 master params, fp16 or fp32 grads)

### `fused_norm_gemv.metal` — fused norm + GEMV (inference)

Eliminates the round-trip of the normalized intermediate: instead of
writing `Norm(X) → memory → GEMV(memory, W) → Y`, the kernels compute
RMSNorm or LayerNorm statistics inline and reapply normalization during
matmul accumulation.

- **One threadgroup per (output column n, row m)**
- **64 threads (2 simdgroups)**
- **TG memory:** tiny scratch for the row reductions
- **Design target:** M ≤ 4 (LLM inference). For training (M ≥ 32),
  callers should use the separate norm-forward + `tc_gemm` path -- per-row
  statistic recompute would dominate.

The hot path inside a Llama decode step is exactly:
```
x_norm = RMSnorm(x)
Q = x_norm @ Wq    K = x_norm @ Wk    V = x_norm @ Wv
```
which is three GEMVs that all consume `x_norm` once. The fused kernel
skips the intermediate write entirely.

## Conv2D

### `conv2d.metal` — im2col forward helper

Transforms `X[N, IC, H, W]` into `col[N, IC*kH*kW, oH*oW]` so the host
can call `tc_gemm` against `W_flat[OC, IC*kH*kW]` to produce `Y`.

- **Layout match:** `col[k * out_hw + n_oh_ow] = x[n, ic, h_in, w_in]`
  with `k = ic*kH*kW + kh*kW + kw`
- **dtype:** fp16

Used by `tc_conv2d_forward`.

### `conv2d_backward.metal` — col2im + dW staging

Two helpers:
- `tc_col2im_f16`: scatter `dCol` back into `dX`, accumulating where
  multiple `(kh, kw)` positions overlap the same input pixel. Uses fp32
  atomic adds into a scratch buffer; finalize casts to fp16.
- `tc_conv2d_dY_to_dCol_setup_f16`: no-op (we feed dY directly into
  GEMM); placeholder for future shape transforms.

The GEMM calls for `dW = sum_n dY[n] @ col[n]^T` and `dCol = W^T @ dY`
go through `tc_gemm` on the host.

## Shared header

### `metal_simdgroup_event.h`

Private AIR intrinsics for `simdgroup_async_copy` — the Apple GPU
equivalent of NVIDIA's `cp.async` (Ampere) or `TMA` (Hopper, but
without the dedicated copy engine). Sourced from Philip Turner's
metal-flash-attention reverse-engineered shim, validated against
Apple's leaked Xcode 14.2 headers.

Exposes:
- `simdgroup_event_t` (opaque event handle)
- `simdgroup_async_copy_2d(...)` (async DMA primitive)
- `__metal_wait_simdgroup_events(...)`

Only included by the `gemm_async*.metal` files. The `__asm("air.…")`
form is rejected by Xcode 17+, so CMake omits the async kernels from
the metallib when SDK ≥ 26.0; the host-side dispatch silently falls
back to the sync path.

## How to add a kernel

The pattern is mechanical:

1. Write the `.metal` source in `kernels/metal/`. Use `simdgroup_matrix`
   for matmul-shaped work; function constants (not preprocessor macros)
   for compile-time switches.
2. Append the path to `TC_METAL_SOURCES` in `CMakeLists.txt`. Gate on
   SDK version if needed.
3. Declare the public ABI in `include/tensorcore/<group>.h`.
4. Wire host dispatch in `lib/ops/<group>.mm`. Use
   `tc_pipeline_get(ctx, @"kernel_name", &err)`.
5. Write a correctness test in `tests/test_<group>.c` using `rms_scaled`
   vs an fp64 CPU reference.
6. Register in `tests/CMakeLists.txt`.
7. Bench it (`bench/bench_*.c` style).

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the long version and the
numerical-guarantee table you'll be measured against.
