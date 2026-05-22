# Glossary

Terms used in the tensorcore codebase + docs. Sorted alphabetically.

### Accelerate

Apple's CPU-side numerical library, including `cblas_sgemm`. tensorcore
uses Accelerate as the last-resort fallback when neither the GPU
`simdgroup_matrix` path nor MPS can serve a call. The fp32 `tc_gemm` is
bit-exact against `cblas_sgemm`.

### accum_dtype

The dtype used for accumulation inside a kernel. Almost always **fp32**
(or i32 for int8 inputs), regardless of input/output dtype. The
fp32-accumulator-everywhere rule is what makes mixed-precision training
stable.

### AIR

Apple Intermediate Representation. The LLVM-IR-like format Metal kernels
compile to. tensorcore's async-copy kernels use private AIR `__asm`
intrinsics; Xcode 17+ rejects this form, so we compile the async kernels
only when SDK < 26.0.

### ALiBi

Attention with Linear Biases. A relative-position scheme where each
attention score gets a `-slope * (i - j)` term added, instead of using
RoPE. BLOOM uses it. `tc_attention_desc.alibi_slopes` enables it.

### Apple7..Apple11

Apple's GPU family identifiers, matching `MTLGPUFamilyApple{N}`. Apple7
= M1, Apple8 = M2, Apple9 = M3 / A17 Pro, Apple10 = M4, Apple11 = M5.
Each family adds capabilities — bf16 MMA on Apple9+, int8 MMA on
Apple10+, `mpp::tensor_ops` on Apple11+.

### autotune

`lib/core/autotune.cpp`'s one-time sweep that picks GEMM and attention
tile shapes per (family, dtype, shape) at first init. Cached on disk for
subsequent inits.

### batch (in batched GEMM)

Number of independent matrix multiplies to run with the same shape and
strides. `tc_gemm_batched` issues one dispatch per batch element with
buffer offset binding.

### bf16

bfloat16 — 16-bit float with 8-bit exponent (same range as fp32) and
7-bit mantissa. Native MMA on Apple9+; software fallback (bit-cast to
fp32) on Apple7..8.

### Br, Bc

Block-row and block-column tile sizes in FlashAttention. At D=64 we use
Br=Bc=32; at D=128 we use Br=Bc=16. See
[attention.md](attention.md).

### BM, BN, BK

Block-M, block-N, block-K tile sizes in GEMM. Default tile is BM=BN=64,
BK=32 — 64 rows of A × 64 cols of B × 32-deep K-block per CTA.

### causal mask

In self-attention, the convention that query token `i` can only attend
to keys at positions `j ≤ i`. Standard for autoregressive LLMs.
`tc_attention_desc.causal = true`.

### cblas_sgemm

The Accelerate BLAS routine for fp32 GEMM. tensorcore's fp32 path is
bit-exact against this; cblas_sgemm is the reference oracle.

### CB / command buffer

Metal's `MTLCommandBuffer`. A list of encoded GPU operations that gets
committed to the device queue. Per-CB setup cost is ~50µs on M2 Ultra;
amortizing this is what the `tc_stream` async pattern is for.

### CTA / threadgroup

Metal's threadgroup. A group of threads (up to 1024) that share
threadgroup memory and can synchronize via barriers. The GEMM kernel
uses 128 threads/CTA arranged as 4 simdgroups.

### cuBLAS / cuDNN / CUTLASS / NCCL

NVIDIA's CUDA-side libraries for matmul (cuBLAS), neural-net ops
(cuDNN), kernel building blocks (CUTLASS), and collective communications
(NCCL). tensorcore replaces all four for Apple Silicon — see
[cuda_comparison.md](cuda_comparison.md).

### DF64

Double-float: an emulated fp64-equivalent using two fp32 values with
unevaluated sum. ~16 decimal digits, ~4× fp32 cost. See
[precision_emulation.md](precision_emulation.md).

### dtype

Data type. `tc_dtype_t` enumerates the supported set: F16, BF16, F32,
I8, I32, F64, SF64, DF64, FP24, FP53. See [dtypes.md](dtypes.md).

### Eshkol

The compiler/runtime hosted in `~/Desktop/eshkol/` (canonical main
branch) and `~/Desktop/eshkol-platform/` (active development).
tensorcore exposes `__tc-*` builtins via the bridge file
`eshkol/bridge/tensorcore_codegen.cpp`. See
[eshkol_integration.md](eshkol_integration.md).

### family / `tc_family_t`

Apple's GPU family classification at runtime. See **Apple7..Apple11**.

### FlashAttention / FA / FA-2

Memory-efficient attention algorithm (Dao 2022 / 2023) that fuses
QK·softmax·PV into one pass without materializing the `S × S` score
matrix. tensorcore's `tc_attention_forward` is FA-2; backward uses the
LSE-saved scheme.

### fp16

IEEE 754 binary16. 16-bit float with 5-bit exponent and 10-bit
mantissa. Native MMA on Apple7+. Default training/inference dtype.

### fp32

IEEE 754 binary32. Default accumulator dtype. The reference precision —
`tc_gemm` fp32 is bit-exact vs Accelerate.

### fp64

IEEE 754 binary64. Apple GPUs have no native fp64 unit. tensorcore
emulates it via SF64 (SoftFloat) or DF64 (double-float).

### FP24 / FP53

Custom precision formats from `eshkol-platform`. FP24 is 24-bit (sign +
8 exp + 15 mantissa); FP53 is 53-bit mantissa with extended exponent
range. See [precision_emulation.md](precision_emulation.md).

### function constant

Metal's compile-time-switched specialization mechanism.
`MTLFunctionConstantValues` set at pipeline-creation time produces a
specialized variant of a kernel with the constants inlined — no runtime
branch. Used in `flash_attention.metal` (`g_causal`, `g_use_lse`,
`g_use_window`, `g_use_alibi`).

### GGML / GGUF

`ggml` is Georgi Gerganov's portable ML library; GGUF is its model file
format (v3 supported). tensorcore reads GGUF via `lib/io/gguf.c` and
shares Q4_0/Q8_0 block layout with ggml.

### GQA / MQA

Grouped-Query / Multi-Query attention. The number of KV heads is less
than the number of Q heads — multiple query heads share a KV head.
Llama-2 70B uses 8:1 GQA; PaLM uses MQA (H:1). Configured via
`tc_attention_desc.kv_heads`.

### im2col

The transformation that turns a Conv2D into a GEMM by unrolling each
output position's receptive field into a column. tensorcore's
`tc_conv2d_forward` does im2col + `tc_gemm`.

### JACCL

Apple's collective comms library (analogue of NCCL) introduced in macOS
26.2 alongside Thunderbolt-5 networking. tensorcore's `TC_DIST_RING`
backend will sit on JACCL once it ships; today the single-host ring
(`tc_distributed_ring`) and portable CPU `TC_DIST_GLOO` TCP backend are
functional.

### LSE

Log-Sum-Exp. The numerically-stable summation in attention's softmax
denominator. tensorcore saves LSE per query token from the forward pass
(when `return_lse=true`) so the backward pass can recompute the
softmax probabilities without re-materializing the score matrix.

### M-series (M1 / M2 / M3 / M4 / M5)

Apple's marketing names for Apple Silicon SoC generations. Maps to
Apple7..Apple11 GPU families respectively. Plus product variants — M*
Max, M* Ultra — that differ in core count but share the family
identifier.

### Mac Studio / unified-memory ceiling

Apple's desktop form factor. M2/M3/M4 Studios ship with up to 192 GB of
unified memory; M5 Ultra is rumored to push higher. The single largest
memory-pool model in the Apple Silicon ecosystem.

### Metal

Apple's GPU compute + graphics API. tensorcore wraps Metal 3 and Metal
4 (the latter via SDK 26.0+ gating).

### Metal 4

The Metal API version that introduced `mpp::tensor_ops` and
`MTL4MachineLearningCommandEncoder` (the M5 path). Requires Xcode
26.0+ and macOS 26.0+.

### Metal Performance Shaders / MPS

Apple's high-level GPU primitives library. tensorcore wraps MPSMatrix as
a fallback when its own kernels don't cover a shape.

### metallib

The compiled output of one or more `.metal` files; contains AIR
bitcode for every kernel. tensorcore ships `tensorcore.metallib`
alongside the dylib.

### MFA / metal-flash-attention

Philip Turner's open-source FlashAttention port to Metal. The async-copy
GEMM kernel patterns in tensorcore are MFA-derived. See
`kernels/metal/metal_simdgroup_event.h`.

### MMA

Matrix-Multiply-Accumulate. The unit operation of tensor cores
(NVIDIA's `mma.sync`) and simdgroup_matrix (Metal's `simdgroup_multiply_accumulate`).
tensorcore's GEMM kernel decomposes a tile into 8×8 MMA fragments.

### `mpp::tensor_ops`

Apple's Metal Performance Primitives namespace for the M5 neural-
accelerator-driven matmul path. Reachable via `matmul2d`.

### MTLBuffer / MTLDevice / MTLCommandQueue

Metal's resource and command primitives. `tc_buffer*` wraps
`MTLBuffer`; `tc_context*` holds the `MTLDevice` + `MTLCommandQueue`.

### MFU

Model FLOPs Utilization. Achieved TFLOPS / theoretical peak TFLOPS. The
v0.2 target is ~75% MFU on fp16 GEMM at 4096³.

### Ozaki-II

A CRT-based exact GEMM scheme (the "Ozaki Scheme") from the precision-
arithmetic literature. tensorcore has a `TC_BACKEND_OZAKI_II` enum
value reserved for a future high-precision research path; not yet
implemented.

### pipeline cache

`lib/core/pipeline_cache.mm`. Stores compiled `MTLComputePipelineState`
objects keyed by (function name, function constants). Pipelines are
compiled lazily on first use (~5-50ms per kernel) and cached for the
process lifetime.

### Q4_0 / Q4_1 / Q8_0 / Q*_K_M

GGML block-quantization formats. Q4_0 = 32 weights/block, fp16 scale,
4-bit weights = 4.5 bits/weight. Q8_0 = 8.5 bits/weight. The Q*_K_M
"k-quant" family (Q4_K_M, Q5_K_M) is more complex; not yet supported
by tensorcore (v0.2).

### rms_scaled (error metric)

`||Y - Y_ref|| / (||Y_ref|| + ε)`. Energy-weighted error metric used
throughout the test suite. Robust to per-cell rounding noise. See
[numerics.md](numerics.md).

### RMSnorm

Root-mean-square normalization. `y = (x / rms(x)) * gamma` where
`rms(x) = sqrt(mean(x²) + eps)`. The norm used by Llama, Mistral,
Qwen, Gemma. `tc_rmsnorm_forward`.

### RoPE

Rotary Position Embedding. Encodes position by rotating pairs of
embedding dimensions by an angle derived from position × frequency.
`tc_rope_forward` operates in place on Q/K.

### scratch buffer

A caller-allocated workspace buffer for ops that need intermediate
storage (e.g. `tc_conv2d_*`'s im2col col-buffer, `scratch_dX_f32`).
Reused across iterations to avoid repeated allocation.

### SF64

SoftFloat-64. Pure-software IEEE 754 binary64 implemented in MSL.
Bit-exact agreement with x86-64 fp64 under the same rounding mode.
~10-20× slower than fp32. Backend reports `TC_BACKEND_SF64_EMULATED`.

### simdgroup / SIMD width

The Metal equivalent of NVIDIA's "warp": 32 threads that execute in
lockstep. Apple7+ uses 32-wide simdgroups. `thread_execution_width = 32`
in `tc_device_info`.

### simdgroup_matrix

Metal's matrix-multiply-accumulate primitive operating on 8×8 fragments.
The default tensorcore GEMM path. fp16 / fp32 supported on Apple7+;
bf16 on Apple9+; int8 on Apple10+. Software fallback via fp32 on older
chips.

### softmax_scale

The factor `1 / sqrt(head_dim)` applied to attention scores before the
softmax. Stored in `tc_attention_desc.softmax_scale`.

### stream

`tc_stream*`. A lane in which async ops accumulate into one
`MTLCommandBuffer`, committed by `tc_stream_sync`. The mechanism that
makes the 7B Q4_0 decode bench reach 186 tok/s vs ~10 tok/s
sync-per-call.

### sw_vers / `xcrun --show-sdk-version`

macOS shell commands that report the OS version and Xcode SDK version
respectively. CMake reads `xcrun --show-sdk-version` to gate the Metal
4 path.

### TB5 / Thunderbolt 5

80 Gbps bidirectional, ~10 GB/s steady-state. The transport layer for
multi-Mac distributed training. tensorcore's `TC_DIST_RING` will use
TB5 + JACCL.

### tensor core (NVIDIA)

The matrix-multiply-accumulate unit on NVIDIA GPUs since Volta
(`sm_70`). Apple Silicon's equivalent is `simdgroup_matrix`; M5+'s
neural accelerator adds `mpp::tensor_ops`.

### `tc_*` (prefix)

Public C ABI prefix. All entry points in `include/tensorcore/*.h` begin
with `tc_`. The Eshkol bindings use `__tc-*` for the underlying
builtins and `tc-*` for the user-facing wrappers (see
[eshkol_integration.md](eshkol_integration.md)).

### TFLOPS

Tera-floating-point-operations-per-second. Standard throughput unit for
GEMM. tensorcore's fp16 4096³ GEMM lands at ~17.88 TFLOPS on M2 Ultra
(~66% of theoretical peak).

### threadgroup memory / TG memory

On-chip scratch memory shared by threads in a CTA. 32 KB budget on
M-series M1..M5. The single biggest constraint on Apple Silicon kernel
design.

### Tile

Subdivision of the GEMM output produced by one threadgroup. tensorcore's
default tile is 64×64; the experimental large tile is 128×128.

### tok/s

Tokens per second. The natural throughput unit for LLM inference. Q4_0
7B decode bench on M2 Ultra: 186 tok/s pure-GEMV.

### ULP

Unit in the Last Place. The smallest representable difference at a given
floating-point value. fp16 results across chips agree within ~1 ULP per
cell; fp32 against Accelerate agrees to 0 ULP (bit-exact).

### unified memory / UMA

The single address space shared by CPU and GPU on Apple Silicon. No
host↔device copies. `tc_buffer_map` returns a CPU-addressable pointer
to the same bytes the GPU reads.

### vec4

A 4-element vector type in Metal (`half4`, `float4`, `char4`). Used for
cooperative loads in the GEMM kernel — one thread loads 4 elements per
issue.

### WM, WN

Simdgroup grid dimensions in a threadgroup. WM=2 × WN=2 = 4 simdgroups
per threadgroup in the default 64×64 GEMM tile.

### YaRN

Yet another Rope Network — a long-context RoPE scaling scheme. Used by
some 1M-token models. Not yet implemented in tensorcore; mentioned in
the context of the `Qwen3` Eshkol-side use case.

### ZeRO-1 / ZeRO-2 / ZeRO-3

DeepSpeed-style optimizer/gradient/parameter sharding tiers. v0.5 of
tensorcore's distributed path ships the primitives needed for all three.
