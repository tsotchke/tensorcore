# Codebase audit — what ICC sees

This page is a snapshot of what the
[Infinite Context Coder](https://github.com/tsotchke/infinite_context_coder)
(ICC) deterministic codebase tool surfaced when indexing tensorcore. It's a
ground-truth pass: structural facts derived from the source by tree-sitter
parsers and cross-file edge resolution, not a hand-written summary.

Re-run whenever the source moves:

```sh
ICC_HOME=~/Desktop/infinite_context_coder
$ICC_HOME/bin/icc register --name tensorcore --path ~/Desktop/tensorcore \
    --skip-dir build --skip-dir CMakeFiles --skip-dir .cache --skip-dir .claude
$ICC_HOME/bin/icc index --repo tensorcore
$ICC_HOME/bin/icc build-memory --repo tensorcore
$ICC_HOME/bin/icc architecture-summary --repo tensorcore --bundle --include-cheatsheet
```

## Indexed surface (v0.1.22+ checkpoint, post-doc-overhaul)

| Metric | Value |
|---|---:|
| Files indexed | 142 |
| Total lines indexed | 29,879 |
| Languages | C, Metal, ObjC++, C headers, Markdown, CMake, Python, C++, Eshkol, TOML, shell, text |
| Public symbols in `include/tensorcore/*.h` | 86 |
| Python binding symbols | 187+ |
| Call-graph edges (total) | 1,777 |
| Call-graph edges (resolved cross-file) | 924 |
| Call-graph edges (unresolved system / runtime) | 822 |
| Call-graph edges (ambiguous) | 31 |
| Doc coverage (public symbols) | 159 / 205 (78%) |
| Unreferenced docs (orphan files) | 0 / 32 |
| Test suite | 22 / 22 passing (20 correctness + 2 example smokes), 2-3s |

## Public module roots

- `lib/core` — device init, pipeline cache, buffer pool, autotune
- `lib/ops` — gemm, attention, training, conv, quantized
- `lib/distributed` — distributed primitives
- `lib/fallback` — MPS / Accelerate fallback paths
- `lib/c_api` — ABI shims
- `lib/io` — GGUF reader
- `lib/tensorops` — M5 Metal 4 path

## Largest files

| Lines | File | Language |
|---:|---|---|
| 1,396 | `python/tensorcore/__init__.py` | Python |
| 803 | `lib/io/gguf.c` | C |
| 782 | `python/tests/test_basic.py` | Python |
| 544 | `kernels/metal/flash_attention_backward.metal` | Metal |
| 513 | `lib/ops/attention.mm` | ObjC++ |

## Most-included headers

| Includers | Header |
|---:|---|
| 37 | `include/tensorcore/tensorcore.h` |
| 13 | `lib/core/internal.h` |
| 11 | `include/tensorcore/status.h` |
| 10 | `include/tensorcore/dtype.h` |
| 9 | `include/tensorcore/device.h` |
| 3 | `include/tensorcore/quantized.h` |
| 3 | `include/tensorcore/distributed.h` |
| 2 | `include/tensorcore/gemm.h` |
| 2 | `include/tensorcore/conv.h` |
| 2 | `include/tensorcore/gguf.h` |

The umbrella header (`tensorcore.h`) is included from 37 sites — the
intended pattern.

## `tc_gemm` call graph (depth 2 callees)

```
lib/ops/gemm.mm::tc_gemm
├── lib/ops/gemm.mm::kernel_for           ← picks the path
│   ├── lib/ops/gemm.mm::use_128_tile     ← reads TC_USE_128_TILE=1
│   └── lib/ops/gemm.mm::use_async_kernel ← reads TC_USE_ASYNC=1 (gated by SDK)
├── lib/ops/gemm.mm::resolve_pipeline     ← pipeline-cache lookup
├── lib/ops/gemm.mm::validate
├── lib/ops/gemm.mm::validate_gemm_buffers
├── lib/core/device.mm::tc_set_last_backend
├── lib/tensorops/tensorops_m5.mm::tc_tensorops_gemm_attempt
│   └── lib/core/device.mm::tc_set_last_backend
└── lib/fallback/mps_gemm.mm::tc_mps_gemm
    ├── lib/fallback/mps_gemm.mm::bf16_via_fp32  ← Apple7..8 bf16 fallback
    ├── lib/fallback/mps_gemm.mm::i8_via_fp32    ← Apple7..9 int8 fallback
    └── lib/fallback/mps_gemm.mm::to_mps_dtype
```

This confirms the [architecture.md](architecture.md) fallback-ladder
description: simdgroup_matrix → TensorOps M5 → MPS (which itself contains
the bf16/int8 software fallbacks).

## `tc_set_last_backend` call sites

Found with `grep -rn`, cross-checked with `icc trace-callers`:

- `lib/ops/gemm.mm` — 5 sites (SIMDGROUP, MPS, NONE for failure)
- `lib/ops/attention.mm` — 3 sites (all SIMDGROUP)
- `lib/tensorops/tensorops_m5.mm` — 2 sites (TENSOROPS_M5)

The training, conv, and quantized ops do **not** update
`tc_last_backend`. This is a v0.2 polish item — and the reason
`tc_last_backend()` is documented as "last-GEMM-like" rather than
"last-call." See [api_reference.md § GEMM](api_reference.md#gemm-gemmh).

## Dead-code candidates

ICC's `find-dead-code` (best-effort; function pointers and CLI dispatch
aren't followed) flagged these. Treat as breadcrumbs, not certainties:

| Symbol | File | Lines | Note |
|---|---|---:|---|
| `tg_sum32` | `kernels/metal/fused_norm_gemv.metal:29-85` | 57 | Helper; check whether the new fused-norm-gemv path is wired |
| `QuantizedMatrix.gemv_async` | `python/tensorcore/__init__.py:1449-…` | — | Convenience method; verify a downstream user before deleting |

Plus ~5 more in test scaffolding (not load-bearing). Not blockers; useful
cleanup for a future tightening pass.

## Header constants — derived from the C ABI

ICC's `architecture-cheatsheet` emits the full enum/define table verbatim.
Excerpts:

- `tc_family_t`: `APPLE7..APPLE11` (M1 → M5)
- `tc_dtype_t`: 10 dtypes; first-class F16/BF16/F32/I8/I32; emulated F64/SF64/DF64/FP24/FP53
- `tc_backend_t`: NONE, SIMDGROUP_MATRIX, TENSOROPS_M5, MPS, ACCELERATE_CPU, SF64_EMULATED, OZAKI_II
- `tc_status_t`: TC_OK plus 11 error codes
- `tc_dist_backend_t`: SINGLE, RING, GLOO
- `tc_reduce_op_t`: SUM, AVG, MAX, MIN
- `tc_gguf_type_t`: F32, F16, Q4_0, Q4_1, Q8_0, BF16, UNSUPPORTED
- `tc_quant_t`: Q4_0, Q8_0

The full table is in
`artifacts/repos/tensorcore/architecture/` after `icc architecture-summary`.

## Why this audit matters

The "doc overhaul" pass that produced this directory was grounded in:
- the headers (read in full),
- the ICC architecture-summary + cheatsheet,
- `icc trace-callees tc_gemm`,
- `icc trace-callers tc_set_last_backend`,
- `icc find-clusters` on the larger files,
- targeted grep verification when ICC's tree-sitter resolution lost edges.

This is the practice we want to keep: docs as a verified projection of
the codebase, re-run when the source moves. ICC is the verification tool;
[memory.md](codebase_audit.md) (this file) is its public summary.
