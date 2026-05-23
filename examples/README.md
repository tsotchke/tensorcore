# examples/

Minimal, compilable demonstrations of the tensorcore C ABI. Each example
is a single `.c` file that links only against `libtensorcore.dylib` and
can be read end-to-end in a sitting.

All examples build automatically as part of the main project:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/examples/<name>
```

| Example | Demonstrates | Runtime |
|---|---|---|
| `hello_gemm.c` | Minimal fp16 GEMM, print first cells, report backend | <100ms |
| `gguf_inspect.c` | Open a GGUF file, walk tensors, dump metadata, optionally copy a tensor into a `tc_buffer` | <1s |
| `decode_step.c` | One full synthetic Llama decode step end-to-end (RMSnorm + Q4_0 GEMVs + RoPE + FlashAttention + SwiGLU + residual) | ~30ms / 2 layers |
| `training_step.c` | One full training iteration: RMSnorm + Linear + softmax forward, backward through softmax+CE / Linear / RMSnorm, AdamW on weights + gamma | ~10ms / step |
| `mesh_training_demo.c` | Split-rank training loop with RMSNorm, GEMM, softmax+CE, AdamW, DiLoCo outer sync, and GLOO rendezvous flags | ~10ms single-rank; network-dependent multi-rank |
| `native_sdk_consumer/` | Standalone C and C++ consumers for an installed native SDK, shared/static CMake targets, and pkg-config smoke source | build-only |

## `hello_gemm.c` (60 lines)

The "is the library wired up" smoke test. Does one fp16 GEMM at 256³,
prints the first 4×4 block of the output, and reports
`tc_backend_name(tc_last_backend())`. If you see `backend=simdgroup_matrix`
you're on the fast path.

Use it to confirm the build works on your chip.

## `gguf_inspect.c` (165 lines)

CLI inspector for GGUF v3 files.

```sh
./build/examples/gguf_inspect model.gguf                 # metadata + tensor list
./build/examples/gguf_inspect model.gguf attn_q          # copy one tensor to a buffer
./build/examples/gguf_inspect model.gguf --load-supported # bulk-load every supported tensor
```

Useful for verifying that a GGUF parses correctly under tensorcore's
reader and for ad-hoc inspection of metadata fields. See
[../docs/gguf.md](../docs/gguf.md) for the reader API.

## `decode_step.c` (270 lines)

The Llama-architecture inference assembly described in
[../docs/inference.md](../docs/inference.md), as compilable C. Runs two
synthetic transformer layers against randomly-initialized Q4_0 weights.

Per-layer call sequence:

1. `tc_fused_rmsnorm_gemv` — Q projection with inlined RMSnorm.
2. `tc_gemv_quantized_async × 2` — K and V projections on a stream.
3. `tc_stream_sync`.
4. `tc_rope_forward × 2` — RoPE on Q and K in-place.
5. `tc_attention_forward` — full FlashAttention with causal mask.
6. `tc_gemv_quantized` — output projection.
7. `tc_fused_rmsnorm_gemv` — MLP gate projection.
8. `tc_gemv_quantized_async` + `tc_stream_sync` — up projection.
9. `tc_swiglu_forward`.
10. `tc_gemv_quantized` — down projection.

No tokenization, sampling, KV-cache management, or real-model
integration — that's the v0.2 deliverable. This is the matrix-layer
plumbing, isolated.

Output:

```
[layer 0] backend after MLP-down: simdgroup_matrix
[layer 1] backend after MLP-down: simdgroup_matrix
[decode] 2 layers ran in 31.1ms
```

## `training_step.c` (280 lines)

The forward + backward + optimizer assembly described in
[../docs/training_loop.md](../docs/training_loop.md), as compilable C.
Runs a small `RMSnorm → Linear → softmax` block on synthetic data for 15
steps. Demonstrates:

- `tc_rmsnorm_forward` + `tc_rmsnorm_backward` (with **fp32** dgamma —
  matches the kernel's accumulator dtype).
- `tc_gemm` forward + `tc_gemm` with `transpose_a` for dW + `tc_gemm`
  with `transpose_b` for dX.
- `tc_softmax_forward`. The fused softmax+CE gradient is computed on the
  host (`probs - one_hot`) and fed directly into `dlogits` — `tc_softmax_backward`
  is **deliberately not called** because the closed-form gradient absorbs
  the Jacobian.
- `tc_adamw_step` with fp32 master weights and fp16/fp32 gradients.

Output:

```
step  1  loss=3.9827
step  5  loss=2.4490
step 10  loss=0.9505
step 15  loss=0.2549
```

Loss decreases monotonically — the assembly is numerically correct.

## `mesh_training_demo.c` (400 lines)

Runnable end-to-end mesh training demo. Each process owns a small local
training shard:

1. `tc_rmsnorm_forward`.
2. `tc_gemm` for the linear projection.
3. `tc_softmax_forward` plus host cross-entropy gradient.
4. `tc_gemm` backward for `dW` and `dX`.
5. `tc_rmsnorm_backward`.
6. `tc_adamw_step` on fp32 master weights and gamma.
7. `tc_diloco_step` / `tc_diloco_apply_outer` for outer synchronization.

Single-rank smoke:

```sh
./build/examples/mesh_training_demo --inner 2 --outer 1
```

Two or more ranks use the same executable with one process per host:

```sh
./mesh_training_demo --rank 0 --world 2 --url tcp://100.x.y.z:9100
./mesh_training_demo --rank 1 --world 2 --url tcp://100.x.y.z:9100
```

Output reports rendezvous time, per-outer loss, DiLoCo bytes sent, final
outer-step count, and elapsed wall time. The single-rank mode is also
registered as `example_mesh_training_demo` in CTest.

## `native_sdk_consumer/`

Out-of-tree consumer fixture for release artifacts and downstream
projects. It uses `find_package(tensorcore CONFIG REQUIRED)` against an
installed prefix, builds shared/static C consumers plus a C++ consumer,
and verifies public ABI helpers without requiring a real GPU. Set
`TC_CONSUMER_RUN_INIT=1` to additionally prove runtime initialization on
the current host.

## Reading order for a new contributor

1. `hello_gemm.c` — proves your environment works.
2. `gguf_inspect.c` — touches the GGUF reader.
3. `decode_step.c` — inference assembly.
4. `training_step.c` — training assembly.
5. `mesh_training_demo.c` — split-rank training + DiLoCo assembly.
6. `native_sdk_consumer/` — proves an installed SDK works out of tree.

Each example assumes you've read the corresponding doc:

- `hello_gemm` ↔ [../docs/api_reference.md](../docs/api_reference.md)
- `gguf_inspect` ↔ [../docs/gguf.md](../docs/gguf.md)
- `decode_step` ↔ [../docs/inference.md](../docs/inference.md)
- `training_step` ↔ [../docs/training_loop.md](../docs/training_loop.md) + [../docs/training_kernels.md](../docs/training_kernels.md)
- `mesh_training_demo` ↔ [../docs/diloco.md](../docs/diloco.md) + [../docs/deployment.md](../docs/deployment.md)

## Extending

Adding a new example follows the same pattern as adding a kernel — see
[../docs/extending.md](../docs/extending.md). Append the executable to
`examples/CMakeLists.txt`, link `tensorcore` + `-lm` (if you use math
functions), and write the `.c` file.
