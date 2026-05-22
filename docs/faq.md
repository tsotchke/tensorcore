# FAQ — common confusions

Field-collected questions that show up over and over. If your question
isn't here, [troubleshooting.md](troubleshooting.md) is the diagnostic
playbook for things that fail outright; this page is for things that
work but look weird.

## "My fp16 GEMM doesn't match cuBLAS to the last bit"

It's not supposed to. fp16 has 10 bits of mantissa; the order of
operations in a 4096-wide inner product is implementation-defined. Two
correct implementations on different hardware can differ by ~1 ULP per
output cell.

The right comparison is **rms_scaled error against an fp64 reference**:

```
rms_scaled = ||Y - Y_ref|| / (||Y_ref|| + ε)
```

`tc_gemm` fp16 lands at rms_scaled ≤ 5e-3 at 4096³. That's the
guarantee. See [numerics.md](numerics.md).

`tc_gemm` fp32 *is* bit-exact against `cblas_sgemm`; if you need
bit-equivalence, that's the path.

## "Why is async-batched 14× faster than sync-per-call?"

Per-call command-buffer overhead on M2 Ultra is ~50µs. A Q4_0 GEMV
kernel for a 7B-decode shape is ~25µs. Sync-per-call you pay 75µs per
call; async-batched you pay 50µs once per stream-sync and then 25µs per
call.

On a 32-layer decode step with ~6 GEMVs per layer (~200 total), sync
pulls ~15ms/token, async pulls ~5ms/token. See
[memory_model.md § Streams](memory_model.md#streams-and-command-buffers).

## "Why does `tc_last_backend()` not change after my training kernel call?"

It should change after served training, Conv2D, quantized, attention, and
GEMM dispatches. If it does not, confirm you are loading the current
native library and that the call actually reached a dispatch path rather
than returning an argument/shape/dtype error. Set `TC_TRACE=1` before
process start to print each served dispatch to stderr.

## "Why is my 128×128 GEMM tile *slower* than 64×64?"

Register pressure. The current `gemm_simdgroup_128.metal` uses 16
simdgroups × 32 = 512 threads per CTA with 16 fp32 accumulator fragments
per simdgroup. That's enough to throttle occupancy on Apple7-9.

The fix is the v0.2 retune: smaller per-simdgroup fragment count (TM=2,
TN=4) with WM=4×WN=2 sg grid. See [gemm.md § 128×128](gemm.md#the-128128-tile-opt-in).

Today: only use `TC_USE_128_TILE=1` if you're benching it; the default
64×64 is faster at all shapes on M1-M4.

## "Why does FlashAttention D=128 have lower throughput than D=64?"

Threadgroup memory budget. At D=128, fitting `sQ` + `sK` + `sV` + `sS` +
`sP` in 32 KB forces Br = Bc = 16. At D=64, Br = Bc = 32 fits
comfortably. Smaller tiles = lower arithmetic intensity = lower TFLOPS.

The v0.2 plan uses aliased threadgroup memory regions (reuse `sK` as `sP`
after K is consumed) to recover Br = Bc = 32 at D=128 on Apple9+. See
[attention.md § Why Br x Bc](attention.md#why-br-bc-32-32-at-d64-and-16-16-at-d128).

## "Why does the bench report different TFLOPS than `2 × M × N × K / time`?"

It doesn't — that's exactly the formula. But on a partial-shape causal
attention, only the lower-triangular portion of the score matrix is
computed; the bench counts that.

For FlashAttention forward with causal=true and S=4096, only ~50% of the
2 × B × H × S² × D FLOPs are real work. The bench harness counts the
full 2 × B × H × S² × D and divides by time; the reported TFLOPS is
"effective" — what the user perceives.

## "I see `[tensorcore] family=Apple8 bf16_sg=no` but I have an M3 Max"

`MTLGPUFamilyApple9` was introduced for M3 (which has bf16
`simdgroup_matrix`). If you see `Apple8` on an M3 chip, your macOS / SDK
combination is reporting the wrong family. Check `xcrun --show-sdk-version`
and update Xcode.

If you have M3 but the runtime classifies as Apple8, the bf16 path goes
through the fp32 fallback (still correct, slightly slower). Doesn't break
anything but you're not getting the native bf16 MMA.

## "Why is the Q4_0 bench `186 tok/s` but llama.cpp reports 55-65?"

Because the synthetic harness measures *Q4_0 GEMV throughput only* —
not full inference. It excludes attention, softmax, RoPE, and RMSnorm.
The kernel itself is faster than llama.cpp's because of the v2 rewrite
(NR0=4 outputs per simdgroup, NSG=2 simdgroups per TG, pre-scaled y
values with mask+FMA dequant) - see [kernels.md § gemm_quantized_v2](kernels.md#gemm_quantized_v2metal-q4_0-gemv-llamacpp-class-default).

End-to-end inference with real KV-cache + attention + sampling will
land lower than 186; the v0.2 integration milestone will tell us how
much lower. The kernel core isn't the bottleneck anymore.

## "My GGUF model has 200 tensors loaded and 50 skipped"

The skipped tensors are using a GGUF encoding tensorcore doesn't handle
yet — most commonly Q4_K_M, Q5_K_M, or one of the other k-quant
variants. Check via:

```c
for (uint64_t i = 0; i < tc_gguf_tensor_count(gguf); ++i) {
    tc_gguf_tensor_info info;
    tc_gguf_tensor_at(gguf, i, &info);
    if (info.type == TC_GGUF_TYPE_UNSUPPORTED)
        printf("skipped: %s\n", info.name);
}
```

`v0.1` supports F32, F16, BF16, Q4_0, Q4_1, Q8_0. The k-quant family is
on the v0.2 list. See [gguf.md](gguf.md).

## "Why does my training step crash after a few iterations with NaN?"

Almost always loss scaling. fp16 has dynamic range too narrow to
represent very small gradients without scaling them up first; if the
loss multiplier scales activations above fp16's ~6.5e4 max, you get NaN.

tensorcore doesn't manage the scaler — your code does:

```c
loss = loss * scale;
/* backward */
/* gradients are now `scale` times too large */
grads = grads / scale;
/* optimizer step uses the un-scaled grads */
```

Dynamic loss scaling (halve the scale when NaN, otherwise double every
N steps) is the standard recipe. See PyTorch's `GradScaler` for the
algorithm; the bookkeeping is yours, the compute primitives are ours.

bf16 has fp32's exponent range and doesn't need loss scaling — that's
why bf16 is the preferred training dtype on chips that support it
natively (Apple9+).

## "Should I use `tc_gemm` or `tc_fused_rmsnorm_gemv` for QKV projection?"

`tc_fused_rmsnorm_gemv` for inference (M ≤ 4). It's measurably faster
because it skips the intermediate `x_norm` write/read.

`tc_rmsnorm_forward + tc_gemm` for training (M ≥ 32). The per-row
`rstd` recomputation cost in the fused kernel scales with M, which
overwhelms the savings at training batch sizes. Plus you need `rstd_out`
for the backward pass, which the fused kernel doesn't expose.

See [training_kernels.md § Fused Norm + GEMV](training_kernels.md#fused-norm-gemv).

## "Why are there separate `_async` versions of everything?"

Because the choice between sync and async is *yours*, not the library's.
`tc_gemm` and `tc_gemm_async` differ only in whether they sync at the
end. Some workloads want sync ergonomics (one-shot computation, debug
visibility), others want async batching (steady-state inference /
training).

The same descriptor, the same kernel, same dispatch path. The async
variants just don't commit the command buffer until you `tc_stream_sync`.

## "Why is FlashAttention faster than attention without it?"

Less memory traffic. A naive implementation materializes the `B × H × S
× S` attention-score matrix; FlashAttention computes it in tiles and
never writes it back to memory. At S=4096, that's an extra 64 MB per
head per batch you don't need to round-trip.

The throughput win on Apple Silicon is bigger than on H100 because we
have lower memory bandwidth (~800 GB/s vs ~3 TB/s). When you save 64 MB
of read+write, you save more relative time on us.

See [attention.md](attention.md) and [the FlashAttention paper](https://arxiv.org/abs/2205.14135).

## "What's the relationship to MLX?"

MLX is Apple's array framework — like NumPy/PyTorch for Apple Silicon.
It has its own kernel library; ours overlaps for matmul and attention.

Practical differences:
- **MLX is a framework.** Arrays, autograd, optimizers, model classes.
- **tensorcore is a kernel library.** C ABI, no autograd, no framework
  abstractions.
- **MLX's matmul is hand-tuned**; ours is within ~10% at fp16 GEMM and
  faster on Q4_0 GEMV (per the synthetic bench).

You can use both: MLX for high-level model code, tensorcore for specific
hot paths where the C ABI integration matters (Eshkol, custom inference
runtimes, C/C++ projects that don't want a Python dependency).

The roadmap (`v0.7`) is to be a backend target for MLX/PyTorch/JAX/ONNX.

## "What's the relationship to MetalPerformanceShaders (MPS)?"

MPS is Apple's official high-level GPU primitive library, callable from
Objective-C/Swift. It includes GEMM, conv, BNN, image ops.

tensorcore wraps MPS as a fallback — when our `simdgroup_matrix` kernel
doesn't cover a shape (e.g. very small or odd-shape GEMM), the dispatch
routes through `MPSMatrix`. `tc_last_backend()` reports `TC_BACKEND_MPS`
when this happens.

Why not just use MPS for everything? MPS is closed-source, doesn't
expose `mpp::tensor_ops` (the M5 path), doesn't support GGUF-style
quantization, and the C ABI is ObjC++ only. We need our own kernels for
the perf-sensitive paths; MPS catches the long tail.

## "Why is the version triple checked across three files?"

Three sources of truth would normally be bad design, but each has a
different role:

- `pyproject.toml` — what `pip install` sees.
- `CMakeLists.txt::project(VERSION ...)` — what `cmake --install` and
  `tensorcore.pc` and `find_package` see.
- `include/tensorcore/tensorcore.h::TENSORCORE_VERSION_*` — what
  downstream C / C++ code can read at compile time without a runtime
  call.

`scripts/check_version_consistency.sh` asserts all three agree. CI runs
it on every push. See [ci_and_scripts.md](ci_and_scripts.md).

## "Why doesn't the Python `__init__.py` use `pybind11` or `cython`?"

Because it doesn't need to. The C ABI is `extern "C"`, the structs are
POD, and `ctypes` is in the standard library. No build step, no C
compilation per Python version, no `pip install` dance with native
extensions.

The wheel does ship the native dylib + metallib inside the package
(v0.1.8+), so end users `pip install` and it works. The native runtime
is built once on macos-15 in CI and packaged; Python is along for the
ride.

## "Where do I file an issue?"

GitHub: `https://github.com/tsotchke/tensorcore/issues`. Include:

- Chip (`tc_device_info.name`)
- macOS version (`sw_vers`)
- Xcode SDK (`xcrun --show-sdk-version`)
- Failing call's shape, dtype, descriptor
- `tc_last_backend()` immediately after the call
- A minimal reproducer if you can produce one (`hello_gemm.c` style)
