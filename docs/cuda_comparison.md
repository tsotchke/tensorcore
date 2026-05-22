# tensorcore for CUDA people

`tensorcore` is what you reach for on Apple Silicon when you would have
reached for CUDA on NVIDIA. This page is the direct mapping. If you have
muscle memory for cuBLAS or cuDNN, this is the cheat sheet that gets you
productive in an afternoon.

## The thesis

NVIDIA's moat on AI is **software stack maturity × silicon × interconnect**.
Two of those three are pure software (cuBLAS, cuDNN, CUTLASS, NCCL,
Triton, TensorRT). On Apple Silicon, MLX and MPS each cover a slice, but
nothing is the single, hardware-aware library that turns the matrix units
(`simdgroup_matrix` on M1-M4, `mpp::tensor_ops` on M5) into a training-grade
foundation.

`tensorcore` is that library. It's the Apple-side answer to *the entire CUDA
software stack* at the kernel layer.

| NVIDIA stack | Apple stack with `tensorcore` |
|---|---|
| CUDA Runtime | Metal 3.1 / Metal 4 (we wrap it) |
| cuBLAS | `tc_gemm` (fp16, bf16, fp32, i8; sync + async + batched) |
| cuDNN convolutions | `tc_conv2d_forward`, `tc_conv2d_backward_input`, `tc_conv2d_backward_weight` |
| cuDNN attention / FlashAttention | `tc_attention_forward`, `tc_attention_backward` (D=64, D=128, causal, GQA, sliding window, ALiBi) |
| cuDNN normalizations | `tc_rmsnorm_forward/backward`, `tc_layernorm_forward/backward` |
| cuDNN activations | `tc_swiglu_forward/backward`, `tc_softmax_forward/backward` |
| Apex / FusedAdam | `tc_adamw_step` (fp32 master, fp16/fp32 grads, single-kernel) |
| CUTLASS | the kernels themselves: `gemm_simdgroup.metal`, `flash_attention.metal`, hand-tuned for `simdgroup_matrix` |
| NCCL | `tc_allreduce`, `tc_broadcast`, `tc_allgather`, `tc_barrier` (single-host + GLOO TCP today; TB5 ring v0.5) |
| cudaMalloc / cudaFree | `tc_buffer_alloc` / `tc_buffer_free` (with a power-of-2 LIFO pool — no `cudaMallocAsync` needed because UMA) |
| cudaMemcpyAsync (host↔device) | not needed — unified memory: `tc_buffer_map` returns a CPU-addressable pointer with no copy |
| CUDA streams | `tc_stream_create`, `tc_stream_sync` |
| Tensor Cores (sm_70+ MMA) | `simdgroup_matrix` on Apple7+ (M1+); `mpp::tensor_ops::matmul2d` on Apple11 (M5) |
| TF32 / FP8 transformer engine | bf16 + fp32 accum today; fp8/fp4 emulation slated for v0.6 |
| ggml (community) | first-class: built-in Q4_0 / Q8_0 GEMV plus a GGUF v3 reader |
| Triton on CUDA | the eshkol bridge — write `.esk` and codegen drops into the same C ABI |

## What's different (you should know)

### Unified memory is the load-bearing primitive

There is no host/device distinction. `tc_buffer_alloc` returns memory that
both the GPU and the CPU can touch directly. `tc_buffer_map` hands you a
`void*` you can read or write. No `cudaMemcpy`. No staging.

This means:
- Zero-copy GGUF: mmap the file, point a `tc_buffer` at the bytes, dispatch.
- KV-cache lives wherever you allocate it; no transfer cost between layers.
- Python interop via ctypes is a pointer cast, not a serialization.

The downside: there is no PCIe to overlap. Pipeline parallelism has to be
designed around the unified-memory bandwidth, not around hiding host↔device
transfers.

### One library, every M-series chip

The same library binary runs on M1 through M5. Family gating is a runtime
detail: bf16 lights up on Apple9+ (M3+), int8 on Apple10+ (M4+), Metal 4
TensorOps on Apple11+ (M5+). Older silicon falls back automatically — to a
software emulation for bf16/int8, or to MPS / Accelerate. See
[family_gating.md](family_gating.md).

NVIDIA solves this with `sm_XX` PTX. Apple solves it with `MTLGPUFamily*`
checks plus `function_constant`s; we wrap both.

### Threadgroup memory is 32 KB. That's the whole budget.

On Apple Silicon, you do not get the 100+ KB of shared memory you have on
H100. The kernel design (`Br = Bc = 32` for D=64 attention, `Br = Bc = 16`
for D=128, 64×64 GEMM tile with 4 simdgroups, etc.) is shaped by that
ceiling. This is the single biggest difference from CUDA kernel design.

### `simdgroup_matrix` is the MMA unit

The Metal equivalent of an `mma.sync` instruction is a `simdgroup_matrix`
load + multiply + store at 8×8 (fp16, bf16, fp32) or 8×8 i8. One simdgroup
(32 threads) cooperatively executes one MMA. We tile 64×64 with 4×4 MMA
fragments per simdgroup, 16 fragments per CTA at the standard tile.

The 128×128 tile (`gemm_simdgroup_128.metal`) is opt-in via
`TC_USE_128_TILE=1`; v0.1 regresses on M2, v0.2 retunes.

### No bespoke compute capability table

You don't pick a kernel by compute capability; you pick by `tc_family_t`
(Apple7..Apple11) and `tc_dtype_t`. The dispatch in `lib/ops/*.mm` does the
work. If a path isn't available on the current chip, you get
`TC_ERR_UNSUPPORTED_FAMILY` or a fallback to MPS / Accelerate (which path
served the call is reported by `tc_last_backend()`).

### MPS and Accelerate are real fallbacks, not stubs

Old chips, exotic shapes, or kernel cache misses fall back to
`MetalPerformanceShaders` (GPU) and `cblas_sgemm` (CPU via Accelerate). The
fallback path is exercised by `test_gemm_bf16` on M-series chips below
Apple9: bf16 is bit-cast to fp32 (bf16 = high 16 bits of fp32), routed
through Accelerate, and the result is correct to ~3e-3 RMS-scaled error.

The point: tensorcore degrades gracefully across chips. CUDA gives you "your
kernel doesn't run." We give you "your kernel ran via a slower path; here's
which path."

## The translation cheat sheet

### Initialize

```cuda
cudaSetDevice(0);
cublasCreate(&handle);
```

```c
tc_context* ctx;
tc_init(&ctx);          /* idempotent; second call returns TC_ERR_ALREADY_INITIALIZED */
```

### Allocate

```cuda
void* d_A;
cudaMalloc(&d_A, bytes);
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
```

```c
tc_buffer* A;
tc_buffer_alloc(ctx, bytes, &A);
void* p;
tc_buffer_map(A, &p);             /* no copy — UMA */
memcpy(p, host_data, bytes);
```

### GEMM (fp16, fp32 accum)

```cuda
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
             &alpha, B_d, CUDA_R_16F, N,
                     A_d, CUDA_R_16F, K,
             &beta,  C_d, CUDA_R_16F, N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

```c
tc_gemm_desc d = {0};
d.M = M; d.N = N; d.K = K;
d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F16;
d.accum_dtype = TC_DTYPE_F32;
d.alpha = 1.f; d.beta = 0.f;
tc_gemm(ctx, &d, A, B, C);        /* dispatches simdgroup_matrix on Apple7+ */
```

### FlashAttention forward (causal, fp16)

```cuda
/* via cuDNN 9 sdpa or FlashAttention-2 kernels */
flashAttention2Fwd(Q, K, V, O, batch, heads, seq, head_dim,
                   /*causal=*/true, scale);
```

```c
tc_attention_desc d = {0};
d.batch = B; d.heads = H; d.seq_q = d.seq_kv = S; d.head_dim = D;
d.io_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
d.softmax_scale = 1.f / sqrtf((float)D);
d.causal = true;
tc_attention_forward(ctx, &d, Q, K, V, O, NULL);
```

### All-reduce

```cuda
ncclAllReduce(buf, buf, n, ncclFloat16, ncclSum, comm, stream);
```

```c
tc_allreduce(dist_ctx, buf, n, TC_DTYPE_F16, TC_REDUCE_SUM);
```

### Async + streams

```cuda
cudaStream_t s;  cudaStreamCreate(&s);
cublasSetStream(handle, s);
cublasGemmEx(...);                /* enqueued on s */
cudaStreamSynchronize(s);
```

```c
tc_stream* s;  tc_stream_create(ctx, &s);
tc_gemm_async(ctx, &d, A, B, C, s);
tc_stream_sync(s);
```

## Where the analogy breaks

- **There's no `cudaGraph` equivalent yet.** Op fusion happens at the kernel
  level (`tc_fused_rmsnorm_gemv`); graph capture is a v0.7 item.
- **There's no Triton on Metal.** The Eshkol toolchain plays that role in
  our ecosystem; targeting tensorcore from PyTorch / JAX / MLX directly is
  on the v0.7 roadmap.
- **There's no `nvidia-smi`.** The closest is `system_profiler SPDisplaysDataType`
  for the GPU spec, and `powermetrics --samplers gpu_power` for live wattage.
  We report family + recommended working-set via `tc_device_info_get`.
- **There's no compute-vs-graphics queue split.** Apple GPUs do compute and
  graphics on the same command queues; we ignore this and just submit
  compute.

## "But how fast is it actually?"

On M2 Ultra (current checkpoint):

- `tc_gemm` fp16, 4096³: **17.88 TFLOPS** (~66% of peak)
- `tc_gemm` fp32, 4096³: 2.46 TFLOPS (bit-exact vs Accelerate)
- `tc_attention_forward` fp16 D=64, B=1, H=32, S=4096: **7.07 TFLOPS**
- Q4_0 GEMV at 7B-decode shape (async-batched): **186 tok/s @ 632 GB/s**
  (~79% of LPDDR5 peak; 3-3.5× ahead of llama.cpp's ~55-65 tok/s on the
  same chip)

H100's fp16 tensor-core peak is ~1500 TFLOPS, so per-chip we are 15× behind
on raw throughput. **Per watt at inference**, an M3 Max at 30 W matches an
H100 at 350 W on tokens/sec; that's a 5-10× per-joule advantage that's pure
silicon physics. See **[benchmarks.md](benchmarks.md)** and
**[ROADMAP.md](../ROADMAP.md)** for the honest competitive map.

## What you give up by leaving CUDA

- The world's best profiler (`nsys`, `ncu`). Metal's `Xcode` GPU profiler is
  good but not at parity. `MTLCounters` works.
- A 15-year-old developer ecosystem. `tensorcore` is v0.1.
- 64+ GB per chip without sharding. M2/M3/M4/M5 Ultra Studios with 192 GB
  unified memory close this for fine-tunes ≤70B params, but each-individual-
  chip cap is similar to H100 80 GB or H200 141 GB.
- Hopper-class fp8 tensor cores. Coming via emulation in v0.6.

## What you get

- A library that runs unchanged from M1 to M5.
- A unified memory model that removes the entire host/device transfer
  problem class.
- Per-watt inference that NVIDIA can't match without changing silicon.
- A 3.5K-line public C ABI you can read end-to-end in an afternoon.
- A roadmap that's explicit about what's silicon-bound and what's not.

If your model fits in ≤32 Macs of unified memory and your power budget is
finite, tensorcore is built for you.
