# Memory model

`tensorcore` lives on Apple Silicon, which means **one address space**:
the GPU and the CPU can touch the same bytes. There is no `cudaMemcpy`,
no PCIe, no staging. This page covers what that means in practice, how
the buffer pool works, how streams order things, and what the threading
guarantees are.

## Unified memory in one paragraph

Every `tc_buffer*` is backed by an `MTLBuffer` with `MTLStorageMode =
Shared`. That memory is mapped into both the CPU's address space and the
GPU's. `tc_buffer_map(buf, &p)` returns a `void*` you can read or write
from the CPU at zero cost; the same bytes are what the GPU kernel sees
on its next dispatch.

The trade-off vs PCIe-attached GPUs:

| | NVIDIA discrete | Apple Silicon (us) |
|---|---|---|
| Host ↔ device transfer | mandatory; staged through PCIe at ~50-300 GB/s | none |
| GPU memory ceiling | per-GPU VRAM (80/141 GB) | unified pool (96/192 GB on a Studio) |
| Latency hiding | overlap compute with copies | overlap compute with compute (no copies) |
| Driver overhead | high per-dispatch | minimal; command buffers are direct |

The implication: pipeline parallelism on Apple Silicon optimizes for
overlapping kernel dispatches, not for hiding host↔device copies. Most
of the kernel design (the async-batched stream pattern; the small per-op
command-buffer cost) reflects this.

## The buffer pool

`lib/core/buffer_pool.mm` is the allocator. Pattern:

1. Round requested size up to the next power of two.
2. Look in that bucket. If a free buffer exists, **pop the most recent**
   (LIFO — warmest cache lines).
3. Otherwise, `[device newBufferWithLength:options:MTLResourceStorageModeShared]`
   and return.

`tc_buffer_free` doesn't release to Metal; it pushes the buffer back
into its size bucket. The pool grows monotonically until `tc_shutdown`,
which drops everything. This means:

- **No allocation cost in the steady state** — repeated allocs of the
  same size hit the bucket, return immediately.
- **LIFO recycling** — same bucket entries get reused frequently, so
  the underlying pages stay hot in cache.
- **Memory growth is monotonic per-context** — if your peak is high,
  the pool holds onto it until shutdown. Re-create the context to
  release.

Power-of-two bucketing means a 5 KB request lands in the 8 KB bucket;
overhead is up to ~2× in pathological cases. Most tensorcore workloads
allocate at predictable sizes (per-shape activations) so the overhead
amortizes.

### Validation

`tests/test_buffer_pool.mm` asserts:
- Allocation followed by free + same-size re-alloc returns the same
  pointer (LIFO).
- Different sizes go to different buckets.
- Concurrent allocate/free from multiple threads is safe (the pool
  is internally guarded).

## `tc_buffer_map` semantics

```c
tc_buffer* buf = NULL;
tc_buffer_alloc(ctx, 4096, &buf);
void* p = NULL;
tc_buffer_map(buf, &p);
/* p is a stable CPU-addressable pointer; same value across calls for
 * the same buf */
memcpy(p, host_data, 4096);
tc_gemm(ctx, &d, A, buf, C);   /* GPU sees the same bytes */
```

The pointer is **stable** for the lifetime of the buffer (until
`tc_buffer_free`). Cache it freely.

The memory is **coherent** without explicit synchronization on Apple
Silicon. Per Apple's documented memory model for `MTLStorageModeShared`:
- CPU writes are visible to subsequent GPU reads.
- GPU writes are visible to subsequent CPU reads.
- "Subsequent" means after `tc_stream_sync` for cross-CPU/GPU ordering;
  within a single thread's CPU-only sequence, normal C memory ordering
  applies.

If you write from the CPU and dispatch a kernel that reads, the kernel
sees what you wrote. If you dispatch a kernel that writes and then read
from the CPU, you must `tc_stream_sync` (or call a synchronous
`tc_gemm` that internally syncs) before the CPU read.

### Zero-copy GGUF

`lib/io/gguf.c` mmaps the GGUF file. `tc_gguf_tensor_info.data` is a
pointer into the mmap; `tc_gguf_tensor_to_buffer` allocates a buffer
and `memcpy`'s the bytes in.

A future optimization: when the GGUF tensor's on-disk alignment
matches the `MTLBuffer` requirement, we could `MTLBuffer` directly over
the mmap'd pages (`newBufferWithBytesNoCopy:`), skipping the copy. That's
on the v0.2 list; today the bulk-load takes a few seconds and 4 GB of
duplicate memory on a 7B Q4_0 model. Real impact: a fresh load takes
~1.5s instead of being instant. Tolerable; not optimal.

## Streams and command buffers

A `tc_stream*` corresponds to a sequence of dispatches sharing one
`MTLCommandBuffer`. The async API (`tc_gemm_async`,
`tc_attention_forward_async`, `tc_gemv_quantized_async`) encodes into
the stream's pending command buffer without committing it. `tc_stream_sync`
commits the command buffer and blocks until completion.

### What this saves you

Per-call command-buffer setup on M2 Ultra is ~50µs. A Q4_0 7B decode
step has ~200 GEMVs across 32 layers. Sync-per-call:

```
200 × (50µs CB + ~25µs kernel) = 15 ms / token = 67 tok/s ceiling
```

Async-batched:

```
1 × 50µs CB + 200 × ~25µs kernel = 5 ms / token = 200 tok/s
```

Measured: 186 tok/s, 632 GB/s effective bandwidth — see
[benchmarks.md](benchmarks.md). The dispatch-overhead difference is the
single biggest source of inference speed on Apple Silicon.

### When to sync

| Situation | Sync? |
|---|---|
| Within a decode step's layers | **Don't** — async every call, sync at the end |
| Between forward and backward in training | Optional; the dependency is implicit through the buffer the backward reads |
| Before reading a result from the CPU | **Required** — the GPU may not have finished |
| Before reusing a buffer for a different op | Implicit in the command-buffer ordering as long as you stay on one stream |

The default stream (NULL passed to a non-async API) commits and waits on
every call. It exists for ergonomics, not for performance. Use it for
"hello world" and one-shot computations; switch to `tc_stream` for
anything that runs in a loop.

### Cross-stream ordering

Two streams are independent. If a buffer is written by stream A and read
by stream B, you must `tc_stream_sync(A)` before scheduling on B (or
fold both calls onto the same stream). v0.1 doesn't expose explicit
inter-stream events; v0.2 will.

## Threading

| Object | Cross-thread? |
|---|---|
| `tc_context*` | shared safely; the pipeline cache and buffer pool are guarded |
| `tc_buffer*` | **caller must serialize** access to the same buffer; different buffers are independent |
| `tc_stream*` | **single-threaded only** — one thread owns the stream's pending CB |
| `tc_last_backend()` | **thread-local** — reports the calling thread's last GEMM/attention dispatch |
| `tc_dist_ctx*` | single-threaded; the distributed surface assumes one caller per rank |

The C ABI is reentrant per-context. Two threads can dispatch concurrently
into the same context as long as they touch different buffers and
different streams. The pipeline cache is read-mostly; new pipelines are
compiled under a mutex but that's a one-time cost per kernel name.

The Python binding doesn't release the GIL inside dispatches; one Python
thread per context is the recommended pattern.

## Buffer lifetimes

The Python wrappers track ownership via weakrefs and tear down in the
right order automatically:

```
Context.__del__
  → close DistContexts
  → close Streams
  → close LoadedModels (which close their owned tc_buffers)
  → close Buffers
  → tc_shutdown
```

In C, you do this manually. The pattern:

1. Allocate all your buffers (the pool absorbs the cost).
2. Run.
3. `tc_buffer_free` everything you allocated.
4. `tc_stream_destroy` your streams.
5. `tc_shutdown`.

Skipping the buffer frees doesn't leak in the C-API sense — `tc_shutdown`
drops the pool — but it does keep the unified memory pinned for the
context's lifetime.

## Memory ceilings

| Chip | Max unified memory | Practical model ceiling (fp16) |
|---|---:|---|
| M1 Max | 64 GB | ~7B fp16 + KV-cache + activations |
| M2 Max | 96 GB | ~13B fp16 |
| M2 Ultra | 192 GB | ~70B fp16 (with sharded KV-cache headroom) |
| M3 Max | 128 GB | ~30B fp16 (Apple9 bf16 native) |
| M4 Max | 128 GB | ~30B fp16 (Apple10 int8 native) |
| M5 Max | 64 GB (rumored) | ~13B fp16 (TensorOps perf > capacity) |
| M5 Ultra | 192-256 GB (rumored) | ~70B fp16 with TensorOps acceleration |

These ceilings are about **fitting the model**; performance on a
192-GB M2 Ultra is bandwidth-bound, not compute-bound. A 70B fp16 model
at decode-step bandwidth (140 GB / token at fp16, ~35 GB / token at
Q4_0) sets the per-token ceiling regardless of TFLOPS.

This is why per-watt at inference matters more than peak TFLOPS for the
Apple Silicon story; the memory bandwidth is the constraint, and Apple
gets ~800 GB/s LPDDR5 in a 30 W package while NVIDIA gets ~3 TB/s HBM3
in a 700 W package. Different design points; same total work served by
very different power profiles.

## See also

- [architecture.md](architecture.md) — where the buffer pool fits in the
  call flow.
- [inference.md](inference.md) — the async-stream pattern in practice.
- [benchmarks.md](benchmarks.md) — measured bandwidth numbers.
- [distributed.md](distributed.md) — when a model doesn't fit in one
  Mac's unified memory.
