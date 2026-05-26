# tensorcore — Roadmap

**Goal:** become **the vendor-neutral compute substrate** for AI training and
inference, HPC scientific computing, graphics, and any workload that needs a
lot of optimized compute — runnable on Apple, NVIDIA, AMD, Intel, ARM, and
heterogeneous meshes of all of them. CUDA is welcome where it makes sense; no
piece of the stack should *require* it.

The narrower v0.1 goal — "make Apple Silicon a first-class training and
inference platform" — is the foundation, not the destination. v0.2 through
v0.6 extend the same C ABI to:

- **chipStar HIP** backend: Intel (Level Zero), NVIDIA (OpenCL), AMD
  (OpenCL), ARM Mali — one HIP source compiled to SPIR-V via chipStar runs
  on all of them.
- **CPU SIMD** backends: AVX2 (with FMA + F16C) and NEON, owned by tensorcore
  so cheap big-RAM Linux boxes can contribute real compute.
- **Cross-continent distributed** via DiLoCo: K=100-1000 inner SGD steps
  between outer-step parameter syncs makes per-step gradient sync over
  100-200 ms RTT links viable.
- **Memory tiering** with `tc_buffer_promote_async` / `_demote_async` across
  L0 (device RAM) → L1 (host RAM) → L2 (remote RAM) → L3 (NVMe) so a 70B
  model can train on a mesh with no single node holding the whole thing.

This roadmap is honest about where we are, where the silicon wins, where
Apple has to ship hardware, and what software work closes the gap.

---

## v0.1 measured baseline (M2 Ultra, this checkpoint)

| Op | Shape | dtype | TFLOPS | % of peak | Reference |
|---|---|---|---|---|---|
| GEMM | 4096³ | fp16 | **17.88** | ~66% | MLX on M3 Max ≈ 13.3 (`philipturner/metal-benchmarks`) |
| GEMM | 4096³ | fp32 | 2.36 | ~60% | — |
| GEMM | 2048³ | fp16 | 10.79 | — | — |
| FlashAttention | S=4096, D=64, H=32 | fp16 | 6.70 | — | MFA-pre-tuning baseline |
| FlashAttention | D=128 | fp16 | correctness verified | — | — |
| Q4_0/Q8_0 GEMV | 7B decode-step harness | fp16 activations | verified | — | — |

Same kernels run unchanged on M1 (Apple7), M2 (Apple8), M3 (Apple9), M4 (Apple10),
M5 (Apple11). bf16 path gated to Apple9+, int8 to Apple10+, Metal-4 TensorOps
to Apple11+. All tests pass; fp32 GEMM is bit-exact vs Accelerate.

---

## The honest "compete-with-NVIDIA" picture

We will not equal an H100 on raw fp16 TFLOPS per chip in v0.x. We don't need
to. The competitive axes are different and play to Apple's silicon strengths.

### Where Apple wins today, with good software

1. **Per-watt at inference** — measured. A 7B-param fp16 llama on M3 Max with
   well-tuned kernels hits ~50-80 tok/s at ~30 W package power. An H100 doing
   the same workload draws ~350 W. **5-10× advantage in tokens-per-joule.**
   This advantage is real silicon physics (unified memory, no PCIe, integrated
   NPU), not software hand-waving.

2. **Unified-memory training (≤192 GB single-host)** — An M2/M3/M4/M5 Ultra
   Studio with 192 GB of unified memory can hold a 70B-param fp16 model with
   gradients **without sharding**. A single H100 is capped at 80 GB. For
   research-scale fine-tunes and academic-scale pretraining, this is decisive
   when the software fully exploits it.

3. **Power-efficient distributed at small-cluster scale** — A 4× M5 Ultra
   Studio cluster over Thunderbolt-5 ring (JACCL/MLX, ~80 Gbps bidirectional)
   has roughly 768 GB unified memory total at ~600 W steady-state. Comparable
   H100 4-pack pulls ~3 kW. With tensorcore + JACCL ZeRO-2 sharding, this is a
   credible 100B-param fine-tune setup.

### Where Apple loses today, and what closes the gap

1. **Per-chip raw fp16 TFLOPS** — H100 ≈ 1500 TFLOPS (FP16 tensor core) vs
   M5 Ultra ≈ 80-100 TFLOPS. **15× gap per chip.** Doesn't go away in
   software. Closed only by buying more Macs or waiting for new silicon.
2. **Inter-host bandwidth** — Thunderbolt-5 = 80 Gbps. NVLink-5 = 1800 Gbps.
   **22× gap.** ZeRO-2 / pipeline-parallel hides most of this for medium
   models (≤100B params, ≤8 nodes). Above that, hardware investment needed.
3. **Datacenter-scale interconnect (>8 nodes)** — Apple ships no equivalent of
   InfiniBand HDR / NVSwitch. The gap is fundamental and requires either
   (a) Apple shipping a high-bandwidth Mac-cluster fabric, or
   (b) consumers cobbling together TB5 rings + 100GbE side-channels.

### The structural opportunity

NVIDIA's moat is "software stack maturity × silicon × interconnect." Apple
already wins on perf-per-watt and unified memory. The software stack is
weak — that's `tensorcore`. The interconnect needs Apple's help long-term.

**`tensorcore` is the bet on closing the software gap completely, so when
Apple does ship better fabric (M5 Ultra Pro? M6?), the entire stack is ready
to consume it on day one.**

---

## Phasing

### Heterogeneous-substrate sprint (in flight, unreleased)

The work below extends v0.1's Apple-only ABI to a vendor-neutral compute
substrate. Status is honest: some shipped already, some scaffolded,
some still queued.

- [x] **Portable CPU backend feature-complete on CPU side** — full
      transformer training stack (RMSnorm/LayerNorm/RoPE/SwiGLU/softmax/
      AdamW), FlashAttention forward + backward, Conv2D forward, plus
      BLAS-delegate GEMM (Accelerate / Intel MKL / OpenBLAS auto-detected).
- [x] **Measured: 2.10 TFLOPS fp32 GEMM on old-donkey (88-core Xeon
      E5-2699 v4) via MKL** — 70% of compute peak, vs 0.66 GFLOPS for the
      original reference triple-loop (~3200× speedup).
- [x] **CPU FlashAttention with AVX2+F16C-vectorized dot product** —
      2× speedup over scalar inner loop; 19.7 GFLOPS for B=1 H=32 S=512
      D=64 on old-donkey.
- [x] **Hand-tuned AVX2 6×16 fp32 GEMM micro-kernel** (`lib/ops/gemm_cpu_avx2.cpp`)
      — opt-in via `TC_USE_AVX2_GEMM=1`. Self-contained; no BLAS dep.
      Shared-library OpenMP scaling validated on old-donkey: 0.74 TFLOPS
      at 4096³ with 64 workers. MKL remains the faster default path at
      1.53 TFLOPS.
- [x] **DiLoCo cross-continent training API** — `include/tensorcore/diloco.h`
      + runtime in `lib/distributed/diloco.cpp`. Outer/inner optimizer split,
      local single-rank steps, and dense multi-rank outer steps over
      `TC_DIST_GLOO` are wired. NONE / FP16-intent / TOPK masks,
      Nesterov / SGD / Adam outer optimizers, and async overlap are present;
      TOPK over GLOO uses sparse packed transport. Dropout-tolerant WAN
      recovery remains queued.
- [x] **HIP/chipStar backend scaffold** — `include/tensorcore/hip.h` +
      `lib/hip/device.cpp` + `lib/hip/gemm.cpp` + `lib/hip/README.md`.
      Public ABI frozen; full runtime implementation lands after chipStar
      runtime is validated on cosbox (NVIDIA driver fix pending).
- [x] **Direct CUDA backend scaffold** — `include/tensorcore/cuda.h` +
      `lib/cuda/device.cpp` + `lib/cuda/gemm.cpp`. Public diagnostics and
      a hidden cuBLAS hook exist so NVIDIA-native support can land without
      changing the ABI.
- [x] **CUDA training closure on RTX 3090** — managed-memory training
      dispatch covers RMSnorm forward/backward, SwiGLU forward/backward,
      softmax forward/backward, and fp32/fp16-gradient AdamW, with
      `test_training_kernels`, `test_e2e_training`, `training_step`, and
      `mesh_training_demo` registered for non-Metal CUDA CTest proof.
      Revalidated on cosbox at `6382b98` (2026-05-23): CUDA build CTest
      18/18 passed and the Python CUDA smoke dispatched fp32/fp16 GEMM to
      cuBLAS managed-memory kernels on RTX 3090 cc=8.6.
- [x] **chipStar 1.1 built on cosbox (RTX 3090 host)** — hipcc + libCHIP.so
      verified; HIP source compiles. Runtime test blocked on NVIDIA driver
      mismatch; restart-pending.
- [x] **HIP/chipStar runtime diagnostics + fp32 GEMM evidence gate** —
      `TC_ENABLE_HIP=ON` now builds HIP device diagnostics when the runtime
      target exists, even if hipBLAS is absent. When hipBLAS is present,
      `tc_gemm` can route to `TC_BACKEND_HIP`. The new `test_hip_device`,
      `test_hip_gemm`, and `scripts/ci_hip_smoke.sh` path records no-build,
      no-runtime, runtime-only, and full-GEMM evidence states.
- [x] **Memory-tier API** — `include/tensorcore/memory_tier.h` with
      `tc_buffer_set_tier_hint`, `tc_buffer_promote_async`, etc. L0 (device)
      stub baseline shipping today; L1-L4 tiers (host RAM, RDMA-remote,
      NVMe) materialize as the runtime grows.
- [ ] **HIP/chipStar runtime expansion** — fp16/bf16/int8 coverage,
      device/managed buffer policy, and cross-vendor perf validation across
      Intel Level Zero, AMD OpenCL, ARM Mali, and any viable NVIDIA OpenCL
      stack. Phase 1 follow-through.
- [x] **NEON GEMM kernel** for aarch64 (xavier, Apple CPU side) — opt-in
      via `TC_USE_NEON_GEMM=1`; CBLAS remains default pending broader
      throughput data.
- [x] **Multi-thread + 1024+ scaling of the AVX2 GEMM kernel**. Phase 2:
      `bench_gemm_shared` links the OpenMP-enabled shared runtime and
      validates owned AVX2 fallback scaling through 4096³ on old-donkey.
      Remaining work is throughput tuning beyond the current 0.74 TFLOPS.
- [x] **`TC_DIST_GLOO` backend** for Apple and portable CPU collectives over TCP:
      fp32/fp16 all-reduce, min/max, broadcast, allgather, barrier, sparse
      packed exchange, and DiLoCo dense outer steps. WAN correctness proof
      exists; long-duration performance soak remains.
- [x] **NAT-tolerant direct GLOO ring** — ranks advertise reachable
      IPv4/IPv6 per-host addresses, bound neighbor-connection timeouts,
      coordinate broker fallback, and emit `TC_GLOO_TRACE=1` route logs.
      Proven on localhost IPv4/IPv6 fork smokes and on the four-rank
      Atlas + Enki + old-donkey + cosbox cross-continent mesh.
- [x] **Sparse top-k compressed all-reduce** in DiLoCo — GLOO
      now ships TOPK deltas as sparse `(idx, fp16)` payloads and validates
      the bandwidth cut with forked localhost and cross-continent sparse
      training smoke coverage.
- [x] **Activation checkpointing runtime + mesh demo** — CPU and Metal
      backends free owned buffer storage on `tc_checkpoint_discard`, keep
      handles valid but unmapped, reallocate before recompute on
      `tc_checkpoint_realize`, and exercise RMSNorm activation
      discard/realize in `mesh_training_demo --checkpoint`.
- [x] **Live end-to-end transformer training demo across the mesh** —
      `scripts/run_live_mesh_training_demo.sh` prepares and launches
      `examples/mesh_training_demo` across Atlas, Enki, old-donkey, and
      cosbox. Validated 2026-05-23 with direct-ring DiLoCo outer sync,
      a 5 outer-step live soak, 40 checkpoint discard/realize cycles on
      all four ranks, and cosbox rank 3 reporting `backend=cuda`.

### v0.1 — Foundation (shipped this checkpoint)
- [x] CMake + metallib precompile + cross-family runtime detect
- [x] Public C ABI (`include/tensorcore/`) — 16 headers, 109 exported symbols
- [x] Device init / pipeline cache / power-of-2 buffer pool, autotune sweep + JSON cache
- [x] **`simdgroup_matrix` GEMM** — fp16/fp32 (64×64 tile, BK=32, vec4 loads,
      f32-accum) — **17.88 TFLOPS @ 4096³ on M2 Ultra, fp32 bit-exact vs Accelerate**
- [x] bf16 GEMM — native on Apple9+, fp32-fallback on Apple7..8 (validated, rms ≤ 3e-3)
- [x] int8 GEMM — native on Apple10+, fp32-widen fallback on Apple7..9 (bit-exact i32 accum)
- [x] 128×128 large-tile GEMM (env-flag opt-in; register-pressure tuning v0.2)
- [x] **Fused FlashAttention forward** — fp16, D=64 and D=128, with causal,
      GQA, sliding-window, and ALiBi via function constants
- [x] **FlashAttention backward** — D=64 (LSE-saved scheme); D=128 backward
      also shipped
- [x] Conv2D forward + backward (dInput + dWeight), im2col + tc_gemm strategy
- [x] Q4_0/Q8_0 quantized GEMV plus GPU quantization (v2 kernel: 186 tok/s @
      632 GB/s on 7B decode, ~3× ahead of llama.cpp)
- [x] GGUF v3 reader, metadata helpers, bulk tensor loading, and quantized
      matrix descriptors
- [x] RMSNorm, LayerNorm, RoPE, SwiGLU, softmax, AdamW, and fused
      RMSNorm+GEMV kernels (fwd + bwd where applicable)
- [x] Python binding — full public ABI parity, owned object
      wrappers (`Context`, `Buffer`, `Stream`, `DistContext`, `GgufFile`,
      `LoadedModel`, `QuantizedMatrix`), NumPy interop
- [x] Distributed primitives — single-host ring all-reduce / broadcast / allgather
      / barrier, both threads-with-shared-mem and fork-with-socketpair transports,
      plus Apple and portable CPU `TC_DIST_GLOO` TCP collectives
- [x] M5 TensorOps GEMM + FlashAttention kernels (`tensorops_*.metal`),
      SDK 26.0+ gated, runtime selector
- [x] MPS + Accelerate fallback paths
- [x] Correctness tests vs `cblas_sgemm` + fp64 reference — default Apple
      CI, macOS 15 CI, Ubuntu portable CPU CI, and macOS portable CPU CI
      pass on every push
- [x] TFLOPS bench harness (GEMM sweep, attention sweep, 7B Q4_0 inference)
- [x] Eshkol binding skeleton + runtime-proven `extern` bridge through
      `tc_eshkol_*` helpers, with optional raw-codegen declaration file
- [x] **Native SDK release artifact** — headers + libraries + metallib +
      CMake config + pkg-config tarball, with consumer verification in CI
- [x] **Wheel packaging** — `tensorcore_apple-*.whl` with dylib + metallib
      vendored; published as GitHub release asset on every `v*` tag
- [x] **Version consistency CI** — `scripts/check_version_consistency.sh`
      asserts `pyproject.toml` / CMake / header triple agreement on every push
- [x] **End-to-end assembly examples** — `examples/decode_step.c` (synthetic
      Llama decode step) and `examples/training_step.c` (synthetic mixed-
      precision training iteration; loss decreases 3.98 → 0.25 over 15 steps)

### v0.2 — Saturation perf on Apple7..9 (next 2-4 weeks of focused work)

- 20+ TFLOPS fp16 4096³ on M2 Ultra (~75% of peak). Hand-tune via:
  - Double-buffered K-block loads (one tile in flight while computing on prev)
  - Async dispatch with multi-CB pipelining
  - 128×128 tile with register-pressure-aware sg layout (WM=4×WN=2, TM=2×TN=4)
- FlashAttention parity with **MFA** (Apple's open metal-flash-attention):
  - Br=64 for D=128 via aliased TG memory regions
  - Causal-mask early-exit pruning at K-block granularity
  - Split-K for short-seq → long-context generation
- Backward/perf closeout: GEMM-gradient throughput and FlashAttention
  backward tuning. RMSnorm, RoPE, SwiGLU, softmax, and AdamW backward/optimizer
  ABI paths have correctness coverage across the active CPU/Metal/CUDA
  backends.
- Full mixed-precision training loop test (small transformer block) matching
  PyTorch-MPS gradient outputs

### v0.3 — Metal 4 / M5 TensorOps (when M5 hardware lands in the lab)

- Metal 4 `mpp::tensor_ops` kernels (macOS SDK 26.0+, M5 runtime)
- Runtime select: TensorOps vs simdgroup_matrix per (shape, dtype)
- Target Apple's reported 4× speedup at small-shape end (M5 "neural accelerators")
- Pre-compile every kernel via both backends; auto-bench at first dispatch

### v0.4 — Eshkol consolidation (closes the three-backend tax)

- `eshkol/tensorcore.esk` + `tc_eshkol_*` C shims — keep `__tc-*` bridge calls
  runtime-proven through Eshkol's `extern` path
- Redirect `eshkol-platform/lib/backend/tensor_*_codegen.cpp` to emit calls
  to `tc_gemm` / `tc_attention_forward` instead of bespoke gpu_memory.mm dispatch
- Migrate `quantum_geometric_tensor/src/metal/` kernels (45+ kernels) onto tensorcore
- Migrate `semiclassical_qllm/src/backend/backend_metal.m` onto tensorcore
- After v0.4: **one** Metal kernel layer across the entire ecosystem
- Keep SF64 / Ozaki-II / FP24 / FP53 paths inside tensorcore as
  `TC_DTYPE_SF64` / `TC_DTYPE_OZAKI` extensions (the existing implementations
  in `eshkol-platform/lib/backend/gpu/gpu_memory.mm` move here verbatim)

### v0.5 — Distributed (Thunderbolt 5 + JACCL)

- Ring all-reduce + tree-reduce over TB5; reuse MLX's JACCL where possible
- NCCL-style API (`tc_distributed_init`, `tc_allreduce`, `tc_allgather`, ...)
- Multi-Mac topology discovery + bandwidth probe + automatic best-route
- ZeRO-1, ZeRO-2, ZeRO-3 parameter sharding
- Pipeline parallelism scheduler (1F1B, interleaved)
- Target on a 4× M5 Ultra cluster: train a 70B-param fp16 fine-tune at >40% MFU,
  competitive with a 4× A100 cluster for the same task on a per-watt basis

### v0.6 — Frontier-scale enablement (the long bet)

This is where "compete at frontier scale" gets serious. The goals are
silicon-bound but software prepares the ground:

- **Multi-precision frontier training stack:**
  - fp8 / fp4 simulation today via custom kernels (Apple silicon doesn't ship
    native fp8 MMA yet; we emulate via per-tile-scale fp16 → bf16)
  - Stable accumulation across thousands of GPUs (per-tile master-weight
    scheme, well-tested for >100B-param models)
- **High-bandwidth fabric integration:** wrappers for any Apple-shipped
  high-bandwidth Mac-cluster interconnect (rumored M-series Pro Ultra fabric;
  no public spec yet). Build a clean abstraction now so we adopt instantly.
- **Activation checkpointing + selective recomputation** kernels with
  zero-overhead async re-materialization on the unified-memory side.
- **3D parallelism** (data + tensor + pipeline) scheduling with awareness of
  per-Mac unified-memory ceiling.
- **Mixture-of-experts** routing kernels tuned for unified memory (the
  routing-then-shard pattern has very different optimal kernels on UMA vs
  PCIe).

Realistic frontier-scale claim: with v0.6 + Apple shipping a 200+ Gbps inter-Mac
fabric, a **32× M5 Ultra Studio cluster** (~6 TB total unified memory, ~10 kW)
is a credible training rig for **100-200B-param dense models** — roughly the
class of GPT-3 175B, Llama-2 70B, or Llama-3 70B. Above that, NVIDIA still wins
on cluster size; below that, the Apple stack wins on $/training-FLOP-per-watt
and on the unified-memory programming model.

We won't claim "Apple beats H100 cluster at 1024-GPU scale" — it's not true
this year. We will claim, supported by measured numbers, that **for any team
training models that fit in ≤32 Macs of unified memory, the Apple stack
becomes the smarter purchase** once tensorcore matures.

### v0.7+ — The compiler layer

- `tensorcore` as a backend target for PyTorch, MLX, JAX, ONNX, GGUF
- Pattern-matched op fusion in Eshkol's tensor codegen
- `torch.compile`-style Triton-on-Metal alternative driven by tensorcore IR

## How to read this roadmap

- **v0.1 — measured.** Every claim ships with a test, a bench number, or
  a correctness validation; see [docs/benchmarks.md](docs/benchmarks.md)
  for reproducibility and [CHANGELOG.md](CHANGELOG.md) for per-checkpoint
  history.
- **v0.2-v0.3 — committed.** The work is bounded; the deliverables are
  shape-tuning, kernel-design wins, and SDK 26 hardware exercises. No
  hardware dependency outside Apple's already-shipped silicon.
- **v0.4 — coordination.** Software-only, but consolidating across three
  sibling projects. See [docs/eshkol_integration.md](docs/eshkol_integration.md).
- **v0.5 — substrate-bound.** Needs macOS 26.2's JACCL surface and a
  Thunderbolt-5 ring to validate against. The single-host ring and portable
  GLOO baseline are in v0.1, so the work is transport-swap + scheduler,
  not algorithm.
- **v0.6+ — silicon-bound.** Honest about what's software-only vs what
  needs Apple to ship.

---

## Open questions we're not pretending to have solved

1. Will Apple ship a high-bandwidth inter-Mac fabric? Required for true
   frontier-scale. No public commitment.
2. Will Metal expose ANE for training? Today it's CoreML/inference only.
   `tensorcore` v0.6 will probe the private `_ANECompiler` route if Apple
   doesn't open it officially.
3. How far can we push fp8/fp4 *emulation* before native hardware lands?
   Per-tile-scaling schemes get us reasonable accuracy at 2-4× memory savings
   but no extra TFLOPS.

The roadmap commits to the work we can do unilaterally. The ambitious endgame
needs Apple to ship matching hardware. The clearest place we add leverage is
making the software stack so good that Apple has commercial reason to ship it.
