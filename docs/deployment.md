# Deploying tensorcore across a heterogeneous mesh

This guide is the steady-state operating handbook for the tensorcore
distributed substrate as validated in v0.1.22+: cross-continent DiLoCo
training across Apple Silicon Macs and Linux x86_64 hosts.

The 4-rank reference deployment used to validate every primitive in this
document is:

| Rank | Host | Hardware | Site | Backend |
|---:|---|---|---|---|
| 0 | Atlas | Apple M2 Ultra (76-core GPU, 192 GB) | Quebec | Metal |
| 1 | Enki | Apple M4 (10-core GPU) | Quebec | Metal |
| 2 | old-donkey | 88-core Xeon E5-2699 v4, 500 GB RAM | Alaska | CPU + MKL |
| 3 | cosbox | i7-5930K + RTX 3090, 31 GB | Alaska | CPU + CUDA scaffold |

Linked by Tailscale over the public internet (about 200 ms RTT, about 11 Mbps
bandwidth Quebec <-> Alaska). The same procedure works on any LAN/WAN
configuration.

## 1. Per-host build

### Apple Silicon (Macs)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Produces `build/libtensorcore.dylib` + `build/tensorcore.metallib`. The
metallib must be deployed alongside the dylib (it's the Metal-shader
binary). Set `TC_METALLIB` to its absolute path on every invocation.

### Linux CPU (any x86_64 / aarch64 / RISC-V host)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTC_ENABLE_METAL=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The portable CPU backend builds without any GPU SDK. BLAS is auto-detected
in this order: Intel MKL -> OpenBLAS -> Netlib BLAS. With MKL on a 22-core
NUMA node, fp32 GEMM hits ~2.1 TFLOPS at 4096^3 - competitive with M2
Ultra's GPU fp32 throughput.

For the AMD64 hand-tuned AVX2 kernel:

```sh
TC_USE_AVX2_GEMM=1 ./build/bench/bench_gemm
```

The AVX2 macro-kernel uses OpenMP tile fanout when the shared library was
built with OpenMP support. Set `TC_AVX2_THREADS=1` to force serial execution
for A/B comparisons, or `TC_AVX2_THREADS=N` to cap the internal worker count.

For the aarch64 NEON kernel (xavier, Apple CPU side):

```sh
TC_USE_NEON_GEMM=1 ./build/bench/bench_gemm
```

### Linux with NVIDIA CUDA toolkit (RTX 30/40, H100, etc.)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
                -DTC_ENABLE_METAL=OFF \
                -DTC_ENABLE_CUDA=ON
cmake --build build -j
```

Requires CUDA Toolkit >= 11.0 (validated against 12.0). The build will
look for `libcudart` and `libcublas` via `find_package(CUDAToolkit)`.
After build, verify the device is detected:

```c
#include "tensorcore/tensorcore.h"
#include "tensorcore/cuda.h"
int main() {
    tc_context* ctx = NULL; tc_init(&ctx);
    tc_cuda_init(ctx);
    tc_cuda_device_info info;
    tc_cuda_device_at(0, &info);
    printf("%s, cc=%s, %.1f GB, fp16=%d bf16=%d tf32=%d\n",
        info.device_name, info.compute_capability,
        info.global_memory_bytes / 1e9,
        info.supports_fp16, info.supports_bf16, info.supports_tf32);
}
```

Expected for an RTX 3090: `NVIDIA GeForce RTX 3090, cc=8.6, 25.3 GB, fp16=1 bf16=1 tf32=1`.

GEMM dispatch to cuBLAS is currently opt-in:

```sh
TC_USE_CUDA_GEMM=1 ./build/bench/bench_gemm
```

Successful opt-in CUDA GEMM reports `TC_BACKEND_CUDA` / `"cuda"` via
`tc_last_backend()`. Buffers allocated while `TC_USE_CUDA_GEMM=1` is set
use CUDA managed memory, so cuBLAS can dereference A/B/C directly. Wrapped
host pointers still use an internal staged-copy fallback. If CUDA dispatch
fails or is not requested, `tc_gemm` falls back to the portable CPU/BLAS path.

### Linux with chipStar (vendor-neutral GPU: Intel Level Zero, AMD OpenCL, ARM Mali)

```sh
# Build chipStar 1.1+ first (see chipStar README; needs LLVM/Clang 19 + SPIRV translator)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
                -DTC_ENABLE_METAL=OFF \
                -DTC_ENABLE_HIP=ON
cmake --build build -j
```

chipStar's HIP backend dispatches to whichever ICD is installed (Intel
`libze`, AMD `libamdocl`, etc.). For NVIDIA hardware, the chipStar+OpenCL
path does **not** work today (NVIDIA's OpenCL driver lacks SPIR-V
ingestion); use the direct CUDA backend above instead.

## 2. Network setup

### Same-LAN: any IP-routable address

For two hosts on the same LAN, just use their LAN IPs. The 4-rank
reference setup uses Tailscale, but identical commands work on a 10 GbE
LAN, RDMA fabric, etc.

### Tailscale (cross-WAN / heterogeneous network)

Install Tailscale on every host, sign into the same tailnet, get each
host's 100.x.y.z address:

```sh
tailscale status
# 100.96.130.16  Atlas
# 100.111.56.36  Enki
# 100.121.14.68  old-donkey
# 100.86.83.35   cosbox
```

The rendezvous URL points at **rank 0's** address.

### Thunderbolt Bridge (Mac<->Mac, 40 Gbps)

When two Macs are connected via a real Thunderbolt cable (lightning-bolt
badge "3" or "4"), macOS auto-creates `bridge0`. Assign IPs to activate:

```sh
# On Mac 0:
sudo networksetup -setmanual "Thunderbolt Bridge" 192.168.42.1 255.255.255.0 192.168.42.1
# On Mac 1:
sudo networksetup -setmanual "Thunderbolt Bridge" 192.168.42.2 255.255.255.0 192.168.42.1
```

Verify activation: `ifconfig bridge0 | grep status` should report `active`
on both. Bandwidth check: `iperf3` should show ~2.5-3 GB/s sustained.

If the bridge shows `inactive` after IPs are assigned, the cable is
USB-only (not TB) - get an actual Thunderbolt cable.

## 3. Launching the substrate

The reference test is `tests/test_dist_remote` - a split-binary that
takes `--rank`, `--world`, and `--url`.

### 2-rank cross-continent run

On rank 0 (any host; the listener):
```sh
TC_METALLIB=/path/to/tensorcore.metallib \
    ./build/tests/test_dist_remote \
    --rank 0 --world 2 \
    --url tcp://<rank-0-ip>:9000
```

On rank 1 (connects to rank 0):
```sh
./build/tests/test_dist_remote \
    --rank 1 --world 2 \
    --url tcp://<rank-0-ip>:9000
```

Expected output on each side:
```
[rank N/2] connecting to tcp://...
[rank N] rendezvous done in X.X sec
[rank N] allreduce 4MB sum: ... ms/iter, ~Y.YY GB/s
[rank N] DiLoCo 3 outer steps x 5 inner: Z.ZZZ sec, bandwidth/step=528.0 bytes
[rank N] OK
```

The DiLoCo bandwidth-per-step number is the punchline: for a 64K-element
fp16 parameter with TOPK_01PCT compression, each outer step ships ~528
bytes. Compare to the dense 4 MB allreduce in the line above to see the
multiplier.

### 4-rank reference deployment

Launch rank 0 first (must start listening before others try to connect):

```sh
# rank 0 (Atlas):
TC_METALLIB=... ./test_dist_remote --rank 0 --world 4 --url tcp://100.96.130.16:9000 &

# rank 1 (Enki):
ssh enki.local 'TC_METALLIB=... /tmp/test_dist_remote --rank 1 --world 4 --url tcp://100.96.130.16:9000' &

# rank 2 (old-donkey):
ssh old-donkey 'cd /tmp/tc && ./build/tests/test_dist_remote --rank 2 --world 4 --url tcp://100.96.130.16:9000' &

# rank 3 (cosbox):
ssh cosbox 'cd /tmp/tc-cuda && LD_LIBRARY_PATH=./build ./build/tests/test_dist_remote --rank 3 --world 4 --url tcp://100.96.130.16:9000' &

wait
```

Validated working state for this exact deployment:
- Rank 0 rendezvous: 17 sec (waits for 3 peers via WAN)
- DiLoCo 3 outer x 5 inner across all four ranks: 4-17 sec depending on
  which rank's network leg is slowest
- All ranks converge to the same theta; bit-correct sum verified

## 4. Backend selection

`tc_init` initializes the default backend for the host:

- On Apple: Metal (always, when `TC_ENABLE_METAL=ON`).
- On Linux: CPU. If `TC_ENABLE_CUDA=ON` was set at build time, `tc_cuda_init`
  can be called explicitly to attach an NVIDIA device and
  `TC_USE_CUDA_GEMM=1` opts supported fp32/fp16/bf16/int8 GEMM calls into
  cuBLAS.

The substrate doesn't currently auto-route compute to the fastest
available backend at `tc_init` time - that's a v0.2 feature. Today, the
build flags determine which backends are linked; CUDA GEMM is explicitly
selected with `TC_USE_CUDA_GEMM=1`, and HIP remains behind `tc_hip_init`.

## 5. Distributed transport selection

| Backend | When to use | Bandwidth | Limitations |
|---|---|---|---|
| `TC_DIST_SINGLE` | World size 1, in-process tests | n/a | Single rank only |
| `TC_DIST_RING` | Apple TB5 ring (future) | 80-120 Gbps | Reserved for v0.5 |
| `TC_DIST_GLOO` | **All real multi-host setups today** | Link-limited | Broker default; opt-in ring fp32 SUM for 3+ ranks |

The rank-0 broker is the default because it only requires peers to reach
the rendezvous host. Set `TC_GLOO_RING=1` to enable direct rank-to-rank
ring sockets for fp32 SUM on networks where every rank can reach its ring
neighbors. If a direct neighbor cannot be reached, all ranks now
coordinate over the rendezvous sockets and fall back to the broker path
instead of failing `tc_dist_init`.

For Tailscale or other overlay networks, `TC_GLOO_ADVERTISE_HOST` can be
set per rank to the address peers should dial for direct ring links. If
unset, each rank reports the local address selected for the rendezvous
connection. `TC_GLOO_RING_CONNECT_TIMEOUT_MS` bounds direct-ring connect
attempts before fallback, and `TC_GLOO_NO_RING=1` forces broker dispatch
if you need to debug a ring-capable build.

## 6. DiLoCo configuration

For cross-continent training, configure DiLoCo with aggressive
compression and async overlap:

```c
tc_diloco_config cfg = {
    .inner_steps = 1000,                         /* K=1000 -> ~1000x sync reduction */
    .outer_lr = 0.7f,                            /* DiLoCo paper default */
    .outer_momentum = 0.9f,                      /* Nesterov */
    .outer_optimizer = TC_DILOCO_OUTER_NESTEROV,
    .compress = TC_DILOCO_COMPRESS_TOPK_01PCT,   /* keeps top 0.1% magnitudes */
    .async_overlap = true,                       /* outer step in background while next K inner */
    .tolerate_dropouts = true,                   /* WAN drops don't deadlock */
};
```

With this config, a 70B fp16 model trains over an 11 Mbps WAN link with
effective zero outer-step overhead (the sync hides inside the 100+
seconds of inner compute between syncs).

## 7. Troubleshooting

### `tc_dist_init GLOO` fails on rank > 0
Rank 0 hasn't started yet. Launch rank 0 first, sleep 2-5 sec for the
listener to bind, then start the other ranks.

### `connection refused` after rank 0 is up
Firewall blocking the port. On Linux:
```sh
sudo ufw allow 9000/tcp
```
On macOS: System Settings -> Network -> Firewall -> Allow incoming for
your test binary.

### `allreduce ... 0.00 GB/s` over WAN
Expected when bandwidth is small fraction of MB/s. Use DiLoCo + sparse
compression instead of dense allreduce; see the DiLoCo numbers in the
same output for the actual cross-continent training throughput.

### CUDA build but `tc_cuda_init` returns `TC_ERR_UNSUPPORTED_FAMILY`
Driver/runtime mismatch. Check `nvidia-smi` works; reboot if you see
"Driver/library version mismatch" (kernel module needs reload).

### Mac<->Mac TB Bridge stays `inactive`
The cable is USB-only, not Thunderbolt. Replace with a TB3/TB4-rated
cable. Software cannot work around a cable that physically lacks the TB
protocol lanes.

### Verification: bit-correct sum across ranks
`test_dist_remote --test allreduce` runs a 4MB allreduce SUM and verifies
each element equals the expected `sum(1..world_size)`. Any mismatch
indicates the transport corrupted bits - usually a code bug, not a
network issue.

## 8. What's running today vs queued

**Working today (validated):**
- Cross-LAN tensorcore distributed training (Mac<->Mac, Mac<->Linux, Linux<->Linux)
- Cross-continent DiLoCo via Tailscale or any IP-routable network
- Sparse top-k compression on the wire (validated 1/1000th-of-dense
  payload at TOPK_01PCT)
- Multi-Apple-generation interop (M2 Ultra + M4 in same training run)
- Heterogeneous-vendor mesh (Apple GPU + Linux CPU + NVIDIA-capable
  host all in one DiLoCo run)
- Direct CUDA backend init + device introspection (RTX 3090 validated)
- Opt-in CUDA GEMM with managed-memory tc_buffer allocations
  (`TC_USE_CUDA_GEMM=1`, RTX 3090 validated for fp32/fp16; bf16/int8 are
  gated by CUDA device capability)
- Managed-memory CUDA training dispatch for RMSNorm forward/backward,
  LayerNorm forward, SwiGLU forward/backward, softmax forward/backward, and
  fp32/fp16-gradient AdamW, with host-buffer fallback to portable CPU kernels

**Coming, not blocking:**
- CUDA GEMM default selection without `TC_USE_CUDA_GEMM=1`; the opt-in
  managed-memory cuBLAS path is validated on RTX 3090, while default
  selection still waits for broader correctness/perf evidence.
- Auto-select backend in `tc_init` (`Metal>CUDA>HIP>CPU` priority)
- Broader CUDA allocator policy for default no-copy GEMM without an env
  flag.
- Larger-network activation-checkpoint policy: the buffer-level CPU/Metal
  discard/realize primitive is implemented; framework-level scheduling
  still needs mesh-aware placement and recompute heuristics.
- AVX2 GEMM default selection and broader throughput tuning; the opt-in
  BLIS-style OpenMP tile fanout is implemented.
- Thunderbolt 4 link validation between two Macs (cable-dependent)

## 9. Where files live

```
include/tensorcore/
    tensorcore.h          umbrella header
    distributed.h         tc_dist_init, tc_allreduce, ...
    diloco.h              tc_diloco_*
    cuda.h                tc_cuda_init, tc_cuda_device_info
    hip.h                 tc_hip_init (chipStar)
    memory_tier.h         tc_buffer_set_tier_hint, promote/demote_async
    checkpoint.h          activation checkpointing API

lib/distributed/
    distributed.mm        Apple build's tc_dist_* + GLOO wiring
    distributed_cpu.cpp   Linux/CPU build's tc_dist_*
    gloo_tcp.cpp          TCP transport (POSIX sockets, both platforms)
    diloco.cpp            DiLoCo runtime (platform-independent)
    sparse_compress.cpp   top-k pack/unpack primitives

lib/cuda/                 direct CUDA backend
lib/hip/                  chipStar HIP backend
lib/ops/*_cpu.cpp         CPU kernels: gemm, attention, conv2d, training
kernels/metal/            Metal shader sources

tests/
    test_dist_remote.c    split-binary distributed test (rank 0 listens,
                          rank N connects; works on real network)
    test_diloco_*.c       fork-based DiLoCo tests (single host)
    test_gloo_fork.c      collective primitives over real TCP
```
