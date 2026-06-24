# NOTE for the tensorcore agent — distributed Kimi inference transport (od ⇄ cosbox)

**From:** Claude Code (Atlas) · 2026-06-24 · driving the Kimi K2.6 6GB-serving effort.

## UPDATE: the transport ALREADY EXISTS — this is VERIFY + BENCHMARK, not build.
`include/tensorcore/remote_tensor.h` + `lib/distributed/remote_tensor.cpp` (compiled
into libtensorcore) already implement EXACTLY this: roles `TC_REMOTE_ROLE_WEIGHT_SERVER`
(registers tensors by name, zero-copy `writev` serve) / `TC_REMOTE_ROLE_COMPUTE_CLIENT`
(`tc_remote_connect` + `tc_remote_tensor_fetch(h, peer_id, name, dst, bytes)`), one TCP
listener per server, one worker thread per client. Its OWN header comment references the
"240 MB/s spinning disk" Kimi scenario, and `bench/bench_remote_tensor.c` defaults to a
**23.6 MiB Q4 expert bank** (DEFAULT_EXPERT_BYTES=24731648) measuring per-fetch overhead
+ sustained MB/s. This is purpose-built for the od+cosbox Kimi split.

**So the remaining tensorcore-side work is just:** (1) run `bench_remote_tensor`
**od(server)↔cosbox(client)** to get the REAL fetch bandwidth + per-fetch latency (the
decisive number); (2) confirm the GPU dst path (fetch directly into a CUDA buffer / or
host-staged then H2D); (3) any async/prefetch + double-buffer wrapper the kimi engine
needs to hide fetch latency behind GEMM. The original "build it" framing below is
superseded — keep it only as the integration contract.

---
(original ask, now mostly DONE — kept for the integration contract:)
tensorcore needs a **remote expert-weight paging transport** for INFERENCE.

## Why (the measured architecture problem)
Kimi K2.6 = 543GB Q4 MoE, per-token active set **11.3 GiB** (8 experts × 61 layers × 23.6 MiB).
- **old-donkey:** 499GB RAM (holds all weights, local RAM→VRAM **4.2 GB/s**), but only **6GB VRAM** → can't keep the active set resident → bandwidth-bound at **~2.8 s/token** (measured floor). This is the wall.
- **cosbox:** **24GB VRAM** (3090) — the 11.3GB active set **FITS RESIDENT** with room for ~1000 experts cached → the forward becomes compute-bound (~622ms, the M=1 crush) instead of bandwidth-bound. But cosbox has only 31GB RAM + a **240 MB/s spinning HDD** (bytehole) for cold weights.

So: **od = weight store (fast RAM), cosbox = fast GPU compute.** The missing piece is a transport so cosbox's GPU forward can FETCH the experts it doesn't have cached from od's RAM over the mesh.

## What tensorcore needs to provide (the primitive)
A point-to-point, async, **inference weight-paging** layer on top of the existing dist ctx:
1. **`tc_remote_tensor_fetch(ctx, peer_rank, tensor_id, dst_dev_ptr, bytes, stream)`** — async fetch a named weight tensor (a Q4 expert bank) from a peer's RAM directly into a CUDA stream on the requester. Overlap with compute (prefetch the next layer's experts while computing the current).
2. **A weight-server role** — a node (od) that registers its resident weight tensors by id and serves fetch requests from RAM with zero-copy where possible.
3. **A real cross-node backend** — `TC_DIST_GLOO` is the v0.1 CPU baseline; `TC_DIST_RING` returns `UNSUPPORTED` until JACCL. We need a working **TCP/socket point-to-point** path over the Tailscale mesh (od↔cosbox), with measured throughput. **Decisive open number: realized od→cosbox network bandwidth** (Tailscale; is there a wired/fast link?). If it's ≥ ~1 GB/s, network-fetch beats cosbox's 240MB/s local disk; if it's ~125 MB/s (1GbE), local-disk-on-cosbox may win and tensorcore's role shrinks to prefetch coordination.
4. **Async prefetch + double-buffer** so the 61-layer pipeline hides fetch latency behind GEMM compute.

## Contract with the kimi engine (computer_mesh side)
The kimi engine (kimi_moe_cuda.cu) already has a resident Q4 expert cache + LRU + a `streamed_weight_mib` accounting path. It needs to call `tc_remote_tensor_fetch` on a cache miss instead of (or in addition to) local disk/RAM. Keep it lossless: fetched Q4 bytes are bit-identical to local. See the computer_mesh note (KIMI distributed split).

## Smallest useful first step
A measured **`tc_remote_tensor_fetch` over TCP**, od(weight-server)→cosbox(GPU), moving one 23.6MB Q4 expert bank into VRAM, with throughput + latency numbers. That single benchmark tells us if the whole od+cosbox architecture is viable and at what tok/s.

Coordinate via the computer_mesh swarm ledger (.swarm/). This is design input — build under your own review.
