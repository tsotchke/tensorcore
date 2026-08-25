# 10 GbE upgrade plan for Kimi K2.6 distributed inference

## Problem statement

Kimi K2.6 (Q4, 543 GB MoE) needs to serve from old-donkey (high-RAM, weight store) → cosbox (high-VRAM RTX 3090, compute). Per-token active set ≈ 11.3 GiB across 8 experts/layer × 61 layers × 23.6 MiB/expert.

`tc_remote_tensor_fetch` (`lib/distributed/remote_tensor.cpp`) is the transport. Current measurement (2026-06-24):

| link | tc_remote_tensor_fetch | raw TCP | local-disk floor (cosbox) |
|---|---:|---:|---:|
| Tailscale od ⇄ cosbox | 103 MB/s | 111 MB/s | 240 MB/s |
| Raw LAN (192.168.1.x) | not tested | 117 MB/s | 240 MB/s |

Both saturate the **1 GbE ethernet ceiling** (~125 MB/s theoretical). Tailscale's userspace WireGuard tax is only ~5 MB/s. Software tuning cannot rescue this — the wire is the bottleneck.

## Decision criterion (from the original NOTE)

Distributed inference is viable if network throughput exceeds cosbox's local-disk throughput (~240 MB/s, the worst case for paging the Q4 expert bank from disk on a cold miss). At 103 MB/s, we're **2.3× SLOWER than the local SSD** — so for K2.6, local-disk paging on cosbox is the correct architecture today.

## Hardware needed to flip the decision

To beat 240 MB/s with margin (target 1+ GB/s for room to absorb p99 spikes):

| component | option A (minimum) | option B (recommended) | option C (future-proof) |
|---|---|---|---|
| NIC, cosbox | Intel X550-T2 (10G-BASE-T) | Mellanox ConnectX-4 Lx (SFP+) | Mellanox ConnectX-6 Dx (25 GbE) |
| NIC, old-donkey | Intel X550-T2 (10G-BASE-T) | Mellanox ConnectX-4 Lx (SFP+) | Mellanox ConnectX-6 Dx (25 GbE) |
| cabling | Cat6a copper, 5m | DAC copper, 1m | DAC copper, 1m |
| switch (if needed) | MikroTik CRS305-1G-4S+IN | MikroTik CRS305-1G-4S+IN | MikroTik CRS510 (25 GbE) |
| approx cost | ~$200 (2 NICs + cable, direct-attach) | ~$300 (2 NICs + DAC) | ~$700 (2 NICs + DAC) |

Option B (Mellanox ConnectX-4 Lx + DAC) is the sweet spot: rock-solid Linux/macOS driver story, 10 GbE wire-rate sustained on both ends, ~10× the current Tailscale throughput.

A direct-attach link between old-donkey and cosbox skips the switch entirely; assumes both machines have a free PCIe x4 slot and the rack distance is < 3m.

## Expected payoff

With option B, projected throughput is **~1.0 GB/s sustained** (typical real-world for Mellanox 10G with WireGuard overhead). That puts us 4× above the local-disk threshold:

| metric | today (1 GbE) | after 10 GbE | unit |
|---|---:|---:|---|
| sustained throughput | 103 | ~1000 | MB/s |
| 23.6 MiB expert fetch latency (p50) | 228 | ~24 | ms |
| 8-expert active set fetch (per token) | 1.82 | ~0.19 | s |
| serving rate (rough) | 0.5 | ~5 | tokens/sec |

`bench/bench_remote_tensor` is the validation tool: re-run it post-upgrade and the numbers above are testable claims, not estimates.

## Failure modes if option A is chosen

10G-BASE-T over Cat6a is fine for cold starts but has higher latency variance under load (~50% extra jitter vs SFP+ DAC), and PHY power is significantly higher. Acceptable for non-time-critical paging; suboptimal for low-latency token serving.

## Recommendation

Buy option B unless rack distance > 5m (then upgrade to fiber DAC, same NIC). Total spend ~$300 unlocks the Kimi distributed-inference track that today's hardware can't support no matter how we tune the software.
