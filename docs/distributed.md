# Distributed

`distributed.h` is the NCCL-equivalent surface for multi-Mac training. v0.1
ships the single-host implementation and a GLOO TCP baseline for default
Apple and portable CPU Ethernet smoke coverage; the high-throughput TB5
ring backend lands in v0.5 once the JACCL substrate is in macOS 26.2 and we
can validate against MLX's reference.

## Surface

```c
typedef enum {
    TC_DIST_SINGLE = 0,    /* one-process emulation (no-op all-reduce)        */
    TC_DIST_RING   = 1,    /* TB5 ring across Macs (v0.5)                     */
    TC_DIST_GLOO   = 2,    /* TCP fallback over Ethernet                      */
} tc_dist_backend_t;

typedef enum {
    TC_REDUCE_SUM = 0,
    TC_REDUCE_AVG = 1,
    TC_REDUCE_MAX = 2,
    TC_REDUCE_MIN = 3,
} tc_reduce_op_t;

tc_status_t tc_dist_init   (tc_context*, tc_dist_backend_t,
                            int world_size, int rank,
                            const char* rendezvous_url,
                            tc_dist_ctx**);
tc_status_t tc_dist_finalize(tc_dist_ctx*);

int tc_dist_world_size(const tc_dist_ctx*);
int tc_dist_rank      (const tc_dist_ctx*);

tc_status_t tc_allreduce(tc_dist_ctx*, tc_buffer*, size_t n, tc_dtype_t, tc_reduce_op_t);
tc_status_t tc_broadcast(tc_dist_ctx*, tc_buffer*, size_t n, tc_dtype_t, int root);
tc_status_t tc_allgather(tc_dist_ctx*, const tc_buffer*, tc_buffer*, size_t n_per_rank, tc_dtype_t);
tc_status_t tc_barrier  (tc_dist_ctx*);
```

## Backends

### `TC_DIST_SINGLE`

`world_size = 1`. All collectives are no-ops (allreduce leaves the buffer
unchanged; broadcast is a no-op for `root == rank == 0`; allgather copies
`in` to the first slot of `out`; barrier returns immediately).

Use this in tests, single-Mac training, and to validate higher-level code
without depending on the multi-Mac substrate. The Eshkol bindings exercise
this path so the FFI layer is testable on one machine.

### `TC_DIST_RING` (v0.5)

Real multi-Mac ring. Each Mac is one rank. Two transport implementations
share the same algorithm:

1. **Thread + shared memory ring** (works today; `tests/test_distributed_ring.c`).
   Each rank is a pthread; the "transport" is shared `tc_buffer` regions
   with mutex synchronization. Correctness-only; not a real test of the
   multi-Mac transport.
2. **Fork + socketpair ring** (works today; `tests/test_distributed_ring_fork.c`).
   Each rank is a forked child process; the "transport" is socketpairs the
   parent set up before fork. **This is the same code path the multi-Mac
   backend will use** — only the socketpair gets swapped for the JACCL /
   TB5 transport.

Both validated bit-exact for 4 ranks × 1024 fp32 elements.

The v0.5 work is the transport swap: socketpair → JACCL (Apple's open
collective comms library, exposed in macOS 26.2). The algorithm — ring
reduce-scatter + ring all-gather — doesn't change.

`tc_dist_init(..., TC_DIST_RING, ...)` today returns
`TC_ERR_UNSUPPORTED_FAMILY` if `world_size > 1` (no real multi-Mac
runtime yet).

### `TC_DIST_GLOO`

CPU-backed all-reduce over Ethernet for default Apple and portable CPU
builds. Slower than the future TB5/JACCL backend but works between any
networked hosts. The current in-tree transport uses `gloo+tcp://host:port`
rendezvous, rank-0 broker collectives by default, and optional direct ring
neighbor sockets for `world_size >= 3` fp32 SUM when `TC_GLOO_RING=1` is
set. The broker rendezvous path accepts IPv4 hosts, DNS names, and
bracketed IPv6 literals such as `tcp://[fd00::10]:29500`. Direct ring setup
is still IPv4-oriented and opportunistic: ranks advertise their reachable
IPv4 address, try bounded neighbor connects, and coordinate fallback over
the rendezvous sockets if any direct edge is blocked by NAT/firewall
policy. Set `TC_GLOO_TRACE=1` to confirm whether a run selected
`route=ring` or `route=broker`. It supports fp32 SUM/AVG/MIN/MAX
all-reduce, fp16 SUM/AVG all-reduce, byte-level broadcast from any root,
allgather, barrier, and the internal sparse TOPK DiLoCo wire path.
bf16/int8 reductions and public generic sparse packed wire-format APIs
still return explicit unsupported statuses.

## All-reduce algorithm

The ring all-reduce is the standard NCCL-style implementation:

```
Reduce-scatter phase:
    for step in 0..world_size-1:
        send my chunk_send  → next rank
        recv chunk_recv     ← prev rank
        reduce chunk_recv into local chunk
Then all-gather phase:
    for step in 0..world_size-1:
        send my chunk_send  → next rank
        recv chunk_recv     ← prev rank
        copy chunk_recv into local chunk
```

Each step moves `n / world_size` elements. Total bandwidth per rank is
`2 * (world_size - 1) / world_size ≈ 2n` for large `world_size` — the
optimal ring bandwidth.

For Thunderbolt 5 (~80 Gbps bidirectional, ~10 GB/s steady-state), the
expected all-reduce throughput on 4 Macs is ~8 GB/s aggregate; ZeRO-2
on a 70B fp16 model bottlenecks on this bandwidth at ~40% MFU — which
is the v0.5 target.

## Rendezvous

`tc_dist_init` takes a rendezvous URL. The scheme determines the
substrate:

- `"single://"` — single-process emulation; ignores URL contents.
- `"tb5://192.168.42.0/cluster"` — TB5 ring; rank 0 advertises on a
  bridge IP, others connect.
- `"gloo+tcp://host0:port"` or `"tcp://host0:port"` — GLOO TCP rendezvous.
- `"tcp://[ipv6-literal]:port"` — GLOO TCP rendezvous over IPv6. Brackets
  are required because colons are part of the address.

`single://` is always functional. `gloo+tcp://` is functional in default
Apple and portable CPU builds when all ranks can reach rank 0's host and
port.

## ZeRO-1 / ZeRO-2 / ZeRO-3 (v0.5 plan)

ZeRO partitions optimizer state, gradients, and parameters across ranks.
v0.5 ships the all-reduce + all-gather + broadcast primitives that ZeRO
needs; the user-side ZeRO scheduler will live in a higher layer
(`tensorcore-train` is the working name).

| ZeRO stage | What's partitioned | tc_* primitives needed |
|---|---|---|
| ZeRO-1 | optimizer states (Adam m, v) | tc_allreduce (grads) |
| ZeRO-2 | optimizer + gradients | tc_reduce_scatter + tc_allreduce (final) |
| ZeRO-3 | optimizer + grads + parameters | tc_allgather (params on demand) |

`tc_reduce_scatter` is not in v0.1 but is essentially the first half of the
ring all-reduce algorithm; will be exposed as a primitive in v0.5.

## Pipeline parallelism (v0.5 plan)

The same `tc_allreduce` + `tc_broadcast` primitives, with the addition of
point-to-point `tc_send` / `tc_recv` (also v0.5). The 1F1B and interleaved
1F1B schedules are scheduling problems, not communication problems — they'll
live in `tensorcore-train`.

## How to use the v0.1 single-Mac path

```c
tc_dist_ctx* d = NULL;
tc_dist_init(ctx, TC_DIST_SINGLE, /*world_size=*/1, /*rank=*/0,
             "single://", &d);

/* train */
tc_allreduce(d, grad_buffer, n_elements, TC_DTYPE_F16, TC_REDUCE_AVG);

tc_dist_finalize(d);
```

`tc_allreduce` is a no-op when `world_size = 1`. Putting it in your training
loop now means v0.5 will be a backend swap, not a code change.

## Tests

- `tests/test_distributed_ring.c`: pthreads + shared memory; algorithm
  correctness for 4 ranks × 1024 fp32 with all four reduce ops. Bit-exact.
- `tests/test_distributed_ring_fork.c`: fork + socketpair; same scenario,
  same result. **This is the real topology** — the only thing that changes
  for multi-Mac is the socket type.
- `tests/test_gloo_fork.c`: localhost smoke with four forked ranks over
  `gloo+tcp://127.0.0.1:port`, covering broker fp32/fp16 allreduce,
  any-root broadcast, allgather, and barrier.
- `tests/test_gloo_ring_fork.c`: localhost smoke with four forked ranks
  and `TC_GLOO_RING=1`, covering direct TCP ring reduce-scatter/all-gather
  for fp32 SUM.

The ring and GLOO TCP fork smokes run in the default Apple suite. The same
GLOO smokes also run in the portable CPU suite.

## What's silicon-bound vs software-bound

| Goal | Bound by | Closes when |
|---|---|---|
| Algorithm correctness | software | v0.1 (done) |
| Single-Mac validation | software | v0.1 (done) |
| Multi-Mac TB5 ring | software (JACCL substrate) | macOS 26.2 + v0.5 |
| 4× M5 Ultra 70B fine-tune | software + TB5 bandwidth | v0.5 |
| >8 Mac frontier-scale | hardware (Apple has to ship better fabric) | v0.6 + Apple |

See [ROADMAP.md](../ROADMAP.md) §v0.5/v0.6 for the realistic competitive
picture.
