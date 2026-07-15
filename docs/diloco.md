# DiLoCo — cross-continent distributed training

When ranks are on different continents, the round-trip is 100-200 ms and
the practical bandwidth between sites is 1-10 MB/s. Standard distributed
training (per-step gradient all-reduce) cannot run over those numbers —
the sync time per step exceeds the compute time per step by 100-1000×.

DiLoCo (**Di**stributed **Lo**w-**Co**mmunication training, Douillard et al.
2024) solves this by **syncing parameter deltas at low duty cycle**, not
gradients per step. Each worker does K=100-1000 normal inner-loop SGD steps
locally before computing Δθ and exchanging it cross-site. The
**communication volume drops by ~K×** without accuracy loss.

Validated at scale: OpenDiLoCo / Prime Intellect's INTELLECT-1 trained a
10B-parameter model across volunteers' machines on the public internet
using this exact algorithm.

## The algorithm

```
For each worker:
    θ_local = θ_global_anchor                   // start aligned

For each outer step t:
    For inner step k in 0..K-1:
        Run normal optimizer (Adam) on θ_local with local minibatch.

    Δθ_i := θ_local_i − θ_global_anchor          // each rank's delta
    Δθ_i' := compress(Δθ_i)                       // fp16, top-k, low-rank, ...
    Δ̄θ := all_reduce_avg(Δθ_i') across ranks      // cross-site comm here
    Δ̄θ := decompress(Δ̄θ)
    θ_global_anchor += outer_lr × Δ̄θ              // Nesterov / SGD / Adam
    θ_local := θ_global_anchor                    // resync, start next outer
```

Crucially, the all-reduce happens *once per K inner steps*, not once per
step. With K=100, the cross-site bandwidth requirement drops 100×.

## Why this works for transcontinental setups

For your specific topology (Quebec Macs ↔ Alaska Linux, ~200 ms RTT,
~1.3 MB/s cross-site bandwidth):

| Model | Δθ raw (fp32) | + fp16 | + top-k 0.1% | Cross-site at 1.3 MB/s |
|---|---:|---:|---:|---:|
| 7B fp16 | 28 GB | 14 GB | 14 MB | **11 sec / outer step** |
| 13B fp16 | 52 GB | 26 GB | 26 MB | 20 sec / outer step |
| 70B fp16 | 280 GB | 140 GB | 140 MB | 110 sec / outer step |

With K=100 inner steps × ~100 ms/step = 10 sec of compute per worker
between outer steps. With **async overlap** (the outer all-reduce runs in
the background while the next K inner steps execute), the bridge cost is
fully hidden if `K × inner_step_time > compress_time + bridge_time +
decompress_time`. For 7B with K=100, that's `10 sec > 11 sec` — barely;
K=200 makes it comfortable. For 13B+, K=500-1000.

**This is what makes cross-continent unified-RAM training actually work.**

## API

See [api_reference.md](api_reference.md) and
[`include/tensorcore/diloco.h`](../include/tensorcore/diloco.h) for the
full surface.
Concise:

```c
/* 1. Set up cross-site distributed context (TC_DIST_GLOO over WAN). */
tc_dist_ctx* cross_site = NULL;
tc_dist_init(ctx, TC_DIST_GLOO, world_size, rank, rendezvous_url, &cross_site);

/* 2. Configure DiLoCo. */
tc_diloco_config cfg = {
    .inner_steps = 100,
    .outer_lr = 0.7f,                /* DiLoCo paper: outer lr ~0.7 for Nesterov */
    .outer_momentum = 0.9f,
    .outer_optimizer = TC_DILOCO_OUTER_NESTEROV,
    .compress = TC_DILOCO_COMPRESS_TOPK_01PCT,
    .async_overlap = true,           /* immutable snapshot + explicit commit */
    .tolerate_dropouts = false,      /* recovery is staged work */
};
tc_diloco_ctx* d = NULL;
tc_diloco_init(cross_site, &cfg, &d);

/* 3. Register parameters once. */
tc_diloco_add_parameter(d, "blk.0.attn_q.weight", theta_q, n_q_elems, TC_DTYPE_F16);
/* ... all parameters ... */

/* 4. Inside the inner training loop, after each inner step: */
bool outer_pending;
tc_diloco_step(d, &outer_pending);
if (outer_pending) {
    tc_diloco_apply_outer(d);   /* snapshots and returns while the worker runs */
}

/* At a caller-owned parameter boundary, before launching another round: */
tc_diloco_async_state_t state;
tc_status_t worker_status;
uint64_t round_id;
tc_diloco_async_poll(d, &state, &worker_status, &round_id);
if (state == TC_DILOCO_ASYNC_READY) {
    tc_diloco_async_commit(d);
} else if (state == TC_DILOCO_ASYNC_FAILED) {
    /* Commit acknowledges and returns the background error without mutation. */
    tc_diloco_async_commit(d);
}
```

## Compression schemes

| Scheme | Intended reduction | Current runtime status |
|---|---:|---|
| `NONE` | 1× | Implemented as dense fp32 |
| `FP16` | 2× | Accepted, but the multi-rank path currently sends dense fp32; no wire reduction yet |
| `FP8` (per-tensor scaled) | 4× | Reserved; `tc_diloco_init` returns unsupported |
| `TOPK_1PCT` | ~100× | Implemented with error feedback and sparse `(index, fp16 value)` Gloo payloads; non-Gloo transports use a dense masked fallback |
| `TOPK_01PCT` | ~1000× | Implemented with the same Gloo-only sparse transport path |
| `LOWRANK` (PowerSGD) | 50-500× | Reserved; `tc_diloco_init` returns unsupported |
| `SIGNSGD` | 32× | Reserved; `tc_diloco_init` returns unsupported |

Error feedback (storing the lost residual locally and adding it to the
next outer Δθ) is what makes top-k stable; the runtime handles this
internally when `compress` is one of the top-k variants.

## Snapshot-safe async overlap

Immutable-snapshot async overlap is the intended mechanism for hiding WAN
latency. The target schedule is:

Without overlap:

```
   |--inner K steps (10 sec)--||---outer sync (11 sec)---||--inner K--||---outer---|
```

With overlap, the outer all-reduce starts when the inner K-th step
completes, runs on a background thread, and the next K inner steps proceed
against the *previous* θ_global_anchor. The new anchor is swapped in at
the next outer-step boundary:

```
   |--inner K (10s)--||--inner K (10s)--||--inner K (10s)--|
                       |---outer sync (11s)---|
                                              |---outer sync (11s)---|
```

Net effect can approach zero wall-clock overhead for the cross-site sync when
`K × inner_step_time > outer_sync_time`.

The runtime realizes this contract with an explicit state machine. `apply_outer`
captures immutable fp32 snapshots on the caller thread. The worker performs
compression, collectives, and the outer optimizer against private parameter
state and never maps a registered live buffer. `tc_diloco_async_poll` and
`tc_diloco_async_wait` expose READY/FAILED state and the round identity.
`tc_diloco_async_commit` runs on the caller thread and preserves inner work made
after the snapshot using:

```
committed_live = new_anchor + (current_live - round_snapshot)
```

Only a successful commit increments `outer_steps_completed`. A failed worker
leaves live parameters and counters unchanged; commit acknowledges and returns
the background error. A second launch before commit returns `TC_ERR_BUSY`.
Finalize refuses to destroy a context with an unacknowledged result. Portable
regressions cover successful rebase, private pre-commit state, round ordering,
and an injected Gloo peer loss.

## Versioned checkpoint state

`TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME` is available when the runtime exposes
the v1 checkpoint blob. The blob is deterministic and little-endian. It owns
only DiLoCo state: anchors, Nesterov/Adam moments, top-k error-feedback
residuals, counters and metrics, round identity, topology/membership epochs,
and a READY or FAILED async result. Model parameter bytes remain owned by the
caller and must be restored before the DiLoCo blob.

```c
tc_diloco_state_set_epochs(d, topology_epoch, membership_epoch);

size_t state_size = 0;
tc_diloco_state_size(d, TC_DILOCO_STATE_ABI_VERSION_CURRENT, &state_size);
void* state = malloc(state_size);
size_t written = 0;
tc_diloco_state_serialize(d, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                          state, state_size, &written);

/* Recreate the same config, dist rank/world, and registered parameters;
 * restore each live model buffer first, then restore DiLoCo-owned state. */
tc_diloco_state_deserialize(resumed,
                            TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                            state, written);
```

Restore validates the exact config, rank/world, parameter name/dtype/size, and
any non-zero local topology epochs before changing the context. Corrupt blobs,
unknown versions, config drift, epoch drift, or parameter-layout drift fail
without partial state mutation. A RUNNING worker returns `TC_ERR_BUSY`; wait
for READY/FAILED before checkpointing. READY state can be committed after
restore, and FAILED state preserves its error until acknowledgement.

## Failure tolerance

Real cross-continent links drop. Tailscale relays through Seattle if a
direct path can't be negotiated; ISPs reset; mid-step crashes happen.
Dropout-tolerant rounds are the target policy: a bounded-time round should
re-form membership, continue with the accepted cohort, and resynchronize a
returning rank from an authenticated checkpoint and anchor.

That behavior is not implemented yet. `tolerate_dropouts` is a reserved config
field and `tc_diloco_init` rejects `true`; current collectives can still block
or fail when a rank disappears. The campaign requires injected rank-loss,
rejoin, replay, checkpoint-lineage, and bounded-time evidence before the flag
is accepted as functional.

## Position in the distributed stack

```
                  ┌── Within Quebec site ─────────────┐
                  │  TC_DIST_RING over Thunderbolt 4   │
                  │  per-step gradient all-reduce      │
                  │  RTT < 1 ms, BW ~3 GB/s             │
                  │  Standard DDP                       │
                  └─────────────────┬──────────────────┘
                                    │  one rank per site is
                                    │  the "outer-step worker"
                                    ▼
                  ┌── DiLoCo cross-site bridge ────────┐
                  │  TC_DIST_GLOO over Tailscale WAN   │
                  │  outer-step Δθ all-reduce          │
                  │  RTT 100-200 ms, BW 1-10 MB/s       │
                  │  K=100-1000 inner steps between    │
                  └─────────────────┬──────────────────┘
                                    ▲
                                    │
                  ┌── Within Alaska site ─────────────┐
                  │  TC_DIST_GLOO over 10 GbE LAN     │
                  │  per-step gradient all-reduce     │
                  │  RTT < 5 ms, BW ~1 GB/s            │
                  │  Standard DDP                      │
                  └────────────────────────────────────┘
```

DiLoCo runs at the top of this stack. The actual cross-site transport
(TC_DIST_GLOO) is a separate concern; DiLoCo just calls `tc_allreduce`
on the dist_ctx the user provided.

## References

- **DiLoCo** (Douillard et al., 2024): https://arxiv.org/abs/2311.08105
- **OpenDiLoCo** (Prime Intellect, 2024): https://github.com/PrimeIntellect-ai/OpenDiLoCo
- **INTELLECT-1** (10B model trained over the public internet): https://www.primeintellect.ai/blog/intellect-1-release
- **PowerSGD** (Vogels et al., 2019): low-rank gradient compression
- **Error feedback for sparsified SGD** (Karimireddy et al., 2019)

## Implementation status

`include/tensorcore/diloco.h` declares the surface and
`lib/distributed/diloco.cpp` implements the local single-rank path plus
dense and sparse TOPK multi-rank outer steps over `TC_DIST_GLOO`.
Sparse TOPK uses `(idx, fp16)` payloads through the
internal GLOO sparse all-reduce hook. Dropout-tolerant WAN recovery and
broader WAN soak remain staged work. See [ROADMAP.md](../ROADMAP.md).
