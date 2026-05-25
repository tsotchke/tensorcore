# Mesh Resource Scheduler

`scripts/mesh_resource_scheduler.py` is the mesh-wide admission controller for
shared hardware such as `cosbox:cuda3090`. It sits above individual projects:
qLLM, GeoRefine, tensorcore demos, and future agents should submit desired
work to this scheduler instead of starting CUDA jobs directly.

The scheduler uses the Tsotchke arbiter as the lease backend. Its policy is
conservative:

- Live work is never killed for priority alone.
- Jobs may target one resource, an explicit `resources` list, or an inventory
  `resource_pool`; pool jobs are resolved to per-resource placements.
- Tenant counts are tracked across resources, so idle resources are assigned
  to the least-scheduled tenant first, with `priority` used inside that fair
  share.
- Known stale leases are released only when that job's probe says dead.
- Unknown leases block the resource unless the live job explicitly declares
  matching adoption metadata.
- A live known job without a lease is adopted by claiming a lease for it.
- A live known job may adopt an otherwise-unknown lease only when
  `adopt_unknown_lease_metadata_keys` is set, the lease has the same tenant,
  and every listed metadata field matches. This recovers manual leases that
  describe the same workload without stealing another user's lease.
- A live CUDA-exclusive job with an existing lease refreshes worker identity
  metadata on heartbeat, so lease records do not stay pending after adoption.
- New CUDA-exclusive work is launched only if its mandatory `admission_cmd`
  exits 0.
- CUDA-exclusive work must also pass a post-start probe and report worker
  identity before the launch is considered healthy.
- A new job is launched only after the scheduler has claimed the resource.
- If a launch fails, the scheduler releases the lease it just claimed.

## Fleet Inventory

Mesh capacity is declared in `configs/mesh_resources.json`. That inventory is
the source of truth for accelerator ownership and scheduling eligibility:

- `cosbox:cuda3090` is the primary exclusive CUDA artifact lane.
- `old-donkey:cuda3050` is the low-VRAM CUDA precompute lane.
- `jack-blupc:cuda3060` is active as a scheduler-registered paused lane.
  Windows SSH/bootstrap, portable CPU smoke, CUDA Toolkit 12.6 redistributable
  discovery, exclusive admission, and CUDA build/CTest smoke are healthy.
  Submitted Jack CUDA jobs must still provide workload-specific start,
  post-start, and Windows worker-identity probes before they can launch.
- `atlas:metal_m2ultra` is active Metal capacity for validation, evaluation,
  generation support, and Tensorcore Metal workloads.
- `enki:metal_m4_tsotchke_chan` is `reserved`; only `tsotchke-chan` owners may
  use it. General mesh jobs must not target the M4 Metal slot.

Each row also declares `control_plane`:

- `tensorcore_scheduler`: jobs for this resource must be present in the
  scheduler jobs file, or the system audit reports an unused scheduler lane.
- `direct_lease`: clients may claim arbiter leases directly without a scheduler
  job, useful for request-scoped generation services.
- `reserved`: capacity is visible but only the `reserved_for` principals may
  use it.
- `blocked`: capacity is inventoried but must not receive scheduled work.

Run the scheduler with `--inventory-json configs/mesh_resources.json` and
`--jobs-json configs/mesh_resource_jobs.json` so jobs targeting unknown,
blocked, or reserved resources fail validation before they can claim an arbiter
lease. Inventory rows with `backend: "cuda"` are also forced through the
`cuda_exclusive` job policy; a CUDA job cannot downgrade itself to `generic`
and bypass admission, post-start, or worker-identity checks.
Validate inventory edits with:

```sh
python3 scripts/check_mesh_resource_inventory.py
python3 scripts/check_mesh_resource_jobs.py
```

Audit the live mesh, including scheduler freshness, arbiter inventory, lease
metadata, and optional CUDA process ownership:

```sh
scripts/mesh_system_audit.py \
  --inventory-json configs/mesh_resources.json \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --scheduler-state-json ~/.tsotchke/state/mesh-resource-scheduler-state.json \
  --arbiter-cmd "scripts/mesh_arbiter_with_inventory.py --inventory-json configs/mesh_resources.json --arbiter-cmd ~/.tsotchke/bin/tsotchke-arbiter -- status --json" \
  --probe-cuda
```

Use `scripts/mesh_arbiter_with_inventory.py` as the scheduler's `--arbiter-cmd`
wrapper when the arbiter status path should also show the inventory resources:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd "scripts/mesh_arbiter_with_inventory.py --inventory-json configs/mesh_resources.json --arbiter-cmd ~/.tsotchke/bin/tsotchke-arbiter --" \
  --inventory-json configs/mesh_resources.json \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json
```

## Jobs File

The jobs file is JSON:

```json
{
  "schema": "tensorcore.mesh_resource_jobs.v1",
  "jobs": [
    {
      "id": "qllm-phase1-cosbox",
      "sync_id": "qllm-phase1-cosbox",
      "resource": "cosbox:cuda3090",
      "resource_class": "cuda_exclusive",
      "owner": "qllm:phase1",
      "priority": 50,
      "desired_state": "running",
      "ttl_sec": 900,
      "probe_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user is-active --quiet qllm-phase1.service"
      ],
      "start_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user start qllm-phase1.service"
      ],
      "admission_cmd": [
        "ssh",
        "cosbox",
        "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
      ],
      "post_start_probe_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user is-active --quiet qllm-phase1.service && nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q qllm"
      ],
      "worker_identity_cmd": [
        "ssh",
        "cosbox",
        "python3 /home/tyr/work/tensorcore/scripts/mesh_worker_identity.py --resource cosbox:cuda3090 --unit qllm-phase1.service --match-regex qllm.train_geometric_lm_torch --require-active-unit --require-matching-process --require-matched-cuda"
      ],
      "metadata": {
        "project": "semiclassical_qllm"
      }
    },
    {
      "id": "georefine-m2-cosbox",
      "sync_id": "georefine-m2-cosbox",
      "resource": "cosbox:cuda3090",
      "resource_class": "cuda_exclusive",
      "owner": "georefine:m2",
      "priority": 100,
      "desired_state": "running",
      "ttl_sec": 900,
      "probe_cmd": [
        "ssh",
        "cosbox",
        "pgrep -af 'experiments.georefine|m2_compress|m2_live_agent' >/dev/null"
      ],
      "completion_cmd": [
        "ssh",
        "cosbox",
        "/home/tyr/.local/bin/check-georefine-qwen-artifact /path/to/run --max-size-ratio 0.10"
      ],
      "admission_cmd": [
        "ssh",
        "cosbox",
        "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
      ],
      "post_start_probe_cmd": [
        "ssh",
        "cosbox",
        "pgrep -af 'experiments.georefine.m2_compress' >/dev/null"
      ],
      "worker_identity_cmd": [
        "ssh",
        "cosbox",
        "python3 /home/tyr/work/tensorcore/scripts/mesh_worker_identity.py --resource cosbox:cuda3090 --match-regex experiments.georefine.m2_compress --artifact-dir /path/to/run --require-matching-process --require-matched-cuda"
      ],
      "metadata": {
        "project": "georefine"
      }
    }
  ]
}
```

Fields:

- `id`: stable scheduler job name.
- `logical_id`: optional stable logical job name. Pool placements use
  `id@resource` as their placement id and keep `logical_id` as the unsuffixed
  job id.
- `sync_id`: stable lease identity. Keep this stable across PID churn.
- `resource`: arbiter resource name, for example `cosbox:cuda3090`.
- `resources`: optional explicit list of candidate resource names. The
  scheduler expands this into one placement per resource.
- `resource_pool`: optional inventory selector. It can be a class/backend/tag
  string such as `validation`, a list of exact resource ids, or an object with
  fields such as `backend`, `class`/`classes`, `node`/`nodes`, `tags`,
  `resources`, and `min_memory_gib`.
- `resource_class`: `generic` or `cuda_exclusive`. If omitted, resources
  containing `:cuda` or starting with `cuda` are treated as `cuda_exclusive`.
- `owner`: human-readable lease owner.
- `tenant`: user or service principal for fair-share accounting. If omitted,
  it defaults to the part of `owner` before the first `:`.
- `priority`: tie-breaker used only when the resource is idle and tenant
  fair-share counts are equal.
- `max_parallel`: maximum number of placements for the same logical job that
  may be active or planned in one scheduler pass. Defaults to `1`; raise it for
  sharded jobs that should use multiple machines.
- `tenant_max_parallel`: optional per-job tenant concurrency cap. The scheduler
  also supports a global `--max-running-per-tenant` cap.
- `adopt_unknown_lease_metadata_keys`: optional list of metadata fields that
  allow the scheduler to adopt a live job's otherwise-unknown lease when the
  lease has the same tenant and all listed metadata fields match. Keep this
  narrow, for example `["project", "service", "run_dir"]`.
- `desired_state`: `running` makes the job launchable; `paused` prevents new
  launches but still lets a live job be adopted and protected.
- `probe_cmd`: optional command that exits 0 when the job is live. If omitted
  or inconclusive, liveness is unknown and known leases block scheduling.
- `completion_cmd`: optional command that exits 0 only when the job's output is
  complete and should not be relaunched. A nonzero exit code means
  "definitively incomplete"; a timeout is treated as unknown and blocks
  relaunch of that job for the current scheduler pass.
- `admission_cmd`: command that exits 0 only when the target host is currently
  eligible for this job. Use this for evidence gates such as CUDA,
  HIP/chipStar, Windows, PyTorch, or live-mesh proof. A nonzero exit code or
  timeout blocks launching that job for the current scheduler pass, but does
  not kill or release already-live work. Required for `cuda_exclusive` jobs.
- `post_start_probe_cmd`: command that proves a just-started worker is actually
  live on the target host. Required for `cuda_exclusive` jobs. If this fails,
  the scheduler releases the lease it just claimed.
- `worker_identity_cmd`: command that prints JSON identifying the worker host,
  PID, service/cgroup, and accelerator process metadata. Required for
  `cuda_exclusive` jobs and stored in adopted lease metadata.
- `start_cmd`: command used only when the job is selected for launch.
- `metadata`: copied into the arbiter lease with `sync_job_id`, `job_id`,
  `logical_job_id`, `tenant`, `resource_class`, scheduler host, and worker
  identity status.

Command arrays and command strings may use these placement placeholders:
`{resource}`, `{node}`, `{backend}`, `{tenant}`, `{owner}`, `{id}`,
`{logical_id}`, `{sync_id}`, and `{resource_class}`. This lets one pool job use
the same command template across Atlas, old-donkey, Jack, or future nodes.

Example multi-user pool job:

```json
{
  "id": "heldout-eval",
  "owner": "alice:evaluator",
  "tenant": "alice",
  "resource_pool": {
    "backend": "service",
    "classes": ["validation", "llm-generation"]
  },
  "max_parallel": 2,
  "priority": 40,
  "desired_state": "running",
  "ttl_sec": 600,
  "probe_cmd": ["ssh", "{node}", "/opt/tensorcore/bin/heldout-eval-live {logical_id}"],
  "start_cmd": ["ssh", "{node}", "/opt/tensorcore/bin/start-heldout-eval {logical_id} {resource}"],
  "metadata": {
    "project": "semiclassical_qllm"
  }
}
```

For the current GeoRefine Qwen target, completion means the run's
`m2_certificate.json` has `completed=true`, a positive final held-out PPL
(`ppl_compressed_eval`), a positive final stored size (`size_compressed_bytes`),
and `size_ratio <= 0.10`. `scripts/check_georefine_qwen_artifact.py` enforces
that gate.

## Running

Dry-run one pass:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd ~/.tsotchke/bin/tsotchke-arbiter \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --dry-run \
  --pretty-json
```

Run the control loop and write last-state evidence:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd ~/.tsotchke/bin/tsotchke-arbiter \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --state-json ~/.tsotchke/state/mesh-resource-scheduler-state.json \
  --loop \
  --admission-timeout-sec 10 \
  --post-start-timeout-sec 30 \
  --post-start-interval-sec 2 \
  --worker-identity-timeout-sec 10 \
  --interval-sec 30
```

Agents should update the jobs file atomically, then let the scheduler loop make
the launch decision. They should not kill another agent's process to free CUDA.

## Evidence Admission

Admission commands should be cheap, read-only checks over evidence already
produced by the host's smoke jobs. Examples:

```json
{
  "admission_cmd": [
    "ssh",
    "cosbox",
    "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
  ]
}
```

```json
{
  "admission_cmd": [
    "ssh",
    "xavier",
    "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --hip-toolchain /tmp/hip-toolchain.json --require-hip-spirv-runtime --require-hip-toolchain-clean-head"
  ]
}
```

This makes scheduler state explicit: a CUDA training job waits for CUDA
evidence, a chipStar job waits for SPIR-V-capable GPU runtime readiness, and
a Windows job can wait for Jack's host-smoke evidence before receiving mesh
work.

For exclusive NVIDIA resources, append `scripts/check_cuda_resource_admission.py`
to the admission command so unmanaged CUDA compute applications block a launch:

```sh
python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090
```

## Worker Identity

Use `scripts/mesh_worker_identity.py` as the `worker_identity_cmd` on Linux
hosts. It emits `tensorcore.mesh_worker_identity.v1` JSON with hostname,
matching worker PIDs, optional user-systemd unit metadata, matching
`nvidia-smi` compute applications, and `cuda_pids`:

```sh
python3 scripts/mesh_worker_identity.py \
  --resource cosbox:cuda3090 \
  --unit qllm-phase1.service \
  --match-regex qllm.train_geometric_lm_torch \
  --require-active-unit \
  --require-matching-process \
  --require-matched-cuda
```

The command exits nonzero if a required unit is inactive, a required process
does not match, `--require-cuda` is set and no CUDA process exists, or
`--require-matched-cuda` is set and no matched worker PID owns CUDA.

## Failure Modes

- `resource_busy_unknown_lease`: another holder is present and not in the jobs
  file. Add a probeable job entry or release the lease manually after verifying
  the process is dead.
- `multiple live holders detected`: two known jobs are live for the same
  resource. The scheduler refuses to pick a victim; stop one job explicitly.
- `live_holder_blocked_by_known_lease_unknown_liveness`: a known lease has no
  reliable probe result. Fix the probe before scheduling more work.
- `idle_admission_blocked`: no live holder exists, but every otherwise
  launchable job failed or timed out its admission check. Refresh the relevant
  evidence artifact or fix the host/toolchain before relaunching.
- `live_holder_identity_failed`: a live CUDA-exclusive job exists, but the
  worker identity command failed. Fix the identity probe before allowing
  automated adoption.
- `claimed_and_launched` with `ok=false`: the start command failed and the
  claimed lease was released unless the worker passed post-start validation.
