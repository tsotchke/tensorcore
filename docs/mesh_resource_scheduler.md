# Mesh Resource Scheduler

`scripts/mesh_resource_scheduler.py` is the mesh-wide admission controller for
shared hardware such as `cosbox:cuda3090`. It sits above individual projects:
qLLM, GeoRefine, tensorcore demos, and future agents should submit desired
work to this scheduler instead of starting CUDA jobs directly.

## Cluster Scheduler V1 Boundary

The scheduler is now the required control-plane API for trusted GPU work. A
client submits a `tensorcore.job.v1` document; the scheduler validates it
against `configs/mesh_resources.json`, converts it to the existing mesh job
row, writes it to the scheduler-owned jobs queue, and only then can a scheduler
loop claim a canonical resource and launch the worker command.

The v1 boundary is intentionally backend-neutral:

- `tensorcore.job.v1` describes owner, tenant, priority, resource selector,
  command, environment, artifact root, quality gates, and preemption policy.
- `submit` validates and queues the job or emits a dry-run launch plan.
- `status` reports inventory, queued jobs, and arbiter leases.
- `cancel` disables a queued job without claiming to have killed live work.
- `drain` and `undrain` update inventory state for operator maintenance.
- `audit` verifies that CUDA jobs have admission, post-start probe, worker
  identity, and run-intent contracts. It can also consume worker GPU
  reconciliation reports so unleased CUDA processes fail the control-plane
  audit before new work is admitted.

This keeps Tensorcore as the first-party scheduler while preserving a clean
adapter boundary for a later Slurm or Kubernetes backend.

Example dry run:

```sh
python3 scripts/mesh_resource_scheduler.py submit \
  --job-json configs/georefine_qwen_job.template.json \
  --jobs-json /var/lib/tensorcore/mesh_resource_jobs.json \
  --inventory-json configs/mesh_resources.json \
  --dry-run --pretty-json
```

Real `submit` and `cancel` mutations require `--event-log-jsonl`; dry runs do
not. Keep that path in the scheduler state directory so queue mutations have an
append-only audit trail. Mutating commands take an advisory lock beside the
queue file, named like `.mesh_resource_jobs.json.lock`, before the
load-modify-write cycle and before appending the queue event. Scheduler-VM
clients should call the submit/cancel CLI instead of editing the queue file
directly.

Example scheduler loop:

```sh
python3 scripts/mesh_resource_scheduler.py \
  --jobs-json /var/lib/tensorcore/mesh_resource_jobs.json \
  --inventory-json configs/mesh_resources.json \
  --state-json /var/lib/tensorcore/mesh_resource_state.json \
  --gpu-reconciliation-audit-json /var/lib/tensorcore/gpu-reconciliation-audit.json \
  --gpu-reconciliation-max-age-sec 120 \
  --loop --json
```

For a dedicated Linux scheduler VM, use
`configs/tensorcore-scheduler.service.example` with
`configs/tensorcore-scheduler.env.example` as the starting systemd unit and
environment file. Install
`configs/tensorcore-gpu-reconciliation-audit.service.example` and
`configs/tensorcore-gpu-reconciliation-audit.timer.example` beside it for the
periodic read-only GPU reconciliation audit.

Trusted GeoRefine Qwen compression jobs should start from
`configs/georefine_qwen_job.template.json` and use
`scripts/start_georefine_qwen_rank_probe.py`. Direct GPU launchers are legacy
emergency bypasses and their artifacts are not trusted unless promoted through
the scheduler/finalizer path; `scripts/start_georefine_qwen_cr025.py` refuses
non-preflight launches unless the explicit legacy override flag or environment
variable is present.

## Dedicated Scheduler VM Contract

The scheduler should run on one always-on control-plane VM with SSH reachability
to workers and arbiter reachability. GPU workers do not need Slurm or
Kubernetes for v1; they need the Tensorcore checkout, worker probes, and the
project harnesses that scheduler jobs call.

Provisioning contract:

- `/opt/tensorcore`: read-only or deployment-managed Tensorcore checkout.
- `/var/lib/tensorcore`: scheduler-owned writable queue/state directory.
- `/etc/tensorcore/scheduler.env`: environment based on
  `configs/tensorcore-scheduler.env.example`.
- `tensorcore-scheduler.service`: systemd unit based on
  `configs/tensorcore-scheduler.service.example`.
- `tensorcore-gpu-reconciliation-audit.timer`: systemd timer based on
  `configs/tensorcore-gpu-reconciliation-audit.timer.example`.
- `mesh_resource_jobs.json`: scheduler-owned queue; project repos submit into
  it through `mesh_resource_scheduler.py submit`, not by editing worker state.

The service account should have no GPU-local shell automation beyond the
scheduler commands. Emergency manual GPU launches are allowed only with an
explicit override and produce untrusted artifacts.

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
- Stale unknown leases still block scheduling, but after the quarantine age
  threshold they are reported as `stale_unknown_quarantine_candidate` with
  lease age and metadata-key evidence so an operator can review them without
  the scheduler taking destructive action.
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

- `cosbox:cuda3090` is the primary exclusive CUDA artifact lane. The current
  GeoRefine CR025 row has a source-controlled starter
  (`scripts/start_georefine_qwen_cr025.py`) but stays paused until the
  GeoRefine checkout and Python environment are deployed on cosbox. The
  qLLM phase-1 row is also adoption-only until `qllm-phase1.service` or an
  equivalent launcher is installed from the qLLM git checkout rather than a
  host-local systemd unit.
- `old-donkey:cuda3050` is the low-VRAM CUDA precompute lane. The current
  precompute-chain row has a source-controlled starter
  (`scripts/start_qllm_olddonkey_precompute_chain.py`) but stays paused until
  the dedicated qLLM checkout, data shards, and Python environment are verified
  on old-donkey.
- `jack-blupc:cuda3060` is active as a scheduler-registered paused lane.
  Windows SSH/bootstrap, portable CPU smoke, CUDA Toolkit 12.6 redistributable
  discovery, exclusive admission, and CUDA build/CTest smoke are healthy.
  Submitted Jack CUDA jobs must still provide workload-specific start,
  post-start, and Windows worker-identity probes before they can launch. The
  repo-owned scheduled-smoke helpers are also paused on Jack until a persistent
  Windows service or credentialed scheduled-task path can keep CUDA work alive
  after an SSH session exits.
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
  --arbiter-cmd "scripts/mesh_arbiter_with_inventory.py --inventory-json configs/mesh_resources.json --arbiter-cmd tsotchke-arbiter -- status --json" \
  --probe-cuda
```

Use `scripts/mesh_arbiter_with_inventory.py` as the scheduler's `--arbiter-cmd`
wrapper when the arbiter status path should also show the inventory resources:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd "scripts/mesh_arbiter_with_inventory.py --inventory-json configs/mesh_resources.json --arbiter-cmd tsotchke-arbiter --" \
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
      "tenant": "qllm",
      "priority": 50,
      "desired_state": "paused",
      "ttl_sec": 900,
      "adopt_unknown_lease_metadata_keys": [
        "project",
        "service"
      ],
      "probe_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user is-active --quiet qllm-phase1.service"
      ],
      "admission_cmd": [
        "ssh",
        "cosbox",
        "cd ~/src/tensorcore && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
      ],
      "post_start_probe_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user is-active --quiet qllm-phase1.service"
      ],
      "worker_identity_cmd": [
        "ssh",
        "cosbox",
        "cd ~/src/tensorcore && python3 scripts/mesh_worker_identity.py --resource cosbox:cuda3090 --unit qllm-phase1.service --match-regex qllm.train_geometric_lm_torch --require-active-unit --require-matching-process --require-matched-cuda"
      ],
      "metadata": {
        "project": "semiclassical_qllm",
        "scheduler_pause_reason": "adoption-only until qllm-phase1.service or an equivalent phase-1 launcher is source-controlled and installed from the qLLM git checkout",
        "service": "qllm-phase1"
      }
    },
    {
      "id": "georefine-m2-cosbox",
      "sync_id": "georefine-m2-cosbox",
      "resource": "cosbox:cuda3090",
      "resource_class": "cuda_exclusive",
      "owner": "georefine:m2",
      "tenant": "georefine",
      "priority": 100,
      "desired_state": "paused",
      "ttl_sec": 900,
      "probe_cmd": [
        "ssh",
        "cosbox",
        "pgrep -af 'experiments.georefine|m2_compress|m2_live_agent' >/dev/null"
      ],
      "completion_cmd": [
        "ssh",
        "cosbox",
        "cd ~/src/tensorcore && python3 scripts/check_georefine_qwen_artifact.py /path/to/run --max-size-ratio 0.10"
      ],
      "admission_cmd": [
        "ssh",
        "cosbox",
        "cd ~/src/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
      ],
      "post_start_probe_cmd": [
        "ssh",
        "cosbox",
        "pgrep -af 'experiments.georefine.m2_compress' >/dev/null"
      ],
      "worker_identity_cmd": [
        "ssh",
        "cosbox",
        "cd ~/src/tensorcore && python3 scripts/mesh_worker_identity.py --resource cosbox:cuda3090 --match-regex experiments.georefine.m2_compress --artifact-dir /path/to/run --require-matching-process --require-matched-cuda"
      ],
      "metadata": {
        "project": "georefine",
        "scheduler_pause_reason": "adoption-only until the GeoRefine launcher is source-controlled and preflighted"
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
  `scripts/check_mesh_resource_jobs.py` requires paused rows to include
  `metadata.scheduler_pause_reason`; paused rows with `start_cmd` must also
  include `preflight_cmd`.
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
- `preflight_cmd`: optional non-launching command for paused rows. It should
  verify prerequisites needed before changing `desired_state` to `running` and
  must emit JSON via `--json`.
- `metadata.preflight_default`: set to `false` when a paused row's preflight is
  safe but should not run in the default preflight sweep. Operators can still
  run it explicitly with `check_mesh_resource_preflights.py --job-id <id>`.
  The preflight runner reports these opt-out rows in `skipped_default_job_ids`.
- `worker_identity_cmd`: command that prints JSON identifying the worker host,
  PID, service/cgroup, and accelerator process metadata. Required for
  `cuda_exclusive` jobs and stored in lease metadata after launch, adoption,
  and heartbeat refresh.
- `start_cmd`: command used only when the job is selected for launch.
- `metadata`: copied into the arbiter lease with `sync_job_id`, `job_id`,
  `logical_job_id`, `tenant`, `resource_class`, scheduler host, and worker
  identity status.

Command arrays and command strings may use these placement placeholders:
`{repo_root}`, `{resource}`, `{node}`, `{backend}`, `{tenant}`, `{owner}`, `{id}`,
`{logical_id}`, `{sync_id}`, and `{resource_class}`. This lets one pool job use
the same command template across Atlas, old-donkey, Jack, or future nodes.
Checked-in configs should call repo-local helpers such as
`python3 scripts/check_windows_cuda_resource_admission.py`; deploy new scheduler
machines by cloning/pulling this repo rather than relying on private wrapper
paths in `~/.tsotchke/bin`.

Deploy or refresh a git checkout on a mesh node before enabling scheduler work:

```sh
python3 scripts/mesh_deploy_git_checkout.py \
  --target cosbox \
  --repo-url https://github.com/tsotchke/tensorcore.git \
  --repo-dir '~/src/tensorcore' \
  --ref master \
  --require-clean \
  --json
```

Use the same pattern for workload-owned repos such as GeoRefine or qLLM before
adding a `running` job for them. If a workload can only be started through
`~/.tsotchke/bin`, `/home/tyr/.local/bin`, or `/data/qllm/runs/*.sh`, keep the
job `paused` and adoption-only until that launcher is committed to the owning
repo. `scripts/check_mesh_resource_jobs.py` rejects those private paths in the
checked-in jobs file.

Before unpausing the current external workload rows, run their non-launching
preflights:

```sh
python3 scripts/start_georefine_qwen_cr025.py --target cosbox --preflight-only --json
python3 scripts/start_qllm_olddonkey_precompute_chain.py --target old-donkey --preflight-only --json
```

Those preflights require the target host to have noninteractive Git access to
the private workload repos. If they fail with `Permission denied (publickey)`,
install a deploy key or point `--repo-url` at a reachable internal git mirror
before changing `desired_state` to `running`.

To run checked-in `preflight_cmd` rows from the jobs file:

```sh
python3 scripts/check_mesh_resource_preflights.py --job-id georefine-m2-cosbox --json
python3 scripts/check_mesh_resource_preflights.py --job-id old-donkey-precompute-chain --json
```

The scheduler's default arbiter command is `tsotchke-arbiter` on `PATH` or the
value of `TC_MESH_ARBITER_CMD`. Install that backend from the `computer_mesh`
git checkout and add its `tsotchke/bin` directory to the service environment
instead of hard-coding `~/.tsotchke/bin` in Tensorcore configs.
When the arbiter must be reached over SSH, set `TC_MESH_ARBITER_CMD` to
`scripts/remote_tsotchke_arbiter.py` and provide
`TC_REMOTE_ARBITER_HOST`/`TC_REMOTE_ARBITER_BIN` in the scheduler environment.
The wrapper intentionally has no checked-in host or key defaults.

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
  "probe_cmd": ["ssh", "{node}", "cd ~/src/heldout_eval && python3 scripts/heldout_eval_live.py --job {logical_id} --json"],
  "start_cmd": ["ssh", "{node}", "cd ~/src/heldout_eval && python3 scripts/start_heldout_eval.py --job {logical_id} --resource {resource} --json"],
  "metadata": {
    "project": "semiclassical_qllm"
  }
}
```

For legacy CR025 GeoRefine Qwen rows, completion means the run's
`m2_certificate.json` has `completed=true`, a positive final held-out PPL
(`ppl_compressed_eval`), a positive final stored size (`size_compressed_bytes`),
`size_ratio <= 0.10`, `quality_gate.passed=true`, and passed
base-vs-artifact chat verification either in the certificate or
`m2_chat_verification.json`. The CR070 rank-probe template relaxes the size
ceiling to `0.30` but requires `ppl_delta <= 0.05` and `target_kl <= 0.80`
through its completion command. `scripts/check_georefine_qwen_artifact.py`
enforces these gates.
`scripts/check_georefine_qwen_live.py` is the repo-owned liveness
probe for the same run family; it checks the supervisor status file and falls
back to process matching without relying on a host-local wrapper.
`scripts/start_georefine_qwen_cr025.py` is the matching source-controlled
starter for the CR025 candidate; it clones or fast-forwards the GeoRefine
checkout on cosbox and invokes `experiments.georefine.m2_supervised_run` with
the production CR025 command. Use `--preflight-only --json` to check checkout,
Python imports, calibration text, and evaluation text without launching.
`scripts/start_georefine_qwen_rank_probe.py` is the scheduler-owned CR070
starter used by `configs/georefine_qwen_job.template.json`; the template pins
a run directory so liveness, post-start, identity, and completion checks cannot
accidentally match unrelated GeoRefine runs.

## Running

Dry-run one pass:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd tsotchke-arbiter \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --dry-run \
  --pretty-json
```

Run the control loop and write last-state evidence:

```sh
scripts/mesh_resource_scheduler.py \
  --arbiter-cmd tsotchke-arbiter \
  --jobs-json ~/.tsotchke/state/mesh-resource-jobs.json \
  --state-json ~/.tsotchke/state/mesh-resource-scheduler-state.json \
  --loop \
  --admission-timeout-sec 10 \
  --post-start-timeout-sec 30 \
  --post-start-interval-sec 2 \
  --worker-identity-timeout-sec 10 \
  --unknown-lease-quarantine-age-sec 900 \
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
    "cd ~/src/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090 --allow-process-regex steamwebhelper$ --allowed-process-max-memory-mib 16 --json"
  ]
}
```

```json
{
  "admission_cmd": [
    "ssh",
    "xavier",
    "cd ~/src/tensorcore && python3 scripts/check_operational_evidence.py --hip-toolchain /tmp/hip-toolchain.json --require-hip-spirv-runtime --require-hip-toolchain-clean-head"
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

Scheduler-owned CUDA inventory rows must also define `gpu_reconciliation`.
`scripts/mesh_worker_gpu_reconcile_sweep.py` uses that per-resource policy to
poll worker GPU state, compare it with arbiter leases, and emit one
reconciliation report per resource. Provide arbiter state through
`--arbiter-status-json`, `--arbiter-cmd`, or `TC_MESH_ARBITER_CMD`; disable
reconciliation only with an explicit reason, for example while a Windows worker
snapshot agent is not available.
`mesh_resource_scheduler.py audit` accepts the sweep output directory via
`--worker-reconciliation-dir`. `scripts/mesh_gpu_reconciliation_audit.py`
wraps the sweep plus scheduler audit when a single control-plane command is
preferred. The systemd timer is the v1 worker-enforcement loop: it writes
fresh per-resource reports, writes sweep/audit JSON artifacts, and exits
nonzero when stale worker state would make placement unsafe.
The scheduler loop should be started with `--gpu-reconciliation-audit-json`;
when that flag is present, new idle CUDA placement is blocked unless the audit
artifact has schema `tensorcore.gpu_reconciliation_audit.v1`, is fresh, and is
`ok=true`. The gate is per CUDA resource: live holders are still probed and
heartbeated, stale/unknown lease state is still reconciled, and non-CUDA
resources can still schedule while a CUDA audit artifact is missing, stale, or
failed.

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
