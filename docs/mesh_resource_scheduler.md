# Mesh Resource Scheduler

`scripts/mesh_resource_scheduler.py` is the mesh-wide admission controller for
shared hardware such as `cosbox:cuda3090`. It sits above individual projects:
qLLM, GeoRefine, tensorcore demos, and future agents should submit desired
work to this scheduler instead of starting CUDA jobs directly.

The scheduler uses the Tsotchke arbiter as the lease backend. Its policy is
conservative:

- Live work is never killed for priority alone.
- Known stale leases are released only when that job's probe says dead.
- Unknown leases block the resource.
- A live known job without a lease is adopted by claiming a lease for it.
- New work is launched only if its optional `admission_cmd` exits 0.
- CUDA-exclusive work must also pass a post-start probe and report worker
  identity before the launch is considered healthy.
- A new job is launched only after the scheduler has claimed the resource.
- If a launch fails, the scheduler releases the lease it just claimed.

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
        "cd /home/tyr/work/qllm-phase0 && systemd-run --user --unit=qllm-phase1 --collect --property=Restart=always --property=RestartSec=60 /bin/bash scripts/launch_phase1_tf32_cosbox.sh"
      ],
      "admission_cmd": [
        "ssh",
        "cosbox",
        "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090"
      ],
      "post_start_probe_cmd": [
        "ssh",
        "cosbox",
        "systemctl --user is-active --quiet qllm-phase1.service && nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q qllm"
      ],
      "worker_identity_cmd": [
        "ssh",
        "cosbox",
        "cd /home/tyr/work/tensorcore && python3 scripts/mesh_cuda_worker_identity.py --unit qllm-phase1.service --process-substring qllm --require-cuda-process"
      ],
      "metadata": {
        "project": "semiclassical_qllm"
      }
    },
    {
      "id": "georefine-m2-cosbox",
      "sync_id": "georefine-m2-cosbox",
      "resource": "cosbox:cuda3090",
      "resource_class": "generic",
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
        "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head"
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
- `sync_id`: stable lease identity. Keep this stable across PID churn.
- `resource`: arbiter resource name, for example `cosbox:cuda3090`.
- `resource_class`: `generic` or `cuda_exclusive`. If omitted, resources
  containing `:cuda` or starting with `cuda` are treated as `cuda_exclusive`.
- `owner`: human-readable lease owner.
- `priority`: tie-breaker used only when the resource is idle.
- `desired_state`: `running` makes the job launchable; `paused` prevents new
  launches but still lets a live job be adopted and protected.
- `probe_cmd`: optional command that exits 0 when the job is live. If omitted
  or inconclusive, liveness is unknown and known leases block scheduling.
- `completion_cmd`: optional command that exits 0 only when the job's output is
  complete and should not be relaunched. A nonzero exit code means
  "definitively incomplete"; a timeout is treated as unknown and blocks
  relaunch of that job for the current scheduler pass.
- `admission_cmd`: optional command that exits 0 only when the target host is
  currently eligible for this job. Use this for evidence gates such as CUDA,
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
  `resource_class`, scheduler host, and worker identity status.

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
    "cd /home/tyr/work/tensorcore && python3 scripts/check_operational_evidence.py --cuda /tmp/tensorcore-cuda-smoke.json --require-cuda --require-cuda-clean-head && python3 scripts/check_cuda_resource_admission.py --resource cosbox:cuda3090"
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

## CUDA Worker Identity

Use `scripts/mesh_cuda_worker_identity.py` as the `worker_identity_cmd` on
Linux NVIDIA hosts. It emits `tensorcore.mesh_cuda_worker_identity.v1` JSON
with hostname, probe PID, cgroup, optional systemd unit metadata, matching
`nvidia-smi` compute processes, and `cuda_pids`:

```sh
python3 scripts/mesh_cuda_worker_identity.py \
  --unit qllm-phase1.service \
  --process-substring qllm \
  --require-cuda-process
```

The command exits nonzero if the requested unit is inactive, `nvidia-smi`
fails, or `--require-cuda-process` is set and no matching CUDA process exists.

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
