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
      "metadata": {
        "project": "semiclassical_qllm"
      }
    },
    {
      "id": "georefine-m2-cosbox",
      "sync_id": "georefine-m2-cosbox",
      "resource": "cosbox:cuda3090",
      "owner": "georefine:m2",
      "priority": 10,
      "desired_state": "paused",
      "ttl_sec": 900,
      "probe_cmd": [
        "ssh",
        "cosbox",
        "pgrep -af 'experiments.georefine|m2_compress|m2_live_agent' >/dev/null"
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
- `owner`: human-readable lease owner.
- `priority`: tie-breaker used only when the resource is idle.
- `desired_state`: `running` makes the job launchable; `paused` prevents new
  launches but still lets a live job be adopted and protected.
- `probe_cmd`: command that exits 0 when the job is live.
- `start_cmd`: command used only when the job is selected for launch.
- `metadata`: copied into the arbiter lease with `sync_job_id` and `job_id`.

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
  --interval-sec 30
```

Agents should update the jobs file atomically, then let the scheduler loop make
the launch decision. They should not kill another agent's process to free CUDA.

## Failure Modes

- `resource_busy_unknown_lease`: another holder is present and not in the jobs
  file. Add a probeable job entry or release the lease manually after verifying
  the process is dead.
- `multiple live holders detected`: two known jobs are live for the same
  resource. The scheduler refuses to pick a victim; stop one job explicitly.
- `live_holder_blocked_by_known_lease_unknown_liveness`: a known lease has no
  reliable probe result. Fix the probe before scheduling more work.
- `claimed_and_launched` with `ok=false`: the start command failed and the
  claimed lease was released.
