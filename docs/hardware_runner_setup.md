# M5/SDK26 hardware runner setup

Operator runbook for closing the M5 / SDK 26 TensorOps runtime evidence
blocker. This is the setup path for `.github/workflows/hardware-evidence.yml`
when it is dispatched with `require_metal4_tensorops=true`.

## Current blocker snapshot

As of the current M5/SDK26 evidence handoff:

- Latest pushed head: `040a221` (resolve to the full SHA locally before
  passing it as `--expected-head`).
- Local release smoke records
  `checks.metal4_tensorops.runtime_compile_status=skipped_sdk_too_old`
  because Atlas has SDK 15.2.
- Local M5 TensorOps preflight is blocked by `display_gpu` and `sdk26`
  because Atlas is M2 Ultra with Xcode 16.2 / SDK 15.2.
- GitHub runner API currently reports zero repository self-hosted runners, so
  the expected Hardware Evidence preflight status is
  `blocked_no_matching_runner` until an M5 runner with the required labels is
  registered and online.

GitHub-hosted CI and local release smoke do not prove the M5 TensorOps runtime
path. The runtime blocker is closed only when the Hardware Evidence artifact
validates with
`checks.metal4_tensorops.compile_status=compiled`,
`checks.metal4_tensorops.runtime_status=passed`, and a clean git head matching
the expected pushed commit.

## Required access

Install and authenticate `gh` on the operator machine:

```sh
gh auth status
```

Use the repository `tsotchke/tensorcore` in all commands below unless a fork or
staging repository is intentionally being tested:

```sh
REPO=tsotchke/tensorcore
EXPECTED_HEAD="$(git rev-parse 040a221)"
```

`EXPECTED_HEAD` must be the full SHA for the pushed commit that the workflow
will run. If `040a221` is not present locally, fetch first and resolve the
pushed ref:

```sh
git fetch origin master
EXPECTED_HEAD="$(git rev-parse origin/master)"
```

## Create `TC_RUNNER_READ_TOKEN`

The Hardware Evidence preflight calls:

```sh
gh api "repos/${GITHUB_REPOSITORY}/actions/runners"
```

The workflow sets `GH_TOKEN` to `secrets.TC_RUNNER_READ_TOKEN` when present and
falls back to `github.token` otherwise. A 403 from this call means the active
token cannot list repository Actions runners.

Create a fine-grained personal access token for the preflight:

1. In GitHub, open Settings -> Developer settings -> Personal access tokens ->
   Fine-grained tokens -> Generate new token.
2. Name it `TC_RUNNER_READ_TOKEN`.
3. Limit repository access to `tsotchke/tensorcore`.
4. Grant repository permission `Administration: Read-only`. This is the narrow
   permission GitHub requires to list repository self-hosted runners. The token
   owner must have admin access to the repository.
5. Generate the token and store it immediately in the repository secret:

```sh
gh secret set TC_RUNNER_READ_TOKEN --repo "$REPO"
```

Paste the token when prompted. Do not store it in shell history.

Verify the same token can list runners before redispatching:

```sh
read -r -s TC_RUNNER_READ_TOKEN
printf '\n'
GH_TOKEN="$TC_RUNNER_READ_TOKEN" gh api "repos/$REPO/actions/runners" \
  --jq '{total_count, runners: [.runners[] | {name, status, busy, labels: [.labels[].name]}]}'
unset TC_RUNNER_READ_TOKEN
```

If the token is only available on the clipboard, skip the local `GH_TOKEN=...`
probe and verify by redispatching Hardware Evidence, then fetching the
preflight artifact. A fixed token changes the preflight from
`runner_api_unavailable` to either `blocked_no_matching_runner`,
`matching_runner_offline`, or `matching_runner_online`.

## Register the M5 self-hosted runner

Use a real M5-or-newer Apple Silicon host with macOS/arm64, Xcode installed,
and SDK 26.0 or newer selected:

```sh
xcodebuild -version
xcrun --show-sdk-version
system_profiler SPDisplaysDataType
```

When `require_metal4_tensorops=true`, the workflow routes the M5 TensorOps job
with:

```yaml
runs-on: [self-hosted, macOS, ARM64, m5, sdk26, metal4-tensorops]
```

The first three labels are default labels applied by GitHub when the runner is
registered on macOS ARM64. Do not configure the runner with
`--no-default-labels`. The final three labels are Tensorcore-specific required
labels for the M5/SDK26 runtime gate. They keep a generic Apple Silicon runner
from accidentally picking up the Metal 4 TensorOps promotion job.

Register from the repository UI:

1. Open `https://github.com/tsotchke/tensorcore/settings/actions/runners`.
2. Click New self-hosted runner.
3. Choose macOS and ARM64.
4. On the M5 host, run the generated download and checksum commands.
5. Run `config.sh` with a stable name, the repository URL, the generated
   one-hour registration token, the default work directory, and optional
   operator labels:

```sh
mkdir -p "$HOME/actions-runner/tensorcore-m5"
cd "$HOME/actions-runner/tensorcore-m5"

./config.sh \
  --url "https://github.com/tsotchke/tensorcore" \
  --token "$REGISTRATION_TOKEN" \
  --name "tensorcore-m5-sdk26-$(scutil --get LocalHostName)" \
  --work _work \
  --labels m5,sdk26,metal4-tensorops
```

Start it persistently on macOS:

```sh
./svc.sh install
./svc.sh start
./svc.sh status
```

For foreground debugging instead of a service:

```sh
./run.sh
```

After registration, verify GitHub sees an online runner with the required
labels:

```sh
gh api "repos/$REPO/actions/runners" \
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

The relevant row must include `self-hosted`, `macOS`, `ARM64`, `m5`, `sdk26`,
and `metal4-tensorops`, and `status` must be `online`.

## Preflight the M5 host locally

From a clean checkout on the M5 host:

```sh
git fetch origin master
git checkout "$EXPECTED_HEAD"
git status --short
python3 scripts/m5_tensorops_runner_preflight.py --json
python3 scripts/check_m5_tensorops_runner_preflight.py \
  build/m5_tensorops_runner_preflight.json \
  --require-candidate \
  --git-head "$EXPECTED_HEAD" \
  --require-clean-head
```

`status=candidate` is acceptable before the runtime test binary exists. It
means the host, Xcode, SDK 26, and display GPU checks passed, but
`test_tensorops_runtime` was not built yet.

To prove the host is fully ready outside GitHub Actions:

```sh
BUILD_DIR=build-m5-tensorops scripts/run_m5_tensorops_runtime_smoke.sh
python3 scripts/m5_tensorops_runner_preflight.py \
  --build-dir build-m5-tensorops \
  --output build-m5-tensorops/m5_tensorops_runner_preflight.json \
  --require-ready
python3 scripts/check_m5_tensorops_runner_preflight.py \
  build-m5-tensorops/m5_tensorops_runner_preflight.json \
  --require-ready \
  --git-head "$EXPECTED_HEAD" \
  --require-clean-head
```

`status=ready` requires `test_tensorops_runtime` to emit
`tensorops_runtime_status=passed`.

## Dispatch Hardware Evidence

Dispatch the required M5 run from a checkout whose dispatch ref resolves to
`EXPECTED_HEAD`:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --dispatch \
  --ref master \
  --expected-head "$EXPECTED_HEAD"
```

The helper dispatches `.github/workflows/hardware-evidence.yml` with
`require_metal4_tensorops=true`. It exits after dispatch. Use the fetch modes
below after the workflow has produced artifacts.

## Fetch or cancel a queued run

If the self-hosted job is queued or the runner state is unknown, fetch the
GitHub-hosted preflight artifact:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --runner-preflight \
  --latest-preflight-for-head \
  --expected-head "$EXPECTED_HEAD"
```

The default output path is:

```text
build/m5-tensorops-hardware-evidence/hardware_runner_preflight.json
```

When no online matching runner exists and the hardware job would stay queued,
fetch the same preflight and cancel the workflow run in one step:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --runner-preflight \
  --latest-preflight-for-head \
  --cancel-if-no-online-runner \
  --expected-head "$EXPECTED_HEAD"
```

To require that the preflight proves an online runner before proceeding:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --runner-preflight \
  --latest-preflight-for-head \
  --require-online-runner \
  --expected-head "$EXPECTED_HEAD"
```

To inspect a known workflow run instead of the latest run for the head:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --runner-preflight \
  --run-id "$RUN_ID" \
  --expected-head "$EXPECTED_HEAD"
```

## Fetch accepted runtime evidence

After the self-hosted job completes successfully:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --latest-for-head \
  --expected-head "$EXPECTED_HEAD"
```

The helper downloads `tensorcore-hardware-evidence` and validates it through
`scripts/check_release_evidence.py` with `--require-gpu`,
`--require-metal4-tensorops`, `--require-clean-head`, and the expected head.
Success prints:

```text
M5 TensorOps hardware evidence accepted: head=<full-sha> compile=compiled runtime=passed artifact=<path>
```

The accepted evidence lands at:

```text
build/m5-tensorops-hardware-evidence/release_smoke_runtime_evidence.json
```

For a known run:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --repo "$REPO" \
  --run-id "$RUN_ID" \
  --expected-head "$EXPECTED_HEAD"
```

## Interpret `hardware_runner_preflight`

The repository-runner preflight is produced by the GitHub-hosted job before the
self-hosted job is scheduled. It validates repository visibility and queue
routing only. It does not prove M5 runtime support.

| Status | Meaning | Operator action |
|---|---|---|
| `runner_api_unavailable` | The workflow could not list repository Actions runners. `runner_api_rc` is non-zero and `runner_api_error` usually contains the GitHub API failure. | Add or fix `TC_RUNNER_READ_TOKEN`, then redispatch. HTTP 403 maps here. |
| `blocked_no_matching_runner` | The runner API was available, but no registered runner had all required labels. | Register an M5/SDK26 macOS ARM64 runner. With the current zero-runner repo state, this is the expected blocker. |
| `matching_runner_offline` | A runner has the required labels, but none are online. | Start the runner service on the M5 host, then redispatch. |
| `matching_runner_online` | At least one required-label runner is online. | Wait for `apple-gpu-release-smoke` to run and upload runtime evidence. |

Important fields:

- `required_labels`: labels used by the workflow and preflight. With
  `require_metal4_tensorops=true`, this must include `self-hosted`, `macOS`,
  `ARM64`, `m5`, `sdk26`, and `metal4-tensorops`.
- `registered_runner_count`: total runners visible to the preflight token.
- `matching_runner_count`: registered runners containing all required labels.
- `online_matching_runner_count`: matching runners with `status=online`.
- `matching_runners[]`: name, status, busy flag, and labels for matched rows.
- `diagnostics[*].diagnostic_class`: one of `token_unavailable`,
  `runner_absent`, `runner_offline`, or `runner_online`.
- `diagnostics[*].recommended_action`: the handoff text for the next operator
  action.

Validate a downloaded preflight:

```sh
python3 scripts/check_hardware_runner_preflight.py \
  build/m5-tensorops-hardware-evidence/hardware_runner_preflight.json \
  --expected-head "$EXPECTED_HEAD" \
  --require-metal4-tensorops
```

Add `--require-runner-api` when the secret is expected to be fixed, and
`--require-online-runner` when the runner is expected to be online.

## Interpret `m5_tensorops_runner_preflight`

The local M5 preflight runs on the self-hosted host. It checks machine
readiness and, when the binary exists, the runtime probe.

| Status | Meaning | Evidence value |
|---|---|---|
| `blocked` | A required host/toolchain check failed, or the runtime probe failed. | Diagnostic only. Does not close the runtime blocker. |
| `candidate` | macOS/arm64, Xcode, SDK 26+, and M5 display checks passed, but the runtime binary was missing or skipped. | Good setup signal before the workflow build. Does not close the runtime blocker. |
| `ready` | All host checks passed and `test_tensorops_runtime` reported `tensorops_runtime_status=passed`. | Strong local readiness signal. The release gate is still the fetched Hardware Evidence artifact. |

Required checks are `host_platform`, `xcode`, `sdk26`, `display_gpu`, and
`tensorops_runtime_probe`. Use `summary.blocked_checks` to find the failing
check. Diagnostic classes separate environment gaps from code/runtime failures:

- `environment_unavailable`: wrong platform, SDK below 26.0, no M5 display GPU,
  or an environment skip such as `skipped_no_m5`.
- `artifact_missing`: `test_tensorops_runtime` is not built yet.
- `source_failed`: the runtime probe ran but failed or did not emit
  `tensorops_runtime_status=passed`.

Validate local readiness evidence:

```sh
python3 scripts/check_m5_tensorops_runner_preflight.py \
  build/m5_tensorops_runner_preflight.json \
  --require-candidate \
  --git-head "$EXPECTED_HEAD" \
  --require-clean-head
```

Use `--require-ready` only after `scripts/run_m5_tensorops_runtime_smoke.sh`
has built and passed the runtime probe on the M5 host.

## Closeout checklist

1. `TC_RUNNER_READ_TOKEN` exists as a repository secret and the next preflight
   is not `runner_api_unavailable`.
2. The repository has an online runner with labels `self-hosted`, `macOS`,
   `ARM64`, `m5`, `sdk26`, and `metal4-tensorops`.
3. The runner host local preflight is at least `candidate`; `ready` is expected
   after the runtime smoke has built the test binary.
4. The Hardware Evidence workflow is dispatched for the full `EXPECTED_HEAD`
   corresponding to pushed head `040a221` or its successor.
5. `scripts/fetch_m5_tensorops_runtime_evidence.py --latest-for-head` accepts
   the artifact with `compile=compiled`, `runtime=passed`, and a clean matching
   head.
