# Contributing Apple hardware evidence

TensorCore needs direct runtime evidence from physical M4 and M5 machines.
You can contribute a one-shot artifact from borrowed or personally owned
hardware without registering a GitHub runner, granting repository access, or
installing a persistent service.

The handoff command builds only the four family-runtime tests, runs the
repository's collector and verifier, and prints the SHA-256 digest of the JSON
artifact to return. It never records the `system_profiler` payload, because
that payload contains the host serial number.

## Requirements

- Physical Apple Silicon reporting the requested M4 or M5 chip class.
- A clean checkout at the exact full commit requested by the maintainer.
- CMake, Python 3, and Xcode command-line tools.
- M5 only: SDK 26.0 or newer selected with `xcode-select`.
- A non-secret resource id and owner id. These are public provenance labels,
  not credentials. For example, `collaborator:alice:macbook` and
  `collaborator:alice`.

Do not include passwords, access tokens, runner registration tokens, serial
numbers, or the raw output of `system_profiler` in an issue or artifact.

## Run the one-shot handoff

The maintainer supplies `EXPECTED_HEAD`. Clone the public repository, check
out that exact commit, and confirm there are no tracked changes:

```sh
git clone https://github.com/tsotchke/tensorcore.git
cd tensorcore
EXPECTED_HEAD=<full-40-character-commit>
git checkout "$EXPECTED_HEAD"
git status --short --untracked-files=no
```

On M4:

```sh
python3 scripts/run_apple_family_evidence_handoff.py \
  --chip M4 \
  --expected-head "$EXPECTED_HEAD" \
  --hardware-resource collaborator:<name>:<asset> \
  --authority-owner collaborator:<name>
```

On M5:

```sh
xcrun --show-sdk-version
python3 scripts/run_apple_family_evidence_handoff.py \
  --chip M5 \
  --expected-head "$EXPECTED_HEAD" \
  --hardware-resource collaborator:<name>:<asset> \
  --authority-owner collaborator:<name>
```

Successful output ends with a single transport record:

```text
APPLE_FAMILY_EVIDENCE_HANDOFF chip=M4 head=<sha> sha256=<digest> path=<json-path>
```

Return the JSON at `path` plus that final line. If the upload surface does not
accept JSON, zip the JSON without modifying it and include the digest in the
submission. The resource and owner values are deliberately embedded in the
evidence; use identifiers you are comfortable making public.

## What the command proves

Before configuration, the handoff rejects:

- a non-Darwin or non-ARM64 host;
- a checkout not matching `EXPECTED_HEAD`;
- tracked worktree changes;
- the wrong physical chip class;
- M5 with an SDK older than 26.0;
- missing resource/owner provenance; and
- an unauthorized claim on the reserved
  `enki:metal_m4_tsotchke_chan` resource.

The collector then rebuilds `test_device`, `test_gemm_bf16`, `test_gemm_i8`,
and `test_tensorops_runtime`, hashes each binary and output, and requires the
family-specific backend markers. M4 must prove Apple9, BF16 simdgroup support,
integer MPS fallback, and no M5 TensorOps selection. M5 must prove Apple10,
SDK 26+, integer MPS fallback, and actual `tensorops_m5` execution.

## Maintainer intake

Recompute the transport digest, then validate the artifact against the same
checkout:

```sh
shasum -a 256 <apple-family-evidence.json>
python3 scripts/check_apple_family_runtime_evidence.py \
  <apple-family-evidence.json> \
  --git-head "$EXPECTED_HEAD" \
  --require-chip M4 \
  --require-clean-head \
  --require-pass
```

Replace `M4` with `M5` as appropriate. Once both artifacts are available, run:

```sh
python3 scripts/check_apple_family_runtime_sprint.py \
  --m4-evidence <m4-evidence.json> \
  --m5-evidence <m5-evidence.json> \
  --require-tracked-clean
```

Only a passing artifact from the exact requested commit closes a physical
gate. Fixtures, edited JSON, screenshots, and evidence from a different commit
remain diagnostic and do not count as release evidence.
