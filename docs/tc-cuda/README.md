# tc-cuda v1 CUDA subset

This directory holds the machine-readable authority for the tc-cuda v1 CUDA
subset. The single source of truth is:

**`docs/tc-cuda/subset.v1.json`**

The tc-cuda compiler's accept-list and all subset documentation are generated
from that one artefact. Doc and code cannot drift because they are one
artefact. This README is a summary only and never overrides the JSON.

## Scope

tc-cuda accepts CUDA kernel source and CUDA host API calls and executes them on
tensorcore's backends. Its default behaviour on a construct it cannot handle is
to **refuse to build** — never to emit a binary whose kernel silently does
nothing. See the design document for the full requirements:

- `docs/design/tc-cuda-universal-substrate-20260827.md`

## Supported constructs — 55

The subset is the union of what `lib/cuda/*.cu` and the Kimi engine's `*.cu`
actually use, surveyed not guessed. Category counts:

| Category | Count |
| --- | ---: |
| A. Function and declaration forms | 4 |
| B. Launch geometry and indexing | 3 |
| C. Memory | 4 |
| D. Synchronisation and warp collectives | 4 |
| E. Atomics | 1 |
| F. Half precision | 8 |
| G. fp32 libdevice math | 13 |
| H. Rounding-explicit and fast intrinsics | 4 |
| I. Integer and bit arithmetic | 2 |
| J. Host runtime API families | 12 |
| **Total** | **55** |

Each supported entry in the JSON carries a stable `id`, the CUDA `name`,
`status: "supported"`, and concise `semantics`.

## Unsupported constructs

Every construct in the design's "explicitly out of scope in v1" list is encoded
in the JSON with `status: "unsupported"` and a stable `id`. Each is a **build
error that names the construct** in CUDA's own vocabulary with file:line. This
includes the performance hints `__ldg` and `__launch_bounds__`, which are
deliberately rejected rather than silently ignored.

## Fail-closed policy

- **Single authority.** `subset.v1.json` is the only source of truth; the
  compiler accept-list is generated from it.
- **Fail closed.** Any construct not listed as `supported` is a build error
  naming the construct. There is no best-effort mode and no silent fallback.
- **Uniqueness.** Every `id` is unique across supported and unsupported.
  Duplicate ids, unknown categories, or any status other than
  `supported`/`unsupported` are validation failures.
- **No silent acceptance.** A construct that is present in source but absent
  from the supported list is never accepted; it is named and rejected.

## Source checker

Run `python3 scripts/tc_cuda.py check SOURCE.cu --manifest-output manifest.json`.
The emitted `checked` status proves source validation only. CUDA-to-Metal
lowering and execution are not claimed until their separate conformance gates
pass.
