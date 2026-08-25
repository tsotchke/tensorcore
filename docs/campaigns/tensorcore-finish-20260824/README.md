# TensorCore finish campaign — production closure reset

As of: 2026-08-24

Machine-readable authority: `configs/full_capability_campaign.json`

ICC task: `tensorcore-production-release-closure-20260824`

## 0.1.23 release-qualification snapshot

The public 0.1.23 release is scoped to capabilities and hardware classes that
have direct evidence. It does not wait for unrelated frontier research, but it
also does not advertise those unfinished capabilities.

| Gate | Current result |
|---|---|
| Public ABI/export/binding parity | Verified: DiLoCo capability query, default state ABI v2, explicit v1 compatibility |
| Sanitizer/input boundaries | Verified: complete portable ASan/UBSan suite and invalid public-enum cases |
| Verification integrity | Verified: script selftests, adversarial evals, source provenance, fail-closed scheduler results |
| Metal | Verified on representative M2 Ultra; 53 tests passed |
| Portable CPU | Verified on arm64 and Windows x64/MSVC |
| Jetson CUDA | Verified on `sm_72`, CUDA 11.4; FP32/FP16/int8 and training kernels |
| Blackwell CUDA | Verified on `sm_120`, CUDA 13; FP32/FP16/BF16/int8 and training kernels |
| M5 / HIP / Ampere / Windows CUDA | Explicitly unqualified or unsupported for 0.1.23 |
| Public-source privacy | Blocking CI/release gate; no personal paths or private fleet coordinates |

The exact public contract is documented in
`docs/releases/0.1.23.md`. Advanced DiLoCo rank-churn recovery,
topology-scale research, consumer-owned trainers, M5 TensorOps, HIP, and other
frontier tasks remain in this campaign as future work. They do not become
release claims merely because source seams exist.

## Mission

Finish TensorCore as an evidence-backed production substrate. “Finished” does
not mean that the local test suite passes or that a backend compiles. It means
one clean revision exposes one coherent public contract, rejects malformed
inputs without undefined behavior, passes every first-party binding and
packaging surface, carries physical evidence for every claimed hardware class,
and can be consumed by qLLM, QGTL, Eshkol, and the mesh control plane without
private symbols or inferred capabilities.

The campaign extends the existing full-capability program rather than creating
a second roadmap. The 2026-08-24 full audit is the closure reset: historical
tasks remain valuable evidence, but a historical `verified` result does not
override a regression found in the current source.

## Baseline at reset

The starting point is stronger than the open-task count suggests:

| Surface | Reset evidence |
|---|---|
| Metal runtime on M2 Ultra | 52/52 CTest entries pass |
| Portable CPU runtime | 35 tests pass; two opt-in raw AMX tests skip |
| Distributed and transport paths | Loopback Gloo, authenticated transport, mesh, remote tensor, DiLoCo failure and checkpoint tests pass |
| First-party bindings | PyTorch, Rust, Swift, and WebAssembly runtime suites pass |
| Packaging and consumers | Wheel, native artifacts, C/C++/CMake/pkg-config consumers pass |
| Memory safety | ASan completes every runnable portable path |
| Release contract | Fails export, Python FFI, Python constant, and cross-language ABI-v2 parity gates |
| Undefined behavior | UBSan finds invalid-enum loads in DiLoCo and remote-role validation |
| Verification integrity | 59/61 script selftests and 5/6 adversarial evals pass |
| Fleet proof | M5/SDK26, CUDA fleet, Jetson, HIP, Windows, and full topology evidence remain incomplete |

The original uncommitted `TC_CAPABILITY_V1_KNOWN_MASK` correction is now part
of the locally committed ABI closure together with Python, Rust,
documentation, exports, and mixed-version tests.

## Definition of finished

TensorCore is finished only when all of the following hold on the same clean
revision:

1. The `tc-production-release-candidate` task and all of its dependencies are
   `verified` in `configs/full_capability_campaign.json` and ICC.
2. Native C headers, macOS/Windows exports, Python ctypes, Rust FFI, API docs,
   and consumer adapters describe one versioned ABI.
3. ASan and UBSan pass without suppressing malformed-input findings.
4. Every advertised capability is implemented and directly evidenced or is
   reported known-but-unavailable. Compilation is never availability evidence.
5. Every class claimed by the release matrix has a commit-bound physical
   receipt. For 0.1.23 that means M2 Metal, portable arm64 CPU, Windows x64
   CPU, Jetson `sm_72`, and Blackwell `sm_120`; M5, HIP, Ampere, and Windows
   CUDA remain declared exclusions.
6. All first-party bindings, packages, native SDK consumers, script selftests,
   adversarial evals, documentation gates, the ICC production audit, the
   completion oracle, and the architecture model pass.
7. No critical or high correctness, security, ABI, sanitizer, or evidence
   integrity finding is waived. Missing evidence is a failure.

The closure evidence bundle emits one `tensorcore_finish_gate` event for each
hard-gate id in `configs/full_capability_campaign.json`. The task-specific ICC
oracle requires all six names with value `PASS`, plus failure-free evidence and
the verified architecture model. `scripts/check_finish_campaign_evidence.py`
joins the exact-revision receipts and emits those events; it rejects stale or
dirty evidence instead of carrying historical verification forward.

## Execution graph

```text
release-contract closure ─┐
                          ├── verification integrity ─┐
sanitizer/input hardening ┘                          │
                                                     ├── DiLoCo + distributed closure
verified identity/topology ──────────────────────────┘
                                                                  │
                                                                  v
consumer adapters ───── representative hardware proof ── production RC
       │                            │                         │
       └──────── mesh operation + real training ─────────────┘
                                                                  │
                                                                  v
                                                     fleet Pareto acceptance
```

The first three TensorCore lanes may run in parallel. Consumer and fleet work
must not start from a moving or internally inconsistent public contract.

## Workstreams

| Campaign task | Owner | State at reset | Exit condition |
|---|---|---|---|
| `tc-release-contract-closure` | tensorcore | implemented, committed | ABI v2, exports, bindings, docs, and contract gates agree |
| `tc-sanitizer-input-hardening` | tensorcore | implemented, committed | ASan/UBSan clean; malformed public inputs fail safely; diagnostics are correctly scoped |
| `tc-verification-integrity` | tensorcore | implemented, limited | 61/61 script selftests and 6/6 adversarial evals pass; ICC boundary/coverage and attestation trust remain scoped gaps |
| `tc-diloco-correctness` | tensorcore | implemented, limited | honest wire modes, state continuity, rank churn, timeout, replay, and recovery evidence |
| `tc-distributed-topology-scale` | tensorcore | planned | bounded state, reconnect, ordering, churn, and topology-aware collective evidence |
| `qllm-tensorcore-dispatch-adapter` | qLLM | in progress | public-only link, AUTO/forced/fallback tests, physical dispatch evidence |
| `qllm-tensorcore-decentralized-trainer` | qLLM | planned | real qLLM loss reduction and exact checkpoint continuity across heterogeneous ranks |
| `qgtl-tensorcore-decentralized-policy-adapter` | QGTL | planned | QGTL-owned adapter drives real TensorCore execution and recovery |
| `eshkol-tensorcore-compiler-adapter` | Eshkol | in progress | compiler-owned lowering consumes only the stable public ABI |
| `mesh-decentralized-training-control-plane` | computer_mesh | planned | continuous placement, leases, admission, health, recovery, and evidence |
| `tc-backend-hardware-proof` | tensorcore | verified for 0.1.23 scope | exact-revision physical receipts for every release-qualified class; exclusions remain explicit |
| `tc-production-release-candidate` | tensorcore | in progress | local gates pass; sanitized-history publication and exact final hardware receipts remain |
| `fleet-pareto-acceptance` | computer_mesh | planned | reproducible fleet matrix and explicit unresolved limitations |

## Wave 1 — close the audited release blockers

### Public contract closure

Owned files begin with:

- `include/tensorcore/{capabilities,diloco,riemannian_adam}.h`
- `cmake/tensorcore.exports`
- `python/tensorcore/__init__.py`
- `bindings/rust/tensorcore-rs/src/ffi.rs`
- `docs/api_reference.md` and `docs/diloco.md`
- `scripts/check_{public_exports,python_ffi_surface,python_constants}.py`

Required outcomes:

- Python and Rust default to state ABI v2 and retain explicit v1 round trips.
- Python exposes the DiLoCo capability structure/query and capability bits 11
  and 12.
- macOS and Windows export every non-inline public reference symbol.
- The export checker understands header-only inline helpers.
- Constant checks parse aliases, shifts, and multiline macros, so future drift
  cannot hide behind parser limitations.
- Cross-language tests prove v1 compatibility, default v2 emission, v2 restore,
  minimum-size callers, oversized callers, and unknown-version rejection.

### Sanitizer and input-boundary closure

Required outcomes:

- Public enum-bearing C boundaries validate raw integral representations
  before C++ enum loads.
- Invalid optimizer, compression, remote-role, dtype, backend, and gate values
  are sanitizer-clean negative cases.
- Malformed GGUF counts/sizes, DiLoCo state headers/digests, sparse payloads,
  and remote frames exercise overflow and truncation rejection under ASan and
  UBSan.
- Off-manifold reporting cannot attribute another optimizer's work and remains
  correct across explicit resets and multiple optimizer instances.

### Verification-integrity closure

Required outcomes:

- Replace the retired `georefine-m2-cosbox` fixture expectation with the
  current declared job contract.
- Make the development-suggestions eval accept nonempty roadmaps only when
  every item has a concrete task identity.
- Prefer explicit source revision/dirty markers over an ambient parent Git
  checkout.
- Treat a scheduler result without `ok is True` as failure.
- Add or deliberately scope ICC architecture-boundary policy.
- Teach the ICC coverage configuration about `*_selftest.py` and record Metal
  parser blind spots until parser support exists.
- Configure an attestation secret wherever attestations influence a release;
  never store that secret in the repository.

Wave 1 exits only when the mapped findings in
`closure_reset.blocking_findings` are closed and the three task states can be
changed to `verified` with direct receipts.

## Wave 2 — finish the substrate, not just the release plumbing

1. Make every accepted DiLoCo compression mode honest about its physical wire
   representation and measured bytes. Unsupported modes remain rejected.
2. Prove deterministic v1/v2 checkpoint continuity for SGD, Nesterov, Adam,
   top-k residuals, READY/FAILED async state, topology epoch, and membership
   epoch.
3. Define and test bounded timeouts, rank loss, rejoin, duplicate/replayed
   messages, resynchronization, and mixed-version behavior before enabling
   dropout tolerance.
4. Bound collective round state and prove reconnect plus concurrent ordering.
5. Split or durably justify the scheduler and scheduler-selftest monoliths
   identified by ICC; refactoring may not weaken their fail-closed behavior.

## Wave 3 — consumers and real workloads

- qLLM owns model layouts, dispatch heuristics, optimizer/sampler checkpoint
  semantics, and the distributed trainer.
- QGTL owns training-island and recovery policy.
- Eshkol owns compiler lowering and code generation.
- `computer_mesh` owns topology, placement, leases, admission, health, and
  recovery.
- TensorCore owns only the stable ABI, generic runtime, bindings, and substrate
  evidence.

Each adapter must compile against an installed TensorCore SDK, use no private
or invented weak symbols, test forced TensorCore and forced fallback paths, and
emit runtime evidence proving which path executed.

## Wave 4 — representative hardware and release

The long-running hardware program is class-based, not host-count based. This
table is the frontier matrix, not a promise that every class is part of
0.1.23:

| Class | Minimum proof |
|---|---|
| Apple Metal pre-M5 | correctness, backend selection, fallback behavior, performance receipt |
| Apple M5 / SDK26 | Metal 4 compilation plus physical TensorOps execution |
| NVIDIA Blackwell | CUDA correctness, sparse/training paths, rank-3 PyTorch dispatch, performance |
| NVIDIA Ampere | CUDA compatibility, sparse behavior, performance |
| Jetson/aarch64 CUDA | capacity-aware build, correctness, thermal/throughput context |
| HIP/SPIR-V | real build and runtime path; no compile-only pass |
| Portable x86_64/arm64 | correctness, AVX2/NEON selection, install consumers |
| Windows CUDA/CPU | build, Python/SDK consumers, scheduled smoke, driver/toolchain provenance |

The production release candidate is cut only after the machine-readable hard
gates for its declared support matrix pass on a clean revision. The broader
fleet campaign may continue after a scoped TensorCore release only if excluded
capabilities and hardware classes are explicitly marked unqualified or
unsupported rather than described as complete.

## Gate commands

Campaign graph and documentation:

```sh
python3 scripts/check_full_capability_campaign.py
python3 scripts/check_full_capability_campaign_selftest.py
python3 scripts/check_docs_links.py
python3 scripts/check_release_privacy.py
icc capability-roadmap \
  --repo tensorcore --format markdown
```

Local release contract:

```sh
scripts/check_version_consistency.sh
scripts/check_public_headers.sh
scripts/check_public_exports.sh
python3 scripts/check_python_ffi_surface.py
python3 scripts/check_python_constants.py
python3 scripts/check_python_abi_layout.py
```

Native and sanitizer matrix:

```sh
scripts/run_release_sanitizer_evidence.sh
scripts/release_smoke.sh
```

Evidence and final readiness:

```sh
python3 scripts/run_release_verification_evidence.py
python3 scripts/check_finish_campaign_evidence.py \
  --release-evidence "$TC_RELEASE_EVIDENCE" \
  --sanitizer-evidence "$TC_SANITIZER_EVIDENCE" \
  --verification-evidence "$TC_VERIFICATION_EVIDENCE" \
  --windows-evidence "$TC_WINDOWS_EVIDENCE" \
  --xavier-evidence "$TC_XAVIER_EVIDENCE" \
  --blackwell-evidence "$TC_BLACKWELL_EVIDENCE" \
  --output "$TC_FINISH_EVIDENCE" --require-clean-head
icc production-audit \
  --repo tensorcore --target tensorcore-production-release-closure-20260824 \
  --trace-file "$TC_FINISH_EVIDENCE" \
  --format markdown
```

## Operating cadence

- Start each tranche with `icc task-show`, `icc work --mode plan`, store
  freshness, worktree conflict checks, and the current campaign validator.
- A task moves `planned → in_progress → implemented_limited → verified`.
  `verified` requires the task's acceptance list and direct evidence; it is not
  a synonym for “merged.”
- Regress a task when current evidence contradicts historical verification.
- Keep changes reviewable by closure lane. ABI, sanitizer, evidence, and
  hardware work should not land as one opaque patch.
- Re-run the closure audit after every wave and before any release tag.

## Immediate first tranche

1. ABI/export/binding/doc parity is committed with mixed-version tests.
2. Enum-boundary and off-manifold fixes are committed with sanitizer coverage.
3. Verification fixtures, source provenance, and scheduler fail-closed behavior
   are committed and adversarially tested.
4. Release packages and evidence are privacy-scanned, including binary debug
   metadata and uploaded JSON receipts.
5. The remaining closure order is strict: establish sanitized public history,
   produce its final commit, rerun every physical hardware receipt on that
   commit, join the six finish gates, then tag and publish.

That tranche is the shortest path from “excellent local runtime with red
release seams” to a trustworthy substrate on which the longer hardware and
cross-repository work can safely proceed.
