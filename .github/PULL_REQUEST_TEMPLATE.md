# What this changes

<!-- One paragraph: what the PR does and why. -->

## Checklist

- [ ] Correctness test added for any new kernel (`tests/test_<group>.c`).
- [ ] Existing tests still pass: `ctest --test-dir build --output-on-failure`.
- [ ] Numerical guarantees in [CONTRIBUTING.md](../CONTRIBUTING.md) are preserved
      (fp32 bit-exact vs Accelerate, fp16 rms_scaled ≤ 5e-3 vs fp64, etc.).
- [ ] Bench numbers reported if the PR touches a perf-critical path,
      with chip and shape called out.
- [ ] No new public symbols escape `cmake/tensorcore.exports`
      (`scripts/check_public_exports.sh` passes).
- [ ] Public headers compile in C and C++ standalone
      (`scripts/check_public_headers.sh` passes).
- [ ] If touching the Python binding: `scripts/check_python_ffi_surface.py`,
      `scripts/check_python_abi_layout.py`, `scripts/check_python_constants.py`
      all pass.
- [ ] [CHANGELOG.md](../CHANGELOG.md) "Unreleased" section updated.
- [ ] [ROADMAP.md](../ROADMAP.md) updated if this closes a v0.x item.
- [ ] Docs updated where relevant (see [docs/](../docs/) and
      [docs/README.md](../docs/README.md)).

## Bench (if applicable)

```
M2 Ultra, current:   <bench output line>
M2 Ultra, before:    <baseline output line>
Δ:                    <X% improvement / regression>
```

## Notes for the reviewer

<!-- Anything non-obvious: a tile-layout trade-off, a deferred follow-up,
     a flag-gated experimental path, etc. -->
