# Release process

How a tensorcore release goes from local commit to published wheel and
cross-platform native SDKs. This page is the runbook.

## Release artifacts

1. **`tensorcore_apple-X.Y.Z-py3-none-macosx_15_0_arm64.whl`** — pip-installable
   Python wheel; ships the native `libtensorcore.dylib` + `tensorcore.metallib`
   inside the package so end users `pip install` and it works.
2. **`tensorcore-native-sdk-X.Y.Z-macos-ARCH.tar.gz`** — headers, static and
   versioned shared libraries, metallib, CMake config, and pkg-config file.
3. **`tensorcore-native-sdk-X.Y.Z-linux-ARCH.tar.gz`** — the same C ABI and
   package metadata with a versioned ELF shared object; no metallib.
4. **`tensorcore-native-sdk-X.Y.Z-windows-ARCH.zip`** — headers, DLL, import
   library, static library, and CMake/pkg-config metadata.

All artifacts plus `SHA256SUMS` are published as assets on the GitHub release.
The Python distribution is intentionally Apple-only in v0.1; Linux and Windows
are public native SDK targets.

## The version triple

The same `X.Y.Z` lives in three places:

- `pyproject.toml::version`
- `CMakeLists.txt::project(tensorcore VERSION X.Y.Z ...)`
- `include/tensorcore/tensorcore.h::TENSORCORE_VERSION_{MAJOR,MINOR,PATCH}`

`scripts/check_version_consistency.sh` enforces agreement. It's the first
step of every CI run and the first step of `release.yml`.

## Steps to ship a release

### 1. Land all the commits

`master` should be green. CI on the latest commit must be passing on
both `macos-14` and `macos-15` runners. Check:

```sh
gh run list -R tsotchke/tensorcore --workflow CI --limit 3
```

### 2. Bump the version in all three files

```sh
# Pick the new version
NEW=0.1.23

# pyproject.toml
sed -i '' "s/^version = .*/version = \"$NEW\"/" pyproject.toml

# CMakeLists.txt
sed -i '' "s/^    VERSION [0-9.]*/    VERSION $NEW/" CMakeLists.txt

# include/tensorcore/tensorcore.h
# (manually edit TENSORCORE_VERSION_{MAJOR,MINOR,PATCH})

# Verify
bash scripts/check_version_consistency.sh   # → "tensorcore version OK: 0.1.23"
```

(Or just have a script: see `scripts/check_version_consistency.sh` for
the source of truth.)

### 3. Update CHANGELOG.md

Add a new section under `## Unreleased` that becomes `## vX.Y.Z`:

```markdown
## v0.1.23 — <one-liner summary>

- Substantive change A — what it does, what it fixes / improves.
- Substantive change B.
- Reference any PR numbers or related ROADMAP items.
```

Move anything that lands in this release out of "Unreleased". The
"Unreleased" section becomes empty (or carries items not in the release).

### 4. Commit the version bump

```sh
git add pyproject.toml CMakeLists.txt include/tensorcore/tensorcore.h \
        CHANGELOG.md
git commit -m "Bump version to $NEW"
git push origin master
```

CI runs on the push. Wait for it to go green.

### 5. Tag and push the tag

```sh
git tag -a v$NEW -m "tensorcore v$NEW"
git push origin v$NEW
```

The `v*` tag push triggers `.github/workflows/release.yml`. The workflow runs
three independent platform jobs: the full macOS release smoke and wheel build,
the portable Linux build/test/install gate, and the Windows MSVC portable CPU
gate. Each native SDK is unpacked, checked for the expected ABI metadata, and
used to build and execute fresh C and C++ consumers. A final job downloads all
artifacts, creates and verifies `SHA256SUMS`, then publishes the complete set.

You can also dispatch the workflow manually on `master` via
`workflow_dispatch` to get a snapshot release without a tag.

### 6. Verify the release published

```sh
gh release view v$NEW -R tsotchke/tensorcore
```

Should show:
- `published: <timestamp>`
- An asset named `tensorcore_apple-X.Y.Z-py3-none-macosx_15_0_arm64.whl`
- Native SDK assets for macOS, Linux, and Windows
- `SHA256SUMS`

The release URL is
`https://github.com/tsotchke/tensorcore/releases/tag/v$NEW`.

### 7. Run hardware-evidence (optional)

For releases that touch the M5 / TensorOps path, manually dispatch the
`hardware-evidence.yml` workflow with `require_metal4_tensorops=true`.
Use [hardware_runner_setup.md](hardware_runner_setup.md) for the operator
runbook that covers `TC_RUNNER_READ_TOKEN`, M5 self-hosted runner
registration, preflight interpretation, dispatch, fetch, and queued-run
cancellation.
The self-hosted runner exercises the deepest hardware path and emits a
`tensorcore-hardware-evidence` artifact (JSON evidence of chip / family
/ TensorOps availability / backend chosen per representative call).
The workflow first emits a GitHub-hosted
`tensorcore-hardware-runner-preflight` artifact that records the required
self-hosted labels and whether a matching runner was visible. With
`require_metal4_tensorops=true`, the hardware job requires
`[self-hosted, macOS, ARM64, m5, sdk26, metal4-tensorops]`.
The artifact's `diagnostics[*].recommended_action` field is the operational
handoff: it distinguishes an unavailable runner-list token from an absent or
offline self-hosted M5 runner.
Validate that artifact after download with:

```sh
python3 scripts/check_hardware_runner_preflight.py \
  build/hardware_runner_preflight.json \
  --expected-head "$(git rev-parse HEAD)" \
  --require-metal4-tensorops
```

Before registering or debugging the runner, run:

```sh
python3 scripts/m5_tensorops_runner_preflight.py --json
python3 scripts/check_m5_tensorops_runner_preflight.py \
  build/m5_tensorops_runner_preflight.json
```

On a built checkout, `--require-ready` additionally requires the local
`test_tensorops_runtime` binary to emit `tensorops_runtime_status=passed`.
Blocked preflight artifacts include `diagnostics[*].diagnostic_class` metadata so
ICC/readiness can distinguish environment problems from source/runtime probe
failures.

After the manual workflow completes, fetch and validate the M5 artifact for the
expected clean head:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py --latest-for-head \
  --expected-head "$(git rev-parse HEAD)"
```

If the self-hosted hardware job stays queued, fetch the runner-preflight
artifact and cancel the queued run when no matching runner is online:

```sh
python3 scripts/fetch_m5_tensorops_runtime_evidence.py \
  --runner-preflight \
  --latest-preflight-for-head \
  --cancel-if-no-online-runner \
  --expected-head "$(git rev-parse HEAD)"
```

## Running the release pipeline locally

To sanity-check before tagging:

```sh
# 1. Build + test
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# 2. Install to a temp prefix
cmake --install build --prefix /tmp/tensorcore-install

# 3. Full smoke (REQUIRE_GPU=1 if on M-series hardware)
REQUIRE_GPU=1 scripts/release_smoke.sh

# 4. Build the wheel
python3 -m venv /tmp/tc-wheel-venv
source /tmp/tc-wheel-venv/bin/activate
pip install --upgrade pip setuptools wheel
mkdir -p /tmp/tc-wheel-out
TENSORCORE_NATIVE_DIR=/tmp/tensorcore-install/lib \
    python -m pip wheel . --no-build-isolation -w /tmp/tc-wheel-out

# 5. Verify the wheel
python -m pip install /tmp/tc-wheel-out/tensorcore_apple-*.whl \
    --force-reinstall --no-deps
TENSORCORE_LIB= TC_METALLIB= python -c \
    "import tensorcore as tc; print(tc.version())"

# 6. Build and validate the host native SDK archive
scripts/create_native_sdk_archive.sh /tmp/tensorcore-install
scripts/check_native_sdk_archive.sh \
    /tmp/tensorcore-native-sdk-X.Y.Z-macos-arm64.tar.gz
```

On Linux, the same commands produce a `linux-ARCH.tar.gz`. On Windows:

```powershell
$Archive = ./scripts/create_native_sdk_archive.ps1 `
  -Prefix C:\temp\tensorcore-install -OutDir C:\temp
./scripts/check_native_sdk_archive.ps1 -Archive $Archive
```

Host-local success proves one platform. The release is publishable only when
the macOS, Linux, and Windows jobs all pass.

## Compatibility commitments per version bump

| Bump | What's allowed |
|---|---|
| Patch (`0.1.X` → `0.1.X+1`) | New entry-point functions (appended). New enum values (appended). New optional descriptor fields (appended). Bug fixes. Perf improvements. |
| Minor (`0.X.Y` → `0.X+1.0`) | Same as patch + new entry-point families (e.g. a new `tc_*.h` header). Behaviorally-different default kernel choices. |
| Major (`X.Y.Z` → `X+1.0.0`) | Anything goes. ABI breaks allowed. Renames allowed. Removed deprecated entry points. |

We are pre-1.0; everything is technically a patch / minor right now. The
v0.1 series has not removed or renamed any public symbol.

## What can go wrong

| Failure mode | Where it surfaces | Fix |
|---|---|---|
| Version triple disagrees | `check_version_consistency.sh` first step of CI | Update all three files |
| Tests fail on macos-14/15 | `ci.yml` build-and-test step | Investigate; reruns rarely help |
| Wheel build fails to find native artifacts | `release.yml` "Build wheel" step | Check `TENSORCORE_NATIVE_DIR` setting |
| Wheel reinstall fails | `release.yml` "Verify wheel" step | The dylib + metallib weren't vendored — re-check `pyproject.toml` `[tool.setuptools.package-data]` |
| Linux SDK has no SONAME | native SDK checker | Keep `VERSION` and `SOVERSION` on `tensorcore_shared` |
| Windows SDK misses an import/static library | PowerShell native SDK checker | Verify the shared and static install targets and MSVC configuration |
| GitHub release upload fails | `release.yml` final step | Usually the tag already has a release — `gh release delete v$NEW && retry` |
| Self-hosted runner offline | `hardware-evidence.yml` preflight artifact reports no online runner with the labels required by that dispatch, then the hardware job queues | Bring the runner online; cancel/re-run the queued hardware job |
| Runner API unavailable | `hardware-evidence.yml` preflight artifact reports `runner_api_unavailable` | Add a repo secret named `TC_RUNNER_READ_TOKEN` with runner-list permission, or use the artifact as a visibility-only diagnostic |

## CI workflows that gate

Three workflows. Two run automatically on every push to master; one is
manual.

- **`ci.yml`** — gate on every push / PR, including macOS/Linux/Windows SDK consumers
- **`release.yml`** — runs only on `v*` tag push (and via
  `workflow_dispatch`)
- **`hardware-evidence.yml`** — manual; runs on the self-hosted M-series
  runner

See [ci_and_scripts.md](ci_and_scripts.md) for what each runs in detail.

## Yanking a bad release

If a release is shipped with a bug bad enough to require yanking:

```sh
# 1. Delete the GitHub release (keeps the tag)
gh release delete v$BROKEN -R tsotchke/tensorcore --yes

# 2. (Optional) Delete the tag too if you want to free up the version
git push origin --delete v$BROKEN
git tag -d v$BROKEN

# 3. Fix forward: bump to the next patch, push, tag
```

PyPI publication isn't currently in the pipeline; if it were, you'd also
`pip yank tensorcore-apple==$BROKEN`. The wheel lives only on GitHub
releases today.

## Future automation

- **Push to PyPI** on release tag — currently the wheel is GitHub-only
  because PyPI doesn't accept binary macOS arm64 wheels uniformly across
  CIBuildWheel versions; v0.2 will set up the pipeline.
- **macOS notarization** — the wheel ships unsigned. For an end-user
  `pip install` over the network, the OS may flag it. Codesigning +
  notarization on release is on the v0.2 list.
- **Multi-version macOS deployment target** — currently the wheel
  targets `macosx_15_0_arm64`. Backward-compat to macOS 13 is a v0.2
  task (requires re-tooling the SDK gates).
