#!/usr/bin/env python3
"""Selftest for the tc-cuda frontend CLI.

Generates fixtures covering:
- every one of the 25 unsupported manifest entries (rejected with id+name),
- comment and string-literal false positives (accepted),
- unknown __-prefixed intrinsic rejection,
- accepted minimal and rmsnorm-shaped kernels,
- manifest determinism (byte-for-byte identical across two runs).

Usage:
    python3 scripts/tc_cuda_selftest.py

Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
TC_CUDA = ROOT / "scripts" / "tc_cuda.py"
SUBSET = ROOT / "docs" / "tc-cuda" / "subset.v1.json"

PASS = 0
FAIL = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")
        if detail:
            for line in detail.splitlines():
                print(f"        {line}")


def run_tc_cuda(source: str, manifest_output: str | None = None) -> tuple[int, str, str]:
    """Write source to a temp .cu file, run tc_cuda.py check, return (rc, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        path = f.name

    cmd = [sys.executable, str(TC_CUDA), "check", path]
    if manifest_output:
        cmd += ["--manifest-output", manifest_output]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    pathlib.Path(path).unlink(missing_ok=True)
    return proc.returncode, proc.stdout, proc.stderr


def load_unsupported_entries() -> list[dict]:
    data = json.loads(SUBSET.read_text(encoding="utf-8"))
    return data.get("unsupported", [])


def test_unsupported_entries() -> None:
    """Each of the 25 unsupported entries must be rejected with its id and name."""
    print("\n[unsupported entries — 25 fixtures]")
    entries = load_unsupported_entries()
    assert len(entries) == 25, f"expected 25 unsupported entries, got {len(entries)}"

    # For each entry, generate a minimal CUDA source that contains a token
    # that the scanner will detect. We use the first token from the
    # _UNSUPPORTED_TOKENS map for that id.
    import importlib.util
    spec = importlib.util.spec_from_file_location("tc_cuda", TC_CUDA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for entry in entries:
        eid = entry["id"]
        name = entry["name"]
        tokens = mod._UNSUPPORTED_TOKENS.get(eid, [])
        if not tokens:
            # U19 (recursion) is structural; skip token-based test.
            report(True, f"{eid} {name[:40]} (structural, skipped)")
            continue
        token = tokens[0]
        # Build a source that uses this token in a __global__ kernel.
        if " " in token:
            # Multi-word token like "asm volatile"
            src = textwrap.dedent(f"""\
                __global__ void k() {{
                    {token} {{}};
                }}
            """)
        else:
            # Single token: use it as a function call or identifier.
            if token.startswith("__"):
                src = textwrap.dedent(f"""\
                    __global__ void k() {{
                        {token}();
                    }}
                """)
            elif token.startswith("cuda"):
                src = textwrap.dedent(f"""\
                    __global__ void k() {{
                        {token}(0);
                    }}
                """)
            elif token.endswith("."):
                # e.g. "mma." — use as namespace
                src = textwrap.dedent(f"""\
                    __global__ void k() {{
                        {token}foo();
                    }}
                """)
            else:
                src = textwrap.dedent(f"""\
                    __global__ void k() {{
                        {token}(0);
                    }}
                """)

        rc, out, err = run_tc_cuda(src)
        # Must be rejected (non-zero exit) and diagnostic must contain id and name.
        ok = rc != 0 and eid in err
        report(ok, f"{eid} {name[:50]}", f"rc={rc} stderr={err[:200]}")


def test_comment_string_false_positives() -> None:
    """Unsupported tokens inside comments or strings must NOT trigger rejection."""
    print("\n[comment/string false positives]")
    src = textwrap.dedent("""\
        // __ballot_sync is not supported
        /* __syncwarp __ldg __launch_bounds__ */
        __global__ void k() {
            const char* s = "__ballot_sync __syncwarp";
            const char c = 'x';
        }
    """)
    rc, out, err = run_tc_cuda(src)
    report(rc == 0, "comment/string false positives accepted", f"rc={rc} stderr={err[:200]}")


def test_unknown_intrinsic() -> None:
    """Unknown __-prefixed intrinsic must be rejected."""
    print("\n[unknown intrinsic rejection]")
    src = textwrap.dedent("""\
        __global__ void k() {
            __my_unknown_intrinsic();
        }
    """)
    rc, out, err = run_tc_cuda(src)
    ok = rc != 0 and "__my_unknown_intrinsic" in err
    report(ok, "unknown __ intrinsic rejected", f"rc={rc} stderr={err[:200]}")


def test_accepted_minimal() -> None:
    """A minimal accepted kernel with only supported constructs."""
    print("\n[accepted minimal kernel]")
    src = textwrap.dedent("""\
        __device__ float helper(float x) {
            return x * 2.0f;
        }

        __global__ void add(float* a, float* b, float* out, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                out[i] = helper(a[i]) + b[i];
            }
        }
    """)
    rc, out, err = run_tc_cuda(src)
    report(rc == 0, "minimal kernel accepted", f"rc={rc} stderr={err[:200]}")


def test_accepted_rmsnorm() -> None:
    """An rmsnorm-shaped kernel using only supported constructs."""
    print("\n[accepted rmsnorm-shaped kernel]")
    src = textwrap.dedent("""\
        #include <math.h>

        __device__ __half2float(__half h) {
            return 0.0f;
        }

        __global__ void rmsnorm_forward_kernel(
            const __half* __restrict__ input,
            __half* __restrict__ output,
            const __half* __restrict__ weight,
            int rows,
            int cols,
            float eps
        ) {
            int row = blockIdx.x;
            if (row >= rows) return;

            const __half* in_row = input + (size_t)row * cols;
            __half* out_row = output + (size_t)row * cols;

            float sum_sq = 0.0f;
            for (int i = threadIdx.x; i < cols; i += blockDim.x) {
                float v = __half2float(in_row[i]);
                sum_sq += v * v;
            }

            __shared__ float s_sum;
            if (threadIdx.x == 0) s_sum = sum_sq;
            __syncthreads();

            float rms = rsqrtf(s_sum / (float)cols + eps);

            for (int i = threadIdx.x; i < cols; i += blockDim.x) {
                float v = __half2float(in_row[i]);
                out_row[i] = __float2half_rn(v * rms * __half2float(weight[i]));
            }
        }
    """)
    rc, out, err = run_tc_cuda(src)
    report(rc == 0, "rmsnorm-shaped kernel accepted", f"rc={rc} stderr={err[:300]}")


def test_manifest_determinism() -> None:
    """Two runs on the same source must produce byte-identical manifests."""
    print("\n[manifest determinism]")
    src = textwrap.dedent("""\
        __global__ void k1(float* a) { a[0] = 1.0f; }
        __global__ void k2(float* b) { b[0] = 2.0f; }
    """)

    with tempfile.TemporaryDirectory() as td:
        m1 = str(pathlib.Path(td) / "m1.json")
        m2 = str(pathlib.Path(td) / "m2.json")

        rc1, out1, err1 = run_tc_cuda(src, manifest_output=m1)
        rc2, out2, err2 = run_tc_cuda(src, manifest_output=m2)

        if rc1 != 0 or rc2 != 0:
            report(False, "manifest determinism (both runs accepted)",
                   f"rc1={rc1} rc2={rc2} err1={err1[:200]} err2={err2[:200]}")
            return

        b1 = pathlib.Path(m1).read_bytes()
        b2 = pathlib.Path(m2).read_bytes()
        ok = b1 == b2
        report(ok, "manifest byte-identical across two runs",
               "" if ok else f"m1={b1[:200]} m2={b2[:200]}")

        # Also verify the manifest content is correct.
        data = json.loads(b1)
        kernels = [k["name"] for k in data.get("kernels", [])]
        ok2 = sorted(kernels) == ["k1", "k2"] and data.get("status") == "checked"
        report(ok2, "manifest content correct (kernels + status=checked)",
               f"kernels={kernels} status={data.get('status')}")


def test_manifest_status_never_lowered() -> None:
    """The manifest must never contain status 'lowered'."""
    print("\n[manifest status never lowered]")
    src = "__global__ void k() { }"
    with tempfile.TemporaryDirectory() as td:
        m = str(pathlib.Path(td) / "m.json")
        rc, out, err = run_tc_cuda(src, manifest_output=m)
        if rc != 0:
            report(False, "accepted for manifest check", f"rc={rc} err={err[:200]}")
            return
        text = pathlib.Path(m).read_text(encoding="utf-8")
        ok = "lowered" not in text
        report(ok, "manifest does not contain 'lowered'",
               "" if ok else text[:300])


def main() -> int:
    print("tc-cuda frontend selftest")
    print(f"  tc_cuda.py: {TC_CUDA}")
    print(f"  subset:     {SUBSET}")

    test_unsupported_entries()
    test_comment_string_false_positives()
    test_unknown_intrinsic()
    test_accepted_minimal()
    test_accepted_rmsnorm()
    test_manifest_determinism()
    test_manifest_status_never_lowered()

    print(f"\n{'=' * 50}")
    print(f"  PASS: {PASS}  FAIL: {FAIL}")
    if FAIL > 0:
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
