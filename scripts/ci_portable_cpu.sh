#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${TC_CPU_BUILD_DIR:-build-portable-cpu}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
tmp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
install_dir="${TC_CPU_INSTALL_DIR:-${tmp_root}/tensorcore-portable-cpu-install}"
consumer_dir="${TC_CPU_CONSUMER_DIR:-${tmp_root}/tensorcore-portable-cpu-consumer}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CC_BIN="${CC:-cc}"

case "$(uname -s)" in
    Darwin) shared_lib_name="libtensorcore.dylib" ;;
    Linux) shared_lib_name="libtensorcore.so" ;;
    *) shared_lib_name="libtensorcore.so" ;;
esac

cmake -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DTC_ENABLE_METAL=OFF \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_BENCH=OFF \
  -DTC_BUILD_EXAMPLES=OFF
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure

cmake -E rm -rf "$install_dir" "$consumer_dir"
cmake --install "$build_dir" --prefix "$install_dir"
cmake -E make_directory "$consumer_dir"

cmake -S "$ROOT/examples/native_sdk_consumer" -B "$consumer_dir/build" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_PREFIX_PATH="$install_dir"
cmake --build "$consumer_dir/build" --parallel
DYLD_LIBRARY_PATH="$install_dir/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
TC_CONSUMER_RUN_INIT=1 \
    "$consumer_dir/build/consumer_shared"
DYLD_LIBRARY_PATH="$install_dir/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$consumer_dir/build/consumer_cxx"
TC_CONSUMER_RUN_INIT=1 "$consumer_dir/build/consumer_static"

if command -v pkg-config >/dev/null 2>&1; then
  PKG_CONFIG_PATH="$install_dir/lib/pkgconfig" pkg-config --modversion tensorcore
  PKG_CONFIG_PATH="$install_dir/lib/pkgconfig" pkg-config --libs --static tensorcore
  "$CC_BIN" "$ROOT/examples/native_sdk_consumer/main.c" \
    $(PKG_CONFIG_PATH="$install_dir/lib/pkgconfig" pkg-config --cflags --libs tensorcore) \
    -o "$consumer_dir/pkg-consumer"
  DYLD_LIBRARY_PATH="$install_dir/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
  LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$consumer_dir/pkg-consumer"
fi

shared_lib="$install_dir/lib/$shared_lib_name"
if [ ! -f "$shared_lib" ]; then
    echo "portable CPU shared library not found: $shared_lib" >&2
    exit 1
fi

PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
TENSORCORE_LIB="$shared_lib" \
"$PYTHON_BIN" - <<'PY'
import ctypes
import math
import os
import subprocess
import signal
import socket
import struct
import sys
import tensorcore as tc


def f16(value):
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def f16_to_f32(bits):
    return struct.unpack("<e", struct.pack("<H", int(bits)))[0]


def _reserve_loopback_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _run_python_gloo_rank(rank, url):
    ctx = tc.init()
    dist = None
    bufs = []
    try:
        dist = tc.dist_init(ctx, "gloo", 2, rank, url)
        if tc.dist_world_size(dist) != 2 or tc.dist_rank(dist) != rank:
            raise RuntimeError("GLOO metadata mismatch")

        vals = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        if rank == 1:
            vals = (ctypes.c_float * 4)(10.0, 20.0, 30.0, 40.0)
        buf = tc.buffer_alloc(ctx, ctypes.sizeof(vals))
        bufs.append(buf)
        ctypes.memmove(tc.buffer_map(buf), vals, ctypes.sizeof(vals))
        view = (ctypes.c_float * 4).from_address(tc.buffer_map(buf).value)

        tc.allreduce(dist, buf, 4, "f32", "sum")
        if any(math.fabs(view[i] - (11.0 * (i + 1))) > 1e-6 for i in range(4)):
            raise RuntimeError(f"GLOO allreduce mismatch on rank {rank}: {list(view)}")

        for i in range(4):
            view[i] = (70.0 + i) if rank == 1 else -1.0
        tc.broadcast(dist, buf, 4, "f32", root=1)
        if any(math.fabs(view[i] - (70.0 + i)) > 1e-6 for i in range(4)):
            raise RuntimeError(f"GLOO broadcast mismatch on rank {rank}: {list(view)}")

        for i in range(4):
            view[i] = rank * 100.0 + i
        gathered = tc.buffer_alloc(ctx, 8 * ctypes.sizeof(ctypes.c_float))
        bufs.append(gathered)
        tc.allgather(dist, buf, gathered, 4, "f32")
        gout = (ctypes.c_float * 8).from_address(tc.buffer_map(gathered).value)
        want = [0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]
        if any(math.fabs(gout[i] - want[i]) > 1e-6 for i in range(8)):
            raise RuntimeError(f"GLOO allgather mismatch on rank {rank}: {list(gout)}")

        theta_vals = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
        theta_buf = tc.buffer_alloc(ctx, ctypes.sizeof(theta_vals))
        bufs.append(theta_buf)
        ctypes.memmove(tc.buffer_map(theta_buf), theta_vals, ctypes.sizeof(theta_vals))
        theta = (ctypes.c_float * 4).from_address(tc.buffer_map(theta_buf).value)
        with tc.DiLoCoContext(dist, inner_steps=2, outer_lr=1.0,
                              outer_optimizer="sgd", compress="none") as diloco:
            diloco.add_parameter("theta", theta_buf, 4, "f32")
            delta = 0.25 if rank == 0 else 0.75
            for step in range(2):
                for i in range(4):
                    theta[i] += delta
                pending = diloco.step()
                if step == 0 and pending:
                    raise RuntimeError("DiLoCo outer step became pending too early")
                if step == 1 and not pending:
                    raise RuntimeError("DiLoCo outer step did not become pending")
            diloco.apply_outer()
            if (diloco.inner_steps_completed != 2 or
                    diloco.outer_steps_completed != 1 or
                    diloco.last_outer_bytes_sent <= 0.0):
                raise RuntimeError("DiLoCo GLOO counters mismatch")
        if any(math.fabs(theta[i] - 2.0) > 1e-6 for i in range(4)):
            raise RuntimeError(f"DiLoCo GLOO theta mismatch on rank {rank}: {list(theta)}")

        tc.barrier(dist)
        return 0
    finally:
        for buf in reversed(bufs):
            tc.buffer_free(ctx, buf)
        if dist is not None:
            tc.dist_finalize(dist)
        tc.shutdown(ctx)


def _run_python_gloo_fork_smoke():
    if not hasattr(os, "fork"):
        print("python GLOO/DiLoCo fork smoke SKIP: no fork")
        return
    try:
        port = _reserve_loopback_port()
    except OSError:
        print("python GLOO/DiLoCo fork smoke SKIP: no loopback port")
        return
    url = f"gloo+tcp://127.0.0.1:{port}"
    child = os.fork()
    if child == 0:
        try:
            signal.alarm(30)
            rc = _run_python_gloo_rank(1, url)
        except BaseException as exc:
            print(f"[rank 1] python GLOO/DiLoCo fork smoke FAIL: {exc}", file=sys.stderr)
            rc = 1
        os._exit(0 if rc == 0 else 1)
    parent_rc = 1
    parent_exc = None
    status = 0
    try:
        signal.alarm(30)
        try:
            parent_rc = _run_python_gloo_rank(0, url)
        except BaseException as exc:
            print(f"[rank 0] python GLOO/DiLoCo fork smoke FAIL: {exc}", file=sys.stderr)
            parent_exc = exc
        _, status = os.waitpid(child, 0)
    finally:
        signal.alarm(0)
    child_ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    if parent_exc is not None or parent_rc != 0 or not child_ok:
        raise SystemExit("python GLOO/DiLoCo fork smoke failed")
    print("python GLOO/DiLoCo fork smoke OK")


def _run_gemm_variant_smoke(name, env_updates, allow_sigill_skip=False, require=False):
    code = r'''
import ctypes
import math
import os
import tensorcore as tc

M = N = K = int(os.environ.get("TC_GEMM_VARIANT_SIZE", "32"))
ctx = tc.init()
bufs = []
try:
    A_vals = [((i * 13 + 7) % 17 - 8) / 7.0 for i in range(M * K)]
    B_vals = [((i * 5 + 3) % 19 - 9) / 9.0 for i in range(K * N)]
    A_arr = (ctypes.c_float * (M * K))(*A_vals)
    B_arr = (ctypes.c_float * (K * N))(*B_vals)
    A = tc.buffer_alloc(ctx, ctypes.sizeof(A_arr))
    B = tc.buffer_alloc(ctx, ctypes.sizeof(B_arr))
    C = tc.buffer_alloc(ctx, M * N * ctypes.sizeof(ctypes.c_float))
    bufs.extend([A, B, C])
    ctypes.memmove(tc.buffer_map(A), A_arr, ctypes.sizeof(A_arr))
    ctypes.memmove(tc.buffer_map(B), B_arr, ctypes.sizeof(B_arr))
    tc.gemm(ctx, A, B, C, M, N, K, dtype="f32", accum="f32")
    out = (ctypes.c_float * (M * N)).from_address(tc.buffer_map(C).value)
    max_err = 0.0
    for m in range(M):
        for n in range(N):
            ref = sum(A_vals[m * K + k] * B_vals[k * N + n] for k in range(K))
            max_err = max(max_err, abs(out[m * N + n] - ref))
    if max_err > 1e-3:
        raise SystemExit(f"{os.environ['TC_GEMM_VARIANT_NAME']} max_err={max_err}")
    print(f"{os.environ['TC_GEMM_VARIANT_NAME']} GEMM variant smoke OK: max_err={max_err:.3g} backend={tc.last_backend_name()}")
finally:
    for buf in reversed(bufs):
        tc.buffer_free(ctx, buf)
    tc.shutdown(ctx)
'''
    env = os.environ.copy()
    env.update(env_updates)
    env["TC_GEMM_VARIANT_NAME"] = name
    proc = subprocess.run([sys.executable, "-c", code],
                          env=env, text=True, capture_output=True)
    if proc.returncode == 0:
        print(proc.stdout.strip())
        return
    if allow_sigill_skip and proc.returncode == -signal.SIGILL:
        msg = f"{name} GEMM variant smoke SKIP: signal {-proc.returncode}"
        if require:
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
            raise SystemExit(msg)
        print(msg)
        return
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    raise SystemExit(f"{name} GEMM variant smoke failed: exit {proc.returncode}")


def _run_gemm_variant_smokes():
    _run_gemm_variant_smoke("AVX2 opt-in", {"TC_USE_AVX2_GEMM": "1"})
    _run_gemm_variant_smoke("NEON opt-in", {"TC_USE_NEON_GEMM": "1"})
    machine = os.uname().machine if hasattr(os, "uname") else ""
    amx_size = "256" if sys.platform == "darwin" and machine in ("arm64", "aarch64") else "32"
    _run_gemm_variant_smoke("AMX opt-in", {
                                "TC_USE_AMX_GEMM": "1",
                                "TC_GEMM_VARIANT_SIZE": amx_size,
                            },
                            allow_sigill_skip=True,
                            require=os.environ.get("REQUIRE_AMX_GEMM") == "1")


_run_python_gloo_fork_smoke()
_run_gemm_variant_smokes()

ctx = tc.init()
bufs = []
try:
    info = tc.device_info(ctx)
    if info.name_str != "portable-cpu" or info.family != tc.TC_FAMILY_UNKNOWN:
        raise SystemExit(f"unexpected portable CPU device: {info.name_str} family={info.family}")

    A_vals = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    B_vals = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
    C_vals = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)

    A = tc.buffer_alloc(ctx, ctypes.sizeof(A_vals))
    B = tc.buffer_alloc(ctx, ctypes.sizeof(B_vals))
    C = tc.buffer_alloc(ctx, ctypes.sizeof(C_vals))
    bufs.extend([A, B, C])
    ctypes.memmove(tc.buffer_map(A), A_vals, ctypes.sizeof(A_vals))
    ctypes.memmove(tc.buffer_map(B), B_vals, ctypes.sizeof(B_vals))
    ctypes.memmove(tc.buffer_map(C), C_vals, ctypes.sizeof(C_vals))

    tc.gemm(ctx, A, B, C, 2, 2, 2, dtype="f32")
    if tc.last_backend_name() != "portable_cpu":
        raise SystemExit(f"unexpected GEMM backend: {tc.last_backend_name()}")

    out = (ctypes.c_float * 4).from_address(tc.buffer_map(C).value)
    expected = (19.0, 22.0, 43.0, 50.0)
    if any(math.fabs(out[i] - expected[i]) > 1e-5 for i in range(4)):
        raise SystemExit(f"unexpected GEMM result: {[out[i] for i in range(4)]}")

    tc.buffer_set_tier_hint(C, "warm")
    if tc.buffer_get_tier(C) != tc.TC_TIER_L0_DEVICE:
        raise SystemExit("portable memory tier should remain L0")
    tc.buffer_promote_async(C, "l0")
    tc.buffer_demote_async(C, "l0")
    tc.buffer_tier_sync(C)
    if tc.memory_tier_usage(ctx, "l0") != (0, 0):
        raise SystemExit("portable memory tier usage should be zero in the stub baseline")

    checkpoint_calls = {"n": 0}
    def recompute(_user_data):
        checkpoint_calls["n"] += 1
        return tc.TC_OK

    checkpoint_id = tc.checkpoint_register(C, recompute)
    if not tc.checkpoint_is_resident(checkpoint_id):
        raise SystemExit("checkpoint should start resident")
    tc.checkpoint_discard(checkpoint_id)
    if (tc.checkpoint_is_resident(checkpoint_id) or
            tc.checkpoint_count_discarded() == 0 or
            tc.checkpoint_total_bytes_discarded() < tc.buffer_size(C)):
        raise SystemExit("checkpoint discard counters mismatch")
    tc.checkpoint_realize(checkpoint_id)
    if (not tc.checkpoint_is_resident(checkpoint_id) or
            checkpoint_calls["n"] != 1 or
            tc.checkpoint_total_bytes_discarded() != 0):
        raise SystemExit("checkpoint realize counters mismatch")
    tc.checkpoint_unregister(checkpoint_id)

    dist = tc.dist_init(ctx, "single", 1, 0, "single://portable-python")
    try:
        if tc.dist_world_size(dist) != 1 or tc.dist_rank(dist) != 0:
            raise SystemExit("portable distributed metadata mismatch")
        tc.allreduce(dist, C, 4, "f32", "sum")
        tc.barrier(dist)

        if tc.hip_device_count() != 0 or tc.hip_last_kernel_name() != "none":
            raise SystemExit("portable HIP inactive diagnostics mismatch")
        try:
            tc.hip_init(ctx)
        except tc.TensorcoreError as exc:
            if exc.status != tc.TC_ERR_UNSUPPORTED_FAMILY:
                raise
        else:
            raise SystemExit("portable CPU should not initialize HIP")

        if tc.cuda_device_count() != 0 or tc.cuda_last_kernel_name() != "none":
            raise SystemExit("portable CUDA inactive diagnostics mismatch")
        try:
            tc.cuda_init(ctx)
        except tc.TensorcoreError as exc:
            if exc.status != tc.TC_ERR_UNSUPPORTED_FAMILY:
                raise
        else:
            raise SystemExit("portable CPU should not initialize CUDA")

        Theta_vals = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        Theta = tc.buffer_alloc(ctx, ctypes.sizeof(Theta_vals))
        bufs.append(Theta)
        ctypes.memmove(tc.buffer_map(Theta), Theta_vals, ctypes.sizeof(Theta_vals))
        with tc.DiLoCoContext(dist, inner_steps=2, outer_lr=0.5,
                              outer_optimizer="sgd", compress="none") as diloco:
            diloco.add_parameter("theta", Theta, 4, "f32")
            theta = (ctypes.c_float * 4).from_address(tc.buffer_map(Theta).value)
            for i in range(4):
                theta[i] += 1.0
            if diloco.step():
                raise SystemExit("DiLoCo outer step became pending too early")
            if not diloco.step():
                raise SystemExit("DiLoCo outer step did not become pending")
            diloco.apply_outer()
            if (diloco.inner_steps_completed != 2 or
                    diloco.outer_steps_completed != 1 or
                    diloco.last_outer_bytes_sent != 0.0):
                raise SystemExit("DiLoCo counters mismatch")
        theta = (ctypes.c_float * 4).from_address(tc.buffer_map(Theta).value)
        expected_theta = (1.5, 2.5, 3.5, 4.5)
        if any(math.fabs(theta[i] - expected_theta[i]) > 1e-6 for i in range(4)):
            raise SystemExit(f"unexpected DiLoCo theta: {[theta[i] for i in range(4)]}")
    finally:
        tc.dist_finalize(dist)

    X16_vals = (ctypes.c_uint16 * 4)(f16(0.0), f16(1.0), f16(-1.0), f16(2.0))
    Y16_vals = (ctypes.c_uint16 * 4)(0, 0, 0, 0)
    X16 = tc.buffer_alloc(ctx, ctypes.sizeof(X16_vals))
    Y16 = tc.buffer_alloc(ctx, ctypes.sizeof(Y16_vals))
    bufs.extend([X16, Y16])
    ctypes.memmove(tc.buffer_map(X16), X16_vals, ctypes.sizeof(X16_vals))
    ctypes.memmove(tc.buffer_map(Y16), Y16_vals, ctypes.sizeof(Y16_vals))
    tc.softmax_forward(ctx, X16, Y16, 1, 4)
    sm = (ctypes.c_uint16 * 4).from_address(tc.buffer_map(Y16).value)
    got = [f16_to_f32(sm[i]) for i in range(4)]
    denom = sum(math.exp(v) for v in (0.0, 1.0, -1.0, 2.0))
    want = [math.exp(v) / denom for v in (0.0, 1.0, -1.0, 2.0)]
    if max(abs(g - w) for g, w in zip(got, want)) > 1e-3:
        raise SystemExit(f"unexpected softmax result: {got}")

    print(f"{tc.version()} python portable CPU smoke OK")
finally:
    for buf in reversed(bufs):
        tc.buffer_free(ctx, buf)
    tc.shutdown(ctx)
PY
