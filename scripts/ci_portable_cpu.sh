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
import struct
import tensorcore as tc


def f16(value):
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def f16_to_f32(bits):
    return struct.unpack("<e", struct.pack("<H", int(bits)))[0]

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
