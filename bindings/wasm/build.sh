#!/usr/bin/env bash
# build.sh — emcc compile of the tensorcore CPU subset → tensorcore.{js,wasm}.
#
# Builds an ES-module shim suitable for `import` from browser or
# Node.js. Same source files as the portable-CPU CMake build path —
# math is byte-identical to the native bindings on the wrapped subset.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$HERE/dist"
mkdir -p "$DIST"

if ! command -v emcc >/dev/null 2>&1; then
    echo "emcc not found on PATH — install emscripten SDK first" >&2
    exit 2
fi

SOURCES=(
    "$HERE/src/wrapper.c"
    "$REPO_ROOT/lib/ops/lorentz_cpu.cpp"
    "$REPO_ROOT/lib/ops/sphere_cpu.cpp"
    "$REPO_ROOT/lib/ops/torus_cpu.cpp"
    "$REPO_ROOT/lib/ops/lie_groups_cpu.cpp"
    "$REPO_ROOT/lib/ops/quantum_gates_cpu.cpp"
    "$REPO_ROOT/lib/ops/quantum_attention_cpu.cpp"
    "$REPO_ROOT/lib/ops/density_matrix_cpu.cpp"
    "$REPO_ROOT/lib/ops/holonomic_cpu.cpp"
)

# Exports: every tc_wasm_* symbol. (emcc requires the leading underscore.)
EXPORTED=(
    "_tc_wasm_lorentz_exp" "_tc_wasm_lorentz_log" "_tc_wasm_lorentz_distance"
    "_tc_wasm_sphere_exp"  "_tc_wasm_sphere_log"  "_tc_wasm_sphere_distance"
    "_tc_wasm_sphere_slerp"
    "_tc_wasm_torus_exp"   "_tc_wasm_torus_log"   "_tc_wasm_torus_distance"
    "_tc_wasm_su2_exp"     "_tc_wasm_su2_log"     "_tc_wasm_so3_exp" "_tc_wasm_su2_to_so3"
    "_tc_wasm_qstate_zero" "_tc_wasm_qstate_apply_1q" "_tc_wasm_qstate_apply_2q"
    "_tc_wasm_qstate_prob_one" "_tc_wasm_gate_1q" "_tc_wasm_gate_2q"
    "_tc_wasm_quantum_attention_score" "_tc_wasm_quantum_entanglement_entropy"
    "_tc_wasm_holonomic_compose_su2"   "_tc_wasm_holonomic_berry_phase"
    "_malloc" "_free"
)

EXPORTED_JSON=$(printf '"%s",' "${EXPORTED[@]}" | sed 's/,$//')

emcc "${SOURCES[@]}" \
    -O2 \
    -I"$REPO_ROOT/include" \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createTensorCore \
    -s ENVIRONMENT='web,node' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=33554432 \
    -s EXPORTED_FUNCTIONS="[$EXPORTED_JSON]" \
    -s EXPORTED_RUNTIME_METHODS='["HEAPF32","HEAP32"]' \
    -o "$DIST/tensorcore.mjs"

# Copy the JS ergonomics shim alongside the emcc output so consumers
# can `import { Lorentz, Sphere, ... } from 'tensorcore-wasm';`
cp "$HERE/src/wrapper.js" "$DIST/index.mjs"

echo "wrote $DIST/tensorcore.mjs"
echo "wrote $DIST/tensorcore.wasm"
echo "wrote $DIST/index.mjs"
