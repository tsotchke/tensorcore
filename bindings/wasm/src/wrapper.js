// wrapper.js — ergonomic JS wrapper over the emcc-built tensorcore
// module. Hides the malloc + HEAPF32 view dance behind functions
// that accept JS typed arrays and return JS typed arrays.
//
// Usage:
//   import init from './index.mjs';
//   const tc = await init();
//   const point = tc.lorentz.exp(base, tangent, 1.0);
//
// All math is byte-identical to the native bindings on the wrapped
// subset (Lorentz, Sphere, Torus, Lie groups, quantum state-vector,
// quantum attention + entanglement, holonomic).

import createTensorCoreModule from './tensorcore.mjs';

/** Internal helper: copy a JS Float32Array into the wasm heap and
 *  return its byte pointer. Caller is responsible for freeing. */
function pushFloats(mod, arr) {
    const f32 = arr instanceof Float32Array ? arr : new Float32Array(arr);
    const bytes = f32.byteLength;
    const ptr = mod._malloc(bytes);
    mod.HEAPF32.set(f32, ptr >> 2);
    return { ptr, length: f32.length };
}
function pullFloats(mod, ptr, length) {
    return new Float32Array(mod.HEAPF32.buffer, ptr, length).slice();
}
function withMalloced(mod, bytes, body) {
    const ptr = mod._malloc(bytes);
    try { return body(ptr); }
    finally { mod._free(ptr); }
}

/** Build the ergonomic API once we have the wasm module instance. */
function buildAPI(mod) {
    return {
        version: () => 'tensorcore-wasm',  // module-level version constant

        // ===== Lorentz =====
        lorentz: {
            exp(base, tangent, curvature = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base);
                const t = pushFloats(mod, tangent);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_lorentz_exp(b.ptr, t.ptr, out, n, curvature);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(t.ptr);
                    return r;
                });
            },
            log(base, point, curvature = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base);
                const p = pushFloats(mod, point);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_lorentz_log(b.ptr, p.ptr, out, n, curvature);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(p.ptr);
                    return r;
                });
            },
            distance(p, q, curvature = 1.0) {
                const pp = pushFloats(mod, p);
                const qq = pushFloats(mod, q);
                const d = mod._tc_wasm_lorentz_distance(pp.ptr, qq.ptr, p.length, curvature);
                mod._free(pp.ptr); mod._free(qq.ptr);
                return d;
            },
        },

        // ===== Sphere =====
        sphere: {
            exp(base, tangent, radius = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base), t = pushFloats(mod, tangent);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_sphere_exp(b.ptr, t.ptr, out, n, radius);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(t.ptr);
                    return r;
                });
            },
            log(base, point, radius = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base), p = pushFloats(mod, point);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_sphere_log(b.ptr, p.ptr, out, n, radius);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(p.ptr);
                    return r;
                });
            },
            distance(p, q, radius = 1.0) {
                const pp = pushFloats(mod, p), qq = pushFloats(mod, q);
                const d = mod._tc_wasm_sphere_distance(pp.ptr, qq.ptr, p.length, radius);
                mod._free(pp.ptr); mod._free(qq.ptr);
                return d;
            },
            slerp(p, q, t, radius = 1.0) {
                const n = p.length;
                const pp = pushFloats(mod, p), qq = pushFloats(mod, q);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_sphere_slerp(pp.ptr, qq.ptr, t, out, n, radius);
                    const r = pullFloats(mod, out, n);
                    mod._free(pp.ptr); mod._free(qq.ptr);
                    return r;
                });
            },
        },

        // ===== Torus =====
        torus: {
            exp(base, tangent, radius = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base), t = pushFloats(mod, tangent);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_torus_exp(b.ptr, t.ptr, out, n, radius);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(t.ptr);
                    return r;
                });
            },
            log(base, point, radius = 1.0) {
                const n = base.length;
                const b = pushFloats(mod, base), p = pushFloats(mod, point);
                return withMalloced(mod, n * 4, (out) => {
                    mod._tc_wasm_torus_log(b.ptr, p.ptr, out, n, radius);
                    const r = pullFloats(mod, out, n);
                    mod._free(b.ptr); mod._free(p.ptr);
                    return r;
                });
            },
            distance(p, q, radius = 1.0) {
                const pp = pushFloats(mod, p), qq = pushFloats(mod, q);
                const d = mod._tc_wasm_torus_distance(pp.ptr, qq.ptr, p.length, radius);
                mod._free(pp.ptr); mod._free(qq.ptr);
                return d;
            },
        },

        // ===== Lie =====
        lie: {
            su2Exp(a, b, c) {
                return withMalloced(mod, 8 * 4, (out) => {
                    mod._tc_wasm_su2_exp(a, b, c, out);
                    return pullFloats(mod, out, 8);
                });
            },
            su2Log(U) {
                const u = pushFloats(mod, U);
                return withMalloced(mod, 3 * 4, (out) => {
                    mod._tc_wasm_su2_log(u.ptr, out);
                    const r = pullFloats(mod, out, 3);
                    mod._free(u.ptr);
                    return [r[0], r[1], r[2]];
                });
            },
            so3Exp(wx, wy, wz) {
                return withMalloced(mod, 9 * 4, (out) => {
                    mod._tc_wasm_so3_exp(wx, wy, wz, out);
                    return pullFloats(mod, out, 9);
                });
            },
            su2ToSo3(U) {
                const u = pushFloats(mod, U);
                return withMalloced(mod, 9 * 4, (out) => {
                    mod._tc_wasm_su2_to_so3(u.ptr, out);
                    const r = pullFloats(mod, out, 9);
                    mod._free(u.ptr);
                    return r;
                });
            },
        },

        // ===== Quantum =====
        quantum: {
            stateZero(nQubits) {
                const dim = 2 * (1 << nQubits);
                return withMalloced(mod, dim * 4, (ptr) => {
                    mod._tc_wasm_qstate_zero(ptr, nQubits);
                    return pullFloats(mod, ptr, dim);
                });
            },
            apply1q(state, nQubits, target, gate) {
                const s = pushFloats(mod, state);
                const g = pushFloats(mod, gate);
                mod._tc_wasm_qstate_apply_1q(s.ptr, nQubits, target, g.ptr);
                const r = pullFloats(mod, s.ptr, s.length);
                mod._free(s.ptr); mod._free(g.ptr);
                return r;
            },
            apply2q(state, nQubits, qa, qb, gate) {
                const s = pushFloats(mod, state);
                const g = pushFloats(mod, gate);
                mod._tc_wasm_qstate_apply_2q(s.ptr, nQubits, qa, qb, g.ptr);
                const r = pullFloats(mod, s.ptr, s.length);
                mod._free(s.ptr); mod._free(g.ptr);
                return r;
            },
            probOne(state, nQubits, qubit) {
                const s = pushFloats(mod, state);
                const p = mod._tc_wasm_qstate_prob_one(s.ptr, nQubits, qubit);
                mod._free(s.ptr);
                return p;
            },
            gate1q(gateType) {
                return withMalloced(mod, 8 * 4, (out) => {
                    mod._tc_wasm_gate_1q(gateType, out);
                    return pullFloats(mod, out, 8);
                });
            },
            gate2q(gateType) {
                return withMalloced(mod, 32 * 4, (out) => {
                    mod._tc_wasm_gate_2q(gateType, out);
                    return pullFloats(mod, out, 32);
                });
            },
            attentionScore(stateQ, stateK, nQubits) {
                const q = pushFloats(mod, stateQ);
                const k = pushFloats(mod, stateK);
                const s = mod._tc_wasm_quantum_attention_score(q.ptr, k.ptr, nQubits);
                mod._free(q.ptr); mod._free(k.ptr);
                return s;
            },
            entanglementEntropy(state, nQubits, qubitKeep) {
                const s = pushFloats(mod, state);
                const e = mod._tc_wasm_quantum_entanglement_entropy(s.ptr, nQubits, qubitKeep);
                mod._free(s.ptr);
                return e;
            },
        },

        // ===== Holonomic =====
        holonomic: {
            composeSU2(generators) {
                const gens = generators instanceof Float32Array
                    ? generators
                    : new Float32Array(generators.flat ? generators.flat() : generators);
                if (gens.length % 3 !== 0) {
                    throw new Error('holonomic.composeSU2: generators length must be multiple of 3');
                }
                const nSeg = gens.length / 3;
                const g = pushFloats(mod, gens);
                const out = withMalloced(mod, 8 * 4, (outPtr) => {
                    const rc = mod._tc_wasm_holonomic_compose_su2(g.ptr, nSeg, outPtr);
                    if (rc !== 0) throw new Error(`compose_su2 returned ${rc}`);
                    return pullFloats(mod, outPtr, 8);
                });
                mod._free(g.ptr);
                return out;
            },
            berryPhase(U) {
                const u = pushFloats(mod, U);
                return withMalloced(mod, 4, (outPtr) => {
                    mod._tc_wasm_holonomic_berry_phase(u.ptr, outPtr);
                    const phase = mod.HEAPF32[outPtr >> 2];
                    mod._free(u.ptr);
                    return phase;
                });
            },
        },
    };
}

export default async function init(options = {}) {
    const mod = await createTensorCoreModule(options);
    return buildAPI(mod);
}
