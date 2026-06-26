// smoke.mjs — Node.js smoke test for the WASM bindings.
//
// Mirrors the Python + Rust + Swift substrate smokes; same probes,
// same expected numbers; we're proving the WASM bridge marshals
// typed arrays correctly + the math is byte-identical.
//
// Run after `./build.sh`:
//   node test/smoke.mjs

import init from '../dist/index.mjs';

const tc = await init();

function approx(a, b, tol = 1e-4) { return Math.abs(a - b) < tol; }
function maxAbsDiff(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; ++i) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
}

const results = {};
function probe(name, body) {
    try {
        const r = body();
        results[name] = { passed: r.passed === true, ...r,
                          runtime_status: r.passed ? 'passed' : 'failed' };
    } catch (e) {
        results[name] = { passed: false, runtime_status: 'failed',
                          error: String(e) };
    }
}

probe('lorentz_roundtrip', () => {
    const base = new Float32Array(8); base[0] = 1.0;
    const tangent = new Float32Array([0, 0.1, -0.2, 0.15, 0.05, 0, 0, 0]);
    const point = tc.lorentz.exp(base, tangent);
    const rec = tc.lorentz.log(base, point);
    const err = maxAbsDiff(tangent, rec);
    return { passed: err < 1e-4, err };
});

probe('sphere_slerp', () => {
    const base = new Float32Array(16); base[0] = 1.0;
    const tangent = new Float32Array(16);
    tangent[1] = 0.1; tangent[2] = 0.2; tangent[3] = -0.15;
    const point = tc.sphere.exp(base, tangent);
    const mid = tc.sphere.slerp(base, point, 0.5);
    const dMid = tc.sphere.distance(base, mid);
    const dFull = tc.sphere.distance(base, point);
    return { passed: approx(2 * dMid, dFull, 1e-3), dMid, dFull };
});

probe('torus_roundtrip', () => {
    const base = new Float32Array([0, 0, 0]);
    const v = new Float32Array([0.4, -0.3, 0.2]);
    const q = tc.torus.exp(base, v);
    const v2 = tc.torus.log(base, q);
    const err = maxAbsDiff(v, v2);
    return { passed: err < 1e-4, err };
});

probe('su2_exp_log', () => {
    const [aIn, bIn, cIn] = [0.1, -0.2, Math.PI / 4];
    const U = tc.lie.su2Exp(aIn, bIn, cIn);
    const [a, b, c] = tc.lie.su2Log(U);
    return { passed: approx(a, aIn) && approx(b, bIn) && approx(c, cIn),
              a, b, c };
});

probe('so3_quarter_turn', () => {
    const R = tc.lie.so3Exp(0, 0, Math.PI / 2);
    return { passed: approx(R[1], -1, 1e-3) && approx(R[3], 1, 1e-3)
                     && approx(R[8], 1, 1e-3) };
});

probe('quantum_hadamard_prob', () => {
    let state = tc.quantum.stateZero(2);
    const H = tc.quantum.gate1q(4);  // H = 4
    state = tc.quantum.apply1q(state, 2, 0, H);
    const p = tc.quantum.probOne(state, 2, 0);
    return { passed: approx(p, 0.5, 1e-5), prob_one: p };
});

probe('quantum_overlap', () => {
    const s0 = tc.quantum.stateZero(2);
    const s1 = new Float32Array(s0); s1[0] = 0; s1[2] = 1; // |10⟩
    const same = tc.quantum.attentionScore(s0, s0, 2);
    const orth = tc.quantum.attentionScore(s0, s1, 2);
    return { passed: approx(same, 1) && approx(orth, 0), same, orth };
});

probe('bell_entanglement', () => {
    const bell = new Float32Array(8);
    bell[0] = 1 / Math.sqrt(2); bell[6] = bell[0];
    const s = tc.quantum.entanglementEntropy(bell, 2, 0);
    return { passed: approx(s, 1, 1e-4), entropy_bits: s };
});

probe('holonomic_phase_pi4', () => {
    const U = tc.holonomic.composeSU2([0, 0, Math.PI / 4]);
    const phase = tc.holonomic.berryPhase(U);
    return { passed: approx(phase, Math.PI / 4, 1e-4), phase };
});

probe('holonomic_closed_loop', () => {
    const U = tc.holonomic.composeSU2([0, 0, Math.PI / 4,
                                         0, 0, -Math.PI / 4]);
    const phase = tc.holonomic.berryPhase(U);
    return { passed: Math.abs(phase) < 1e-3, phase };
});

// Print results in a way the ICC smoke wrapper can parse.
const passed = Object.values(results).filter(r => r.passed).length;
const total = Object.values(results).length;
for (const [name, r] of Object.entries(results)) {
    const mark = r.passed ? 'PASS' : 'FAIL';
    console.log(`${mark} ${name}`);
}
console.log(`\n${passed}/${total} probes passed`);
console.log(JSON.stringify({
    checks: {
        wasm_bindings: {
            runtime_status: passed === total ? 'passed' : 'failed',
            total_passed: passed, total_failed: total - passed,
            probes: results,
        }
    }
}));

process.exit(passed === total ? 0 : 2);
