// TensorCore.swift — safe Swift surface over libtensorcore.
//
// Inputs are [Float] (or UnsafePointer<Float> for tight inner loops);
// outputs are [Float] for variable-size returns or fixed-size arrays
// where the C ABI's shape is known statically (e.g. SU(2) unitary =
// 8 floats, SO(3) rotation = 9 floats).
//
// Math is bit-identical to the Python + Rust bindings — they all
// dispatch into the same lib/ops/*.cpp kernels.

import CTensorCore
import Foundation

// MARK: - Context lifecycle

public enum TensorCoreError: Error {
    case initFailed(status: Int)
    case opFailed(status: Int, message: String)
}

public final class Context {
    fileprivate var ptr: UnsafeMutableRawPointer?

    public init() throws {
        var p: UnsafeMutableRawPointer? = nil
        let rc = withUnsafeMutablePointer(to: &p) { tc_init($0) }
        if rc != 0 || p == nil {
            throw TensorCoreError.initFailed(status: Int(rc))
        }
        self.ptr = p
    }

    deinit {
        if let p = ptr {
            _ = tc_shutdown(p)
        }
    }
}

public func tensorcoreVersion() -> String {
    guard let cstr = tc_version() else { return "(unknown)" }
    return String(cString: cstr)
}

public func tensorcoreStatusString(_ status: Int) -> String {
    guard let cstr = tc_status_string(Int32(status)) else { return "status \(status)" }
    return String(cString: cstr)
}

// MARK: - Helpers

@inline(__always)
private func withFloat<T>(_ a: [Float], _ body: (UnsafePointer<Float>) -> T) -> T {
    return a.withUnsafeBufferPointer { body($0.baseAddress!) }
}

@inline(__always)
private func withFloatMut<T>(_ a: inout [Float], _ body: (UnsafeMutablePointer<Float>) -> T) -> T {
    return a.withUnsafeMutableBufferPointer { body($0.baseAddress!) }
}

// MARK: - Lorentz

public enum Lorentz {
    public static func exp(base: [Float], tangent: [Float], curvature: Float = 1.0) -> [Float] {
        precondition(base.count == tangent.count, "Lorentz.exp: base/tangent length mismatch")
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(tangent) { tp in
            withFloatMut(&out) { op in
                tc_lorentz_exp(bp, tp, op, UInt(n), curvature)
            }
        }}
        return out
    }
    public static func log(base: [Float], point: [Float], curvature: Float = 1.0) -> [Float] {
        precondition(base.count == point.count)
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(point) { pp in
            withFloatMut(&out) { op in
                tc_lorentz_log(bp, pp, op, UInt(n), curvature)
            }
        }}
        return out
    }
    public static func distance(_ p: [Float], _ q: [Float], curvature: Float = 1.0) -> Float {
        precondition(p.count == q.count)
        let n = p.count
        return withFloat(p) { pp in withFloat(q) { qp in
            tc_lorentz_distance(pp, qp, UInt(n), curvature)
        }}
    }
    public static func minkowski(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        return withFloat(a) { ap in withFloat(b) { bp in
            tc_lorentz_minkowski(ap, bp, UInt(a.count))
        }}
    }
}

// MARK: - Sphere

public enum Sphere {
    public static func exp(base: [Float], tangent: [Float], radius: Float = 1.0) -> [Float] {
        precondition(base.count == tangent.count)
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(tangent) { tp in
            withFloatMut(&out) { op in
                tc_sphere_exp(bp, tp, op, UInt(n), radius)
            }
        }}
        return out
    }
    public static func log(base: [Float], point: [Float], radius: Float = 1.0) -> [Float] {
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(point) { pp in
            withFloatMut(&out) { op in
                tc_sphere_log(bp, pp, op, UInt(n), radius)
            }
        }}
        return out
    }
    public static func distance(_ p: [Float], _ q: [Float], radius: Float = 1.0) -> Float {
        return withFloat(p) { pp in withFloat(q) { qp in
            tc_sphere_distance(pp, qp, UInt(p.count), radius)
        }}
    }
    public static func slerp(_ p: [Float], _ q: [Float], t: Float, radius: Float = 1.0) -> [Float] {
        var out = [Float](repeating: 0, count: p.count)
        let n = p.count
        withFloat(p) { pp in withFloat(q) { qp in
            withFloatMut(&out) { op in
                tc_sphere_slerp(pp, qp, t, op, UInt(n), radius)
            }
        }}
        return out
    }
}

// MARK: - Torus

public enum Torus {
    public static func exp(base: [Float], tangent: [Float], radius: Float = 1.0) -> [Float] {
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(tangent) { tp in
            withFloatMut(&out) { op in tc_torus_exp(bp, tp, op, UInt(n), radius) }
        }}
        return out
    }
    public static func log(base: [Float], point: [Float], radius: Float = 1.0) -> [Float] {
        var out = [Float](repeating: 0, count: base.count)
        let n = base.count
        withFloat(base) { bp in withFloat(point) { pp in
            withFloatMut(&out) { op in tc_torus_log(bp, pp, op, UInt(n), radius) }
        }}
        return out
    }
    public static func distance(_ p: [Float], _ q: [Float], radius: Float = 1.0) -> Float {
        return withFloat(p) { pp in withFloat(q) { qp in
            tc_torus_distance(pp, qp, UInt(p.count), radius)
        }}
    }
}

// MARK: - Lie groups (SU(2) + SO(3))

public enum Lie {
    public static func su2Exp(_ a: Float, _ b: Float, _ c: Float) -> [Float] {
        var u = [Float](repeating: 0, count: 8)
        withFloatMut(&u) { up in tc_su2_exp(a, b, c, up) }
        return u
    }
    public static func su2Log(_ U: [Float]) -> (Float, Float, Float) {
        precondition(U.count == 8)
        var a: Float = 0, b: Float = 0, c: Float = 0
        withFloat(U) { up in tc_su2_log(up, &a, &b, &c) }
        return (a, b, c)
    }
    public static func su2Mul(_ U: [Float], _ V: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: 8)
        withFloat(U) { up in withFloat(V) { vp in
            withFloatMut(&out) { op in tc_su2_mul(up, vp, op) }
        }}
        return out
    }
    public static func so3Exp(_ wx: Float, _ wy: Float, _ wz: Float) -> [Float] {
        var R = [Float](repeating: 0, count: 9)
        withFloatMut(&R) { rp in tc_so3_exp(wx, wy, wz, rp) }
        return R
    }
    public static func so3Log(_ R: [Float]) -> (Float, Float, Float) {
        var wx: Float = 0, wy: Float = 0, wz: Float = 0
        withFloat(R) { rp in tc_so3_log(rp, &wx, &wy, &wz) }
        return (wx, wy, wz)
    }
    public static func su2ToSo3(_ U: [Float]) -> [Float] {
        var R = [Float](repeating: 0, count: 9)
        withFloat(U) { up in withFloatMut(&R) { rp in tc_su2_to_so3(up, rp) } }
        return R
    }
}

// MARK: - Quantum

public enum Quantum {
    public static func stateZero(_ nQubits: Int) -> [Float] {
        var state = [Float](repeating: 0, count: 2 * (1 << nQubits))
        withFloatMut(&state) { sp in tc_qstate_zero(sp, Int32(nQubits)) }
        return state
    }

    public static func apply1q(state: inout [Float], nQubits: Int, target: Int, gate: [Float]) {
        precondition(gate.count == 8)
        withFloatMut(&state) { sp in withFloat(gate) { gp in
            tc_qstate_apply_1q_unitary(sp, Int32(nQubits), Int32(target), gp)
        }}
    }

    public static func apply2q(state: inout [Float], nQubits: Int, qa: Int, qb: Int, gate: [Float]) {
        precondition(gate.count == 32)
        withFloatMut(&state) { sp in withFloat(gate) { gp in
            tc_qstate_apply_2q_unitary(sp, Int32(nQubits), Int32(qa), Int32(qb), gp)
        }}
    }

    public static func probOne(state: [Float], nQubits: Int, qubit: Int) -> Float {
        return withFloat(state) { sp in
            tc_qstate_prob_one(sp, Int32(nQubits), Int32(qubit))
        }
    }

    public static func normSq(state: [Float], nQubits: Int) -> Float {
        return withFloat(state) { sp in tc_qstate_norm_sq(sp, Int32(nQubits)) }
    }

    public static func gate1q(_ gateType: Int32) -> [Float] {
        var out = [Float](repeating: 0, count: 8)
        withFloatMut(&out) { op in tc_gate_matrix_1q(gateType, nil, op) }
        return out
    }

    public static func gate2q(_ gateType: Int32) -> [Float] {
        var out = [Float](repeating: 0, count: 32)
        withFloatMut(&out) { op in tc_gate_matrix_2q(gateType, nil, op) }
        return out
    }

    // Quantum attention
    public static func attentionScore(_ stateQ: [Float], _ stateK: [Float],
                                       nQubits: Int) -> Float {
        return withFloat(stateQ) { qp in withFloat(stateK) { kp in
            tc_quantum_attention_score(qp, kp, Int32(nQubits))
        }}
    }
    public static func entanglementEntropy(state: [Float], nQubits: Int,
                                            qubitKeep: Int) -> Float {
        return withFloat(state) { sp in
            tc_quantum_entanglement_entropy(sp, Int32(nQubits), Int32(qubitKeep))
        }
    }
}

// MARK: - Holonomic

public enum Holonomic {
    /// Compose a sequence of SU(2) generators (flat 3 N floats) into
    /// one loop unitary. Returns 8 floats.
    public static func composeSU2(_ generators: [Float]) throws -> [Float] {
        precondition(generators.count % 3 == 0)
        let n = generators.count / 3
        var U = [Float](repeating: 0, count: 8)
        let rc = withFloat(generators) { gp in withFloatMut(&U) { up in
            tc_holonomic_compose_su2(gp, Int32(n), up)
        }}
        if rc != 0 {
            throw TensorCoreError.opFailed(status: Int(rc), message: "tc_holonomic_compose_su2")
        }
        return U
    }

    public static func berryPhase(_ U: [Float]) -> Float {
        var phase: Float = 0
        withFloat(U) { up in tc_holonomic_berry_phase(up, &phase) }
        return phase
    }
}
