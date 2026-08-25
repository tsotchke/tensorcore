// TensorCoreTests.swift — XCTest mirror of the Python + Rust
// substrate smokes. Same probes, same expected numbers; we're
// proving the Swift bridge is correct, not the underlying math.

import XCTest
@testable import TensorCore

final class TensorCoreTests: XCTestCase {

    func testVersionNonEmpty() {
        XCTAssertFalse(tensorcoreVersion().isEmpty)
    }

    func testContextLifecycle() throws {
        // tc_init -> tc_shutdown via Context's deinit.
        _ = try Context()
    }

    func testLorentzRoundtrip() {
        var base = [Float](repeating: 0, count: 8); base[0] = 1.0
        let tangent: [Float] = [0, 0.1, -0.2, 0.15, 0.05, 0, 0, 0]
        let point = Lorentz.exp(base: base, tangent: tangent)
        let recovered = Lorentz.log(base: base, point: point)
        let err = zip(tangent, recovered).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThan(err, 1e-4, "Lorentz roundtrip err \(err) too large")
        XCTAssertGreaterThan(Lorentz.distance(base, point), 0)
    }

    func testSphereSlerpHalvesGeodesic() {
        var base = [Float](repeating: 0, count: 16); base[0] = 1.0
        var tangent = [Float](repeating: 0, count: 16)
        tangent[1] = 0.1; tangent[2] = 0.2; tangent[3] = -0.15
        let point = Sphere.exp(base: base, tangent: tangent)
        let mid = Sphere.slerp(base, point, t: 0.5)
        let dMid = Sphere.distance(base, mid)
        let dFull = Sphere.distance(base, point)
        XCTAssertLessThan(abs(2 * dMid - dFull), 1e-3,
                          "slerp(t=0.5) should halve the geodesic: 2·\(dMid) ≠ \(dFull)")
    }

    func testTorusRoundtrip() {
        let base: [Float] = [0, 0, 0]
        let tangent: [Float] = [0.4, -0.3, 0.2]
        let point = Torus.exp(base: base, tangent: tangent)
        let recovered = Torus.log(base: base, point: point)
        let err = zip(tangent, recovered).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThan(err, 1e-4)
    }

    func testSU2ExpLog() {
        let aIn: Float = 0.1, bIn: Float = -0.2, cIn: Float = .pi / 4
        let U = Lie.su2Exp(aIn, bIn, cIn)
        let (a, b, c) = Lie.su2Log(U)
        XCTAssertLessThan(abs(a - aIn), 1e-4)
        XCTAssertLessThan(abs(b - bIn), 1e-4)
        XCTAssertLessThan(abs(c - cIn), 1e-4)
    }

    func testSO3QuarterTurn() {
        let R = Lie.so3Exp(0, 0, .pi / 2)
        // R = [[0,-1,0],[1,0,0],[0,0,1]]
        XCTAssertLessThan(abs(R[1] - (-1)), 1e-3)
        XCTAssertLessThan(abs(R[3] - 1), 1e-3)
        XCTAssertLessThan(abs(R[8] - 1), 1e-3)
        let (wx, wy, wz) = Lie.so3Log(R)
        XCTAssertLessThan(abs(wx), 1e-3)
        XCTAssertLessThan(abs(wy), 1e-3)
        XCTAssertLessThan(abs(wz - .pi / 2), 1e-3)
    }

    func testHadamardProb() {
        var state = Quantum.stateZero(2)
        let H = Quantum.gate1q(4)  // tc_gate_type_t H = 4
        Quantum.apply1q(state: &state, nQubits: 2, target: 0, gate: H)
        let p = Quantum.probOne(state: state, nQubits: 2, qubit: 0)
        XCTAssertLessThan(abs(p - 0.5), 1e-5)
        let n = Quantum.normSq(state: state, nQubits: 2)
        XCTAssertLessThan(abs(n - 1), 1e-5)
    }

    func testHolonomicSingleSegmentPhase() throws {
        let U = try Holonomic.composeSU2([0, 0, .pi / 4])
        let phase = Holonomic.berryPhase(U)
        XCTAssertLessThan(abs(phase - .pi / 4), 1e-4)
    }

    func testHolonomicClosedLoopPhaseZero() throws {
        let U = try Holonomic.composeSU2([0, 0, .pi / 4,
                                            0, 0, -.pi / 4])
        let phase = Holonomic.berryPhase(U)
        XCTAssertLessThan(abs(phase), 1e-3)
    }

    func testQuantumAttentionOverlap() {
        let state = Quantum.stateZero(2)
        var other = state; other[0] = 0; other[2] = 1  // |10⟩
        let same = Quantum.attentionScore(state, state, nQubits: 2)
        let orth = Quantum.attentionScore(state, other, nQubits: 2)
        XCTAssertLessThan(abs(same - 1), 1e-5)
        XCTAssertLessThan(abs(orth), 1e-5)
    }

    func testBellEntanglementEntropy() {
        // |Φ+⟩ = (|00⟩ + |11⟩) / √2
        var bell = [Float](repeating: 0, count: 8)
        bell[0] = Float(1.0 / 2.0.squareRoot())
        bell[6] = bell[0]
        let s = Quantum.entanglementEntropy(state: bell, nQubits: 2, qubitKeep: 0)
        XCTAssertLessThan(abs(s - 1), 1e-4, "Bell entanglement should be 1 bit, got \(s)")
    }
}
