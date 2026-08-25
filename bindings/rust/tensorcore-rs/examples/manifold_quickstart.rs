//! Quick demo of the tensorcore Rust crate.
//!
//! Run with the dylib visible at runtime:
//!
//!     cargo run --example manifold_quickstart
//!
//! `build.rs` finds the dylib via $TENSORCORE_LIB_DIR / $TENSORCORE_LIB
//! or falls back to <repo>/build/.

use std::f32::consts::PI;

fn main() {
    println!("libtensorcore version: {}", tensorcore::version());

    // Sphere geodesic.
    let mut base = vec![0f32; 4];
    base[0] = 1.0;
    let tangent = vec![0.0, 0.3, 0.2, 0.0];
    let point = tensorcore::sphere::exp(&base, &tangent, 1.0);
    println!("sphere exp_map: tangent {:?} -> point {:?}", tangent, point);
    println!("sphere distance(base, point): {}",
             tensorcore::sphere::distance(&base, &point, 1.0));

    // SU(2) holonomy.
    let u = tensorcore::holonomic::compose_su2(
        &[[0.0, 0.0, PI / 4.0],
          [0.0, 0.0, PI / 4.0]]   // total π/2 around z
    ).unwrap();
    let phase = tensorcore::holonomic::berry_phase(&u);
    println!("Berry phase for total π/2 z-rotation: {}", phase);

    // Quantum: Bell state preparation
    let mut state = tensorcore::quantum::state_zero(2);
    let h = tensorcore::quantum::gate_1q(4);     // H
    let cnot = tensorcore::quantum::gate_2q(10); // CNOT (tc_gate_type_t = 10)
    tensorcore::quantum::apply_1q(&mut state, 2, 0, &h);
    tensorcore::quantum::apply_2q(&mut state, 2, 0, 1, &cnot);
    println!("Bell |Φ+⟩ amplitudes (interleaved complex):");
    for i in 0..4 {
        println!("  |{:02b}⟩ = {} + {}i", i, state[2*i], state[2*i + 1]);
    }
}
