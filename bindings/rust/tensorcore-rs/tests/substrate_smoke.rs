//! Integration smoke for the tensorcore Rust crate.
//!
//! Exercises every wrapped op family against libtensorcore.{dylib,so}.
//! Mirrors the Python substrate smoke in
//! scripts/check_python_substrate_runtime.py.

use std::f32::consts::PI;

#[test]
fn lorentz_roundtrip() {
    let mut base = vec![0f32; 8];
    base[0] = 1.0;
    let v = vec![0.0, 0.1, -0.2, 0.15, 0.05, 0.0, 0.0, 0.0];
    let q = tensorcore::lorentz::exp(&base, &v, 1.0);
    let v2 = tensorcore::lorentz::log(&base, &q, 1.0);
    let err: f32 = v.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(err < 1e-4, "lorentz roundtrip err {} too large", err);
    let d = tensorcore::lorentz::distance(&base, &q, 1.0);
    assert!(d > 0.0, "lorentz distance to non-origin should be > 0");
}

#[test]
fn sphere_roundtrip_and_slerp() {
    let mut base = vec![0f32; 16];
    base[0] = 1.0;
    let mut v = vec![0f32; 16];
    v[1] = 0.1; v[2] = 0.2; v[3] = -0.15;
    let q = tensorcore::sphere::exp(&base, &v, 1.0);
    let v2 = tensorcore::sphere::log(&base, &q, 1.0);
    let err: f32 = v.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(err < 1e-4, "sphere roundtrip err {}", err);
    let mid = tensorcore::sphere::slerp(&base, &q, 0.5, 1.0);
    let d_mid = tensorcore::sphere::distance(&base, &mid, 1.0);
    let d_full = tensorcore::sphere::distance(&base, &q, 1.0);
    assert!((2.0 * d_mid - d_full).abs() < 1e-3,
            "slerp midpoint should halve the geodesic: 2·{}≠{}", d_mid, d_full);
}

#[test]
fn torus_roundtrip() {
    let base = vec![0.0f32, 0.0, 0.0];
    let v = vec![0.4f32, -0.3, 0.2];
    let q = tensorcore::torus::exp(&base, &v, 1.0);
    let v2 = tensorcore::torus::log(&base, &q, 1.0);
    let err: f32 = v.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(err < 1e-4, "torus roundtrip err {}", err);
}

#[test]
fn su2_exp_log_roundtrip() {
    let (a, b, c) = (0.1f32, -0.2, PI / 4.0);
    let u = tensorcore::lie::su2_exp(a, b, c);
    let (a2, b2, c2) = tensorcore::lie::su2_log(&u);
    let err = (a - a2).abs().max((b - b2).abs()).max((c - c2).abs());
    assert!(err < 1e-4, "su2 log err {}: ({},{},{}) -> ({},{},{})", err, a, b, c, a2, b2, c2);
}

#[test]
fn so3_z_quarter_turn() {
    let r = tensorcore::lie::so3_exp(0.0, 0.0, PI / 2.0);
    // R should be the 90° z-rotation: [[0,-1,0],[1,0,0],[0,0,1]]
    assert!((r[0] - 0.0).abs() < 1e-3); assert!((r[1] - (-1.0)).abs() < 1e-3);
    assert!((r[3] - 1.0).abs() < 1e-3); assert!((r[4] - 0.0).abs() < 1e-3);
    assert!((r[8] - 1.0).abs() < 1e-3);
    let (wx, wy, wz) = tensorcore::lie::so3_log(&r);
    assert!(wx.abs() < 1e-3 && wy.abs() < 1e-3);
    assert!((wz - PI / 2.0).abs() < 1e-3, "so3 log z err: {} vs π/2", wz);
}

#[test]
fn su2_double_cover_to_so3() {
    // SU(2) z-rotation by θ → SO(3) z-rotation by 2θ (double cover).
    let u = tensorcore::lie::su2_exp(0.0, 0.0, PI / 4.0);
    let r = tensorcore::lie::su2_to_so3(&u);
    // Determinant ≈ +1, R Rᵀ ≈ I.
    let det = r[0]*(r[4]*r[8] - r[5]*r[7])
             - r[1]*(r[3]*r[8] - r[5]*r[6])
             + r[2]*(r[3]*r[7] - r[4]*r[6]);
    assert!((det - 1.0).abs() < 1e-3, "su2→so3 det {} ≠ 1", det);
}

#[test]
fn quantum_h_on_zero_prob() {
    let mut state = tensorcore::quantum::state_zero(2);
    let h = tensorcore::quantum::gate_1q(4); // H
    tensorcore::quantum::apply_1q(&mut state, 2, 0, &h);
    let p = tensorcore::quantum::prob_one(&state, 2, 0);
    assert!((p - 0.5).abs() < 1e-5, "H on |0⟩ should give prob_one=0.5, got {}", p);
    let n = tensorcore::quantum::norm_sq(&state, 2);
    assert!((n - 1.0).abs() < 1e-5, "norm² after H should be 1, got {}", n);
}

#[test]
fn metric_euclidean_identity_and_inverse() {
    let m = tensorcore::metric::euclidean();
    let p = vec![0.5f32, 0.5, 0.5];
    let v = vec![1f32, 2.0, 3.0];
    let w = vec![4f32, 5.0, 6.0];
    let g_vw = unsafe { tensorcore::metric::apply(m, std::ptr::null_mut(), &p, &v, &w) };
    // Euclidean → g_vw = v·w
    let expect = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;
    assert!((g_vw - expect).abs() < 1e-4, "g_vw {} ≠ {}", g_vw, expect);

    // Inverse of identity is identity
    let id: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let inv = tensorcore::metric::inverse(&id, 3).expect("identity inverse");
    let err: f32 = id.iter().zip(inv.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(err < 1e-5);
}

#[test]
fn geodesic_euclidean_straight_line() {
    let m = tensorcore::metric::euclidean();
    let pos0 = vec![0.0f32; 3];
    let vel0 = vec![1f32, 0.0, 0.0];
    let (p, v) = unsafe {
        tensorcore::geodesic::integrate(m, std::ptr::null_mut(),
                                          &pos0, &vel0, 0.01, 100, 1e-3)
            .expect("integrate")
    };
    // T = 1.0 → position should be ≈ (1, 0, 0), velocity unchanged.
    assert!((p[0] - 1.0).abs() < 1e-3 && p[1].abs() < 1e-3 && p[2].abs() < 1e-3,
            "geodesic pos {:?}", p);
    assert!((v[0] - 1.0).abs() < 1e-3 && v[1].abs() < 1e-3 && v[2].abs() < 1e-3,
            "geodesic vel {:?}", v);
}

#[test]
fn holonomic_single_segment_phase() {
    let u = tensorcore::holonomic::compose_su2(&[[0.0, 0.0, PI / 4.0]]).unwrap();
    let phase = tensorcore::holonomic::berry_phase(&u);
    assert!((phase - PI / 4.0).abs() < 1e-4,
            "Berry phase {} ≠ π/4 = {}", phase, PI / 4.0);

    // Closed loop π/4 then -π/4 → phase ≈ 0.
    let u2 = tensorcore::holonomic::compose_su2(&[[0.0, 0.0, PI / 4.0],
                                                    [0.0, 0.0, -PI / 4.0]]).unwrap();
    let phase2 = tensorcore::holonomic::berry_phase(&u2);
    assert!(phase2.abs() < 1e-3, "closed loop phase {} ≠ 0", phase2);
}

#[test]
fn context_lifecycle() {
    let ctx = tensorcore::Context::new().expect("tc_init");
    assert!(!ctx.raw().is_null());
    // Drop runs tc_shutdown automatically.
}

#[test]
fn version_is_non_empty() {
    let v = tensorcore::version();
    assert!(!v.is_empty(), "version() returned empty string");
}
