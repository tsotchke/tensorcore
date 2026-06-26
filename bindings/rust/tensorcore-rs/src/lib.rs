//! Safe Rust bindings to the tensorcore unified math substrate.
//!
//! Wraps libtensorcore.{dylib,so} — the C/Eshkol substrate that
//! powers qLLM, Noesis, tsotchke-chan, QGTL, and moonlab — so Rust
//! callers can use the geometric / quantum / mesh primitives without
//! hand-rolling FFI.
//!
//! Inputs are `&[f32]`, outputs are `Vec<f32>` (or scalars for
//! distance/norm queries). All math runs through the production
//! libtensorcore code path — same numerics as the Python and Eshkol
//! bindings.
//!
//! # Linking
//!
//! By default `build.rs` looks for `libtensorcore.{dylib,so}` in:
//! 1. `$TENSORCORE_LIB_DIR`
//! 2. `$TENSORCORE_LIB`'s parent dir
//! 3. `<repo>/build/`
//!
//! If you're consuming the crate outside the tensorcore repo, set
//! `TENSORCORE_LIB_DIR` to wherever the dylib lives.
//!
//! # Example
//!
//! ```no_run
//! use tensorcore::sphere;
//!
//! let base = vec![1.0f32, 0.0, 0.0, 0.0];
//! let tangent = vec![0.0f32, 0.1, 0.2, 0.0];
//! let point = sphere::exp(&base, &tangent, 1.0);
//! let recovered = sphere::log(&base, &point, 1.0);
//! // |tangent - recovered| < 1e-4
//! ```

pub mod ffi;

use libc::{c_float, c_int, c_void};
use std::ffi::CStr;

// ---- Status / lifecycle ----

/// Tensorcore status code (0 == OK).
pub type Status = i32;

/// Owning handle to a `tc_context*`. Drops via `tc_shutdown`.
pub struct Context(*mut c_void);

impl Context {
    pub fn new() -> Result<Self, Status> {
        let mut p: *mut c_void = std::ptr::null_mut();
        let s = unsafe { ffi::tc_init(&mut p) };
        if s == ffi::TC_OK { Ok(Context(p)) } else { Err(s) }
    }

    pub fn raw(&self) -> *mut c_void { self.0 }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::tc_shutdown(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// Human-readable string for a tensorcore status code.
pub fn status_string(status: Status) -> String {
    let p = unsafe { ffi::tc_status_string(status as c_int) };
    if p.is_null() { return format!("status {}", status); }
    unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// libtensorcore version string.
pub fn version() -> String {
    let p = unsafe { ffi::tc_version() };
    if p.is_null() { return "(unknown)".to_string(); }
    unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
}

// ---- Lorentz / hyperboloid ----

pub mod lorentz {
    use super::*;

    pub fn exp(base: &[f32], tangent: &[f32], curvature: f32) -> Vec<f32> {
        assert_eq!(base.len(), tangent.len(), "lorentz::exp: base/tangent length mismatch");
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_lorentz_exp(base.as_ptr(), tangent.as_ptr(), out.as_mut_ptr(),
                                  n, curvature);
        }
        out
    }

    pub fn log(base: &[f32], point: &[f32], curvature: f32) -> Vec<f32> {
        assert_eq!(base.len(), point.len(), "lorentz::log: base/point length mismatch");
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_lorentz_log(base.as_ptr(), point.as_ptr(), out.as_mut_ptr(),
                                  n, curvature);
        }
        out
    }

    pub fn distance(p: &[f32], q: &[f32], curvature: f32) -> f32 {
        assert_eq!(p.len(), q.len(), "lorentz::distance: length mismatch");
        unsafe { ffi::tc_lorentz_distance(p.as_ptr(), q.as_ptr(), p.len(), curvature) }
    }

    pub fn minkowski(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "lorentz::minkowski: length mismatch");
        unsafe { ffi::tc_lorentz_minkowski(a.as_ptr(), b.as_ptr(), a.len()) }
    }
}

// ---- Sphere ----

pub mod sphere {
    use super::*;

    pub fn exp(base: &[f32], tangent: &[f32], radius: f32) -> Vec<f32> {
        assert_eq!(base.len(), tangent.len(), "sphere::exp: base/tangent length mismatch");
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_sphere_exp(base.as_ptr(), tangent.as_ptr(), out.as_mut_ptr(),
                                 n, radius);
        }
        out
    }

    pub fn log(base: &[f32], point: &[f32], radius: f32) -> Vec<f32> {
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_sphere_log(base.as_ptr(), point.as_ptr(), out.as_mut_ptr(),
                                 n, radius);
        }
        out
    }

    pub fn distance(p: &[f32], q: &[f32], radius: f32) -> f32 {
        unsafe { ffi::tc_sphere_distance(p.as_ptr(), q.as_ptr(), p.len(), radius) }
    }

    pub fn slerp(p: &[f32], q: &[f32], t: f32, radius: f32) -> Vec<f32> {
        let n = p.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_sphere_slerp(p.as_ptr(), q.as_ptr(), t, out.as_mut_ptr(), n, radius);
        }
        out
    }
}

// ---- Torus ----

pub mod torus {
    use super::*;

    pub fn exp(base: &[f32], tangent: &[f32], radius: f32) -> Vec<f32> {
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_torus_exp(base.as_ptr(), tangent.as_ptr(), out.as_mut_ptr(),
                                n, radius);
        }
        out
    }

    pub fn log(base: &[f32], point: &[f32], radius: f32) -> Vec<f32> {
        let n = base.len();
        let mut out = vec![0f32; n];
        unsafe {
            ffi::tc_torus_log(base.as_ptr(), point.as_ptr(), out.as_mut_ptr(),
                                n, radius);
        }
        out
    }

    pub fn distance(p: &[f32], q: &[f32], radius: f32) -> f32 {
        unsafe { ffi::tc_torus_distance(p.as_ptr(), q.as_ptr(), p.len(), radius) }
    }
}

// ---- Lie groups ----

pub mod lie {
    use super::*;

    /// SU(2) exp: exp(i (a σ_x + b σ_y + c σ_z)). Returns 8 floats
    /// (row-major interleaved complex 2×2).
    pub fn su2_exp(a: f32, b: f32, c: f32) -> [f32; 8] {
        let mut u = [0f32; 8];
        unsafe { ffi::tc_su2_exp(a, b, c, u.as_mut_ptr()); }
        u
    }

    /// SU(2) log → (a, b, c) algebra vector, principal branch θ ∈ [0, π].
    pub fn su2_log(u: &[f32; 8]) -> (f32, f32, f32) {
        let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
        unsafe { ffi::tc_su2_log(u.as_ptr(), &mut a, &mut b, &mut c); }
        (a, b, c)
    }

    pub fn su2_mul(u: &[f32; 8], v: &[f32; 8]) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { ffi::tc_su2_mul(u.as_ptr(), v.as_ptr(), out.as_mut_ptr()); }
        out
    }

    /// Rodrigues rotation R = exp(ω×). Returns the 3×3 matrix
    /// as 9 floats in row-major order (R[3*i + j] = R_{ij}).
    pub fn so3_exp(wx: f32, wy: f32, wz: f32) -> [f32; 9] {
        let mut r = [0f32; 9];
        unsafe { ffi::tc_so3_exp(wx, wy, wz, r.as_mut_ptr()); }
        r
    }

    pub fn so3_log(r: &[f32; 9]) -> (f32, f32, f32) {
        let (mut wx, mut wy, mut wz) = (0f32, 0f32, 0f32);
        unsafe { ffi::tc_so3_log(r.as_ptr(), &mut wx, &mut wy, &mut wz); }
        (wx, wy, wz)
    }

    /// SU(2) → SO(3) via the double-cover adjoint representation.
    pub fn su2_to_so3(u: &[f32; 8]) -> [f32; 9] {
        let mut r = [0f32; 9];
        unsafe { ffi::tc_su2_to_so3(u.as_ptr(), r.as_mut_ptr()); }
        r
    }
}

// ---- Quantum state-vector + gates ----

pub mod quantum {
    use super::*;

    /// |0...0⟩ on `n_qubits`. State length = 2 * 2^n_qubits floats
    /// (interleaved complex).
    pub fn state_zero(n_qubits: u32) -> Vec<f32> {
        let n = n_qubits as usize;
        let mut state = vec![0f32; 2 * (1 << n)];
        unsafe { ffi::tc_qstate_zero(state.as_mut_ptr(), n as c_int); }
        state
    }

    /// In-place: apply 1-qubit gate (8-float unitary) to `state` at `target`.
    pub fn apply_1q(state: &mut [f32], n_qubits: u32, target: u32, gate: &[f32; 8]) {
        unsafe {
            ffi::tc_qstate_apply_1q_unitary(state.as_mut_ptr(), n_qubits as c_int,
                                              target as c_int, gate.as_ptr());
        }
    }

    /// In-place: apply 2-qubit gate (32-float unitary).
    pub fn apply_2q(state: &mut [f32], n_qubits: u32, qa: u32, qb: u32, gate: &[f32]) {
        assert_eq!(gate.len(), 32, "quantum::apply_2q: 2-qubit gate must be 32 floats");
        unsafe {
            ffi::tc_qstate_apply_2q_unitary(state.as_mut_ptr(), n_qubits as c_int,
                                              qa as c_int, qb as c_int, gate.as_ptr());
        }
    }

    pub fn prob_one(state: &[f32], n_qubits: u32, qubit: u32) -> f32 {
        unsafe { ffi::tc_qstate_prob_one(state.as_ptr(), n_qubits as c_int, qubit as c_int) }
    }

    pub fn norm_sq(state: &[f32], n_qubits: u32) -> f32 {
        unsafe { ffi::tc_qstate_norm_sq(state.as_ptr(), n_qubits as c_int) }
    }

    /// 1-qubit gate enum (matches tc_gate_type_t — see
    /// include/tensorcore/quantum_gates.h for the full table): X=1,
    /// Y=2, Z=3, H=4, S=5, T=6, RX=7, RY=8, RZ=9.
    pub fn gate_1q(gate_type: i32) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { ffi::tc_gate_matrix_1q(gate_type as c_int, std::ptr::null(), out.as_mut_ptr()); }
        out
    }

    /// 2-qubit gate enum (matches tc_gate_type_t): CNOT=10, CY=11, CZ=12, SWAP=13.
    pub fn gate_2q(gate_type: i32) -> [f32; 32] {
        let mut out = [0f32; 32];
        unsafe { ffi::tc_gate_matrix_2q(gate_type as c_int, std::ptr::null(), out.as_mut_ptr()); }
        out
    }
}

// ---- Metric / Geodesic / Holonomic (Phase 4) ----

pub mod metric {
    use super::*;

    /// A `tc_metric_fn` callback the C side can call back to evaluate
    /// the metric at a point.
    pub type Fn_ = ffi::TcMetricFn;

    /// `g_{ij}(point) v^i w^j` for the given metric callback.
    /// SAFETY: caller guarantees `fn_`/`user` form a valid (function,
    /// user-data) pair acceptable to libtensorcore.
    pub unsafe fn apply(fn_: Fn_, user: *mut c_void, point: &[f32], v: &[f32], w: &[f32]) -> f32 {
        ffi::tc_metric_apply(fn_, user, point.as_ptr(), point.len() as c_int,
                              v.as_ptr(), w.as_ptr())
    }

    /// Invert a `d×d` symmetric positive-definite matrix.
    /// Returns `Ok(g_inv)` (length d²) or `Err(status)`.
    pub fn inverse(g: &[f32], dim: usize) -> Result<Vec<f32>, i32> {
        let mut out = vec![0f32; dim * dim];
        let rc = unsafe { ffi::tc_metric_inverse(g.as_ptr(), dim as c_int, out.as_mut_ptr()) };
        if rc == 0 { Ok(out) } else { Err(rc) }
    }

    /// Numerical Christoffel symbols. Returns d³ floats in
    /// `out[k * d * d + i * d + j] = Γ^k_{ij}` layout, or `Err` if
    /// the metric is singular at `point`.
    /// SAFETY: see `apply`.
    pub unsafe fn christoffel(
        fn_: Fn_,
        user: *mut c_void,
        point: &[f32],
        h: f32,
    ) -> Result<Vec<f32>, i32> {
        let d = point.len();
        let mut out = vec![0f32; d * d * d];
        let rc = ffi::tc_metric_christoffel(fn_, user, point.as_ptr(), d as c_int,
                                              h, out.as_mut_ptr());
        if rc == 0 { Ok(out) } else { Err(rc) }
    }

    /// Stock Euclidean metric callback.
    pub fn euclidean() -> Fn_ { ffi::tc_metric_euclidean }
    /// Stock Poincaré ball metric callback. `user` should point at a
    /// `f32` curvature held by the caller.
    pub fn poincare_fn() -> Fn_ { ffi::tc_metric_poincare }
    /// Stock sphere stereographic metric callback. `user` should point
    /// at a `f32` radius held by the caller.
    pub fn sphere_stereographic_fn() -> Fn_ { ffi::tc_metric_sphere_stereographic }
}

pub mod geodesic {
    use super::*;

    /// RK4 integrate `n_steps` of length `dt` from `(pos0, vel0)`.
    /// Returns `(pos_final, vel_final)` or `Err(status)` if the metric
    /// is singular at some substep.
    /// SAFETY: caller guarantees `fn_`/`user` form a valid metric pair.
    pub unsafe fn integrate(
        fn_: ffi::TcMetricFn,
        user: *mut c_void,
        pos0: &[f32],
        vel0: &[f32],
        dt: f32,
        n_steps: u32,
        h_christoffel: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), i32> {
        let d = pos0.len();
        assert_eq!(d, vel0.len(), "geodesic::integrate: pos/vel length mismatch");
        let mut pos_out = vec![0f32; d];
        let mut vel_out = vec![0f32; d];
        let rc = ffi::tc_geodesic_integrate(
            fn_, user, d as c_int, dt, n_steps as c_int, h_christoffel,
            pos0.as_ptr(), vel0.as_ptr(),
            pos_out.as_mut_ptr(), vel_out.as_mut_ptr(),
        );
        if rc == 0 { Ok((pos_out, vel_out)) } else { Err(rc) }
    }
}

pub mod holonomic {
    use super::*;

    /// Compose a sequence of SU(2) generators (a, b, c per row) into a
    /// single loop unitary. Returns the 8-float unitary or `Err(status)`.
    pub fn compose_su2(generators: &[[f32; 3]]) -> Result<[f32; 8], i32> {
        let n = generators.len() as i32;
        // Generators are stored as a contiguous [N][3] array — &[[f32; 3]]
        // is layout-compatible with [f32; 3N] thanks to repr(transparent)
        // on inner arrays + no padding.
        let flat: &[f32] = unsafe {
            std::slice::from_raw_parts(generators.as_ptr() as *const c_float,
                                         generators.len() * 3)
        };
        let mut u = [0f32; 8];
        let rc = unsafe {
            ffi::tc_holonomic_compose_su2(flat.as_ptr(), n, u.as_mut_ptr())
        };
        if rc == 0 { Ok(u) } else { Err(rc) }
    }

    /// Extract the signed SU(2) Berry phase (rotation angle) from a
    /// composed loop unitary. Returns a phase in (-π, π].
    pub fn berry_phase(u: &[f32; 8]) -> f32 {
        let mut out = 0f32;
        unsafe { ffi::tc_holonomic_berry_phase(u.as_ptr(), &mut out); }
        out
    }
}
