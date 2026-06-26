//! Raw extern "C" prototypes mirroring the public symbols in
//! libtensorcore.{dylib,so}. Hand-written from include/tensorcore/*.h
//! rather than generated, so the wire ABI stays explicit + reviewable.

use libc::{c_char, c_float, c_int, c_void, size_t};

pub const TC_OK: c_int = 0;
pub const TC_ERR_INVALID_ARG: c_int = -7;
pub const TC_ERR_INTERNAL: c_int = -99;

// ---------- Lifecycle ----------
extern "C" {
    pub fn tc_init(out_ctx: *mut *mut c_void) -> c_int;
    pub fn tc_shutdown(ctx: *mut c_void) -> c_int;
    pub fn tc_status_string(status: c_int) -> *const c_char;
    pub fn tc_version() -> *const c_char;
}

// ---------- Lorentz / hyperboloid ----------
extern "C" {
    pub fn tc_lorentz_exp(
        base: *const c_float,
        tangent: *const c_float,
        point: *mut c_float,
        n: size_t,
        curvature: c_float,
    );
    pub fn tc_lorentz_log(
        base: *const c_float,
        point: *const c_float,
        tangent: *mut c_float,
        n: size_t,
        curvature: c_float,
    );
    pub fn tc_lorentz_distance(
        p: *const c_float,
        q: *const c_float,
        n: size_t,
        curvature: c_float,
    ) -> c_float;
    pub fn tc_lorentz_minkowski(a: *const c_float, b: *const c_float, n: size_t) -> c_float;
}

// ---------- Sphere ----------
extern "C" {
    pub fn tc_sphere_exp(
        base: *const c_float,
        tangent: *const c_float,
        point: *mut c_float,
        n: size_t,
        radius: c_float,
    );
    pub fn tc_sphere_log(
        base: *const c_float,
        point: *const c_float,
        tangent: *mut c_float,
        n: size_t,
        radius: c_float,
    );
    pub fn tc_sphere_distance(
        p: *const c_float,
        q: *const c_float,
        n: size_t,
        radius: c_float,
    ) -> c_float;
    pub fn tc_sphere_slerp(
        p: *const c_float,
        q: *const c_float,
        t: c_float,
        out: *mut c_float,
        n: size_t,
        radius: c_float,
    );
}

// ---------- Torus ----------
extern "C" {
    pub fn tc_torus_exp(
        base: *const c_float,
        tangent: *const c_float,
        point: *mut c_float,
        n: size_t,
        radius: c_float,
    );
    pub fn tc_torus_log(
        base: *const c_float,
        point: *const c_float,
        tangent: *mut c_float,
        n: size_t,
        radius: c_float,
    );
    pub fn tc_torus_distance(
        p: *const c_float,
        q: *const c_float,
        n: size_t,
        radius: c_float,
    ) -> c_float;
}

// ---------- Lie groups (SU(2) / SO(3)) ----------
extern "C" {
    pub fn tc_su2_exp(a: c_float, b: c_float, c: c_float, u: *mut c_float);
    pub fn tc_su2_log(
        u: *const c_float,
        out_a: *mut c_float,
        out_b: *mut c_float,
        out_c: *mut c_float,
    );
    pub fn tc_su2_mul(u: *const c_float, v: *const c_float, out: *mut c_float);
    pub fn tc_so3_exp(wx: c_float, wy: c_float, wz: c_float, r: *mut c_float);
    pub fn tc_so3_log(
        r: *const c_float,
        out_wx: *mut c_float,
        out_wy: *mut c_float,
        out_wz: *mut c_float,
    );
    pub fn tc_su2_to_so3(u: *const c_float, r: *mut c_float);
}

// ---------- Quantum state-vector + gates ----------
extern "C" {
    pub fn tc_qstate_zero(state: *mut c_float, n_qubits: c_int);
    pub fn tc_qstate_apply_1q_unitary(
        state: *mut c_float,
        n_qubits: c_int,
        target: c_int,
        gate: *const c_float,
    );
    pub fn tc_qstate_apply_2q_unitary(
        state: *mut c_float,
        n_qubits: c_int,
        qa: c_int,
        qb: c_int,
        gate: *const c_float,
    );
    pub fn tc_qstate_prob_one(state: *const c_float, n_qubits: c_int, qubit: c_int) -> c_float;
    pub fn tc_qstate_norm_sq(state: *const c_float, n_qubits: c_int) -> c_float;
    pub fn tc_gate_matrix_1q(gate_type: c_int, params: *const c_float, out: *mut c_float);
    pub fn tc_gate_matrix_2q(gate_type: c_int, params: *const c_float, out: *mut c_float);
}

// ---------- Riemannian metric tensor ----------
pub type TcMetricFn = unsafe extern "C" fn(
    point: *const c_float,
    dim: c_int,
    out_g: *mut c_float,
    user: *mut c_void,
);

extern "C" {
    pub fn tc_metric_apply(
        fn_: TcMetricFn,
        user: *mut c_void,
        point: *const c_float,
        dim: c_int,
        v: *const c_float,
        w: *const c_float,
    ) -> c_float;
    pub fn tc_metric_inverse(g: *const c_float, dim: c_int, g_inv: *mut c_float) -> c_int;
    pub fn tc_metric_christoffel(
        fn_: TcMetricFn,
        user: *mut c_void,
        point: *const c_float,
        dim: c_int,
        h: c_float,
        out_christoffel: *mut c_float,
    ) -> c_int;
    pub fn tc_metric_euclidean(point: *const c_float, dim: c_int, out_g: *mut c_float, user: *mut c_void);
    pub fn tc_metric_poincare(point: *const c_float, dim: c_int, out_g: *mut c_float, user: *mut c_void);
    pub fn tc_metric_sphere_stereographic(
        point: *const c_float,
        dim: c_int,
        out_g: *mut c_float,
        user: *mut c_void,
    );
}

// ---------- Geodesic ODE solver ----------
extern "C" {
    pub fn tc_geodesic_integrate(
        fn_: TcMetricFn,
        user: *mut c_void,
        dim: c_int,
        dt: c_float,
        n_steps: c_int,
        h_christoffel: c_float,
        pos0: *const c_float,
        vel0: *const c_float,
        pos_out: *mut c_float,
        vel_out: *mut c_float,
    ) -> c_int;
}

// ---------- Holonomic gates ----------
extern "C" {
    pub fn tc_holonomic_compose_su2(
        generators: *const c_float,
        n_segments: i32,
        out_u: *mut c_float,
    ) -> c_int;
    pub fn tc_holonomic_berry_phase(u: *const c_float, out_phase: *mut c_float);
}
