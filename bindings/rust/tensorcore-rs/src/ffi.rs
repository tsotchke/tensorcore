//! Raw extern "C" prototypes used by this crate's selected safe substrate
//! wrappers. The complete supported ABI is the installed C header set; these
//! declarations are hand-written rather than generated so their wire layout
//! stays explicit and reviewable.

use libc::{c_char, c_float, c_int, c_void, size_t};

pub const TC_OK: c_int = 0;
pub const TC_ERR_INVALID_ARG: c_int = -7;
pub const TC_ERR_ABI_MISMATCH: c_int = -12;
pub const TC_ERR_BUSY: c_int = -13;
pub const TC_ERR_AUTH: c_int = -14;
pub const TC_ERR_INTERNAL: c_int = -99;

pub const TC_RUNTIME_CAPABILITIES_ABI_VERSION_CURRENT: u32 = 1;
pub const TC_CAPABILITY_GEMM_F32: u64 = 1 << 0;
pub const TC_CAPABILITY_DISTRIBUTED_SINGLE: u64 = 1 << 1;
pub const TC_CAPABILITY_DISTRIBUTED_GLOO: u64 = 1 << 2;
pub const TC_CAPABILITY_DILOCO: u64 = 1 << 3;
pub const TC_CAPABILITY_DILOCO_SPARSE_GLOO: u64 = 1 << 4;
pub const TC_CAPABILITY_REMOTE_TENSOR: u64 = 1 << 5;
pub const TC_CAPABILITY_DILOCO_ASYNC_SNAPSHOT_SAFE: u64 = 1 << 6;
pub const TC_CAPABILITY_DILOCO_CHECKPOINT_RESUME: u64 = 1 << 7;
pub const TC_CAPABILITY_DILOCO_ELASTIC_MEMBERSHIP: u64 = 1 << 8;
pub const TC_CAPABILITY_DILOCO_FP16_WIRE: u64 = 1 << 9;
pub const TC_CAPABILITY_TRANSPORT_IDENTITY_AUTH: u64 = 1 << 10;
pub const TC_CAPABILITY_DILOCO_CAPABILITY_QUERY: u64 = 1 << 11;
pub const TC_CAPABILITY_DILOCO_STATE_ABI_V2: u64 = 1 << 12;
pub const TC_CAPABILITY_GEMM_F16: u64 = 1 << 13;
pub const TC_CAPABILITY_GEMM_BF16: u64 = 1 << 14;
pub const TC_CAPABILITY_GEMM_I8: u64 = 1 << 15;

pub const TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT: u32 = 1;

pub const TC_DILOCO_COMPRESS_NONE: c_int = 0;
pub const TC_DILOCO_COMPRESS_FP16: c_int = 1;
pub const TC_DILOCO_COMPRESS_FP8: c_int = 2;
pub const TC_DILOCO_COMPRESS_TOPK_1PCT: c_int = 3;
pub const TC_DILOCO_COMPRESS_TOPK_01PCT: c_int = 4;
pub const TC_DILOCO_COMPRESS_LOWRANK: c_int = 5;
pub const TC_DILOCO_COMPRESS_SIGNSGD: c_int = 6;

pub const TC_DILOCO_OUTER_SGD: c_int = 0;
pub const TC_DILOCO_OUTER_NESTEROV: c_int = 1;
pub const TC_DILOCO_OUTER_ADAM: c_int = 2;

pub const TC_DILOCO_ASYNC_IDLE: c_int = 0;
pub const TC_DILOCO_ASYNC_RUNNING: c_int = 1;
pub const TC_DILOCO_ASYNC_READY: c_int = 2;
pub const TC_DILOCO_ASYNC_FAILED: c_int = 3;

pub const TC_DILOCO_STATE_ABI_VERSION_1: u32 = 1;
pub const TC_DILOCO_STATE_ABI_VERSION_2: u32 = 2;
pub const TC_DILOCO_STATE_ABI_VERSION_CURRENT: u32 = TC_DILOCO_STATE_ABI_VERSION_2;
pub const TC_DILOCO_CAPABILITIES_ABI_VERSION_CURRENT: u32 = 1;

pub const TC_DILOCO_STATE_FEATURE_PAYLOAD_SHA256: u64 = 1 << 7;
pub const TC_DILOCO_STATE_FEATURE_DETERMINISTIC_LAYOUT: u64 = 1 << 8;
pub const TC_DILOCO_STATE_FEATURE_ATOMIC_RESTORE: u64 = 1 << 9;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TcRuntimeCapabilities {
    pub struct_size: u32,
    pub abi_version: u32,
    pub runtime_version_major: u32,
    pub runtime_version_minor: u32,
    pub runtime_version_patch: u32,
    pub reserved0: u32,
    pub known_capability_mask: u64,
    pub available_capability_mask: u64,
    pub compiled_backend_mask: u64,
    pub available_backend_mask: u64,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TcDiLoCoConfig {
    pub inner_steps: c_int,
    pub outer_lr: c_float,
    pub outer_momentum: c_float,
    pub outer_beta2: c_float,
    pub outer_eps: c_float,
    pub outer_optimizer: c_int,
    pub compress: c_int,
    pub async_overlap: u8,
    pub tolerate_dropouts: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TcDiLoCoCapabilities {
    pub struct_size: u32,
    pub abi_version: u32,
    pub runtime_version_major: u32,
    pub runtime_version_minor: u32,
    pub runtime_version_patch: u32,
    pub reserved0: u32,
    pub state_abi_version_min: u32,
    pub state_abi_version_max: u32,
    pub state_abi_version_current: u32,
    pub state_header_size: u32,
    pub state_payload_digest_bytes: u32,
    pub reserved1: u32,
    pub outer_optimizer_supported_mask: u64,
    pub outer_optimizer_serializable_mask: u64,
    pub compress_supported_mask: u64,
    pub compress_serializable_mask: u64,
    pub compress_sparse_wire_mask: u64,
    pub compress_dense_fp32_wire_mask: u64,
    pub state_feature_mask: u64,
    pub async_overlap_supported: u32,
    pub async_state_serializable: u32,
    pub tolerate_dropouts_supported: u32,
    pub reserved2: u32,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TcTransportAuthKey {
    pub identity: *const c_char,
    pub key_id: u64,
    pub secret: *const c_void,
    pub secret_bytes: size_t,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TcTransportAuthConfig {
    pub abi_version: u32,
    pub local_identity: *const c_char,
    pub active_key_id: u64,
    pub keys: *const TcTransportAuthKey,
    pub key_count: size_t,
}

// ---------- Lifecycle ----------
extern "C" {
    pub fn tc_init(out_ctx: *mut *mut c_void) -> c_int;
    pub fn tc_shutdown(ctx: *mut c_void) -> c_int;
    pub fn tc_status_string(status: c_int) -> *const c_char;
    pub fn tc_version() -> *const c_char;
    pub fn tc_runtime_capabilities_get(
        ctx: *mut c_void,
        requested_abi_version: u32,
        out: *mut TcRuntimeCapabilities,
        out_size: size_t,
    ) -> c_int;
    pub fn tc_cuda_is_active() -> c_int;
}

// ---------- Authenticated distributed transports ----------
extern "C" {
    pub fn tc_dist_init_authenticated(
        ctx: *mut c_void,
        backend: c_int,
        world_size: c_int,
        rank: c_int,
        rendezvous_url: *const c_char,
        rank_identities: *const *const c_char,
        rank_identity_count: size_t,
        auth: *const TcTransportAuthConfig,
        out_dist: *mut *mut c_void,
    ) -> c_int;
    pub fn tc_remote_init_authenticated(
        ctx: *mut c_void,
        role: c_int,
        bind_url: *const c_char,
        auth: *const TcTransportAuthConfig,
        out_remote: *mut *mut c_void,
    ) -> c_int;
    pub fn tc_remote_connect_authenticated(
        remote: *mut c_void,
        peer_url: *const c_char,
        expected_peer_identity: *const c_char,
        out_peer_id: *mut c_int,
    ) -> c_int;
    pub fn tc_remote_auth_rotate(
        remote: *mut c_void,
        auth: *const TcTransportAuthConfig,
    ) -> c_int;
    pub fn tc_remote_peer_identity(remote: *mut c_void, peer_id: c_int) -> *const c_char;
    pub fn tc_remote_peer_key_id(remote: *mut c_void, peer_id: c_int) -> u64;
    pub fn tc_remote_auth_failure_count(remote: *mut c_void) -> u64;
    pub fn tc_mesh_group_init_authenticated(
        ctx: *mut c_void,
        n_peers: i32,
        my_rank: i32,
        peer_urls: *const *const c_char,
        peer_identities: *const *const c_char,
        auth: *const TcTransportAuthConfig,
        out_group: *mut *mut c_void,
    ) -> c_int;
}

// ---------- DiLoCo ----------
extern "C" {
    pub fn tc_diloco_init(
        dist_ctx: *mut c_void,
        config: *const TcDiLoCoConfig,
        out: *mut *mut c_void,
    ) -> c_int;
    pub fn tc_diloco_finalize(diloco: *mut c_void) -> c_int;
    pub fn tc_diloco_add_parameter(
        diloco: *mut c_void,
        name: *const c_char,
        theta_local: *mut c_void,
        num_elements: size_t,
        dtype: c_int,
    ) -> c_int;
    pub fn tc_diloco_step(diloco: *mut c_void, out_outer_step_pending: *mut bool) -> c_int;
    pub fn tc_diloco_apply_outer(diloco: *mut c_void) -> c_int;
    pub fn tc_diloco_async_poll(
        diloco: *const c_void,
        out_state: *mut c_int,
        out_worker_status: *mut c_int,
        out_round_id: *mut u64,
    ) -> c_int;
    pub fn tc_diloco_async_wait(diloco: *mut c_void) -> c_int;
    pub fn tc_diloco_async_commit(diloco: *mut c_void) -> c_int;
    pub fn tc_diloco_capability_query(
        diloco: *const c_void,
        requested_abi_version: u32,
        out: *mut TcDiLoCoCapabilities,
        out_size: size_t,
    ) -> c_int;
    pub fn tc_diloco_state_set_epochs(
        diloco: *mut c_void,
        topology_epoch: u64,
        membership_epoch: u64,
    ) -> c_int;
    pub fn tc_diloco_state_get_epochs(
        diloco: *const c_void,
        out_topology_epoch: *mut u64,
        out_membership_epoch: *mut u64,
    ) -> c_int;
    pub fn tc_diloco_state_size(
        diloco: *const c_void,
        requested_abi_version: u32,
        out_size: *mut size_t,
    ) -> c_int;
    pub fn tc_diloco_state_serialize(
        diloco: *const c_void,
        requested_abi_version: u32,
        out_data: *mut c_void,
        out_size: size_t,
        out_written: *mut size_t,
    ) -> c_int;
    pub fn tc_diloco_state_deserialize(
        diloco: *mut c_void,
        requested_abi_version: u32,
        data: *const c_void,
        data_size: size_t,
    ) -> c_int;
    pub fn tc_diloco_outer_steps_completed(diloco: *const c_void) -> u64;
    pub fn tc_diloco_inner_steps_completed(diloco: *const c_void) -> u64;
    pub fn tc_diloco_last_outer_step_seconds(diloco: *const c_void) -> f64;
    pub fn tc_diloco_last_outer_bytes_sent(diloco: *const c_void) -> f64;
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

// ---------- Remote shard (Phase 4 + B2 push protocol) ----------

#[repr(C)]
pub struct TcShardPlan {
    pub n_peers: i32,
    pub rows: i32,
    pub cols: i32,
    pub dtype: i32,
}

extern "C" {
    pub fn tc_remote_shard_owner(plan: *const TcShardPlan, row: i32) -> i32;
    pub fn tc_remote_shard_local_range(
        plan: *const TcShardPlan,
        my_rank: i32,
        out_lo: *mut i32,
        out_hi: *mut i32,
    );
    pub fn tc_remote_shard_publish_put(
        group: *mut c_void,
        plan: *const TcShardPlan,
        name: *const c_char,
        row_start: i32,
        row_end: i32,
        src: *const c_void,
    ) -> c_int;
    pub fn tc_remote_shard_drain_puts(
        group: *mut c_void,
        plan: *const TcShardPlan,
        name: *const c_char,
        owner_mut_buf: *mut c_void,
        out_applied: *mut i32,
    ) -> c_int;
}
