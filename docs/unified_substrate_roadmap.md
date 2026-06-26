# Tensorcore unified substrate — multi-project op roadmap

Tensorcore's mandate: be the **single C/Eshkol math substrate** that every sibling project (qLLM, Noesis, tsotchke-chan, QGTL, moonlab) pulls from, replacing each project's bespoke kernels with a shared, optimized, cross-device implementation. The ICC oracles in `.icc/completion-oracles.yaml` define production-readiness for the public substrate; this doc tracks the capability surface needed to actually become the substrate.

## Coverage matrix

`✓` = in tree and dispatched cross-device; `~` = partial (Python-only or one backend); `✗` = not yet covered

### Geometric / Riemannian (for qLLM, GeoRefine, Noesis)

| op | tensorcore | qLLM has | QGTL has | priority |
|---|:---:|:---:|:---:|:---:|
| Poincaré ball (mobius_add, exp/log, distance, parallel_transport, conformal_factor) | ✓ | ✓ | — | shipped |
| Sphere (exp/log, geodesic distance, parallel transport, slerp) | ✓ | ✓ | — | shipped (`sphere.h`/`sphere_cpu.cpp`, qLLM `spherical_fast.c` portable replacement) |
| Lorentz / hyperboloid (Minkowski inner product, exp/log on hyperboloid) | ✓ | ✓ (`lorentz_fast.c`) | — | shipped (`lorentz.h`/`lorentz_cpu.cpp`) |
| Torus (periodic manifold) | ✓ | ✓ (`torus_fast.c`) | — | shipped (`torus.h`/`torus_cpu.cpp`, wrap-aware exp/log/distance/PT with intrinsic-dim periods) |
| Product manifold (H × S × R, mixed curvature) | ✓ | ✓ (`mixed_curvature.c`) | — | shipped (`product_manifold.h`/`product_manifold_cpu.cpp`, factor-wise dispatch over Euclidean/Poincaré/Sphere/Lorentz/Torus) |
| Lie groups (matrix exp, log, group action) | ✓ | ✓ (`lie_groups.c`) | ✓ (`holonomic_gates.h`) | shipped (`lie_groups.h`/`lie_groups_cpu.cpp`; closed-form SU(2) Pauli exp/log + SO(3) Rodrigues + double-cover bridge) |
| Riemannian metric tensor (general manifold) | ✓ | ✓ (`metric_tensor.c`, `riemannian_metrics.c`) | ✓ (`quantum_geometric_metric.h`) | shipped (`metric.h`/`metric_cpu.cpp`; metric/inverse-metric/numerical Christoffel via central diff over the manifold's metric callback) |
| Geodesic ODE solver (RK4 on manifolds) | ✓ | ✓ (`geodesic_solver.c`, `fast_geodesic.c`) | ✗ | shipped (`geodesic.h`/`geodesic_cpu.cpp`; RK4 integration of γ̈ + Γ γ̇γ̇ = 0 against arbitrary tc_metric_fn) |
| Differential forms (wedge, exterior derivative) | ✗ | ✓ (`differential_forms.c`) | ✗ | LOW |
| Fiber bundle ops | ✗ | ✓ (`fiber_bundles.c`) | ✓ (`quantum_geometric_connection.h`) | LOW |
| Bakry-Émery curvature | ✗ | ✓ (`bakry_emery.c`) | ✗ | LOW |
| RiemannianAdam optimizer (per-manifold dispatch) | ✓ | — | — | shipped |
| Phase attention + Born-rule output | ✓ | — | — | shipped |

### Quantum primitives (for QGTL, moonlab, Noesis-state-prep)

| op | tensorcore | QGTL has | moonlab has | priority |
|---|:---:|:---:|:---:|:---:|
| Pauli gates (X, Y, Z, H, S, T, RX, RY, RZ) | ✓ | ✓ (`quantum_gate_operations.h`) | ✓ | shipped (`quantum_gates.h`/`quantum_gates_cpu.cpp`; tc_gate_type_t enum values match QGTL's gate_type_t) |
| CNOT, CZ, SWAP (2-qubit gates) | ✓ | ✓ | ✓ | shipped (same file; row-major interleaved-complex 4x4) |
| State-vector apply_gate (bit-twiddling on amplitudes) | ✓ | ✓ (`quantum_circuit_operations.h`) | ✓ | shipped (`tc_qstate_apply_1q_unitary` / `tc_qstate_apply_2q_unitary`; bit-pair sweep, fp32 interleaved complex) |
| Density matrix evolve (Lindblad / open systems) | ✓ | ✓ | ✓ | shipped (`density_matrix.h`/`density_matrix_cpu.cpp`; ρ → UρU† conjugation, Kraus channels, partial trace, purity, Lindblad master equation with single-qubit jump operators) |
| Trotter step (e^{-iHt} via Suzuki decomp) | ✓ | ✓ (`quantum_circuit_creation.h`) | ✓ | shipped (`tc_qstate_trotter_step` in `quantum_gates.h`; Pauli-string evolution via basis change + CNOT staircase + Rz + reverse) |
| Quantum geometric tensor (QGT, Fubini-Study) | ✓ | ✓ (`quantum_geometric_metric.h`) | ✗ | shipped (`tc_qstate_qgt` in `quantum_gates.h`; G_ij = <∂iψ|∂jψ> - <∂iψ|ψ><ψ|∂jψ>, Fubini-Study metric) |
| Holonomic gates (Berry phase via parallel transport) | ✓ | ✓ (`holonomic_gates.h`) | ✗ | shipped (`holonomic.h`/`holonomic_cpu.cpp`; SU(2) holonomy from closed loop on parameter manifold via Lie-group composition + Berry-phase accumulator) |
| Hierarchical / matrix-product tensor ops | ✗ | ✓ (`hierarchical_tensor.h`) | ✓ | LOW |
| Quantum attention (Born-rule + entanglement measure) | ~ Born-rule only | ✓ (`quantum_attention.h`) | — | MED |

### Standard NN ops (for everything)

| op | tensorcore | priority |
|---|:---:|:---:|
| GEMM (fp16/bf16/fp32/i8, cuBLAS/Accelerate/Metal mma) | ✓ | shipped |
| GEMM batched (BMM) | ✓ | shipped |
| FlashAttention forward (CPU + Metal + CUDA half2 + CUDA wmma) | ✓ | shipped |
| FlashAttention backward (CPU + CUDA atomic + warp-reduce) | ✓ | shipped |
| RMSnorm / LayerNorm / SwiGLU / RoPE / Softmax (fwd + bwd) | ✓ | shipped |
| Conv2D | ✓ | shipped |
| 2:4 sparse tensor-core GEMM (cuSPARSELt) | ✓ | shipped |
| AdamW / RiemannianAdam | ✓ | shipped |

### Mesh / distributed (for tsotchke-chan swarm + mesh-resource-scheduler)

| op | tensorcore | priority |
|---|:---:|:---:|
| tc_remote_tensor_fetch (TCP, fp16/raw bytes) | ✓ | shipped |
| tc_remote_collective (AllReduce, AllGather, Broadcast across N peers) | ✓ | shipped (`mesh_collective.h`/`.cpp`; centralised rank-0 reduce with per-collective snapshot cache + rendezvous retry; fork-based 2-peer test ALL PASS) |
| tc_remote_shard (split tensor across N peers, owner-based routing) | ✓ | shipped (`remote_shard.h`/`remote_shard.cpp`; row-major shard plan + owner-based get/put + per-peer routing on tc_mesh transport) |
| DiLoCo gradient sync (every K local steps → cross-machine sync) | ~ documented | MED |
| Mesh resource scheduler integration (`scripts/mesh_resource_scheduler.py` → C-callable) | ~ Python | MED |
| Weight server with versioning (live param updates while serving) | ✗ | LOW |

### Bindings (cross-language consumption)

| binding | tensorcore | priority |
|---|:---:|:---:|
| Python ctypes (`python/tensorcore`) | ✓ | shipped (NumPy-friendly wrappers for Lorentz/Sphere/Torus/SU(2)/SO(3)/quantum gates/metric/geodesic/holonomic/shard/mesh — Phase 5) |
| PyTorch autograd (matmul/bmm/einsum auto-engage) | ✓ | shipped |
| Eshkol native (`eshkol/*.esk` flat-ABI bindings) | ✓ | shipped |
| Rust crate | ✗ | MED |
| Swift package (for Noesis/iOS) | ✗ | LOW |
| JavaScript / WASM (for moonlab demo gallery) | ✗ | LOW |

## ICC oracle coverage needed

Current oracles cover: cuda-for-apple-public-integration, pytorch-bridge-runtime-evidence, eshkol-bridge-runtime-evidence. To prove the unified-substrate vision, add:

- `geometric-ops-runtime-evidence` — Poincaré + Sphere + Lorentz + Product round-trip correctness on each backend
- `quantum-ops-runtime-evidence` — single-qubit gates, CNOT, state-vector evolution
- `mesh-collective-runtime-evidence` — AllReduce sum across N≥2 peers, byte-identical
- `qllm-bridge-runtime-evidence` — qLLM/src/geometric/ links against tensorcore (`qgtl_bridge.c` already exists as a touchpoint)
- `qgtl-bridge-runtime-evidence` — QGTL pulls geometric ops from tensorcore (replaces ~40 .c files)
- `moonlab-bridge-runtime-evidence` — quantum gates + state vectors come from tensorcore

## Phase plan

**Phase 1 — Geometric expansion (this week's pace):**
- Lorentz / hyperboloid C kernel + Eshkol binding + Python
- Sphere C kernel (port from Python) + Eshkol + Python (already)
- Mixed-curvature ProductManifold C kernel (replaces qLLM's `mixed_curvature.c`)

**Phase 2 — Quantum primitives [shipped]:**
- Pauli gates + state-vector apply (the most-called QGTL/moonlab op) — ✓
- Quantum geometric tensor (QGT) computation — ✓
- Trotter step for Hamiltonian evolution — ✓
- Density matrix + Lindblad open-systems evolution — ✓
- Controlled rotations (CRX/CRY/CRZ/CH), Toffoli (CCX), Fredkin (CSWAP), ISWAP, U1/U2/U3, ECR, SX, XX/YY/ZZ — ✓

**Phase 3 — Mesh collectives [shipped]:**
- AllReduce / AllGather / Broadcast over tc_remote transport — ✓

**Phase 4 — Higher-order geometry + sharding (this batch):**
- Riemannian metric tensor (general manifold via metric callback) — ✓ shipped
- Geodesic ODE solver (RK4 with numerical Christoffels) — ✓ shipped
- Holonomic gates (Berry phase from closed-loop parallel transport) — ✓ shipped
- tc_remote_shard (owner-based row-shard routing) — ✓ shipped
- New ICC oracles: `geometry-higher-order-runtime-evidence`, `remote-shard-runtime-evidence`, plus bridge oracles for qllm / qgtl / moonlab — ✓ shipped

**Phase 5 — Bindings + ICC oracle expansion (in progress):**
- ✓ NumPy-friendly Python wrappers for every shipped substrate op family
  (Lorentz / Sphere / Torus / SU(2) / SO(3) / quantum state-vector +
  gate matrices / metric tensor / geodesic ODE / holonomic gates /
  shard plan + register/get / mesh group lifecycle). New oracle
  `python-substrate-runtime-evidence` with 12 criteria — all PASS.
- Rust crate (small surface, big consumer story) — TODO
- Swift package (Noesis / iOS) — TODO
- WASM bindings (moonlab demo gallery) — TODO

**Phase 6 — Production rollout:**
- qLLM replaces `src/geometric/` calls with tensorcore via `qgtl_bridge.c` (the bridge already exists, just needs wiring)
- QGTL replaces `src/quantum_geometric/core/*` math with tensorcore
- moonlab adopts tensorcore for `src/algorithms/` quantum circuits

The phasing means we stop building parallel implementations across 5+ repos and centralize the math in tensorcore where ICC can grade it.
