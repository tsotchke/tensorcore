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
| Torus (periodic manifold) | ✗ | ✓ (`torus_fast.c`) | — | MED |
| Product manifold (H × S × R, mixed curvature) | ✓ | ✓ (`mixed_curvature.c`) | — | shipped (`product_manifold.h`/`product_manifold_cpu.cpp`, factor-wise dispatch over Euclidean/Poincaré/Sphere/Lorentz) |
| Lie groups (matrix exp, log, group action) | ✗ | ✓ (`lie_groups.c`) | ✓ (`holonomic_gates.h`) | MED |
| Riemannian metric tensor (general manifold) | ✗ | ✓ (`metric_tensor.c`, `riemannian_metrics.c`) | ✓ (`quantum_geometric_metric.h`) | MED |
| Geodesic ODE solver (RK45 on manifolds) | ✗ | ✓ (`geodesic_solver.c`, `fast_geodesic.c`) | ✗ | MED |
| Differential forms (wedge, exterior derivative) | ✗ | ✓ (`differential_forms.c`) | ✗ | LOW |
| Fiber bundle ops | ✗ | ✓ (`fiber_bundles.c`) | ✓ (`quantum_geometric_connection.h`) | LOW |
| Bakry-Émery curvature | ✗ | ✓ (`bakry_emery.c`) | ✗ | LOW |
| RiemannianAdam optimizer (per-manifold dispatch) | ✓ | — | — | shipped |
| Phase attention + Born-rule output | ✓ | — | — | shipped |

### Quantum primitives (for QGTL, moonlab, Noesis-state-prep)

| op | tensorcore | QGTL has | moonlab has | priority |
|---|:---:|:---:|:---:|:---:|
| Pauli gates (X, Y, Z, H, S, T, RX, RY, RZ) | ✗ | ✓ (`quantum_gate_operations.h`) | ✓ | **HIGH** |
| CNOT, CZ, SWAP (2-qubit gates) | ✗ | ✓ | ✓ | **HIGH** |
| State-vector apply_gate (bit-twiddling on amplitudes) | ✗ | ✓ (`quantum_circuit_operations.h`) | ✓ | **HIGH** |
| Density matrix evolve (Lindblad / open systems) | ✗ | ✓ | ✓ | MED |
| Trotter step (e^{-iHt} via Suzuki decomp) | ✗ | ✓ (`quantum_circuit_creation.h`) | ✓ | MED |
| Quantum geometric tensor (QGT, Fubini-Study) | ✗ | ✓ (`quantum_geometric_metric.h`) | ✗ | **HIGH** |
| Holonomic gates (Berry phase via parallel transport) | ✗ | ✓ (`holonomic_gates.h`) | ✗ | MED |
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
| tc_remote_collective (AllReduce, AllGather, Broadcast across N peers) | ✗ | **HIGH** |
| tc_remote_shard (split tensor across N peers, owner-based routing) | ✗ | **HIGH** |
| DiLoCo gradient sync (every K local steps → cross-machine sync) | ~ documented | MED |
| Mesh resource scheduler integration (`scripts/mesh_resource_scheduler.py` → C-callable) | ~ Python | MED |
| Weight server with versioning (live param updates while serving) | ✗ | LOW |

### Bindings (cross-language consumption)

| binding | tensorcore | priority |
|---|:---:|:---:|
| Python ctypes (`python/tensorcore`) | ✓ | shipped |
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

**Phase 2 — Quantum primitives:**
- Pauli gates + state-vector apply (the most-called QGTL/moonlab op)
- Quantum geometric tensor (QGT) computation
- Trotter step for Hamiltonian evolution

**Phase 3 — Mesh collectives:**
- AllReduce / AllGather over tc_remote transport
- Shard owner-based routing for parameter servers

**Phase 4 — Bindings + ICC oracle expansion:**
- Rust crate (small surface, big consumer story)
- New oracles for each capability area; smoke scripts that prove engagement

**Phase 5 — Production rollout:**
- qLLM replaces `src/geometric/` calls with tensorcore via `qgtl_bridge.c` (the bridge already exists, just needs wiring)
- QGTL replaces `src/quantum_geometric/core/*` math with tensorcore
- moonlab adopts tensorcore for `src/algorithms/` quantum circuits

The phasing means we stop building parallel implementations across 5+ repos and centralize the math in tensorcore where ICC can grade it.
