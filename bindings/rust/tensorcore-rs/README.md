# tensorcore-rs

Safe Rust bindings to the **tensorcore** unified math substrate
(`libtensorcore.{dylib,so}`).

Wraps every shipped op family:
- **Geometric** — Lorentz, Sphere, Torus (exp / log / distance / slerp /
  parallel transport)
- **Lie groups** — SU(2) + SO(3) closed-form exp / log + double-cover bridge
- **Quantum** — state-vector |0…0⟩ allocator, 1- and 2-qubit unitary
  apply, gate-matrix dispatch (X/Y/Z/H/S/T/RX/RY/RZ/…), prob_one / norm²
- **Phase 4** — Riemannian metric tensor (apply / inverse /
  numerical Christoffels), RK4 geodesic ODE solver, SU(2) holonomic
  gate composition + Berry phase extraction

Inputs are `&[f32]`, outputs are `Vec<f32>` (or scalars for queries).
All math runs through the production C path — byte-identical numerics
to the Python and Eshkol bindings.

## Linking

`build.rs` finds `libtensorcore.{dylib,so}` in this order:

1. `$TENSORCORE_LIB_DIR` — explicit directory
2. `$TENSORCORE_LIB`'s parent dir
3. `<tensorcore-repo>/build/` — for in-repo development

For consumers outside the tensorcore repo:

```bash
TENSORCORE_LIB_DIR=/path/to/lib cargo build
```

## Example

```rust
use tensorcore::sphere;

let base = vec![1.0f32, 0.0, 0.0, 0.0];
let tangent = vec![0.0f32, 0.1, 0.2, 0.0];
let point = sphere::exp(&base, &tangent, 1.0);
let recovered = sphere::log(&base, &point, 1.0);
// |tangent - recovered| < 1e-4
```

See `examples/manifold_quickstart.rs` for a complete demo.

## Status

Phase 5 of the unified-substrate roadmap. Synchronous fp32 surface
(matching the C ABI). Async / GPU dispatch happens inside libtensorcore
— no Rust changes needed on the consumer side.
