# tensorcore-wasm

WebAssembly build of the tensorcore math substrate (subset) for
browser-side demo galleries — moonlab's quantum-circuit visualisers,
qLLM's interactive manifold demos, geometric-attention playgrounds.

This binding compiles the **portable CPU** subset of libtensorcore
through emscripten — no Metal, no CUDA, no networking. The surface
covers every op the demo galleries need:

- Lorentz / Sphere / Torus exp / log / distance
- Lie groups SU(2) + SO(3) closed-form exp / log + double-cover
- Quantum state-vector + 1q / 2q gates + Born-rule overlap
- Quantum entanglement entropy (single-qubit)
- Holonomic gate composition + Berry phase

## Build

You need a recent emscripten SDK on `$PATH` (we build with `emcc`).

```bash
cd bindings/wasm
./build.sh
```

Output: `dist/tensorcore.js` + `dist/tensorcore.wasm` — a single
ES-module shim that JS code can `import` directly:

```javascript
import createTensorCore from './tensorcore.js';

const tc = await createTensorCore();
const score = tc.quantum_attention_score(stateQ, stateK, 2);  // typed-array IO
```

## Linking + surface

`src/wrapper.c` is a thin C file that re-exports the subset of
tensorcore symbols we want exposed to JS via `EMSCRIPTEN_KEEPALIVE`.
Each export takes typed-array pointers (Module.HEAPF32) and a
length; the JS shim in `src/wrapper.js` provides ergonomic
typed-array conversions on top.

Tests live under `test/`. Run via Node.js after building:

```bash
node test/smoke.mjs
```

## Status

Demonstration build — same math substrate as the native bindings
(byte-identical to Python / Rust / Swift on the wrapped op
subset). For production browser deployment you'd add a CDN-hosted
build + a TypeScript `.d.ts` shim; both follow-ups.
