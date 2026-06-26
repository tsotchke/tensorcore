# TensorCore (Swift)

Swift package wrapping `libtensorcore.{dylib,a}` for Apple-platform
consumers (macOS, iOS, visionOS). Same math substrate as the Python,
Rust, and Eshkol bindings.

## Building / testing

The package needs to find both the header surface (vendored under
`Sources/CTensorCore/include/shim.h`) and the dylib at link time.
For the in-repo path:

```bash
cd bindings/swift/TensorCore
swift test \
    -Xcc -I../../../include \
    -Xlinker -L../../../build \
    -Xlinker -rpath -Xlinker ../../../build
```

For external consumers, point `-Xlinker -L` at wherever
`libtensorcore.dylib` lives and add an rpath.

## Surface

- `Lorentz.exp / log / distance / minkowski`
- `Sphere.exp / log / distance / slerp`
- `Torus.exp / log / distance`
- `Lie.su2Exp / su2Log / su2Mul / so3Exp / so3Log / su2ToSo3`
- `Quantum.stateZero / apply1q / apply2q / probOne / normSq
  / gate1q / gate2q / attentionScore / entanglementEntropy`
- `Holonomic.composeSU2 / berryPhase`
- `Context` (RAII tc_init/tc_shutdown), `tensorcoreVersion()`,
  `tensorcoreStatusString(_:)`

XCTest target (`TensorCoreTests`) mirrors the Python + Rust substrate
smokes — same probes, same expected numbers.

## Linking model

The `CTensorCore` system-library target declares
`link "tensorcore"`. Swift's auto-link emits `-ltensorcore`; the
consumer is responsible for getting that library into the linker's
search path. Future work: ship a binary `.xcframework` for SPM users
who don't want to build the substrate themselves.
