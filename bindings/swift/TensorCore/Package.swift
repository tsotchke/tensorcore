// swift-tools-version:5.7
//
// TensorCore — Swift package wrapping the libtensorcore.{dylib,a}
// C ABI. Designed for Noesis / Apple-platform consumers (macOS, iOS,
// visionOS). Same math substrate as the Python and Rust bindings —
// they all dispatch into the canonical lib/ops/*.cpp + lib/distributed/*
// kernels.
//
// Linking model: the C system module (`CTensorCore`) advertises a
// header search path; the Swift target imports it and re-exports
// safe Swift wrappers. At link time the consumer points at the
// libtensorcore artifact via `-Ltensorcore-build-dir` (see README).

import PackageDescription

let package = Package(
    name: "TensorCore",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
    ],
    products: [
        .library(name: "TensorCore", targets: ["TensorCore"]),
    ],
    targets: [
        .systemLibrary(
            name: "CTensorCore",
            path: "Sources/CTensorCore"
        ),
        .target(
            name: "TensorCore",
            dependencies: ["CTensorCore"]
        ),
        .testTarget(
            name: "TensorCoreTests",
            dependencies: ["TensorCore"]
        ),
    ]
)
