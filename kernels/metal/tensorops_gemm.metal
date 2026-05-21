/*
 * tensorcore — Metal 4 tensor-ops GEMM (M5+ Neural Accelerator path).
 *
 * Uses Apple's `mpp::tensor_ops::matmul2d` to drive the dedicated tensor units
 * on M5 GPUs. On M1-M4 silicon the same kernel compiles and runs but is
 * slightly slower than `simdgroup_matrix` (no dedicated tensor unit yet); the
 * host-side dispatch in lib/tensorops/tensorops_m5.mm gates this kernel on
 * MTLGPUFamilyMetal4 + device-name contains "M5"/"M6" for the perf-positive
 * case, matching llama.cpp's PR #16634 hardening pattern.
 *
 * Requires:
 *   - Xcode 26.0+ / macOS 26.0+ SDK (the headers do not exist before)
 *   - -std=metal4.0 at compile (gated via CMake when SDK >= 26.0)
 *
 * Reference: liuliu/example_matmul_metal4, Apple WWDC25 session 262,
 * Metal Performance Primitives Programming Guide.
 *
 * Reported perf on M5 Max (Draw Things MFA v2.5 / Apple ML Research):
 *   ~110 TFLOPS fp16, ~4-5x M4 Max via this path.
 */

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/* Tile constants. matmul2d_descriptor accepts static or dynamic dims; we use
 * static M_tile / N_tile (compiler can optimize layouts) with K dynamic.
 * SDK26's matmul2d implementation accepts the 64x32 output tile shape used
 * by Apple's public examples; 64x64 trips a destination-type static_assert. */
constant constexpr int TC4_M_TILE = 64;
constant constexpr int TC4_N_TILE = 32;

/* Number of simdgroups cooperating on a single matmul2d. M5's tensor unit
 * splits work across simdgroups; 4 simdgroups (128 threads) matches the
 * existing simdgroup_matrix layout for fp16. */
constant constexpr int TC4_SG_COUNT = 4;

/* ====================================================================== *
 * tc4_gemm_f16  —  fp16 inputs, fp16 output, fp32 cooperative-tensor accum *
 * ====================================================================== */
kernel void tc4_gemm_f16(
    device       half* A_buf       [[buffer(0)]],
    device       half* B_buf       [[buffer(1)]],
    device       half* C_buf       [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& K               [[buffer(5)]],
    constant float& alpha          [[buffer(6)]],
    constant float& beta           [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]])
{
    /* Wrap the device buffers as 2D tensors. Element order is (col, row) per
     * the mpp::tensor_ops convention: dextents<int32_t, 2>(cols, rows). */
    auto A = tensor<device const half, dextents<int32_t, 2>, tensor_inline>(
                A_buf, dextents<int32_t, 2>(int32_t(K), int32_t(M)));
    auto B = tensor<device const half, dextents<int32_t, 2>, tensor_inline>(
                B_buf, dextents<int32_t, 2>(int32_t(N), int32_t(K)));
    auto C = tensor<device       half, dextents<int32_t, 2>, tensor_inline>(
                C_buf, dextents<int32_t, 2>(int32_t(N), int32_t(M)));

    /* Descriptor: M_tile, N_tile, K=dynamic (auto-sliced through the K loop).
     * mode: multiply overwrites the destination tile. v0.1 only dispatches
     * this path for alpha=1, beta=0. */
    constexpr auto md = matmul2d_descriptor(
        /*M*/ TC4_M_TILE, /*N*/ TC4_N_TILE, /*K*/ dynamic_length_v<int>,
        /*transA*/ false, /*transB*/ false, /*transC*/ false,
        matmul2d_descriptor::mode::multiply);

    matmul2d<md, execution_simdgroups<TC4_SG_COUNT>> mm;

    /* Slice this threadgroup's piece of A (M_tile rows × K cols) and B
     * (K rows × N_tile cols). */
    const uint row0 = tgid.y * TC4_M_TILE;
    const uint col0 = tgid.x * TC4_N_TILE;

    auto mA = A.slice(0,    row0);
    auto mB = B.slice(col0, 0);

    /* Run the matmul. mpp::tensor_ops handles the K-loop internally based on
     * the dynamic K extent in mA's dextents. */
    auto cSlice = C.slice(col0, row0);
    mm.run(mA, mB, cSlice);

    (void)alpha; (void)beta;
}

/* ====================================================================== *
 * tc4_gemm_bf16 — bf16 in/out, fp32 cooperative-tensor accum               *
 * ====================================================================== */
kernel void tc4_gemm_bf16(
    device       bfloat* A_buf     [[buffer(0)]],
    device       bfloat* B_buf     [[buffer(1)]],
    device       bfloat* C_buf     [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& K               [[buffer(5)]],
    constant float& alpha          [[buffer(6)]],
    constant float& beta           [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]])
{
    auto A = tensor<device const bfloat, dextents<int32_t, 2>, tensor_inline>(
                A_buf, dextents<int32_t, 2>(int32_t(K), int32_t(M)));
    auto B = tensor<device const bfloat, dextents<int32_t, 2>, tensor_inline>(
                B_buf, dextents<int32_t, 2>(int32_t(N), int32_t(K)));
    auto C = tensor<device       bfloat, dextents<int32_t, 2>, tensor_inline>(
                C_buf, dextents<int32_t, 2>(int32_t(N), int32_t(M)));

    constexpr auto md = matmul2d_descriptor(
        TC4_M_TILE, TC4_N_TILE, dynamic_length_v<int>,
        false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<md, execution_simdgroups<TC4_SG_COUNT>> mm;

    const uint row0 = tgid.y * TC4_M_TILE;
    const uint col0 = tgid.x * TC4_N_TILE;

    auto mA = A.slice(0,    row0);
    auto mB = B.slice(col0, 0);

    auto cSlice = C.slice(col0, row0);
    mm.run(mA, mB, cSlice);

    (void)alpha; (void)beta;   /* v0.1: alpha=1/beta=0 only on this path */
}

/* ====================================================================== *
 * tc4_gemm_f32 — fp32 in/out, fp32 cooperative-tensor accum                *
 * ====================================================================== */
kernel void tc4_gemm_f32(
    device       float* A_buf      [[buffer(0)]],
    device       float* B_buf      [[buffer(1)]],
    device       float* C_buf      [[buffer(2)]],
    constant uint& M               [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& K               [[buffer(5)]],
    constant float& alpha          [[buffer(6)]],
    constant float& beta           [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]])
{
    auto A = tensor<device const float, dextents<int32_t, 2>, tensor_inline>(
                A_buf, dextents<int32_t, 2>(int32_t(K), int32_t(M)));
    auto B = tensor<device const float, dextents<int32_t, 2>, tensor_inline>(
                B_buf, dextents<int32_t, 2>(int32_t(N), int32_t(K)));
    auto C = tensor<device       float, dextents<int32_t, 2>, tensor_inline>(
                C_buf, dextents<int32_t, 2>(int32_t(N), int32_t(M)));

    constexpr auto md = matmul2d_descriptor(
        TC4_M_TILE, TC4_N_TILE, dynamic_length_v<int>,
        false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<md, execution_simdgroups<TC4_SG_COUNT>> mm;

    const uint row0 = tgid.y * TC4_M_TILE;
    const uint col0 = tgid.x * TC4_N_TILE;

    auto mA = A.slice(0,    row0);
    auto mB = B.slice(col0, 0);

    auto cSlice = C.slice(col0, row0);
    mm.run(mA, mB, cSlice);

    (void)alpha; (void)beta;
}
