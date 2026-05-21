/*
 * tensorcore — Metal 4 FlashAttention forward via mpp::tensor_ops.
 *
 * Placeholder compile target for a future FlashAttention-2 implementation on
 * top of matmul2d invocations. Host dispatch currently returns
 * TC_ERR_UNSUPPORTED_FAMILY before these entry points are used.
 *
 * Requires Xcode 26.0+ SDK (gated at CMake time).
 *
 * Layout:
 *   - One threadgroup per (batch, head, query-block)
 *   - Br = Bc = 64 query/kv block size
 *   - 4 simdgroups cooperate on each matmul2d
 *   - D = head_dim, supported via two entry points: D=64 and D=128
 *
 * For v0.1, only TensorOps GEMM is exposed at runtime. Keeping these symbols
 * compile-clean proves SDK26 headers and source gating without advertising an
 * unvalidated attention backend.
 */

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TC4_FA_BR = 64;
constant constexpr int TC4_FA_BC = 64;
constant constexpr int TC4_FA_SG_COUNT = 4;

constant bool g_tc4_causal     [[function_constant(0)]];
constant bool g_tc4_return_lse [[function_constant(1)]];

/* ====================================================================== *
 * Template body — instantiated per head_dim (D=64, D=128).                 *
 * ====================================================================== */
template <int D>
inline void flash_attention_tensorops_impl(
    device const half*  Q,
    device const half*  K,
    device const half*  V,
    device       half*  O,
    device       float* LSE,
    uint batch, uint heads, uint kv_heads, uint seq_q, uint seq_kv,
    float softmax_scale,
    uint3 group_id)
{
    const uint q_block_idx = group_id.x;
    const uint head_idx    = group_id.y;
    const uint batch_idx   = group_id.z;
    const uint kv_head_idx = (kv_heads > 0 && kv_heads != heads)
                             ? (head_idx * kv_heads / heads) : head_idx;

    const uint row0 = q_block_idx * TC4_FA_BR;
    if (row0 >= seq_q) return;

    const uint q_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;
    const uint k_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint v_base   = ((batch_idx * kv_heads + kv_head_idx) * seq_kv + 0) * D;
    const uint o_base   = ((batch_idx * heads    + head_idx)    * seq_q  + 0) * D;

    auto Qt = tensor<device const half, dextents<int32_t, 2>, tensor_inline>(
                Q + q_base, dextents<int32_t, 2>(int32_t(D), int32_t(seq_q)));
    auto Qb = Qt.slice(0, row0);
    auto Kt = tensor<device const half, dextents<int32_t, 2>, tensor_inline>(
                K + k_base, dextents<int32_t, 2>(int32_t(D), int32_t(seq_kv)));
    auto Kb = Kt.slice(0, 0);
    auto Ot = tensor<device       half, dextents<int32_t, 2>, tensor_inline>(
                O + o_base, dextents<int32_t, 2>(int32_t(D), int32_t(seq_q)));
    auto Ob = Ot.slice(0, row0);

    constexpr auto placeholder_md = matmul2d_descriptor(
        TC4_FA_BR, D, D,
        false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<placeholder_md, execution_simdgroups<TC4_FA_SG_COUNT>> placeholder_op;
    placeholder_op.run(Qb, Kb, Ob);

    (void)V;
    (void)LSE;
    (void)batch;
    (void)softmax_scale;
    (void)g_tc4_causal;
    (void)g_tc4_return_lse;
}

/* ====================================================================== *
 * Kernel entry points                                                      *
 * ====================================================================== */

kernel void tc4_flash_attention_f16_d64(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device       float* LSE       [[buffer(4), function_constant(g_tc4_return_lse)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& heads          [[buffer(6)]],
    constant uint& kv_heads       [[buffer(7)]],
    constant uint& seq_q          [[buffer(8)]],
    constant uint& seq_kv         [[buffer(9)]],
    constant float& softmax_scale [[buffer(10)]],
    uint3 group_id                [[threadgroup_position_in_grid]])
{
    flash_attention_tensorops_impl<64>(Q, K, V, O, LSE, batch, heads, kv_heads,
                                       seq_q, seq_kv, softmax_scale, group_id);
}

kernel void tc4_flash_attention_f16_d128(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device       float* LSE       [[buffer(4), function_constant(g_tc4_return_lse)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& heads          [[buffer(6)]],
    constant uint& kv_heads       [[buffer(7)]],
    constant uint& seq_q          [[buffer(8)]],
    constant uint& seq_kv         [[buffer(9)]],
    constant float& softmax_scale [[buffer(10)]],
    uint3 group_id                [[threadgroup_position_in_grid]])
{
    flash_attention_tensorops_impl<128>(Q, K, V, O, LSE, batch, heads, kv_heads,
                                        seq_q, seq_kv, softmax_scale, group_id);
}
