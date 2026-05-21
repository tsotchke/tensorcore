#!/usr/bin/env python3
"""Smoke test for the tensorcore Python binding.

Builds tensorcore as a shared library, then runs a small fp16 GEMM through
the Python API and compares against numpy.
"""

import os
import sys
import struct
import tempfile
import ctypes

# Make our local checkout importable.
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("TENSORCORE_TEST_INSTALLED") != "1":
    sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import tensorcore as tc


def _write_test_gguf(path):
    def w_u32(f, v):
        f.write(struct.pack("<I", v))

    def w_u64(f, v):
        f.write(struct.pack("<Q", v))

    def w_str(f, s):
        b = s.encode("utf-8")
        w_u64(f, len(b))
        f.write(b)

    def w_kv_str(f, key, value):
        w_str(f, key)
        w_u32(f, 8)  # GGUF_TYPE_STRING
        w_str(f, value)

    def w_kv_u32(f, key, value):
        w_str(f, key)
        w_u32(f, 4)  # GGUF_TYPE_UINT32
        w_u32(f, value)

    def w_kv_f32(f, key, value):
        w_str(f, key)
        w_u32(f, 6)  # GGUF_TYPE_FLOAT32
        f.write(struct.pack("<f", value))

    def w_kv_str_array2(f, key, a, b):
        w_str(f, key)
        w_u32(f, 9)  # GGUF_TYPE_ARRAY
        w_u32(f, 8)  # GGUF_TYPE_STRING
        w_u64(f, 2)
        w_str(f, a)
        w_str(f, b)

    def w_kv_f32_array2(f, key, a, b):
        w_str(f, key)
        w_u32(f, 9)  # GGUF_TYPE_ARRAY
        w_u32(f, 6)  # GGUF_TYPE_FLOAT32
        w_u64(f, 2)
        f.write(struct.pack("<f", a))
        f.write(struct.pack("<f", b))

    with open(path, "wb") as f:
        w_u32(f, 0x46554747)  # GGUF
        w_u32(f, 3)
        w_u64(f, 1)
        w_u64(f, 12)
        w_kv_str(f, "general.architecture", "llama")
        w_kv_str(f, "general.name", "python-test")
        w_kv_u32(f, "llama.context_length", 2048)
        w_kv_u32(f, "llama.embedding_length", 4096)
        w_kv_u32(f, "llama.feed_forward_length", 11008)
        w_kv_u32(f, "llama.block_count", 32)
        w_kv_u32(f, "llama.attention.head_count", 32)
        w_kv_u32(f, "llama.attention.head_count_kv", 8)
        w_kv_u32(f, "llama.rope.dimension_count", 128)
        w_kv_f32(f, "llama.attention.layer_norm_rms_epsilon", 0.125)
        w_kv_str_array2(f, "tokenizer.ggml.tokens", "<unk>", "hello")
        w_kv_f32_array2(f, "tokenizer.ggml.scores", -1000.0, 0.25)
        w_str(f, "weight.test")
        w_u32(f, 2)
        w_u64(f, 32)
        w_u64(f, 1)
        w_u32(f, tc.TC_GGUF_TYPE_Q4_0)
        w_u64(f, 0)
        pad = (32 - (f.tell() % 32)) % 32
        f.write(b"\0" * pad)
        f.write(struct.pack("<H", 0x3800))
        f.write(bytes([0xAB]) * 16)


def _dequant_q4_0(raw, N, K):
    blocks = K // 32
    W = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for b in range(blocks):
            off = (n * blocks + b) * 18
            scale = np.frombuffer(raw[off:off + 2], dtype=np.float16)[0].astype(np.float32)
            qs = raw[off + 2:off + 18]
            for i, packed in enumerate(qs):
                W[n, b * 32 + i] = (float(packed & 0x0F) - 8.0) * scale
                W[n, b * 32 + i + 16] = (float(packed >> 4) - 8.0) * scale
    return W


def _dequant_q8_0(raw, N, K):
    blocks = K // 32
    W = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for b in range(blocks):
            off = (n * blocks + b) * 34
            scale = np.frombuffer(raw[off:off + 2], dtype=np.float16)[0].astype(np.float32)
            qs = np.frombuffer(raw[off + 2:off + 34], dtype=np.int8).astype(np.float32)
            W[n, b * 32:b * 32 + 32] = qs * scale
    return W


def _scaled_rms(got, ref):
    got32 = got.astype(np.float32)
    ref32 = ref.astype(np.float32)
    err = got32 - ref32
    return float(np.sqrt((err * err).mean()) / (np.sqrt((ref32 * ref32).mean()) + 1e-9))


def _run_diagnostic_api_check():
    dtype_ok = (
        tc.dtype_name("f16") == "f16" and
        tc.dtype_name(tc.TC_DTYPE_BF16) == "bf16" and
        tc.dtype_name(tc.TC_DTYPE_FP53) == "fp53" and
        tc.dtype_name(9999) == "?"
    )
    status_ok = (
        tc.status_string(tc.TC_OK) == "ok" and
        tc.status_string(tc.TC_ERR_NO_DEVICE) == "no Metal device available" and
        tc.status_string(-12345) == "unknown status"
    )
    backend_ok = (
        tc.backend_name(tc.TC_BACKEND_NONE) == "none" and
        tc.backend_name(tc.TC_BACKEND_TENSOROPS_M5) == "tensorops_m5" and
        tc.backend_name(9999) == "?" and
        tc.last_backend() == tc.TC_BACKEND_NONE and
        tc.last_backend_name() == "none"
    )
    tensorops_ok = (
        tc.tensorops_gemm_kernel_name("f16") == "tc4_gemm_f16" and
        tc.tensorops_gemm_kernel_name("bf16") == "tc4_gemm_bf16" and
        tc.tensorops_gemm_kernel_name("f32") == "tc4_gemm_f32" and
        tc.tensorops_gemm_kernel_name("i8", "i32") is None
    )
    return dtype_ok and status_ok and backend_ok and tensorops_ok


def _run_distributed_wrapper_check(ctx):
    values = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    gathered = np.zeros_like(values)
    buf = tc.buffer_alloc(ctx, values.nbytes)
    out = tc.buffer_alloc(ctx, gathered.nbytes)
    dist = None
    try:
        tc.buffer_write(buf, values)
        dist = tc.dist_init(ctx, tc.TC_DIST_SINGLE, 1, 0, "single://python-test")
        metadata_ok = tc.dist_world_size(dist) == 1 and tc.dist_rank(dist) == 0
        tc.allreduce(dist, buf, values.size, "f32", "sum")
        tc.broadcast(dist, buf, values.size, "f32", root=0)
        tc.barrier(dist)
        after = np.empty_like(values)
        tc.buffer_read(buf, after)
        tc.allgather(dist, buf, out, values.size, "f32")
        tc.buffer_read(out, gathered)

        with tc.DistContext(ctx, "ring", 1, 0, "tb5://single") as owned_dist:
            owned_ok = owned_dist.world_size == 1 and owned_dist.rank == 0
            owned_dist.barrier()

        return (
            metadata_ok and owned_ok and
            np.array_equal(after, values) and
            np.array_equal(gathered, values)
        )
    finally:
        if dist:
            tc.dist_finalize(dist)
        tc.buffer_free(ctx, out)
        tc.buffer_free(ctx, buf)


def _run_attention_wrapper_check(ctx):
    bufs = []

    def make(arr):
        b = tc.buffer_alloc(ctx, arr.nbytes)
        bufs.append(b)
        tc.buffer_write(b, arr)
        return b

    def empty(arr):
        b = tc.buffer_alloc(ctx, arr.nbytes)
        bufs.append(b)
        return b

    try:
        B, H, S, D = 1, 1, 64, 64
        scale = 1.0 / np.sqrt(float(D))
        Q = (np.random.randn(B, H, S, D) * 0.25).astype(np.float16)
        K = (np.random.randn(B, H, S, D) * 0.25).astype(np.float16)
        V = (np.random.randn(B, H, S, D) * 0.25).astype(np.float16)
        O = np.zeros_like(Q)
        O_async = np.zeros_like(Q)
        LSE = np.zeros((B, H, S), dtype=np.float32)
        dO = (np.random.randn(B, H, S, D) * 0.125).astype(np.float16)
        dQ = np.zeros_like(Q)
        dK = np.zeros_like(K)
        dV = np.zeros_like(V)

        qf = Q.astype(np.float32)
        kf = K.astype(np.float32)
        vf = V.astype(np.float32)
        scores = np.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
        causal_mask = np.triu(np.ones((S, S), dtype=bool), 1)
        scores = np.where(causal_mask[None, None, :, :], -np.inf, scores)
        m = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - m)
        denom = np.sum(exp_scores, axis=-1, keepdims=True)
        probs = exp_scores / denom
        O_ref = np.einsum("bhqk,bhkd->bhqd", probs, vf)
        LSE_ref = (m[..., 0] + np.log(denom[..., 0])).astype(np.float32)

        qb = make(Q)
        kb = make(K)
        vb = make(V)
        ob = empty(O)
        oab = empty(O_async)
        lseb = empty(LSE)
        dob = make(dO)
        dqb = empty(dQ)
        dkb = empty(dK)
        dvb = empty(dV)

        tc.attention_forward(ctx, qb, kb, vb, ob, B, H, S, S, D,
                             LSE=lseb, return_lse=True)
        tc.buffer_read(ob, O)
        tc.buffer_read(lseb, LSE)

        stream = tc.stream_create(ctx)
        try:
            tc.attention_forward_async(ctx, qb, kb, vb, oab, B, H, S, S, D, stream)
            tc.stream_sync(stream)
        finally:
            tc.stream_destroy(ctx, stream)
        tc.buffer_read(oab, O_async)

        tc.attention_backward(ctx, qb, kb, vb, ob, dob, lseb, dqb, dkb, dvb,
                              B, H, S, S, D)
        tc.buffer_read(dqb, dQ)
        tc.buffer_read(dkb, dK)
        tc.buffer_read(dvb, dV)

        out_err = _scaled_rms(O, O_ref)
        lse_err = float(np.max(np.abs(LSE - LSE_ref)))
        async_err = float(np.max(np.abs(O_async.astype(np.float32) - O.astype(np.float32))))
        grad_max = max(
            float(np.max(np.abs(dQ.astype(np.float32)))),
            float(np.max(np.abs(dK.astype(np.float32)))),
            float(np.max(np.abs(dV.astype(np.float32)))),
        )
        backward_ok = (
            np.all(np.isfinite(dQ.astype(np.float32))) and
            np.all(np.isfinite(dK.astype(np.float32))) and
            np.all(np.isfinite(dV.astype(np.float32))) and
            grad_max > 0.0
        )
        ok = (
            out_err < 2e-2 and
            lse_err < 2e-2 and
            async_err < 1e-3 and
            backward_ok
        )
        return ok, {"out": out_err, "lse": lse_err, "async": async_err, "bwd": grad_max}
    finally:
        for b in reversed(bufs):
            tc.buffer_free(ctx, b)


def _run_batched_gemm_wrapper_check(ctx):
    batch, M, N, K = 3, 32, 24, 32
    stride_a = M * K + 7
    stride_b = K * N + 5
    stride_c = M * N + 3
    total_a = (batch - 1) * stride_a + M * K
    total_b = (batch - 1) * stride_b + K * N
    total_c = (batch - 1) * stride_c + M * N

    A = np.zeros(total_a, dtype=np.float16)
    B = np.zeros(total_b, dtype=np.float16)
    C = np.zeros(total_c, dtype=np.float16)
    C_ref = np.zeros(total_c, dtype=np.float16)
    for b in range(batch):
        a0 = b * stride_a
        b0 = b * stride_b
        c0 = b * stride_c
        A[a0:a0 + M * K] = np.random.randn(M * K).astype(np.float16)
        B[b0:b0 + K * N] = np.random.randn(K * N).astype(np.float16)
        Am = A[a0:a0 + M * K].reshape(M, K).astype(np.float32)
        Bm = B[b0:b0 + K * N].reshape(K, N).astype(np.float32)
        C_ref[c0:c0 + M * N] = (Am @ Bm).astype(np.float16).reshape(-1)

    ab = tc.buffer_alloc(ctx, A.nbytes)
    bb = tc.buffer_alloc(ctx, B.nbytes)
    cb = tc.buffer_alloc(ctx, C.nbytes)
    try:
        tc.buffer_write(ab, A)
        tc.buffer_write(bb, B)
        tc.gemm_batched(ctx, ab, bb, cb, batch, M, N, K,
                        stride_a=stride_a, stride_b=stride_b, stride_c=stride_c)
        tc.buffer_read(cb, C)
        max_abs = 0.0
        for b in range(batch):
            c0 = b * stride_c
            err = np.max(np.abs(
                C[c0:c0 + M * N].astype(np.float32) -
                C_ref[c0:c0 + M * N].astype(np.float32)
            ))
            max_abs = max(max_abs, float(err))
        return max_abs == 0.0, max_abs
    finally:
        tc.buffer_free(ctx, ab)
        tc.buffer_free(ctx, bb)
        tc.buffer_free(ctx, cb)


def _run_padded_gemm_wrapper_check(ctx):
    M, N, K = 37, 41, 29
    lda, ldb, ldc = M + 3, K + 5, N + 7
    alpha, beta = 0.75, -0.25

    A = np.full((K, lda), -37.0, dtype=np.float32)
    B = np.full((N, ldb), 19.0, dtype=np.float32)
    C0 = np.full((M, ldc), -11.0, dtype=np.float32)
    A[:, :M] = (np.random.randn(K, M) * 0.2).astype(np.float32)
    B[:, :K] = (np.random.randn(N, K) * 0.2).astype(np.float32)
    C0[:, :N] = (np.random.randn(M, N) * 0.1).astype(np.float32)

    C_ref = C0.copy()
    C_ref[:, :N] = alpha * (A[:, :M].T @ B[:, :K].T) + beta * C0[:, :N]

    ab = tc.buffer_alloc(ctx, A.nbytes)
    bb = tc.buffer_alloc(ctx, B.nbytes)
    cb = tc.buffer_alloc(ctx, C0.nbytes)
    C = np.empty_like(C0)
    C_async = np.empty_like(C0)
    try:
        tc.buffer_write(ab, A)
        tc.buffer_write(bb, B)

        tc.buffer_write(cb, C0)
        tc.gemm(ctx, ab, bb, cb, M, N, K, dtype="f32",
                alpha=alpha, beta=beta, transpose_a=True, transpose_b=True,
                lda=lda, ldb=ldb, ldc=ldc)
        tc.buffer_read(cb, C)

        tc.buffer_write(cb, C0)
        stream = tc.stream_create(ctx)
        try:
            tc.gemm_async(ctx, ab, bb, cb, M, N, K, stream, dtype="f32",
                          alpha=alpha, beta=beta,
                          transpose_a=True, transpose_b=True,
                          lda=lda, ldb=ldb, ldc=ldc)
            tc.stream_sync(stream)
        finally:
            tc.stream_destroy(ctx, stream)
        tc.buffer_read(cb, C_async)

        sync_err = float(np.max(np.abs(C[:, :N] - C_ref[:, :N])))
        async_err = float(np.max(np.abs(C_async[:, :N] - C_ref[:, :N])))
        padding_ok = (
            np.array_equal(C[:, N:], C0[:, N:]) and
            np.array_equal(C_async[:, N:], C0[:, N:])
        )
        return sync_err < 5e-4 and async_err < 5e-4 and padding_ok, {
            "sync": sync_err,
            "async": async_err,
        }
    finally:
        tc.buffer_free(ctx, ab)
        tc.buffer_free(ctx, bb)
        tc.buffer_free(ctx, cb)


def _run_batched_padded_gemm_wrapper_check(ctx):
    batch, M, N, K = 2, 16, 12, 16
    lda, ldb, ldc = M + 3, K + 5, N + 7
    storage_a = (K - 1) * lda + M
    storage_b = (N - 1) * ldb + K
    storage_c = (M - 1) * ldc + N
    total_a = (batch - 1) * storage_a + storage_a
    total_b = (batch - 1) * storage_b + storage_b
    total_c = (batch - 1) * storage_c + storage_c

    A = np.full(total_a, -3.0, dtype=np.float16)
    B = np.full(total_b, 5.0, dtype=np.float16)
    C = np.full(total_c, -7.0, dtype=np.float16)
    C_ref = C.copy()

    for b_i in range(batch):
        a0 = b_i * storage_a
        b0 = b_i * storage_b
        c0 = b_i * storage_c
        Aphys = (np.random.randn(K, M) * 0.2).astype(np.float16)
        Bphys = (np.random.randn(N, K) * 0.2).astype(np.float16)
        Cm = (Aphys.astype(np.float32).T @ Bphys.astype(np.float32).T).astype(np.float16)
        for k in range(K):
            A[a0 + k * lda:a0 + k * lda + M] = Aphys[k]
        for n in range(N):
            B[b0 + n * ldb:b0 + n * ldb + K] = Bphys[n]
        for m in range(M):
            C_ref[c0 + m * ldc:c0 + m * ldc + N] = Cm[m]

    ab = tc.buffer_alloc(ctx, A.nbytes)
    bb = tc.buffer_alloc(ctx, B.nbytes)
    cb = tc.buffer_alloc(ctx, C.nbytes)
    try:
        tc.buffer_write(ab, A)
        tc.buffer_write(bb, B)
        tc.buffer_write(cb, C)
        tc.gemm_batched(ctx, ab, bb, cb, batch, M, N, K,
                        transpose_a=True, transpose_b=True,
                        lda=lda, ldb=ldb, ldc=ldc)
        tc.buffer_read(cb, C)
        max_abs = 0.0
        padding_ok = True
        for b_i in range(batch):
            c0 = b_i * storage_c
            for m in range(M):
                row = slice(c0 + m * ldc, c0 + m * ldc + N)
                err = np.max(np.abs(
                    C[row].astype(np.float32) - C_ref[row].astype(np.float32)
                ))
                max_abs = max(max_abs, float(err))
                if m < M - 1:
                    pad = slice(c0 + m * ldc + N, c0 + (m + 1) * ldc)
                    padding_ok = padding_ok and np.array_equal(
                        C[pad], np.full(ldc - N, -7.0, dtype=np.float16),
                    )
        return max_abs < 1e-3 and padding_ok, max_abs
    finally:
        tc.buffer_free(ctx, ab)
        tc.buffer_free(ctx, bb)
        tc.buffer_free(ctx, cb)


def _run_conv_wrapper_check(ctx):
    B, IC, OC = 2, 2, 3
    H, W_in, kH, kW = 8, 8, 3, 3
    pad, stride = 1, 1
    out_H, out_W = tc.conv2d_output_shape(H, W_in, kH, kW, pad, pad, stride, stride)
    X = (np.random.randn(B, IC, H, W_in) * 0.25).astype(np.float16)
    W = (np.random.randn(OC, IC, kH, kW) * 0.2).astype(np.float16)
    bias = (np.random.randn(OC) * 0.05).astype(np.float16)
    dY = (np.random.randn(B, OC, out_H, out_W) * 0.1).astype(np.float16)
    Y = np.zeros((B, OC, out_H, out_W), dtype=np.float16)
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    Y_ref = np.zeros_like(Y, dtype=np.float32)
    dX_ref = np.zeros_like(X, dtype=np.float32)
    dW_ref = np.zeros_like(W, dtype=np.float32)

    Xf = X.astype(np.float32)
    Wf = W.astype(np.float32)
    bf = bias.astype(np.float32)
    dYf = dY.astype(np.float32)
    for b_i in range(B):
        for oc in range(OC):
            for oh in range(out_H):
                for ow in range(out_W):
                    acc = float(bf[oc])
                    for ic in range(IC):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * stride - pad + kh
                                iw = ow * stride - pad + kw
                                if 0 <= ih < H and 0 <= iw < W_in:
                                    acc += float(Xf[b_i, ic, ih, iw] * Wf[oc, ic, kh, kw])
                    Y_ref[b_i, oc, oh, ow] = acc
                    dy = dYf[b_i, oc, oh, ow]
                    for ic in range(IC):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * stride - pad + kh
                                iw = ow * stride - pad + kw
                                if 0 <= ih < H and 0 <= iw < W_in:
                                    dX_ref[b_i, ic, ih, iw] += Wf[oc, ic, kh, kw] * dy
                                    dW_ref[oc, ic, kh, kw] += Xf[b_i, ic, ih, iw] * dy

    xb = tc.buffer_alloc(ctx, X.nbytes)
    wb = tc.buffer_alloc(ctx, W.nbytes)
    bb = tc.buffer_alloc(ctx, bias.nbytes)
    dyb = tc.buffer_alloc(ctx, dY.nbytes)
    yb = tc.buffer_alloc(ctx, Y.nbytes)
    dxb = tc.buffer_alloc(ctx, dX.nbytes)
    dwb = tc.buffer_alloc(ctx, dW.nbytes)
    scratch = tc.buffer_alloc(
        ctx,
        tc.conv2d_scratch_bytes(B, IC, H, W_in, kH, kW, pad, pad, stride, stride),
    )
    scratch_dx = tc.buffer_alloc(ctx, tc.conv2d_backward_input_scratch_bytes(B, IC, H, W_in))
    try:
        tc.buffer_write(xb, X)
        tc.buffer_write(wb, W)
        tc.buffer_write(bb, bias)
        tc.buffer_write(dyb, dY)
        tc.conv2d_forward(ctx, xb, wb, bb, yb, scratch,
                          B, IC, OC, H, W_in, kH, kW,
                          pad_h=pad, pad_w=pad, stride_h=stride, stride_w=stride)
        tc.buffer_read(yb, Y)
        fwd_err = _scaled_rms(Y, Y_ref)

        tc.conv2d_backward_input(ctx, dyb, wb, dxb, scratch, scratch_dx,
                                 B, IC, OC, H, W_in, kH, kW,
                                 pad_h=pad, pad_w=pad, stride_h=stride, stride_w=stride)
        tc.buffer_read(dxb, dX)
        dx_err = _scaled_rms(dX, dX_ref)

        tc.conv2d_backward_weight(ctx, xb, dyb, dwb, scratch,
                                  B, IC, OC, H, W_in, kH, kW,
                                  pad_h=pad, pad_w=pad, stride_h=stride, stride_w=stride)
        tc.buffer_read(dwb, dW)
        dw_err = _scaled_rms(dW, dW_ref)

        err = max(fwd_err, dx_err, dw_err)
        return err < 2e-2, {"forward": fwd_err, "dX": dx_err, "dW": dw_err}
    finally:
        tc.buffer_free(ctx, xb)
        tc.buffer_free(ctx, wb)
        tc.buffer_free(ctx, bb)
        tc.buffer_free(ctx, dyb)
        tc.buffer_free(ctx, yb)
        tc.buffer_free(ctx, dxb)
        tc.buffer_free(ctx, dwb)
        tc.buffer_free(ctx, scratch)
        tc.buffer_free(ctx, scratch_dx)


def _run_buffer_layout_check(ctx):
    src_base = np.arange(48, dtype=np.float32).reshape(6, 8)
    src_view = src_base[:, ::2]
    buf = tc.buffer_alloc(ctx, src_view.nbytes)
    try:
        tc.buffer_write(buf, src_view)
        got = np.empty(src_view.shape, dtype=src_view.dtype)
        tc.buffer_read(buf, got)
        write_ok = np.array_equal(got, np.ascontiguousarray(src_view))

        out_base = np.full((6, 8), -1.0, dtype=np.float32)
        out_view = out_base[:, ::2]
        tc.buffer_read(buf, out_view)
        read_ok = (
            np.array_equal(out_view, np.ascontiguousarray(src_view)) and
            np.all(out_base[:, 1::2] == -1.0)
        )
        return write_ok and read_ok
    finally:
        tc.buffer_free(ctx, buf)


def _run_training_wrapper_checks(ctx):
    bufs = []

    def make(arr):
        b = tc.buffer_alloc(ctx, arr.nbytes)
        bufs.append(b)
        tc.buffer_write(b, arr)
        return b

    def empty(arr):
        b = tc.buffer_alloc(ctx, arr.nbytes)
        bufs.append(b)
        return b

    try:
        eps = 1e-5

        rows, dim = 2, 64
        X = np.random.randn(rows, dim).astype(np.float16)
        gamma = (0.5 + np.random.rand(dim)).astype(np.float16)
        beta = ((np.random.rand(dim) - 0.5) * 0.1).astype(np.float16)
        Y = np.zeros_like(X)
        rstd = np.zeros(rows, dtype=np.float32)
        mean = np.zeros(rows, dtype=np.float32)

        xb = make(X)
        gb = make(gamma)
        bb = make(beta)
        yb = empty(Y)
        rstdb = empty(rstd)
        meanb = empty(mean)

        Xf = X.astype(np.float32)
        gf = gamma.astype(np.float32)
        bf = beta.astype(np.float32)
        rms_ref = Xf * (1.0 / np.sqrt(np.mean(Xf * Xf, axis=1, keepdims=True) + eps)) * gf
        tc.rmsnorm_forward(ctx, xb, gb, yb, rstdb, rows, dim, eps)
        tc.buffer_read(yb, Y)
        rms_err = _scaled_rms(Y, rms_ref)

        tc.layernorm_forward(ctx, xb, gb, bb, yb, meanb, rstdb, rows, dim, eps)
        tc.buffer_read(yb, Y)
        mu = np.mean(Xf, axis=1, keepdims=True)
        var = np.mean((Xf - mu) * (Xf - mu), axis=1, keepdims=True)
        layer_ref = (Xf - mu) * (1.0 / np.sqrt(var + eps)) * gf + bf
        layer_err = _scaled_rms(Y, layer_ref)

        n = 256
        gate = np.random.randn(n).astype(np.float16)
        up = np.random.randn(n).astype(np.float16)
        out = np.zeros(n, dtype=np.float16)
        gateb = make(gate)
        upb = make(up)
        outb = empty(out)
        tc.swiglu_forward(ctx, gateb, upb, outb, n)
        tc.buffer_read(outb, out)
        gatef = gate.astype(np.float32)
        swiglu_ref = gatef / (1.0 + np.exp(-gatef)) * up.astype(np.float32)
        swiglu_err = _scaled_rms(out, swiglu_ref)

        B, H, S, D = 1, 2, 4, 32
        rope_x = np.random.randn(B, H, S, D).astype(np.float16)
        rope_ref = rope_x.astype(np.float32).copy()
        cos_t = np.zeros((S, D // 2), dtype=np.float32)
        sin_t = np.zeros((S, D // 2), dtype=np.float32)
        for p in range(S):
            for d2 in range(D // 2):
                theta = p / (10000.0 ** (2.0 * d2 / D))
                cos_t[p, d2] = np.cos(theta)
                sin_t[p, d2] = np.sin(theta)
        for b_i in range(B):
            for h_i in range(H):
                for p in range(S):
                    x0 = rope_ref[b_i, h_i, p, :D // 2].copy()
                    x1 = rope_ref[b_i, h_i, p, D // 2:].copy()
                    rope_ref[b_i, h_i, p, :D // 2] = x0 * cos_t[p] - x1 * sin_t[p]
                    rope_ref[b_i, h_i, p, D // 2:] = x0 * sin_t[p] + x1 * cos_t[p]
        ropeb = make(rope_x)
        cosb = make(cos_t)
        sinb = make(sin_t)
        tc.rope_forward(ctx, ropeb, cosb, sinb, B, H, S, D)
        rope_out = np.zeros_like(rope_x)
        tc.buffer_read(ropeb, rope_out)
        rope_err = _scaled_rms(rope_out, rope_ref)

        sm_rows, sm_dim = 2, 64
        sm_x = (np.random.randn(sm_rows, sm_dim) * 3.0).astype(np.float16)
        sm_y = np.zeros_like(sm_x)
        smxb = make(sm_x)
        smyb = empty(sm_y)
        tc.softmax_forward(ctx, smxb, smyb, sm_rows, sm_dim)
        tc.buffer_read(smyb, sm_y)
        smf = sm_x.astype(np.float32)
        sm_exp = np.exp(smf - np.max(smf, axis=1, keepdims=True))
        sm_ref = sm_exp / np.sum(sm_exp, axis=1, keepdims=True)
        softmax_err = _scaled_rms(sm_y, sm_ref)

        fM, fN, fK = 1, 8, 64
        fx = np.random.randn(fM, fK).astype(np.float16)
        fg = (0.5 + np.random.rand(fK)).astype(np.float16)
        fw = (np.random.randn(fK, fN) * 0.1).astype(np.float16)
        fy = np.zeros((fM, fN), dtype=np.float16)
        fxb = make(fx)
        fgb = make(fg)
        fwb = make(fw)
        fyb = empty(fy)
        tc.fused_rmsnorm_gemv(ctx, fxb, fgb, fwb, fyb, fM, fN, fK, eps)
        tc.buffer_read(fyb, fy)
        fxf = fx.astype(np.float32)
        fnorm = fxf * (1.0 / np.sqrt(np.mean(fxf * fxf, axis=1, keepdims=True) + eps)) * fg.astype(np.float32)
        fused_ref = fnorm @ fw.astype(np.float32)
        fused_err = _scaled_rms(fy, fused_ref)

        opt_n = 64
        params = np.random.randn(opt_n).astype(np.float32)
        moments = np.zeros(opt_n, dtype=np.float32)
        variance = np.zeros(opt_n, dtype=np.float32)
        grads = (np.random.randn(opt_n) * 0.1).astype(np.float32)
        p_ref = params.copy()
        m_ref = moments.copy()
        v_ref = variance.copy()
        lr, beta1, beta2, adam_eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
        bc1, bc2 = 1.0 - beta1, 1.0 - beta2
        m_ref = beta1 * m_ref + (1.0 - beta1) * grads
        v_ref = beta2 * v_ref + (1.0 - beta2) * grads * grads
        p_ref = p_ref - lr * ((m_ref / bc1) / (np.sqrt(v_ref / bc2) + adam_eps) + wd * p_ref)
        pb = make(params)
        mb = make(moments)
        vb = make(variance)
        gradb = make(grads)
        tc.adamw_step(ctx, pb, mb, vb, gradb, "f32", opt_n,
                      lr, beta1, beta2, adam_eps, wd, bc1, bc2)
        tc.buffer_read(pb, params)
        adam_err = _scaled_rms(params, p_ref)

        errs = {
            "rms": rms_err,
            "layer": layer_err,
            "swiglu": swiglu_err,
            "rope": rope_err,
            "softmax": softmax_err,
            "fused": fused_err,
            "adamw": adam_err,
        }
        ok = (
            rms_err < 5e-3 and
            layer_err < 5e-3 and
            swiglu_err < 5e-3 and
            rope_err < 5e-3 and
            softmax_err < 5e-3 and
            fused_err < 1e-2 and
            adam_err < 1e-5
        )
        return ok, errs
    finally:
        for b in reversed(bufs):
            tc.buffer_free(ctx, b)


def _run_owned_api_check():
    M, N, K = 32, 32, 32
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    C_async = np.zeros_like(C)
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    sm_rows, sm_dim = 2, 16
    X = (np.random.randn(sm_rows, sm_dim) * 0.5).astype(np.float16)
    Y = np.zeros_like(X)
    dY = (np.random.randn(sm_rows, sm_dim) * 0.25).astype(np.float16)
    dX = np.zeros_like(X)
    Xf = X.astype(np.float32)
    exp = np.exp(Xf - np.max(Xf, axis=1, keepdims=True))
    Y_ref = exp / np.sum(exp, axis=1, keepdims=True)
    dX_ref = Y_ref * (dY.astype(np.float32) - np.sum(dY.astype(np.float32) * Y_ref, axis=1, keepdims=True))

    rms_rows, rms_dim = 2, 64
    eps = 1e-5
    R = np.random.randn(rms_rows, rms_dim).astype(np.float16)
    gamma = (0.5 + np.random.rand(rms_dim)).astype(np.float16)
    dR = (np.random.randn(rms_rows, rms_dim) * 0.125).astype(np.float16)
    rstd = np.zeros(rms_rows, dtype=np.float32)
    RY = np.zeros_like(R)
    dR_out = np.zeros_like(R)
    dgamma = np.zeros(rms_dim, dtype=np.float32)
    Rf = R.astype(np.float32)
    gf = gamma.astype(np.float32)
    dRf = dR.astype(np.float32)
    rstd_ref = 1.0 / np.sqrt(np.mean(Rf * Rf, axis=1) + eps)
    RY_ref = Rf * rstd_ref[:, None] * gf
    dot = np.sum(dRf * gf[None, :] * Rf, axis=1)
    dR_ref = dRf * gf[None, :] * rstd_ref[:, None] - (
        Rf * dot[:, None] * (rstd_ref[:, None] ** 3) / float(rms_dim)
    )
    dgamma_ref = np.sum(dRf * Rf * rstd_ref[:, None], axis=0)

    with tc.Context() as ctx:
        a = ctx.buffer_from_array(A)
        b = ctx.buffer_from_array(B)
        c = ctx.buffer(C.nbytes)
        owned_nbytes_ok = a.nbytes == A.nbytes and b.nbytes == B.nbytes and c.nbytes == C.nbytes
        ctx.gemm(a, b, c, M, N, K)
        C = c.to_numpy((M, N), np.float16)

        with ctx.stream() as stream:
            ctx.gemm_async(a, b, c, M, N, K, stream)
            stream.sync()
        C_async = c.to_numpy((M, N), np.float16)

        xb = ctx.buffer_from_array(X)
        yb = ctx.buffer(Y.nbytes)
        dyb = ctx.buffer_from_array(dY)
        dxb = ctx.buffer(dX.nbytes)
        ctx.softmax_forward(xb, yb, sm_rows, sm_dim)
        Y = yb.to_numpy(Y.shape, Y.dtype)
        ctx.softmax_backward(yb, dyb, dxb, sm_rows, sm_dim)
        dX = dxb.to_numpy(dX.shape, dX.dtype)

        rb = ctx.buffer_from_array(R)
        gb = ctx.buffer_from_array(gamma)
        drb = ctx.buffer_from_array(dR)
        ryb = ctx.buffer(RY.nbytes)
        rstdb = ctx.buffer(rstd.nbytes)
        droutb = ctx.buffer(dR_out.nbytes)
        dgb = ctx.buffer(dgamma.nbytes)
        ctx.rmsnorm_forward(rb, gb, ryb, rstdb, rms_rows, rms_dim, eps)
        ctx.rmsnorm_backward(rb, gb, drb, rstdb, droutb, dgb, rms_rows, rms_dim)
        RY = ryb.to_numpy(RY.shape, RY.dtype)
        rstd = rstdb.to_numpy(rstd.shape, rstd.dtype)
        dR_out = droutb.to_numpy(dR_out.shape, dR_out.dtype)
        dgamma = dgb.to_numpy(dgamma.shape, dgamma.dtype)

    err = np.max(np.abs(C.astype(np.float32) - C_ref.astype(np.float32)))
    err_async = np.max(np.abs(C_async.astype(np.float32) - C_ref.astype(np.float32)))
    sm_err = _scaled_rms(Y, Y_ref)
    sm_bwd_err = _scaled_rms(dX, dX_ref)
    rms_fwd_err = _scaled_rms(RY, RY_ref)
    rms_rstd_err = float(np.max(np.abs(rstd - rstd_ref)))
    rms_dx_err = _scaled_rms(dR_out, dR_ref)
    rms_dg_err = _scaled_rms(dgamma.astype(np.float32), dgamma_ref)
    backward_ok = (
        sm_bwd_err < 5e-3 and
        rms_fwd_err < 5e-3 and
        rms_rstd_err < 1e-4 and
        rms_dx_err < 5e-3 and
        rms_dg_err < 1e-5 and
        np.all(np.isfinite(dX.astype(np.float32))) and
        np.all(np.isfinite(dR_out.astype(np.float32))) and
        np.all(np.isfinite(dgamma))
    )
    return (
        owned_nbytes_ok and err == 0.0 and err_async == 0.0 and sm_err < 5e-3 and backward_ok
    ), max(float(err), float(err_async), sm_err, sm_bwd_err, rms_fwd_err, rms_rstd_err,
           rms_dx_err, rms_dg_err)


def main():
    diagnostic_ok = _run_diagnostic_api_check()
    print(f"Diagnostic API:       {'OK' if diagnostic_ok else 'FAIL'}")
    if not diagnostic_ok:
        return 5

    print(f"tensorcore: {tc.version()}")
    try:
        ctx = tc.init()
    except tc.TensorcoreError as e:
        if e.status == tc.TC_ERR_NO_DEVICE:
            print("SKIP: no Metal device available")
            return 77
        raise
    info = tc.device_info(ctx)
    print(f"device   : {info.name_str}")
    print(f"family   : Apple{info.family}")
    print(f"unified  : {info.unified_memory}")
    print(f"bf16 sg  : {info.supports_bf16_simdgroup}")
    print(f"i8   sg  : {info.supports_i8_simdgroup}")
    print(f"tensorops: {info.supports_tensorops_m5}")
    print()

    M, N, K = 256, 256, 256
    np.random.seed(0xCA75)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    a = tc.buffer_alloc(ctx, A.nbytes)
    b = tc.buffer_alloc(ctx, B.nbytes)
    c = tc.buffer_alloc(ctx, C.nbytes)
    tc.buffer_write(a, A)
    tc.buffer_write(b, B)

    small = tc.buffer_alloc(ctx, 2)
    try:
        try:
            tc.buffer_write(small, np.zeros(4, dtype=np.uint8))
            write_bounds_ok = False
        except ValueError:
            write_bounds_ok = True
        try:
            tc.buffer_read(small, np.zeros(4, dtype=np.uint8))
            read_bounds_ok = False
        except ValueError:
            read_bounds_ok = True
    finally:
        tc.buffer_free(ctx, small)
    host_bounds_ok = write_bounds_ok and read_bounds_ok

    tc.gemm(ctx, a, b, c, M, N, K, dtype="f16")
    tc.buffer_read(c, C)

    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    err = np.abs(C.astype(np.float32) - C_ref.astype(np.float32))
    rms = np.sqrt((err * err).mean())
    ref_rms = np.sqrt((C_ref.astype(np.float32) ** 2).mean())
    scaled = rms / (ref_rms + 1e-9)
    print(f"GEMM fp16 {M}x{N}x{K}:  max_abs={err.max():.3e}  scaled_rms={scaled:.3e}  "
          f"{'OK' if scaled < 1e-2 else 'FAIL'}")
    gemm_backend = tc.last_backend_name()
    gemm_backend_ok = gemm_backend in (
        "simdgroup_matrix", "tensorops_m5", "mps", "accelerate_cpu"
    )
    print(f"GEMM backend:         {gemm_backend}  "
          f"{'OK' if gemm_backend_ok else 'FAIL'}")

    tc.buffer_write(c, np.zeros_like(C))
    stream = tc.stream_create(ctx)
    tc.gemm_async(ctx, a, b, c, M, N, K, stream, dtype="f16")
    tc.stream_sync(stream)
    tc.stream_destroy(ctx, stream)
    tc.buffer_read(c, C)
    err_async = np.abs(C.astype(np.float32) - C_ref.astype(np.float32))
    rms_async = np.sqrt((err_async * err_async).mean())
    scaled_async = rms_async / (ref_rms + 1e-9)
    print(f"GEMM async fp16:       max_abs={err_async.max():.3e}  scaled_rms={scaled_async:.3e}  "
          f"{'OK' if scaled_async < 1e-2 else 'FAIL'}")
    print(f"Host buffer bounds:    {'OK' if host_bounds_ok else 'FAIL'}")

    buffer_layout_ok = _run_buffer_layout_check(ctx)
    print(f"Host buffer layouts:   {'OK' if buffer_layout_ok else 'FAIL'}")

    distributed_ok = _run_distributed_wrapper_check(ctx)
    print(f"Distributed wrapper:   {'OK' if distributed_ok else 'FAIL'}")

    batched_ok, batched_err = _run_batched_gemm_wrapper_check(ctx)
    print(f"GEMM batched fp16:     max_abs={batched_err:.3e}  "
          f"{'OK' if batched_ok else 'FAIL'}")

    padded_ok, padded_errs = _run_padded_gemm_wrapper_check(ctx)
    print(f"GEMM padded f32:       sync={padded_errs['sync']:.3e}  "
          f"async={padded_errs['async']:.3e}  "
          f"{'OK' if padded_ok else 'FAIL'}")

    batched_padded_ok, batched_padded_err = _run_batched_padded_gemm_wrapper_check(ctx)
    print(f"GEMM batched padded:   max_abs={batched_padded_err:.3e}  "
          f"{'OK' if batched_padded_ok else 'FAIL'}")

    conv_ok, conv_errs = _run_conv_wrapper_check(ctx)
    print(f"Conv2D wrapper:        fwd={conv_errs['forward']:.3e}  "
          f"dX={conv_errs['dX']:.3e}  dW={conv_errs['dW']:.3e}  "
          f"{'OK' if conv_ok else 'FAIL'}")

    attention_ok, attention_errs = _run_attention_wrapper_check(ctx)
    print(f"Attention wrapper:     scaled={attention_errs['out']:.3e}  "
          f"lse={attention_errs['lse']:.3e}  async={attention_errs['async']:.3e}  "
          f"bwd={attention_errs['bwd']:.3e}  "
          f"{'OK' if attention_ok else 'FAIL'}")

    qM, qN, qK = 1, 4, 64
    Xq_np = np.random.randn(qM, qK).astype(np.float16)
    Wq_fp16_np = np.random.randn(qN, qK).astype(np.float16)
    Yq_np = np.zeros((qM, qN), dtype=np.float16)
    q_bytes = tc.quantized_size("q4_0", qN, qK)
    q8_bytes = tc.quantized_size("q8_0", qN, qK)
    q_size_ok = q_bytes == qN * (qK // 32) * 18 and q8_bytes == qN * (qK // 32) * 34

    xq = tc.buffer_alloc(ctx, Xq_np.nbytes)
    wfp16 = tc.buffer_alloc(ctx, Wq_fp16_np.nbytes)
    wq = tc.buffer_alloc(ctx, q_bytes)
    wq8 = tc.buffer_alloc(ctx, q8_bytes)
    yq = tc.buffer_alloc(ctx, Yq_np.nbytes)
    yq_async = tc.buffer_alloc(ctx, Yq_np.nbytes)
    yq8 = tc.buffer_alloc(ctx, Yq_np.nbytes)
    tc.buffer_write(xq, Xq_np)
    tc.buffer_write(wfp16, Wq_fp16_np)
    tc.quantize_weights(ctx, wfp16, wq, "q4_0", qN, qK)
    tc.gemv_quantized(ctx, xq, wq, yq, "q4_0", qM, qN, qK)
    tc.buffer_read(yq, Yq_np)
    Yq_async_np = np.zeros((qM, qN), dtype=np.float16)
    q_stream = tc.stream_create(ctx)
    try:
        tc.gemv_quantized_async(ctx, xq, wq, yq_async, "q4_0", qM, qN, qK, q_stream)
        tc.stream_sync(q_stream)
    finally:
        tc.stream_destroy(ctx, q_stream)
    tc.buffer_read(yq_async, Yq_async_np)
    raw_q4 = ctypes.string_at(tc.buffer_map(wq), q_bytes)
    W_deq = _dequant_q4_0(raw_q4, qN, qK)
    Y_ref = (Xq_np.astype(np.float32) @ W_deq.T).astype(np.float16)
    q_err = np.abs(Yq_np.astype(np.float32) - Y_ref.astype(np.float32))
    q_async_err = np.abs(Yq_async_np.astype(np.float32) - Y_ref.astype(np.float32))
    q_ok = q_size_ok and q_err.max() < 2e-2 and q_async_err.max() < 2e-2
    print(f"Q4_0 GEMV wrapper:    max_abs={q_err.max():.3e}  "
          f"async={q_async_err.max():.3e}  "
          f"{'OK' if q_ok else 'FAIL'}")

    Yq8_np = np.zeros((qM, qN), dtype=np.float16)
    tc.quantize_weights(ctx, wfp16, wq8, "q8_0", qN, qK)
    tc.gemv_quantized(ctx, xq, wq8, yq8, "q8_0", qM, qN, qK)
    tc.buffer_read(yq8, Yq8_np)
    raw_q8 = ctypes.string_at(tc.buffer_map(wq8), q8_bytes)
    W8_deq = _dequant_q8_0(raw_q8, qN, qK)
    Y8_ref = (Xq_np.astype(np.float32) @ W8_deq.T).astype(np.float16)
    q8_err = np.abs(Yq8_np.astype(np.float32) - Y8_ref.astype(np.float32))
    q8_ok = q_size_ok and q8_err.max() < 2e-2
    print(f"Q8_0 GEMV wrapper:    max_abs={q8_err.max():.3e}  "
          f"{'OK' if q8_ok else 'FAIL'}")

    training_ok, training_errs = _run_training_wrapper_checks(ctx)
    max_training_err = max(training_errs.values())
    print(f"Training wrappers:    max_scaled_rms={max_training_err:.3e}  "
          f"{'OK' if training_ok else 'FAIL'}")

    tc.buffer_free(ctx, xq)
    tc.buffer_free(ctx, wfp16)
    tc.buffer_free(ctx, wq)
    tc.buffer_free(ctx, wq8)
    tc.buffer_free(ctx, yq)
    tc.buffer_free(ctx, yq_async)
    tc.buffer_free(ctx, yq8)

    gguf_path = tempfile.NamedTemporaryFile(prefix="tc_py_", suffix=".gguf", delete=False).name
    gguf_ok = False
    gb = None
    g = None
    loaded = None
    try:
        _write_test_gguf(gguf_path)
        g = tc.gguf_open(gguf_path)
        tensor = tc.gguf_get_tensor(g, "weight.test")
        tensor0 = tc.gguf_tensor_at(g, 0)
        qinfo = tc.gguf_tensor_quantized_matrix_info(tensor)
        gb = tc.gguf_tensor_to_buffer(ctx, g, "weight.test")
        copied = ctypes.string_at(tc.buffer_map(gb), tc.buffer_size(gb))
        loaded = tc.gguf_load_supported_tensors(ctx, g)
        loaded_tensor = tc.gguf_loaded_get_tensor(loaded, "weight.test")
        loaded_qinfo = tc.gguf_loaded_tensor_quantized_matrix_info(loaded_tensor)
        config = tc.gguf_get_llama_config(g)
        loaded_copied = ctypes.string_at(
            tc.buffer_map(loaded_tensor["buffer"]),
            loaded_tensor["n_bytes"],
        )
        with tc.GgufFile(gguf_path) as owned_g:
            owned_tensor = owned_g.get_tensor("weight.test")
            owned_tensor0 = owned_g.tensor_at(0)
            owned_config = owned_g.llama_config()
            owned_meta_ok = (
                owned_g.tensor_count() == 1 and
                owned_g.metadata_count() == 12 and
                owned_g.meta_get_str("general.architecture") == "llama" and
                owned_g.meta_get_str("general.name") == "python-test" and
                owned_g.meta_get_i64("llama.context_length", -1) == 2048 and
                abs(owned_g.meta_get_f64("llama.attention.layer_norm_rms_epsilon", -1.0) - 0.125) < 1e-12 and
                owned_g.meta_array_count("tokenizer.ggml.tokens") == 2 and
                owned_g.meta_array_get_str("tokenizer.ggml.tokens", 1) == "hello" and
                owned_config["context_length"] == 2048 and
                owned_config["vocab_size"] == 2
            )
            with owned_g.tensor_to_buffer(ctx, "weight.test") as owned_gb:
                owned_copied = ctypes.string_at(owned_gb.map(), owned_gb.size())
            with tc.Context() as owned_ctx:
                with owned_g.load_supported_tensors(owned_ctx) as owned_loaded:
                    owned_loaded_tensor = owned_loaded.get_tensor("weight.test")
                    owned_loaded_tensor0 = owned_loaded.tensor_at(0)
                    owned_loaded_buffer = owned_loaded_tensor.get("buffer")
                    owned_loaded_property_buffer = owned_loaded_tensor.buffer
                    owned_loaded_qinfo = tc.gguf_loaded_tensor_quantized_matrix_info(owned_loaded_tensor)
                    qmat = owned_loaded.quantized_matrix("weight.test")
                    x_ones = np.ones((1, 32), dtype=np.float16)
                    with owned_ctx.buffer_from_array(x_ones) as xbuf, qmat.output() as ybuf:
                        qmat.gemv(xbuf, ybuf)
                        qmat_y = ybuf.to_numpy((1, qmat.N), np.float16)
                    owned_loaded_ok = (
                        owned_loaded.tensor_count() == 1 and
                        owned_loaded.skipped_tensor_count() == 0 and
                        owned_loaded_tensor0["name"] == owned_loaded_tensor["name"] and
                        owned_loaded_qinfo["N"] == 1 and
                        owned_loaded_qinfo["K"] == 32 and
                        qmat.N == 1 and
                        qmat.K == 32 and
                        qmat.quant_type == tc.TC_QUANT_Q4_0 and
                        owned_loaded_buffer == owned_loaded_tensor["buffer"] and
                        owned_loaded_property_buffer == owned_loaded_tensor["buffer"] and
                        qmat_y[0, 0] == np.float16(40.0)
                    )
                try:
                    _ = owned_loaded_tensor["buffer"]
                    owned_tensor_lifetime_ok = False
                except RuntimeError:
                    owned_tensor_lifetime_ok = True
            with tc.Context() as owned_ctx:
                with owned_ctx.open_gguf(gguf_path) as ctx_g:
                    ctx_open_tensor = ctx_g.get_tensor("weight.test")
                    with owned_ctx.load_supported_tensors(ctx_g) as ctx_loaded:
                        ctx_loaded_tensor = ctx_loaded.tensor_at(0)
                        ctx_loaded_buffer = ctx_loaded_tensor.buffer
                        ctx_loaded_ok = (
                            ctx_g.tensor_count() == 1 and
                            ctx_g.metadata_count() == 12 and
                            ctx_open_tensor["name"] == "weight.test" and
                            ctx_loaded.tensor_count() == 1 and
                            ctx_loaded.skipped_tensor_count() == 0 and
                            ctx_loaded_tensor["name"] == "weight.test" and
                            ctx_loaded_buffer == ctx_loaded_tensor["buffer"]
                        )
        gguf_ok = (
            tc.gguf_tensor_count(g) == 1 and
            tc.gguf_metadata_count(g) == 12 and
            tc.gguf_meta_get_str(g, "general.architecture") == "llama" and
            tc.gguf_meta_get_str(g, "general.name") == "python-test" and
            tc.gguf_meta_get_i64(g, "llama.context_length", -1) == 2048 and
            abs(tc.gguf_meta_get_f64(g, "llama.attention.layer_norm_rms_epsilon", -1.0) - 0.125) < 1e-12 and
            tc.gguf_meta_array_count(g, "tokenizer.ggml.tokens") == 2 and
            tc.gguf_meta_array_get_str(g, "tokenizer.ggml.tokens", 1) == "hello" and
            abs(tc.gguf_meta_array_get_f64(g, "tokenizer.ggml.scores", 1, -1.0) - 0.25) < 1e-12 and
            config["context_length"] == 2048 and
            config["embedding_length"] == 4096 and
            config["feed_forward_length"] == 11008 and
            config["block_count"] == 32 and
            config["attention_head_count"] == 32 and
            config["attention_head_count_kv"] == 8 and
            config["rope_dimension_count"] == 128 and
            config["vocab_size"] == 2 and
            abs(config["rms_norm_epsilon"] - 0.125) < 1e-12 and
            tensor["name"] == "weight.test" and
            tensor["dims"] == (32, 1) and
            tensor["type"] == tc.TC_GGUF_TYPE_Q4_0 and
            tensor["n_bytes"] == 18 and
            owned_meta_ok and
            owned_tensor["name"] == "weight.test" and
            owned_tensor["dims"] == (32, 1) and
            owned_tensor0["name"] == owned_tensor["name"] and
            owned_copied == copied and
            owned_loaded_ok and
            owned_tensor_lifetime_ok and
            ctx_loaded_ok and
            qinfo["N"] == 1 and
            qinfo["K"] == 32 and
            qinfo["quant_type"] == tc.TC_QUANT_Q4_0 and
            qinfo["n_bytes"] == 18 and
            qinfo["buffer"] is None and
            tensor0["name"] == tensor["name"] and
            copied == struct.pack("<H", 0x3800) + bytes([0xAB]) * 16 and
            tc.gguf_loaded_tensor_count(loaded) == 1 and
            tc.gguf_loaded_skipped_tensor_count(loaded) == 0 and
            loaded_tensor["name"] == "weight.test" and
            loaded_tensor["type"] == tc.TC_GGUF_TYPE_Q4_0 and
            loaded_qinfo["N"] == 1 and
            loaded_qinfo["K"] == 32 and
            loaded_qinfo["quant_type"] == tc.TC_QUANT_Q4_0 and
            loaded_qinfo["buffer"] == loaded_tensor["buffer"] and
            loaded_copied == copied
        )
    finally:
        if loaded:
            tc.gguf_loaded_model_free(ctx, loaded)
        if gb:
            tc.buffer_free(ctx, gb)
        if g:
            tc.gguf_close(g)
        os.unlink(gguf_path)
    print(f"GGUF wrapper:         {'OK' if gguf_ok else 'FAIL'}")

    tc.buffer_free(ctx, a)
    tc.buffer_free(ctx, b)
    tc.buffer_free(ctx, c)
    tc.shutdown(ctx)

    owned_ok, owned_err = _run_owned_api_check()
    print(f"Owned Python API:     max_abs={owned_err:.3e}  "
          f"{'OK' if owned_ok else 'FAIL'}")

    ok = (
        scaled < 1e-2 and
        gemm_backend_ok and
        scaled_async < 1e-2 and
        host_bounds_ok and
        buffer_layout_ok and
        distributed_ok and
        batched_ok and
        padded_ok and
        batched_padded_ok and
        conv_ok and
        attention_ok and
        q_ok and
        q8_ok and
        training_ok and
        gguf_ok and
        owned_ok
    )
    return 0 if ok else 5

if __name__ == "__main__":
    sys.exit(main())
