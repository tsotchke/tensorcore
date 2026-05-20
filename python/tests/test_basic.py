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

    with tc.Context() as ctx:
        a = ctx.buffer(A.nbytes).write(A)
        b = ctx.buffer(B.nbytes).write(B)
        c = ctx.buffer(C.nbytes)
        ctx.gemm(a, b, c, M, N, K)
        c.read(C)

        with ctx.stream() as stream:
            ctx.gemm_async(a, b, c, M, N, K, stream)
            stream.sync()
        c.read(C_async)

    err = np.max(np.abs(C.astype(np.float32) - C_ref.astype(np.float32)))
    err_async = np.max(np.abs(C_async.astype(np.float32) - C_ref.astype(np.float32)))
    return err == 0.0 and err_async == 0.0, max(float(err), float(err_async))


def main():
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

    tc.gemm(ctx, a, b, c, M, N, K, dtype="f16")
    tc.buffer_read(c, C)

    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    err = np.abs(C.astype(np.float32) - C_ref.astype(np.float32))
    rms = np.sqrt((err * err).mean())
    ref_rms = np.sqrt((C_ref.astype(np.float32) ** 2).mean())
    scaled = rms / (ref_rms + 1e-9)
    print(f"GEMM fp16 {M}x{N}x{K}:  max_abs={err.max():.3e}  scaled_rms={scaled:.3e}  "
          f"{'OK' if scaled < 1e-2 else 'FAIL'}")

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
    yq8 = tc.buffer_alloc(ctx, Yq_np.nbytes)
    tc.buffer_write(xq, Xq_np)
    tc.buffer_write(wfp16, Wq_fp16_np)
    tc.quantize_weights(ctx, wfp16, wq, "q4_0", qN, qK)
    tc.gemv_quantized(ctx, xq, wq, yq, "q4_0", qM, qN, qK)
    tc.buffer_read(yq, Yq_np)
    raw_q4 = ctypes.string_at(tc.buffer_map(wq), q_bytes)
    W_deq = _dequant_q4_0(raw_q4, qN, qK)
    Y_ref = (Xq_np.astype(np.float32) @ W_deq.T).astype(np.float16)
    q_err = np.abs(Yq_np.astype(np.float32) - Y_ref.astype(np.float32))
    q_ok = q_size_ok and q_err.max() < 2e-2
    print(f"Q4_0 GEMV wrapper:    max_abs={q_err.max():.3e}  "
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
            with owned_g.tensor_to_buffer(ctx, "weight.test") as owned_gb:
                owned_copied = ctypes.string_at(owned_gb.map(), owned_gb.size())
            with owned_g.load_supported_tensors(ctx) as owned_loaded:
                owned_loaded_tensor = owned_loaded.get_tensor("weight.test")
                owned_loaded_qinfo = tc.gguf_loaded_tensor_quantized_matrix_info(owned_loaded_tensor)
                owned_loaded_ok = (
                    owned_loaded.tensor_count() == 1 and
                    owned_loaded.skipped_tensor_count() == 0 and
                    owned_loaded_qinfo["N"] == 1 and
                    owned_loaded_qinfo["K"] == 32
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
            owned_tensor["name"] == "weight.test" and
            owned_tensor["dims"] == (32, 1) and
            owned_copied == copied and
            owned_loaded_ok and
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
        scaled_async < 1e-2 and
        q_ok and
        q8_ok and
        training_ok and
        gguf_ok and
        owned_ok
    )
    return 0 if ok else 5

if __name__ == "__main__":
    sys.exit(main())
