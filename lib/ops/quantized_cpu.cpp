/*
 * tensorcore - portable CPU quantized GEMV.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"
#include "../core/cpu_float.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool fp16_matrix_bytes(int rows, int cols, size_t* out) {
    size_t elems = 0;
    if (rows <= 0 || cols <= 0) return false;
    if (!checked_mul((size_t)rows, (size_t)cols, &elems)) return false;
    return checked_mul(elems, sizeof(uint16_t), out);
}

tc_status_t validate_quantize_buffers(tc_context* ctx,
                                      const tc_buffer* W_fp16,
                                      tc_buffer* W_quant,
                                      tc_quant_t fmt,
                                      int N,
                                      int K) {
    size_t fp16_bytes = 0;
    if (!fp16_matrix_bytes(N, K, &fp16_bytes)) return TC_ERR_INVALID_ARG;
    const size_t quant_bytes = tc_quantized_size(fmt, N, K);
    if (quant_bytes == 0) return TC_ERR_INVALID_ARG;

    tc_status_t s = tc_buffer_validate(ctx, W_fp16, fp16_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, W_quant, quant_bytes);
}

tc_status_t validate_gemv_quantized_buffers(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* W_quant,
                                            tc_buffer* Y,
                                            tc_quant_t fmt,
                                            int M,
                                            int N,
                                            int K) {
    size_t x_bytes = 0;
    size_t y_bytes = 0;
    if (!fp16_matrix_bytes(M, K, &x_bytes) ||
        !fp16_matrix_bytes(M, N, &y_bytes)) {
        return TC_ERR_INVALID_ARG;
    }
    const size_t w_bytes = tc_quantized_size(fmt, N, K);
    if (w_bytes == 0) return TC_ERR_INVALID_ARG;

    tc_status_t s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, W_quant, w_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, Y, y_bytes);
}

tc_status_t validate_fused_rmsnorm_gemv_quantized_buffers(tc_context* ctx,
                                                          const tc_buffer* X,
                                                          const tc_buffer* gamma,
                                                          const tc_buffer* W_quant,
                                                          tc_buffer* Y,
                                                          tc_quant_t fmt,
                                                          int M,
                                                          int N,
                                                          int K) {
    tc_status_t s = validate_gemv_quantized_buffers(ctx, X, W_quant, Y, fmt, M, N, K);
    if (s != TC_OK) return s;

    size_t gamma_bytes = 0;
    if (!fp16_matrix_bytes(1, K, &gamma_bytes)) return TC_ERR_INVALID_ARG;
    return tc_buffer_validate(ctx, gamma, gamma_bytes);
}

} /* namespace */

extern "C" size_t tc_quantized_size(tc_quant_t fmt, int N, int K) {
    if (N <= 0 || K <= 0 || K % 32 != 0) return 0;
    const size_t nblocks = (size_t)(K / 32);
    switch (fmt) {
        case TC_QUANT_Q4_0:
            return (size_t)N * nblocks * 18u;
        case TC_QUANT_Q8_0:
            return (size_t)N * nblocks * 34u;
    }
    return 0;
}

extern "C" tc_status_t tc_quantize_weights(tc_context* ctx,
                                           const tc_buffer* W_fp16,
                                           tc_buffer* W_quant,
                                           tc_quant_t fmt,
                                           int N,
                                           int K) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!W_fp16 || !W_quant || N <= 0 || K <= 0 || K % 32 != 0) return TC_ERR_INVALID_ARG;
    tc_status_t s = validate_quantize_buffers(ctx, W_fp16, W_quant, fmt, N, K);
    if (s != TC_OK) return s;

    void* w_src = nullptr;
    void* w_dst = nullptr;
    s = tc_buffer_map((tc_buffer*)W_fp16, &w_src);
    if (s != TC_OK) return s;
    s = tc_buffer_map(W_quant, &w_dst);
    if (s != TC_OK) return s;

    const uint16_t* W = (const uint16_t*)w_src;
    uint8_t* Wq = (uint8_t*)w_dst;
    const int nblocks = K / 32;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < nblocks; ++b) {
            const int k0 = b * 32;
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i) {
                max_abs = std::max(max_abs, std::fabs(tc_cpu_f16_to_f32(W[(size_t)n * K + k0 + i])));
            }
            if (fmt == TC_QUANT_Q4_0) {
                uint8_t* dst = Wq + ((size_t)n * nblocks + b) * 18u;
                const float scale = max_abs / 8.0f;
                const float inv = scale > 0.0f ? 8.0f / max_abs : 0.0f;
                ((uint16_t*)dst)[0] = tc_cpu_f32_to_f16(scale);
                for (int i = 0; i < 16; ++i) {
                    const float lo = tc_cpu_f16_to_f32(W[(size_t)n * K + k0 + i]);
                    const float hi = tc_cpu_f16_to_f32(W[(size_t)n * K + k0 + i + 16]);
                    int q_lo = (int)std::lround(lo * inv) + 8;
                    int q_hi = (int)std::lround(hi * inv) + 8;
                    q_lo = std::max(0, std::min(15, q_lo));
                    q_hi = std::max(0, std::min(15, q_hi));
                    dst[2 + i] = (uint8_t)((q_hi << 4) | q_lo);
                }
            } else if (fmt == TC_QUANT_Q8_0) {
                uint8_t* dst = Wq + ((size_t)n * nblocks + b) * 34u;
                const float scale = max_abs / 127.0f;
                const float inv = scale > 0.0f ? 127.0f / max_abs : 0.0f;
                ((uint16_t*)dst)[0] = tc_cpu_f32_to_f16(scale);
                int8_t* qs = (int8_t*)(dst + 2);
                for (int i = 0; i < 32; ++i) {
                    const float value = tc_cpu_f16_to_f32(W[(size_t)n * K + k0 + i]);
                    int q = (int)std::lround(value * inv);
                    q = std::max(-128, std::min(127, q));
                    qs[i] = (int8_t)q;
                }
            } else {
                return TC_ERR_UNSUPPORTED_DTYPE;
            }
        }
    }
    return tc_record_dispatch("tc_quantize_weights", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_gemv_quantized(tc_context* ctx,
                                         const tc_buffer* X,
                                         const tc_buffer* W_quant,
                                         tc_buffer* Y,
                                         tc_quant_t fmt,
                                         int M,
                                         int N,
                                         int K) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !W_quant || !Y || M <= 0 || N <= 0 || K <= 0 || K % 32 != 0) {
        return TC_ERR_INVALID_ARG;
    }
    tc_status_t s = validate_gemv_quantized_buffers(ctx, X, W_quant, Y, fmt, M, N, K);
    if (s != TC_OK) return s;

    void* x_ptr = nullptr;
    void* w_ptr = nullptr;
    void* y_ptr = nullptr;
    s = tc_buffer_map((tc_buffer*)X, &x_ptr);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)W_quant, &w_ptr);
    if (s != TC_OK) return s;
    s = tc_buffer_map(Y, &y_ptr);
    if (s != TC_OK) return s;

    const uint16_t* Xp = (const uint16_t*)x_ptr;
    const uint8_t* Wq = (const uint8_t*)w_ptr;
    uint16_t* Yp = (uint16_t*)y_ptr;
    const int nblocks = K / 32;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            if (fmt == TC_QUANT_Q4_0) {
                const uint8_t* row = Wq + (size_t)n * nblocks * 18u;
                for (int b = 0; b < nblocks; ++b) {
                    const uint8_t* block = row + (size_t)b * 18u;
                    const float scale = tc_cpu_f16_to_f32(((const uint16_t*)block)[0]);
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t packed = block[2 + i];
                        const int k_lo = b * 32 + i;
                        const int k_hi = k_lo + 16;
                        const float x_lo = tc_cpu_f16_to_f32(Xp[(size_t)m * K + k_lo]);
                        const float x_hi = tc_cpu_f16_to_f32(Xp[(size_t)m * K + k_hi]);
                        acc += x_lo * scale * (float)((packed & 0x0f) - 8);
                        acc += x_hi * scale * (float)((packed >> 4) - 8);
                    }
                }
            } else if (fmt == TC_QUANT_Q8_0) {
                const uint8_t* row = Wq + (size_t)n * nblocks * 34u;
                for (int b = 0; b < nblocks; ++b) {
                    const uint8_t* block = row + (size_t)b * 34u;
                    const float scale = tc_cpu_f16_to_f32(((const uint16_t*)block)[0]);
                    const int8_t* qs = (const int8_t*)(block + 2);
                    for (int i = 0; i < 32; ++i) {
                        const int k = b * 32 + i;
                        const float x = tc_cpu_f16_to_f32(Xp[(size_t)m * K + k]);
                        acc += x * scale * (float)qs[i];
                    }
                }
            } else {
                return TC_ERR_UNSUPPORTED_DTYPE;
            }
            Yp[(size_t)m * N + n] = tc_cpu_f32_to_f16(acc);
        }
    }
    return tc_record_dispatch("tc_gemv_quantized", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_fused_rmsnorm_gemv_quantized(tc_context* ctx,
                                                       const tc_buffer* X,
                                                       const tc_buffer* gamma,
                                                       const tc_buffer* W_quant,
                                                       tc_buffer* Y,
                                                       tc_quant_t fmt,
                                                       int M,
                                                       int N,
                                                       int K,
                                                       float eps) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!X || !gamma || !W_quant || !Y || M <= 0 || N <= 0 || K <= 0 || K % 32 != 0) {
        return TC_ERR_INVALID_ARG;
    }
    tc_status_t s = validate_fused_rmsnorm_gemv_quantized_buffers(
        ctx, X, gamma, W_quant, Y, fmt, M, N, K);
    if (s != TC_OK) return s;

    void* x_ptr = nullptr;
    void* g_ptr = nullptr;
    void* w_ptr = nullptr;
    void* y_ptr = nullptr;
    s = tc_buffer_map((tc_buffer*)X, &x_ptr);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)gamma, &g_ptr);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)W_quant, &w_ptr);
    if (s != TC_OK) return s;
    s = tc_buffer_map(Y, &y_ptr);
    if (s != TC_OK) return s;

    const uint16_t* Xp = (const uint16_t*)x_ptr;
    const uint16_t* Gp = (const uint16_t*)g_ptr;
    const uint8_t* Wq = (const uint8_t*)w_ptr;
    uint16_t* Yp = (uint16_t*)y_ptr;
    const int nblocks = K / 32;

    static thread_local std::vector<float> tls_xnorm;
    if ((int)tls_xnorm.size() < M * K) tls_xnorm.resize((size_t)M * K);

    for (int m = 0; m < M; ++m) {
        const uint16_t* x_row = Xp + (size_t)m * K;
        float* x_norm = tls_xnorm.data() + (size_t)m * K;

        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            const float x = tc_cpu_f16_to_f32(x_row[k]);
            ss += (double)x * (double)x;
        }
        const float rstd = 1.0f / std::sqrt((float)(ss / (double)K) + eps);
        for (int k = 0; k < K; ++k) {
            x_norm[k] = tc_cpu_f16_to_f32(x_row[k]) * rstd * tc_cpu_f16_to_f32(Gp[k]);
        }

        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            if (fmt == TC_QUANT_Q4_0) {
                const uint8_t* row = Wq + (size_t)n * nblocks * 18u;
                for (int b = 0; b < nblocks; ++b) {
                    const uint8_t* block = row + (size_t)b * 18u;
                    const float scale = tc_cpu_f16_to_f32(((const uint16_t*)block)[0]);
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t packed = block[2 + i];
                        const int k_lo = b * 32 + i;
                        const int k_hi = k_lo + 16;
                        acc += x_norm[k_lo] * scale * (float)((packed & 0x0f) - 8);
                        acc += x_norm[k_hi] * scale * (float)((packed >> 4) - 8);
                    }
                }
            } else if (fmt == TC_QUANT_Q8_0) {
                const uint8_t* row = Wq + (size_t)n * nblocks * 34u;
                for (int b = 0; b < nblocks; ++b) {
                    const uint8_t* block = row + (size_t)b * 34u;
                    const float scale = tc_cpu_f16_to_f32(((const uint16_t*)block)[0]);
                    const int8_t* qs = (const int8_t*)(block + 2);
                    for (int i = 0; i < 32; ++i) {
                        const int k = b * 32 + i;
                        acc += x_norm[k] * scale * (float)qs[i];
                    }
                }
            } else {
                return TC_ERR_UNSUPPORTED_DTYPE;
            }
            Yp[(size_t)m * N + n] = tc_cpu_f32_to_f16(acc);
        }
    }

    return tc_record_dispatch("tc_fused_rmsnorm_gemv_quantized", TC_BACKEND_PORTABLE_CPU, TC_OK);
}

extern "C" tc_status_t tc_gemv_quantized_async(tc_context* ctx,
                                               const tc_buffer* X,
                                               const tc_buffer* W_quant,
                                               tc_buffer* Y,
                                               tc_quant_t fmt,
                                               int M,
                                               int N,
                                               int K,
                                               tc_stream* stream) {
    if (!stream) return TC_ERR_INVALID_ARG;
    return tc_gemv_quantized(ctx, X, W_quant, Y, fmt, M, N, K);
}
