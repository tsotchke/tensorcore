/*
 * tensorcore — Lie group exp/log/mul for SU(2) and SO(3).
 *
 * Math: see include/tensorcore/lie_groups.h. Both groups are 3-d so
 * closed-form (Rodrigues / Pauli) formulas suffice. fp32 throughout.
 *
 * Numerical safety:
 *   - Small-θ branches use Taylor (1 - θ²/6 for sinc, 1 - θ²/2 for cos)
 *     to avoid 0/0 in sin(θ)/θ.
 *   - cos θ clamped to [-1, 1] before acos in log paths.
 *   - SU(2) log handles the U = I corner cleanly (θ = 0 → zero vector).
 */

#include "tensorcore/lie_groups.h"
#include <cmath>
#include <cstring>

namespace {
constexpr float kSmallTheta = 1e-6f;

/* Write a 2×2 SU(2) matrix into U (8 floats, row-major interleaved complex). */
inline void set_su2(float* U,
                     float u00r, float u00i, float u01r, float u01i,
                     float u10r, float u10i, float u11r, float u11i) {
    U[0] = u00r; U[1] = u00i; U[2] = u01r; U[3] = u01i;
    U[4] = u10r; U[5] = u10i; U[6] = u11r; U[7] = u11i;
}
}  // namespace

extern "C" void tc_su2_exp(float a, float b, float c, float* U) {
    /* H = i (a σ_x + b σ_y + c σ_z), v = (a, b, c), θ = ||v||.
     *   exp(H) = cos θ · I + i sin θ / θ · (a σ_x + b σ_y + c σ_z)
     *
     * σ_x = [[0,1],[1,0]],  σ_y = [[0,-i],[i,0]],  σ_z = [[1,0],[0,-1]].
     *
     * So with s = sin θ / θ:
     *   U[0,0] = cos θ + i s · c
     *   U[0,1] = i s · a + s · b      = (s·b) + i (s·a)
     *   U[1,0] = i s · a - s · b      = -(s·b) + i (s·a)
     *   U[1,1] = cos θ - i s · c
     */
    const float theta = std::sqrt(a*a + b*b + c*c);
    float ct, sinc;
    if (theta < kSmallTheta) {
        ct   = 1.0f - 0.5f * theta * theta;
        sinc = 1.0f - theta * theta / 6.0f;
    } else {
        ct   = std::cos(theta);
        sinc = std::sin(theta) / theta;
    }
    const float sc = sinc;
    set_su2(U,
            ct,   sc * c,    sc * b,   sc * a,
           -sc * b, sc * a,  ct,      -sc * c);
}

extern "C" void tc_su2_log(const float* U, float* out_a, float* out_b, float* out_c) {
    /* U = cos θ · I + i sinc(θ) · (a σ_x + b σ_y + c σ_z).
     * Re(U[0,0]) = cos θ; θ = acos(clamp(Re(U[0,0]), -1, 1)).
     * From the off-diagonal terms:
     *   Im(U[0,0]) = sinc · c       (since I and σ_z contributions to (0,0))
     *   Re(U[0,1]) = sinc · b
     *   Im(U[0,1]) = sinc · a
     * Recover (a, b, c) by dividing by sinc(θ). */
    const float u00r = U[0], u00i = U[1];
    const float u01r = U[2], u01i = U[3];
    float ct = u00r;
    if (ct >  1.0f) ct =  1.0f;
    if (ct < -1.0f) ct = -1.0f;
    const float theta = std::acos(ct);
    float inv_sinc;
    if (theta < kSmallTheta) {
        /* sinc(θ) ≈ 1 - θ²/6 → 1/sinc ≈ 1 + θ²/6. */
        inv_sinc = 1.0f + theta * theta / 6.0f;
    } else {
        inv_sinc = theta / std::sin(theta);
    }
    *out_a = u01i * inv_sinc;
    *out_b = u01r * inv_sinc;
    *out_c = u00i * inv_sinc;
}

extern "C" void tc_su2_mul(const float* U, const float* V, float* out_UV) {
    /* (U · V)[r][c] = Σ_k U[r][k] · V[k][c]. */
    float W[8] = {0};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < 2; ++k) {
                const float ur = U[4*r + 2*k];
                const float ui = U[4*r + 2*k + 1];
                const float vr = V[4*k + 2*c];
                const float vi = V[4*k + 2*c + 1];
                W[4*r + 2*c]     += ur * vr - ui * vi;
                W[4*r + 2*c + 1] += ur * vi + ui * vr;
            }
        }
    }
    std::memcpy(out_UV, W, 8 * sizeof(float));
}

extern "C" void tc_so3_exp(float wx, float wy, float wz, float* R) {
    /* Rodrigues:  R = I + sinc(θ) K + (1-cos θ)/θ² · K². */
    const float theta_sq = wx*wx + wy*wy + wz*wz;
    const float theta = std::sqrt(theta_sq);

    float sinc, one_minus_cos_over_t2;
    if (theta < kSmallTheta) {
        sinc                    = 1.0f - theta_sq / 6.0f;
        one_minus_cos_over_t2   = 0.5f - theta_sq / 24.0f;
    } else {
        sinc                    = std::sin(theta) / theta;
        one_minus_cos_over_t2   = (1.0f - std::cos(theta)) / theta_sq;
    }

    /* K = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]].
     * Compute K and K² entries directly. */
    const float K[9] = {
         0.0f, -wz,    wy,
         wz,    0.0f, -wx,
        -wy,    wx,    0.0f,
    };
    /* K² entries (symmetric): */
    const float K2[9] = {
        -(wy*wy + wz*wz),  wx*wy,             wx*wz,
         wx*wy,           -(wx*wx + wz*wz),   wy*wz,
         wx*wz,            wy*wz,            -(wx*wx + wy*wy),
    };

    for (int i = 0; i < 9; ++i) {
        R[i] = ((i % 4) == 0 ? 1.0f : 0.0f)   /* I_{ij} */
             + sinc * K[i]
             + one_minus_cos_over_t2 * K2[i];
    }
}

extern "C" void tc_so3_log(const float* R, float* out_wx, float* out_wy, float* out_wz) {
    /* θ = acos((tr(R) - 1) / 2). ω = θ/(2 sin θ) · (R - R^T) skew-sym vector. */
    const float tr = R[0] + R[4] + R[8];
    float ct = (tr - 1.0f) * 0.5f;
    if (ct >  1.0f) ct =  1.0f;
    if (ct < -1.0f) ct = -1.0f;
    const float theta = std::acos(ct);

    /* Skew vector from R - R^T:
     *   2 ω = (R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]). */
    const float vx = R[7] - R[5];
    const float vy = R[2] - R[6];
    const float vz = R[3] - R[1];

    float scale;
    if (theta < kSmallTheta) {
        /* For small θ, R ≈ I + K with K = θ · K̂; (R - R^T)/2 = sin(θ)·K̂.
         * In the limit, ω ≈ (R - R^T)/2 (sin θ / θ → 1, ratio → 1/2). */
        scale = 0.5f;
    } else {
        scale = 0.5f * theta / std::sin(theta);
    }
    *out_wx = scale * vx;
    *out_wy = scale * vy;
    *out_wz = scale * vz;
}

extern "C" void tc_so3_mul(const float* R1, const float* R2, float* out_R12) {
    float W[9] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                W[3*i + j] += R1[3*i + k] * R2[3*k + j];
            }
        }
    }
    std::memcpy(out_R12, W, 9 * sizeof(float));
}

extern "C" void tc_su2_to_so3(const float* U, float* R) {
    /* Adjoint of U: R_{ij} = (1/2) tr(σ_i U σ_j U†).
     * Equivalently, parametrise U = (q0 + i q3, q2 + i q1; -q2 + i q1, q0 - i q3)
     * (quaternion (q0, q1, q2, q3)) and use the standard quaternion→rotation. */
    const float q0 =  U[0];   /* Re U[0,0] */
    const float q3 =  U[1];   /* Im U[0,0] */
    const float q2 =  U[2];   /* Re U[0,1] */
    const float q1 =  U[3];   /* Im U[0,1] */

    const float xx = q1*q1, yy = q2*q2, zz = q3*q3;
    const float xy = q1*q2, xz = q1*q3, yz = q2*q3;
    const float wx = q0*q1, wy = q0*q2, wz = q0*q3;

    R[0] = 1.0f - 2.0f*(yy + zz);
    R[1] = 2.0f*(xy - wz);
    R[2] = 2.0f*(xz + wy);
    R[3] = 2.0f*(xy + wz);
    R[4] = 1.0f - 2.0f*(xx + zz);
    R[5] = 2.0f*(yz - wx);
    R[6] = 2.0f*(xz - wy);
    R[7] = 2.0f*(yz + wx);
    R[8] = 1.0f - 2.0f*(xx + yy);
}
