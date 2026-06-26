/*
 * tensorcore — Riemannian metric tensor + numerical Christoffels.
 *
 * Math + ABI: include/tensorcore/metric.h.
 *
 * Inversion: Gauss-Jordan with partial pivoting (no LAPACK dep).
 * Det check uses |pivot| < 1e-12 as the singular threshold; that's
 * generous for fp32 metrics on practical curvatures (Poincaré
 * λ²(x) grows fast as ‖x‖ → 1/√c so the inverse stays small but
 * never explodes).
 *
 * Christoffels are evaluated by symmetric central difference (O(h²)).
 * Default h = 1e-3 is the sweet spot for fp32: smaller h amplifies
 * cancellation in (g(p+h) - g(p-h)); larger h loses truncation
 * accuracy. Adjustable per-call if the curvature scale demands it.
 */

#include "tensorcore/metric.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr float kDefaultStep   = 1e-3f;
constexpr float kSingularTol   = 1e-12f;
}  // namespace

extern "C" float tc_metric_apply(tc_metric_fn fn, void* user,
                                 const float* point, int dim,
                                 const float* v, const float* w) {
    std::vector<float> g(static_cast<size_t>(dim) * dim);
    fn(point, dim, g.data(), user);
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            acc += static_cast<double>(g[i * dim + j]) *
                   static_cast<double>(v[i]) *
                   static_cast<double>(w[j]);
        }
    }
    return static_cast<float>(acc);
}

extern "C" int tc_metric_inverse(const float* g, int dim, float* g_inv) {
    const int n = dim;
    std::vector<float> aug(static_cast<size_t>(n) * 2 * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i * 2 * n + j]       = g[i * n + j];
            aug[i * 2 * n + n + j]   = (i == j) ? 1.0f : 0.0f;
        }
    }
    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        float pivot_abs = std::fabs(aug[col * 2 * n + col]);
        for (int r = col + 1; r < n; ++r) {
            const float a = std::fabs(aug[r * 2 * n + col]);
            if (a > pivot_abs) { pivot_abs = a; pivot_row = r; }
        }
        if (pivot_abs < kSingularTol) {
            return 1;
        }
        if (pivot_row != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(aug[col * 2 * n + j], aug[pivot_row * 2 * n + j]);
            }
        }
        const float inv_pivot = 1.0f / aug[col * 2 * n + col];
        for (int j = 0; j < 2 * n; ++j) aug[col * 2 * n + j] *= inv_pivot;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const float factor = aug[r * 2 * n + col];
            if (factor == 0.0f) continue;
            for (int j = 0; j < 2 * n; ++j) {
                aug[r * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            g_inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    return 0;
}

extern "C" float tc_metric_inverse_apply(tc_metric_fn fn, void* user,
                                         const float* point, int dim,
                                         const float* alpha,
                                         const float* beta) {
    std::vector<float> g(static_cast<size_t>(dim) * dim);
    std::vector<float> g_inv(static_cast<size_t>(dim) * dim);
    fn(point, dim, g.data(), user);
    if (tc_metric_inverse(g.data(), dim, g_inv.data()) != 0) {
        return 0.0f;
    }
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            acc += static_cast<double>(g_inv[i * dim + j]) *
                   static_cast<double>(alpha[i]) *
                   static_cast<double>(beta[j]);
        }
    }
    return static_cast<float>(acc);
}

extern "C" int tc_metric_christoffel(tc_metric_fn fn, void* user,
                                     const float* point, int dim,
                                     float h, float* out_christoffel) {
    if (h <= 0.0f) h = kDefaultStep;
    const int d = dim;
    const size_t d2 = static_cast<size_t>(d) * d;

    std::vector<float> g0(d2);
    fn(point, d, g0.data(), user);
    std::vector<float> g_inv(d2);
    if (tc_metric_inverse(g0.data(), d, g_inv.data()) != 0) {
        return 1;
    }

    /* Partial ∂_a g_{ij} via central difference over each coordinate. */
    std::vector<float> dg(static_cast<size_t>(d) * d2);  /* [a][i][j] */
    std::vector<float> p_plus(d), p_minus(d);
    std::vector<float> g_plus(d2), g_minus(d2);
    for (int a = 0; a < d; ++a) {
        std::memcpy(p_plus.data(),  point, sizeof(float) * d);
        std::memcpy(p_minus.data(), point, sizeof(float) * d);
        p_plus[a]  += h;
        p_minus[a] -= h;
        fn(p_plus.data(),  d, g_plus.data(),  user);
        fn(p_minus.data(), d, g_minus.data(), user);
        const float inv_2h = 0.5f / h;
        for (size_t k = 0; k < d2; ++k) {
            dg[a * d2 + k] = (g_plus[k] - g_minus[k]) * inv_2h;
        }
    }

    /* Γ^k_{ij} = ½ g^{kℓ} ( ∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij} ). */
    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                double sum = 0.0;
                for (int l = 0; l < d; ++l) {
                    const float di_gjl = dg[i * d2 + j * d + l];
                    const float dj_gil = dg[j * d2 + i * d + l];
                    const float dl_gij = dg[l * d2 + i * d + j];
                    sum += static_cast<double>(g_inv[k * d + l]) *
                           (di_gjl + dj_gil - dl_gij);
                }
                out_christoffel[k * d * d + i * d + j] =
                    static_cast<float>(0.5 * sum);
            }
        }
    }
    return 0;
}

/* ---- Stock metric callbacks ---- */

extern "C" void tc_metric_euclidean(const float* point, int dim,
                                    float* out_g, void* user) {
    (void)point; (void)user;
    std::memset(out_g, 0, sizeof(float) * static_cast<size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i) out_g[i * dim + i] = 1.0f;
}

extern "C" void tc_metric_poincare(const float* point, int dim,
                                   float* out_g, void* user) {
    const float c = *static_cast<const float*>(user);
    double norm2 = 0.0;
    for (int i = 0; i < dim; ++i) norm2 += static_cast<double>(point[i]) * point[i];
    const double denom = 1.0 - static_cast<double>(c) * norm2;
    /* λ²(x) = (2 / (1 - c ‖x‖²))². If x escapes the ball we clamp denom. */
    const double safe_denom = (denom > 1e-12) ? denom : 1e-12;
    const double lambda2 = 4.0 / (safe_denom * safe_denom);
    std::memset(out_g, 0, sizeof(float) * static_cast<size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i) {
        out_g[i * dim + i] = static_cast<float>(lambda2);
    }
}

extern "C" void tc_metric_sphere_stereographic(const float* point, int dim,
                                               float* out_g, void* user) {
    const float r = *static_cast<const float*>(user);
    const double r2 = static_cast<double>(r) * r;
    double norm2 = 0.0;
    for (int i = 0; i < dim; ++i) norm2 += static_cast<double>(point[i]) * point[i];
    const double denom = r2 + norm2;
    const double factor = (2.0 * r2) / denom;
    const double scale  = factor * factor;
    std::memset(out_g, 0, sizeof(float) * static_cast<size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i) {
        out_g[i * dim + i] = static_cast<float>(scale);
    }
}
