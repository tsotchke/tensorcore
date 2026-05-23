/* tensorcore_torch_ext.cpp — minimal PyTorch ↔ tensorcore bridge.
 *
 * Exposes `matmul(A, B)` returning A @ B via tc_gemm and an opt-in
 * `set_default_matmul()` dispatcher hook for torch.matmul.
 *
 * v0.1 scope:
 *   - fp32 and bf16 2-D matmul.
 *   - 2-D inputs, no batching, no transpose (caller transposes upstream).
 *   - CPU tensors only — Apple unified memory means tensorcore's CPU
 *     backend reads from the same physical RAM PyTorch is using.
 *     Uses `tc_buffer_from_ptr` when the runtime can wrap PyTorch's
 *     allocator output directly, with an alloc-and-copy fallback for
 *     runtimes that require stricter wrapper alignment.
 *
 * Backend selection is honored via the same env vars as the rest of
 * tensorcore: `TC_USE_AMX_GEMM=1` picks the reverse-engineered AMX
 * matrix-coprocessor backend, `TC_USE_NEON_GEMM=1` picks the OpenMP+NEON
 * BLIS-style backend, default falls through to CBLAS (Accelerate on
 * macOS, OpenBLAS/MKL on Linux).
 *
 * Build via setup.py; consumes the tensorcore static library + headers
 * from $TENSORCORE_ROOT (defaults to ../..).
 */

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/ops/matmul_native.h>
#include <c10/core/DeviceType.h>
#include <pybind11/stl.h>

extern "C" {
#include "tensorcore/tensorcore.h"
}

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace py = pybind11;

/* Process-wide tensorcore context, lazily constructed. The C API's
 * `tc_context*` is reusable across many GEMMs; we don't want to pay
 * `tc_init` cost per matmul. */
tc_context* g_ctx = nullptr;
std::atomic<bool> g_default_matmul{false};

void ensure_ctx() {
    static std::atomic<bool> initialized{false};
    if (initialized.load(std::memory_order_acquire)) return;
    /* Race tolerant: tc_init is idempotent against a global only because
     * we serialize first-call via the static `initialized` flag below. */
    static std::atomic_flag init_lock = ATOMIC_FLAG_INIT;
    while (init_lock.test_and_set(std::memory_order_acquire)) {}
    if (!initialized.load(std::memory_order_relaxed)) {
        const auto rc = tc_init(&g_ctx);
        const bool ok =
            (rc == TC_OK || rc == TC_ERR_ALREADY_INITIALIZED) &&
            g_ctx != nullptr;
        if (!ok) {
            init_lock.clear(std::memory_order_release);
            throw std::runtime_error(
                std::string("tc_init failed: ") +
                std::to_string(static_cast<int>(rc)));
        }
        initialized.store(true, std::memory_order_release);
    }
    init_lock.clear(std::memory_order_release);
}

std::string tc_matmul_eligibility_reason(const at::Tensor& A, const at::Tensor& B) {
    if (A.dtype() != B.dtype()) return "dtype_mismatch";
    if (A.dtype() != torch::kFloat32 && A.dtype() != torch::kBFloat16) {
        return "unsupported_dtype";
    }
    if (A.layout() != torch::kStrided || B.layout() != torch::kStrided) {
        return "non_strided_layout";
    }
    if (A.dim() != 2 || B.dim() != 2) return "rank_mismatch";
    if (A.size(1) != B.size(0)) return "shape_mismatch";
    if (!A.device().is_cpu() || !B.device().is_cpu()) return "non_cpu_device";
    return "eligible";
}

bool is_tc_matmul_eligible(const at::Tensor& A, const at::Tensor& B) {
    return tc_matmul_eligibility_reason(A, B) == "eligible";
}

std::vector<int64_t> tensor_sizes(const at::Tensor& t) {
    return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

void register_privateuse1_name() {
    if (!c10::is_privateuse1_backend_registered()) {
        c10::register_privateuse1_backend("tensorcore");
    }
}

void check_dim_fits_tc(const char* name, int64_t value) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
                "tc_matmul dimension ", name, " is outside tensorcore int32 range: ",
                value);
}

size_t checked_matrix_bytes(const char* name, int64_t rows, int64_t cols,
                            size_t elem_size) {
    const uint64_t max_size = std::numeric_limits<size_t>::max();
    const uint64_t r = static_cast<uint64_t>(rows);
    const uint64_t c = static_cast<uint64_t>(cols);
    TORCH_CHECK(c == 0 || r <= max_size / c,
                "tc_matmul byte-size overflow for ", name);
    const uint64_t elems = r * c;
    TORCH_CHECK(elem_size == 0 || elems <= max_size / elem_size,
                "tc_matmul byte-size overflow for ", name);
    return static_cast<size_t>(elems * elem_size);
}

}  // namespace

py::dict tc_matmul_eligibility(const at::Tensor& A, const at::Tensor& B) {
    const std::string reason = tc_matmul_eligibility_reason(A, B);
    py::dict result;
    result["eligible"] = (reason == "eligible");
    result["reason"] = reason;
    result["a_sizes"] = tensor_sizes(A);
    result["b_sizes"] = tensor_sizes(B);
    result["a_dtype"] = std::string(c10::toString(A.scalar_type()));
    result["b_dtype"] = std::string(c10::toString(B.scalar_type()));
    result["a_device"] = A.device().str();
    result["b_device"] = B.device().str();
    result["default_matmul_enabled"] =
        g_default_matmul.load(std::memory_order_acquire);
    return result;
}

at::Tensor tc_matmul_fp32(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dtype() == B.dtype(),
                "tc_matmul requires A and B to share dtype");
    TORCH_CHECK(A.dtype() == torch::kFloat32 || A.dtype() == torch::kBFloat16,
                "tc_matmul supports fp32 and bf16; got ", A.dtype());
    TORCH_CHECK(A.dim() == 2, "tc_matmul requires 2-D A; got dim=", A.dim());
    TORCH_CHECK(B.dim() == 2, "tc_matmul requires 2-D B; got dim=", B.dim());
    TORCH_CHECK(A.size(1) == B.size(0),
                "shape mismatch: A is ", A.sizes(), " B is ", B.sizes());
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(),
                "tc_matmul currently CPU only (unified memory on Apple)");

    /* Contiguous, row-major inputs are required by the wire format. */
    const auto A_c = A.contiguous();
    const auto B_c = B.contiguous();

    const int64_t M64 = A_c.size(0);
    const int64_t K64 = A_c.size(1);
    const int64_t N64 = B_c.size(1);
    check_dim_fits_tc("M", M64);
    check_dim_fits_tc("N", N64);
    check_dim_fits_tc("K", K64);

    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    const bool is_bf16 = (A.dtype() == torch::kBFloat16);
    const size_t elem  = is_bf16 ? sizeof(uint16_t) : sizeof(float);
    const tc_dtype_t tc_dt = is_bf16 ? TC_DTYPE_BF16 : TC_DTYPE_F32;

    auto out = torch::empty({M64, N64}, A_c.options());

    /* PyTorch matmul accepts empty result dimensions. The tensorcore C ABI
     * intentionally rejects zero-byte buffers, so handle those cases at the
     * bridge boundary. For K==0 and a non-empty output, BLAS semantics are
     * C := 0 for this alpha=1, beta=0 wrapper. */
    if (M == 0 || N == 0) return out;
    if (K == 0) return out.zero_();

    ensure_ctx();

    /* Prefer zero-copy tc_buffer_from_ptr when the runtime accepts the
     * pointer. Metal builds require page-aligned no-copy wrappers, so fall
     * back to alloc+memcpy for ordinary PyTorch allocator outputs. */
    tc_buffer *bA = nullptr, *bB = nullptr, *bC = nullptr;
    bool c_direct = false;
    auto cleanup = [&]() {
        if (bA) tc_buffer_free(g_ctx, bA);
        if (bB) tc_buffer_free(g_ctx, bB);
        if (bC) tc_buffer_free(g_ctx, bC);
    };

    auto make_input_buffer = [&](const void* src, size_t bytes,
                                 tc_buffer** out_buf) -> bool {
        if (tc_buffer_from_ptr(g_ctx, const_cast<void*>(src), bytes, out_buf) == TC_OK) {
            return true;
        }
        if (tc_buffer_alloc(g_ctx, bytes, out_buf) != TC_OK) return false;
        void* dst = nullptr;
        if (tc_buffer_map(*out_buf, &dst) != TC_OK || !dst) return false;
        std::memcpy(dst, src, bytes);
        return true;
    };

    auto make_output_buffer = [&](void* dst, size_t bytes,
                                  tc_buffer** out_buf, bool* direct) -> bool {
        if (tc_buffer_from_ptr(g_ctx, dst, bytes, out_buf) == TC_OK) {
            *direct = true;
            return true;
        }
        *direct = false;
        return tc_buffer_alloc(g_ctx, bytes, out_buf) == TC_OK;
    };

    const size_t bytes_a = checked_matrix_bytes("A", M64, K64, elem);
    const size_t bytes_b = checked_matrix_bytes("B", K64, N64, elem);
    const size_t bytes_c = checked_matrix_bytes("C", M64, N64, elem);

    if (!make_input_buffer(A_c.data_ptr(), bytes_a, &bA) ||
        !make_input_buffer(B_c.data_ptr(), bytes_b, &bB) ||
        !make_output_buffer(out.data_ptr(), bytes_c, &bC, &c_direct)) {
        cleanup();
        throw std::runtime_error("tensorcore PyTorch bridge buffer setup failed");
    }

    tc_gemm_desc desc{};
    desc.M = M; desc.N = N; desc.K = K;
    desc.a_dtype = tc_dt;
    desc.b_dtype = tc_dt;
    desc.c_dtype = tc_dt;
    desc.accum_dtype = TC_DTYPE_F32;     /* bf16 in/out, fp32 accum (CBLAS) */
    desc.alpha = 1.0f;
    desc.beta  = 0.0f;
    desc.transpose_a = false;
    desc.transpose_b = false;
    desc.lda = K;
    desc.ldb = N;
    desc.ldc = N;

    const auto rc = tc_gemm(g_ctx, &desc, bA, bB, bC);
    if (rc != TC_OK) {
        cleanup();
        throw std::runtime_error(
            std::string("tc_gemm failed: ") +
            std::to_string(static_cast<int>(rc)));
    }

    if (!c_direct) {
        void* cp = nullptr;
        if (tc_buffer_map(bC, &cp) != TC_OK || !cp) {
            cleanup();
            throw std::runtime_error("tensorcore PyTorch bridge output map failed");
        }
        std::memcpy(out.data_ptr(), cp, bytes_c);
    }

    cleanup();
    return out;
}

at::Tensor tc_matmul_bf16(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
                "tc_matmul_bf16 requires both inputs to be torch.bfloat16; got ",
                A.dtype(), " and ", B.dtype());
    return tc_matmul_fp32(A, B);
}

const char* tc_last_backend_name() {
    return tc_backend_name(tc_last_backend());
}

at::Tensor tc_matmul_dispatch(const at::Tensor& A, const at::Tensor& B) {
    if (g_default_matmul.load(std::memory_order_acquire) &&
        is_tc_matmul_eligible(A, B)) {
        return tc_matmul_fp32(A, B);
    }

    return at::native::matmul(A, B);
}

at::Tensor tc_matmul_autograd_cpu(const at::Tensor& A, const at::Tensor& B) {
    if (!A.requires_grad() &&
        !B.requires_grad() &&
        g_default_matmul.load(std::memory_order_acquire) &&
        is_tc_matmul_eligible(A, B)) {
        return tc_matmul_fp32(A, B);
    }

    return at::native::matmul(A, B);
}

at::Tensor tc_matmul_privateuse1(const at::Tensor& A, const at::Tensor& B) {
    return tc_matmul_fp32(A, B);
}

bool tc_set_default_matmul(bool enabled = true) {
    register_privateuse1_name();
    return g_default_matmul.exchange(enabled, std::memory_order_acq_rel);
}

bool tc_default_matmul_enabled() {
    return g_default_matmul.load(std::memory_order_acquire);
}

std::string tc_privateuse1_backend_name() {
    register_privateuse1_name();
    return c10::get_privateuse1_backend(true);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_dispatch));
}

TORCH_LIBRARY_IMPL(aten, AutogradCPU, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_autograd_cpu));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_privateuse1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_privateuse1_name();
    m.def("matmul", &tc_matmul_fp32,
          "tc_matmul(A: Tensor[fp32|bf16, MxK], B: Tensor[fp32|bf16, KxN]) -> Tensor[MxN]");
    m.def("matmul_bf16", &tc_matmul_bf16,
          "tc_matmul_bf16(A: Tensor[bf16, MxK], B: Tensor[bf16, KxN]) -> Tensor[bf16, MxN]");
    m.def("is_matmul_eligible", &is_tc_matmul_eligible,
          "Return whether A and B can route through tensorcore's torch.matmul dispatcher hook");
    m.def("matmul_eligibility", &tc_matmul_eligibility,
          "Return a structured reason for tensorcore torch.matmul dispatcher eligibility");
    m.def("set_default_matmul", &tc_set_default_matmul,
          py::arg("enabled") = true,
          "Enable or disable the opt-in torch.matmul dispatcher hook; returns the previous state");
    m.def("default_matmul_enabled", &tc_default_matmul_enabled,
          "Return whether torch.matmul is currently routed through tensorcore for eligible fp32/bf16 CPU matrices");
    m.def("privateuse1_backend_name", &tc_privateuse1_backend_name,
          "Return the registered PrivateUse1 backend name used by tensorcore");
    m.def("last_backend_name", &tc_last_backend_name,
          "Return the tensorcore backend name that served the last GEMM");
}
