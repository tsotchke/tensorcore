/* tensorcore_torch_ext.cpp — minimal PyTorch ↔ tensorcore bridge.
 *
 * Exposes `matmul(A, B)` returning A @ B via tc_gemm and an opt-in
 * `set_default_matmul()` dispatcher hook for torch.matmul.
 *
 * v0.1 scope:
 *   - fp32 only (matches the AMX backend's session-1.5 support).
 *   - 2-D inputs, no batching, no transpose (caller transposes upstream).
 *   - CPU tensors only — Apple unified memory means tensorcore's CPU
 *     backend reads from the same physical RAM PyTorch is using.
 *     Uses `tc_buffer_from_ptr` to wrap PyTorch's allocator output
 *     in a zero-copy tc_buffer view, eliminating the alloc-and-memcpy
 *     that dominated v0.1 perf at small sizes.
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

extern "C" {
#include "tensorcore/tensorcore.h"
}

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

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
        if (rc != TC_OK || g_ctx == nullptr) {
            init_lock.clear(std::memory_order_release);
            throw std::runtime_error(
                std::string("tc_init failed: ") +
                std::to_string(static_cast<int>(rc)));
        }
        initialized.store(true, std::memory_order_release);
    }
    init_lock.clear(std::memory_order_release);
}

bool is_tc_matmul_eligible(const at::Tensor& A, const at::Tensor& B) {
    return A.dtype() == torch::kFloat32 &&
           B.dtype() == torch::kFloat32 &&
           A.layout() == torch::kStrided &&
           B.layout() == torch::kStrided &&
           A.dim() == 2 &&
           B.dim() == 2 &&
           A.size(1) == B.size(0) &&
           A.device().is_cpu() &&
           B.device().is_cpu();
}

void register_privateuse1_name() {
    if (!c10::is_privateuse1_backend_registered()) {
        c10::register_privateuse1_backend("tensorcore");
    }
}

}  // namespace

at::Tensor tc_matmul_fp32(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "tc_matmul requires fp32 A");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "tc_matmul requires fp32 B");
    TORCH_CHECK(A.dim() == 2, "tc_matmul requires 2-D A; got dim=", A.dim());
    TORCH_CHECK(B.dim() == 2, "tc_matmul requires 2-D B; got dim=", B.dim());
    TORCH_CHECK(A.size(1) == B.size(0),
                "shape mismatch: A is ", A.sizes(), " B is ", B.sizes());
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(),
                "tc_matmul currently CPU only (unified memory on Apple)");

    /* Contiguous, row-major inputs are required by the wire format. */
    const auto A_c = A.contiguous();
    const auto B_c = B.contiguous();

    const int M = static_cast<int>(A_c.size(0));
    const int K = static_cast<int>(A_c.size(1));
    const int N = static_cast<int>(B_c.size(1));

    ensure_ctx();

    /* Zero-copy wrap: tc_buffer_from_ptr returns a tc_buffer that aliases
     * the PyTorch tensor's data buffer (and the output tensor's data
     * buffer for C). Apple unified memory makes this trivially correct.
     * Lifetime: A_c, B_c, and `out` outlive the GEMM call (synchronous),
     * so the wrapped pointers stay valid. */
    tc_buffer *bA = nullptr, *bB = nullptr, *bC = nullptr;
    auto cleanup = [&]() {
        if (bA) tc_buffer_free(g_ctx, bA);
        if (bB) tc_buffer_free(g_ctx, bB);
        if (bC) tc_buffer_free(g_ctx, bC);
    };

    const size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

    auto out = torch::empty({M, N}, A_c.options());

    if (tc_buffer_from_ptr(g_ctx, const_cast<float*>(A_c.data_ptr<float>()),
                           bytes_a, &bA) != TC_OK ||
        tc_buffer_from_ptr(g_ctx, const_cast<float*>(B_c.data_ptr<float>()),
                           bytes_b, &bB) != TC_OK ||
        tc_buffer_from_ptr(g_ctx, out.data_ptr<float>(),
                           bytes_c, &bC) != TC_OK) {
        cleanup();
        throw std::runtime_error("tc_buffer_from_ptr failed (build against "
                                 "tensorcore with the from_ptr API)");
    }

    tc_gemm_desc desc{};
    desc.M = M; desc.N = N; desc.K = K;
    desc.a_dtype = TC_DTYPE_F32;
    desc.b_dtype = TC_DTYPE_F32;
    desc.c_dtype = TC_DTYPE_F32;
    desc.accum_dtype = TC_DTYPE_F32;
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

    /* C was written directly into `out.data_ptr` — no copy needed. */
    cleanup();
    return out;
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
          "tc_matmul(A: Tensor[fp32, MxK], B: Tensor[fp32, KxN]) -> Tensor[fp32, MxN]");
    m.def("set_default_matmul", &tc_set_default_matmul,
          py::arg("enabled") = true,
          "Enable or disable the opt-in torch.matmul dispatcher hook; returns the previous state");
    m.def("default_matmul_enabled", &tc_default_matmul_enabled,
          "Return whether torch.matmul is currently routed through tensorcore for eligible fp32 CPU matrices");
    m.def("privateuse1_backend_name", &tc_privateuse1_backend_name,
          "Return the registered PrivateUse1 backend name used by tensorcore");
    m.def("last_backend_name", &tc_last_backend_name,
          "Return the tensorcore backend name that served the last GEMM");
}
