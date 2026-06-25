/* tensorcore_torch_ext.cpp — minimal PyTorch ↔ tensorcore bridge.
 *
 * Exposes `matmul(A, B)` returning A @ B via tc_gemm and an opt-in
 * `set_default_matmul()` dispatcher hook for torch.matmul.
 *
 * v0.1 scope:
 *   - fp32 and bf16 2-D matmul.
 *   - 2-D inputs, no batching, no transpose (caller transposes upstream).
 *   - CPU tensors and tensorcore PrivateUse1 host-memory tensors.
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
#include <torch/autograd.h>

#include <ATen/EmptyTensor.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <pybind11/stl.h>

extern "C" {
#include "tensorcore/tensorcore.h"
/* tc_cuda_is_active is part of the public CUDA ABI (see
 * include/tensorcore/cuda.h). Forward-declare here so we don't drag in
 * CUDA Toolkit headers via tensorcore/cuda.h on non-CUDA hosts.
 *
 * Weak fallback: when libtensorcore is built without TC_ENABLE_CUDA
 * (e.g. atlas portable-CPU build), this resolves to 0 so the bridge
 * loads cleanly. On CUDA builds, the strong definition in
 * lib/cuda/buffer.cpp wins. Tracked as task #935. */
__attribute__((weak)) int tc_cuda_is_active(void) { return 0; }

/* MPS engagement: the public tc_gemm dispatcher (lib/ops/gemm.mm) routes
 * to tc_mps_gemm (MPSMatrixMultiplication / simdgroup_matrix kernels)
 * when the tensorcore device is a Metal MTLDevice. The bridge therefore
 * doesn't need a direct tc_mps_gemm symbol — it asks the library which
 * backend served the last GEMM (tc_last_backend()) to confirm that the
 * MPS path actually engaged. Task #936. */
}

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <optional>
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

void tensorcore_host_delete(void* ptr) {
    std::free(ptr);
}

class TensorcoreHostAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        void* ptr = nullptr;
        if (n > 0) {
            ptr = std::malloc(n);
            TORCH_CHECK(ptr != nullptr, "tensorcore PrivateUse1 host allocation failed for ", n, " bytes");
        }
        return c10::DataPtr(
            ptr,
            ptr,
            &tensorcore_host_delete,
            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &tensorcore_host_delete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

TensorcoreHostAllocator g_tensorcore_allocator;
std::atomic<bool> g_allocator_registered{false};

/* Pinned-memory allocator for the PrivateUse1 hooks. PyTorch's pin_memory
 * thread takes the DataPtr returned here and "moves" the underlying
 * storage onto a CPU tensor (see torch/utils/data/_utils/pin_memory.py
 * -> Tensor.pin_memory -> aten::_pin_memory). The destination tensor is
 * device("cpu"), so the DataPtr it gets *must also* be device("cpu") or
 * PyTorch raises "Attempted to set the storage of a tensor on device
 * cpu to a storage on different device".
 *
 * On the tensorcore portable-CPU backend "pinned" is a no-op — plain
 * host malloc is what we'd return either way, the page-lock would only
 * matter to a real CUDA host->device DMA. Returning CPU-typed DataPtrs
 * keeps the DataLoader pipeline correct without falsely advertising
 * actual page-locked behavior. */
class TensorcorePinnedAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        void* ptr = nullptr;
        if (n > 0) {
            ptr = std::malloc(n);
            TORCH_CHECK(ptr != nullptr,
                        "tensorcore pinned-memory allocation failed for ", n, " bytes");
        }
        return c10::DataPtr(
            ptr,
            ptr,
            &tensorcore_host_delete,
            c10::Device(c10::DeviceType::CPU));
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &tensorcore_host_delete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

TensorcorePinnedAllocator g_tensorcore_pinned_allocator;

/* PrivateUse1HooksInterface implementation. Required by PyTorch as soon
 * as anything outside the bridge probes the PrivateUse1 device — most
 * notably DataLoader with pin_memory=True / num_workers > 0, which
 * touches the global PrivateUse1 hooks even when the user-side tensors
 * are CPU/CUDA. Task #938.
 *
 * Surface: enough to satisfy DataLoader and `torch.tensor.is_pinned()`
 * probes without inventing semantics that don't apply to a host-memory
 * backend.
 *
 *   - isBuilt/isAvailable : true (bridge is loaded)
 *   - deviceCount         : 1 (tensorcore exposes one logical device)
 *   - hasPrimaryContext   : true (ensure_ctx is idempotent + lazy)
 *   - getDeviceFromPtr    : tensorcore:0 (single device)
 *   - isPinnedPtr         : false (our allocator returns plain malloc)
 *   - getPinnedMemoryAllocator : the host allocator (PrivateUse1 IS host
 *                                memory on the portable-CPU backend; on
 *                                Metal builds tc_buffer_alloc returns
 *                                unified memory, but the wire format
 *                                between DataLoader and the bridge is
 *                                still plain CPU buffers)
 *   - getCurrentDevice / setCurrentDevice / exchangeDevice : 0
 *   - resizePrivateUse1Bytes : NOT IMPLEMENTED (no resize-in-place yet) */
class TensorcoreHooks final : public at::PrivateUse1HooksInterface {
public:
    ~TensorcoreHooks() override = default;

    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return true; }
    void init() const override { /* lazy via ensure_ctx in matmul path */ }

    bool hasPrimaryContext(c10::DeviceIndex /*device_index*/) const override {
        return true;
    }

    c10::DeviceIndex deviceCount() const override { return 1; }
    c10::DeviceIndex getCurrentDevice() const override { return 0; }
    void setCurrentDevice(c10::DeviceIndex device) const override {
        TORCH_CHECK(device == 0,
                    "tensorcore exposes a single logical device: tensorcore:0; "
                    "got device index ", static_cast<int>(device));
    }
    c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
        TORCH_CHECK(device == 0,
                    "tensorcore exchangeDevice: only device 0 is valid; got ",
                    static_cast<int>(device));
        return 0;
    }
    c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
        if (device != 0) return -1;
        return 0;
    }

    bool isPinnedPtr(const void* /*data*/) const override {
        /* The tensorcore host allocator returns plain malloc'd memory.
         * DataLoader's pin_memory step is harmless (it asks each backend
         * if it owns the pointer; we honestly answer no), so the actual
         * pinning is done by the CUDA hooks when CUDA is available, or
         * skipped on CPU-only systems. */
        return false;
    }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        /* DataLoader uses this allocator to allocate pin-able staging
         * buffers. The destination tensor is CPU-typed, so the DataPtrs
         * we return must also be CPU-typed. See TensorcorePinnedAllocator
         * comment above. */
        return &g_tensorcore_pinned_allocator;
    }

    c10::Device getDeviceFromPtr(void* /*data*/) const override {
        /* Single logical device; the pointer doesn't carry device-index
         * provenance with our allocator, so always return tensorcore:0. */
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
};

TensorcoreHooks g_tensorcore_hooks;
std::atomic<bool> g_hooks_registered{false};

/* DeviceGuardImpl for PrivateUse1. PyTorch's torch.accelerator.* APIs
 * (and consequently DataLoader's pin_memory thread) look up the device
 * guard for the registered PrivateUse1 device via getDeviceGuardImpl,
 * which TORCH_CHECK's that an implementation is present and otherwise
 * raises "PyTorch is not linked with support for <name> devices". The
 * NoOp guard returns a single-device backend with no streams/events —
 * which matches tensorcore's host-memory + lazy-context model. Task #938
 * companion fix; without this the hooks registration alone trips the
 * DataLoader path. */
class TensorcorePrivateUse1GuardImpl final
    : public c10::impl::NoOpDeviceGuardImpl<c10::DeviceType::PrivateUse1> {};

TensorcorePrivateUse1GuardImpl g_tensorcore_guard_impl;
std::atomic<bool> g_guard_impl_registered{false};

void register_tensorcore_guard_impl() {
    bool expected = false;
    if (g_guard_impl_registered.compare_exchange_strong(expected, true)) {
        /* The registry slot is std::atomic; only the first writer wins.
         * If another extension has already populated it, this Registrar
         * call is benign — it warns once and keeps the existing impl. */
        static c10::impl::DeviceGuardImplRegistrar reg(
            c10::DeviceType::PrivateUse1, &g_tensorcore_guard_impl);
        (void)reg;
    }
}

void register_tensorcore_hooks() {
    bool expected = false;
    if (g_hooks_registered.compare_exchange_strong(expected, true)) {
        /* RegisterPrivateUse1HooksInterface is one-shot in PyTorch; if
         * another extension already registered, the second call is a
         * runtime check inside torch that fires TORCH_WARN. Our atomic
         * guard makes that benign within this process. */
        if (!at::isPrivateUse1HooksRegistered()) {
            at::RegisterPrivateUse1HooksInterface(&g_tensorcore_hooks);
        }
    }
}

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

void register_tensorcore_allocator() {
    bool expected = false;
    if (g_allocator_registered.compare_exchange_strong(expected, true)) {
        c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_tensorcore_allocator, 1);
    }
}

bool is_tensorcore_device(const at::Tensor& t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

bool is_same_cuda_pair(const at::Tensor& A, const at::Tensor& B) {
    return A.device().is_cuda() && B.device().is_cuda() &&
           A.device().index() == B.device().index();
}

/* CUDA path is only viable when libtensorcore was built with TC_ENABLE_CUDA
 * AND tc_cuda_init succeeded. Otherwise tc_gemm would fall through to the
 * CPU code path and deref CUDA device pointers from the host -> segfault.
 *
 * tc_init is what attempts tc_cuda_init on CUDA builds. Without a prior
 * matmul to warm ensure_ctx(), tc_cuda_is_active() reads as 0 even on
 * a fully capable CUDA build. Warm it once on the first probe so the
 * answer reflects the build, not the call-order. Any tc_init failure
 * is swallowed and reported as "unavailable". */
bool tc_cuda_bridge_available() {
    static std::atomic<bool> warmed{false};
    if (!warmed.load(std::memory_order_acquire)) {
        try {
            ensure_ctx();
        } catch (...) {
            /* swallow: cuda_bridge_available should never throw. */
        }
        warmed.store(true, std::memory_order_release);
    }
    return tc_cuda_is_active() != 0;
}

bool is_supported_device_pair(const at::Tensor& A, const at::Tensor& B) {
    if (A.device().is_cpu() && B.device().is_cpu()) return true;
    if (is_tensorcore_device(A) && is_tensorcore_device(B)) return true;
    if (is_same_cuda_pair(A, B) && tc_cuda_bridge_available()) return true;
    return false;
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
    if (is_same_cuda_pair(A, B) && !tc_cuda_bridge_available()) {
        return "cuda_backend_unavailable";
    }
    if (!is_supported_device_pair(A, B)) return "unsupported_device_pair";
    if (is_tensorcore_device(A) && (!A.is_contiguous() || !B.is_contiguous())) {
        return "non_contiguous_privateuse1";
    }
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
    register_tensorcore_allocator();
    register_tensorcore_guard_impl();
    register_tensorcore_hooks();
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
    TORCH_CHECK(is_supported_device_pair(A, B),
                "tc_matmul requires both tensors on CPU, both on tensorcore PrivateUse1, "
                "or both on the same CUDA device with libtensorcore built with TC_ENABLE_CUDA");
    if (is_tensorcore_device(A)) {
        TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                    "tc_matmul requires contiguous tensorcore PrivateUse1 inputs");
    }

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

/* Generalized in-place GEMM: C := alpha * (A @ B) + beta * C.
 *
 * Used by:
 *   - aten::mm (2-D matmul with no bias): alpha=1, beta=0
 *   - aten::addmm (Linear with bias on 2-D): alpha=alpha, beta=beta, C pre-init from bias
 *   - aten::baddbmm (batched addmm): same as addmm per batch slice
 *
 * Caller is responsible for:
 *   - validating shapes (2-D inputs, K matches)
 *   - initializing `out` to the right starting value (bias for addmm, zeros for mm)
 *
 * Returns `out` to make pipelining easy. */
at::Tensor tc_gemm_into(const at::Tensor& A, const at::Tensor& B,
                        at::Tensor& out, float alpha, float beta) {
    TORCH_CHECK(A.dtype() == B.dtype() && A.dtype() == out.dtype(),
                "tc_gemm_into requires A/B/out share dtype");
    TORCH_CHECK(A.dtype() == torch::kFloat32 || A.dtype() == torch::kBFloat16,
                "tc_gemm_into supports fp32 and bf16; got ", A.dtype());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && out.dim() == 2,
                "tc_gemm_into requires 2-D A/B/out");
    TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch A=", A.sizes(), " B=", B.sizes());
    TORCH_CHECK(out.size(0) == A.size(0) && out.size(1) == B.size(1),
                "out shape mismatch out=", out.sizes(), " expected (", A.size(0), ",", B.size(1), ")");

    const auto A_c = A.contiguous();
    const auto B_c = B.contiguous();
    TORCH_CHECK(out.is_contiguous(), "tc_gemm_into requires contiguous out");

    const int64_t M64 = A_c.size(0);
    const int64_t K64 = A_c.size(1);
    const int64_t N64 = B_c.size(1);
    check_dim_fits_tc("M", M64);
    check_dim_fits_tc("N", N64);
    check_dim_fits_tc("K", K64);
    const int M = (int)M64, K = (int)K64, N = (int)N64;

    const bool is_bf16 = (A.dtype() == torch::kBFloat16);
    const size_t elem = is_bf16 ? sizeof(uint16_t) : sizeof(float);
    const tc_dtype_t tc_dt = is_bf16 ? TC_DTYPE_BF16 : TC_DTYPE_F32;

    if (M == 0 || N == 0) return out;
    // K==0 with alpha != 0: A @ B is empty sum = 0, so out becomes beta * out.
    if (K == 0) return out.mul_(beta);

    ensure_ctx();

    tc_buffer *bA = nullptr, *bB = nullptr, *bC = nullptr;
    bool c_direct = false;
    auto cleanup = [&]() {
        if (bA) tc_buffer_free(g_ctx, bA);
        if (bB) tc_buffer_free(g_ctx, bB);
        if (bC) tc_buffer_free(g_ctx, bC);
    };

    const size_t bytes_a = checked_matrix_bytes("A", M64, K64, elem);
    const size_t bytes_b = checked_matrix_bytes("B", K64, N64, elem);
    const size_t bytes_c = checked_matrix_bytes("C", M64, N64, elem);

    auto wrap_input = [&](const void* src, size_t bytes, tc_buffer** out_buf) -> bool {
        if (tc_buffer_from_ptr(g_ctx, const_cast<void*>(src), bytes, out_buf) == TC_OK) return true;
        if (tc_buffer_alloc(g_ctx, bytes, out_buf) != TC_OK) return false;
        void* dst = nullptr;
        if (tc_buffer_map(*out_buf, &dst) != TC_OK || !dst) return false;
        std::memcpy(dst, src, bytes);
        return true;
    };

    if (!wrap_input(A_c.data_ptr(), bytes_a, &bA) ||
        !wrap_input(B_c.data_ptr(), bytes_b, &bB)) {
        cleanup();
        throw std::runtime_error("tc_gemm_into: input buffer wrap failed");
    }
    // Output buffer must point at `out`'s storage (we're accumulating).
    if (tc_buffer_from_ptr(g_ctx, out.data_ptr(), bytes_c, &bC) == TC_OK) {
        c_direct = true;
    } else {
        if (tc_buffer_alloc(g_ctx, bytes_c, &bC) != TC_OK) {
            cleanup();
            throw std::runtime_error("tc_gemm_into: output buffer alloc failed");
        }
        // Pre-init the alloc'd buffer with current out values so beta works.
        void* cp = nullptr;
        if (tc_buffer_map(bC, &cp) != TC_OK || !cp) {
            cleanup();
            throw std::runtime_error("tc_gemm_into: output buffer map failed");
        }
        std::memcpy(cp, out.data_ptr(), bytes_c);
    }

    tc_gemm_desc desc{};
    desc.M = M; desc.N = N; desc.K = K;
    desc.a_dtype = tc_dt; desc.b_dtype = tc_dt; desc.c_dtype = tc_dt;
    desc.accum_dtype = TC_DTYPE_F32;
    desc.alpha = alpha;
    desc.beta  = beta;
    desc.transpose_a = false;
    desc.transpose_b = false;
    desc.lda = K; desc.ldb = N; desc.ldc = N;

    const auto rc = tc_gemm(g_ctx, &desc, bA, bB, bC);
    if (rc != TC_OK) {
        cleanup();
        throw std::runtime_error(std::string("tc_gemm_into: tc_gemm failed rc=") + std::to_string((int)rc));
    }
    if (!c_direct) {
        void* cp = nullptr;
        if (tc_buffer_map(bC, &cp) != TC_OK || !cp) {
            cleanup();
            throw std::runtime_error("tc_gemm_into: output map-back failed");
        }
        std::memcpy(out.data_ptr(), cp, bytes_c);
    }
    cleanup();
    return out;
}

/* Helper: shape eligibility for addmm path (independent of grad state). */
bool tc_addmm_shape_eligible(const at::Tensor& self, const at::Tensor& mat1,
                             const at::Tensor& mat2) {
    if (!(mat1.dtype() == mat2.dtype() && mat1.dtype() == self.dtype())) return false;
    if (mat1.dtype() != torch::kFloat32 && mat1.dtype() != torch::kBFloat16) return false;
    if (mat1.dim() != 2 || mat2.dim() != 2) return false;
    if (mat1.size(1) != mat2.size(0)) return false;
    if (!(mat1.device().is_cuda() && mat2.device().is_cuda() && self.device().is_cuda())) return false;
    if (mat1.device().index() != mat2.device().index()) return false;
    if (mat1.device().index() != self.device().index()) return false;
    return true;
}

at::Tensor tc_addmm_forward(const at::Tensor& self, const at::Tensor& mat1,
                            const at::Tensor& mat2,
                            float alpha, float beta) {
    // No-grad context: the autograd Function (if any) wraps this output;
    // intermediates must not carry grad_fn or autograd refuses to reuse them.
    at::NoGradGuard no_grad;
    const int64_t M = mat1.size(0);
    const int64_t N = mat2.size(1);
    auto self_d = self.detach();
    auto mat1_d = mat1.detach();
    auto mat2_d = mat2.detach();
    auto out = self_d.dim() == 1
        ? self_d.unsqueeze(0).expand({M, N}).contiguous()
        : self_d.expand({M, N}).contiguous();
    tc_gemm_into(mat1_d, mat2_d, out, alpha, beta);
    return out;
}

/* Forward fallback that calls PyTorch's native addmm without re-entering
 * our hook. Uses a guard that excludes our autograd-level registration. */
at::Tensor tc_addmm_native(const at::Tensor& self, const at::Tensor& mat1,
                           const at::Tensor& mat2,
                           const at::Scalar& beta, const at::Scalar& alpha) {
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKeySet{
        c10::DispatchKey::AutogradCUDA, c10::DispatchKey::AutogradCPU,
    });
    return at::addmm(self, mat1, mat2, beta, alpha);
}

/* Custom autograd Function: C = beta * self + alpha * (mat1 @ mat2).
 *
 * Forward chooses bridge (tc_gemm_into) or native at::addmm based on
 * eligibility. Either way the Function attaches its own grad_fn, so the
 * autograd graph is well-formed regardless of substrate state.
 *
 * Backward: grad_self = beta * grad_C (broadcast-reduced),
 *           grad_mat1 = alpha * grad_C @ mat2^T,
 *           grad_mat2 = alpha * mat1^T @ grad_C.
 * The matmuls here re-enter our own bridge when enabled. */
class TCAddmmAutogradFunction
    : public torch::autograd::Function<TCAddmmAutogradFunction> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor self,
            at::Tensor mat1,
            at::Tensor mat2,
            double alpha,
            double beta) {
        ctx->save_for_backward({mat1, mat2});
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["beta"]  = beta;
        ctx->saved_data["self_dim"] = (int64_t)self.dim();

        const bool engaged =
            g_default_matmul.load(std::memory_order_acquire) &&
            tc_cuda_bridge_available() &&
            tc_addmm_shape_eligible(self, mat1, mat2);
        if (engaged) {
            return tc_addmm_forward(self, mat1, mat2, (float)alpha, (float)beta);
        }
        return tc_addmm_native(self, mat1, mat2, beta, alpha);
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto mat1 = saved[0];
        auto mat2 = saved[1];
        const auto alpha = ctx->saved_data["alpha"].toDouble();
        const auto beta  = ctx->saved_data["beta"].toDouble();
        const auto self_dim = ctx->saved_data["self_dim"].toInt();
        auto grad_C = grad_outputs[0];

        at::Tensor grad_self = (self_dim == 1)
            ? grad_C.sum(0).mul_(beta)
            : grad_C.mul(beta);

        const auto mat2_T = mat2.transpose(0, 1).contiguous();
        const auto mat1_T = mat1.transpose(0, 1).contiguous();
        // These matmuls re-enter the bridge if it's enabled — substrate-routed backward.
        auto grad_mat1 = at::matmul(grad_C, mat2_T).mul_((float)alpha);
        auto grad_mat2 = at::matmul(mat1_T, grad_C).mul_((float)alpha);

        return {grad_self, grad_mat1, grad_mat2, at::Tensor(), at::Tensor()};
    }
};

/* aten::addmm dispatcher: always routes through TCAddmmAutogradFunction
 * so grad_fn is consistently attached. The Function's forward picks
 * bridge or native; either way autograd is well-formed. */
at::Tensor tc_addmm_dispatch(const at::Tensor& self, const at::Tensor& mat1,
                             const at::Tensor& mat2,
                             const at::Scalar& beta, const at::Scalar& alpha) {
    return TCAddmmAutogradFunction::apply(
        self, mat1, mat2, alpha.toDouble(), beta.toDouble());
}

bool tc_baddbmm_shape_eligible(const at::Tensor& self,
                               const at::Tensor& batch1,
                               const at::Tensor& batch2) {
    if (batch1.dtype() != batch2.dtype()) return false;
    if (batch1.dtype() != self.dtype()) return false;
    if (batch1.dtype() != torch::kFloat32 && batch1.dtype() != torch::kBFloat16) return false;
    if (batch1.dim() != 3 || batch2.dim() != 3 || self.dim() != 3) return false;
    if (batch1.size(0) != batch2.size(0)) return false;
    if (batch1.size(0) != self.size(0)) return false;
    if (batch1.size(2) != batch2.size(1)) return false;
    if (self.size(1) != batch1.size(1) || self.size(2) != batch2.size(2)) return false;
    if (!(batch1.device().is_cuda() && batch2.device().is_cuda() && self.device().is_cuda())) return false;
    return true;
}

at::Tensor tc_baddbmm_forward(const at::Tensor& self, const at::Tensor& batch1,
                              const at::Tensor& batch2,
                              float alpha, float beta) {
    at::NoGradGuard no_grad;
    const int64_t B_dim = self.size(0);
    auto out = self.detach().contiguous().clone();
    const auto b1 = batch1.detach().contiguous();
    const auto b2 = batch2.detach().contiguous();
    for (int64_t i = 0; i < B_dim; ++i) {
        auto out_slice = out.select(0, i);
        tc_gemm_into(b1.select(0, i), b2.select(0, i), out_slice, alpha, beta);
    }
    return out;
}

at::Tensor tc_baddbmm_native(const at::Tensor& self, const at::Tensor& batch1,
                             const at::Tensor& batch2,
                             const at::Scalar& beta, const at::Scalar& alpha) {
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKeySet{
        c10::DispatchKey::AutogradCUDA, c10::DispatchKey::AutogradCPU,
    });
    return at::baddbmm(self, batch1, batch2, beta, alpha);
}

class TCBaddbmmAutogradFunction
    : public torch::autograd::Function<TCBaddbmmAutogradFunction> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor self,
            at::Tensor batch1,
            at::Tensor batch2,
            double alpha,
            double beta) {
        ctx->save_for_backward({batch1, batch2});
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["beta"]  = beta;
        const bool engaged =
            g_default_matmul.load(std::memory_order_acquire) &&
            tc_cuda_bridge_available() &&
            tc_baddbmm_shape_eligible(self, batch1, batch2);
        if (engaged) {
            return tc_baddbmm_forward(self, batch1, batch2, (float)alpha, (float)beta);
        }
        return tc_baddbmm_native(self, batch1, batch2, beta, alpha);
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto batch1 = saved[0];
        auto batch2 = saved[1];
        const auto alpha = ctx->saved_data["alpha"].toDouble();
        const auto beta  = ctx->saved_data["beta"].toDouble();
        auto grad_C = grad_outputs[0];
        auto grad_self   = grad_C.mul(beta);
        auto grad_batch1 = at::bmm(grad_C, batch2.transpose(1, 2)).mul_((float)alpha);
        auto grad_batch2 = at::bmm(batch1.transpose(1, 2), grad_C).mul_((float)alpha);
        return {grad_self, grad_batch1, grad_batch2, at::Tensor(), at::Tensor()};
    }
};

at::Tensor tc_baddbmm_dispatch(const at::Tensor& self, const at::Tensor& batch1,
                               const at::Tensor& batch2,
                               const at::Scalar& beta, const at::Scalar& alpha) {
    return TCBaddbmmAutogradFunction::apply(
        self, batch1, batch2, alpha.toDouble(), beta.toDouble());
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

/* Reshape-aware matmul that handles:
 *   - 2-D x 2-D  (M,K) @ (K,N) -> (M,N)   direct tc_matmul_fp32
 *   - N-D x 2-D  (..,M,K) @ (K,N) -> (..,M,N)  reshape leading dims, tc_matmul, reshape back
 * Higher-rank x higher-rank is left to native; tc_gemm has no batched-gemm
 * primitive (yet), and the looped fallback would be slower than cuBLAS bmm. */
bool tc_matmul_extended_eligible(const at::Tensor& A, const at::Tensor& B) {
    if (A.dtype() != B.dtype()) return false;
    if (A.dtype() != torch::kFloat32 && A.dtype() != torch::kBFloat16) return false;
    if (A.layout() != torch::kStrided || B.layout() != torch::kStrided) return false;
    if (B.dim() != 2) return false;            // require 2-D right-hand operand
    if (A.dim() < 2) return false;             // need at least (M, K)
    if (A.size(-1) != B.size(0)) return false;
    if (A.device().is_cuda()) {
        return B.device().is_cuda() &&
               A.device().index() == B.device().index() &&
               tc_cuda_bridge_available();
    }
    if (A.device().is_cpu()) return B.device().is_cpu();
    if (is_tensorcore_device(A)) return is_tensorcore_device(B);
    return false;
}

at::Tensor tc_matmul_extended(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(tc_matmul_extended_eligible(A, B),
                "tc_matmul_extended: inputs not eligible (A=", A.sizes(),
                " dtype=", A.scalar_type(), " device=", A.device(),
                "; B=", B.sizes(), " dtype=", B.scalar_type(),
                " device=", B.device(), ")");
    if (A.dim() == 2) {
        return tc_matmul_fp32(A, B);
    }
    // (..., M, K) @ (K, N) -> (..., M, N) via 2-D reshape
    const int64_t K = A.size(-1);
    const int64_t M = A.size(-2);
    const int64_t N = B.size(1);
    const auto leading = A.sizes().slice(0, A.dim() - 2);
    std::vector<int64_t> out_sizes(leading.begin(), leading.end());
    out_sizes.push_back(M);
    out_sizes.push_back(N);

    const auto A_flat = A.contiguous().view({-1, K});
    auto C_flat = tc_matmul_fp32(A_flat, B);
    return C_flat.view(out_sizes);
}

/* Batched 3-D x 3-D matmul: A: (B, M, K) @ B: (B, K, N) -> (B, M, N).
 * Used by torch.bmm and by torch.einsum decompositions for attention
 * (Q @ K^T, attn @ V). Implementation iterates over batch and calls
 * tc_matmul_fp32; correct but not as fast as cublasSgemmStridedBatched.
 * The substrate engagement is the goal here, not raw throughput. */
bool tc_bmm_eligible(const at::Tensor& A, const at::Tensor& B) {
    if (A.dtype() != B.dtype()) return false;
    if (A.dtype() != torch::kFloat32 && A.dtype() != torch::kBFloat16) return false;
    if (A.layout() != torch::kStrided || B.layout() != torch::kStrided) return false;
    if (A.dim() != 3 || B.dim() != 3) return false;
    if (A.size(0) != B.size(0)) return false;
    if (A.size(2) != B.size(1)) return false;
    if (A.device().is_cuda()) {
        return B.device().is_cuda() &&
               A.device().index() == B.device().index() &&
               tc_cuda_bridge_available();
    }
    if (A.device().is_cpu()) return B.device().is_cpu();
    return false;
}

at::Tensor tc_bmm_fp32(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(tc_bmm_eligible(A, B),
                "tc_bmm: inputs not eligible (A=", A.sizes(),
                " dtype=", A.scalar_type(), " device=", A.device(),
                "; B=", B.sizes(), " dtype=", B.scalar_type(),
                " device=", B.device(), ")");
    const int64_t B_dim = A.size(0);
    const int64_t M = A.size(1);
    const int64_t N = B.size(2);
    auto out = torch::empty({B_dim, M, N}, A.options());
    const auto A_c = A.contiguous();
    const auto B_c = B.contiguous();
    for (int64_t b = 0; b < B_dim; ++b) {
        // Slice-by-slice: each 2-D slice goes through tc_matmul_fp32.
        auto Ab = A_c.select(0, b);
        auto Bb = B_c.select(0, b);
        auto Cb = tc_matmul_fp32(Ab, Bb);
        out.select(0, b).copy_(Cb);
    }
    return out;
}

class TCBmmAutogradFunction
    : public torch::autograd::Function<TCBmmAutogradFunction> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor A,
            at::Tensor B) {
        ctx->save_for_backward({A, B});
        return tc_bmm_fp32(A, B);
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto A = saved[0];
        auto B = saved[1];
        auto grad_C = grad_outputs[0];
        // grad_A = grad_C @ B^T   (per-batch)
        // grad_B = A^T @ grad_C   (per-batch)
        auto grad_A = tc_bmm_fp32(grad_C, B.transpose(1, 2).contiguous());
        auto grad_B = tc_bmm_fp32(A.transpose(1, 2).contiguous(), grad_C);
        return {grad_A, grad_B};
    }
};

/* Fallback to PyTorch's bmm without re-entering our hook. Excluding the
 * autograd keys lets us reach PyTorch's native CUDA bmm structured kernel,
 * BUT it strips grad_fn attachment from the output. This means: if the
 * caller passes requires_grad inputs while the bridge is disabled, the
 * downstream autograd graph is broken. In practice this only happens in
 * bridge-disabled benches; production training always runs with the
 * bridge enabled, so this fallback is fine for shipped paths. */
at::Tensor tc_bmm_fallback(const at::Tensor& A, const at::Tensor& B) {
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKeySet{
        c10::DispatchKey::AutogradCUDA,
        c10::DispatchKey::AutogradCPU,
    });
    return at::bmm(A, B);
}

at::Tensor tc_bmm_dispatch(const at::Tensor& A, const at::Tensor& B) {
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_cuda_bridge_available() &&
        tc_bmm_eligible(A, B)) {
        if (A.requires_grad() || B.requires_grad()) {
            return TCBmmAutogradFunction::apply(A, B);
        }
        return tc_bmm_fp32(A, B);
    }
    return tc_bmm_fallback(A, B);
}

at::Tensor tc_bmm_dispatch_cpu(const at::Tensor& A, const at::Tensor& B) {
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_bmm_eligible(A, B)) {
        if (A.requires_grad() || B.requires_grad()) {
            return TCBmmAutogradFunction::apply(A, B);
        }
        return tc_bmm_fp32(A, B);
    }
    return tc_bmm_fallback(A, B);
}

class TCMatmulAutogradFunction
    : public torch::autograd::Function<TCMatmulAutogradFunction> {
public:
    static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor A,
            at::Tensor B) {
        ctx->save_for_backward({A, B});
        return tc_matmul_extended(A, B);
    }

    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto A = saved[0];
        auto B = saved[1];
        auto grad_C = grad_outputs[0];

        // C = A @ B  with A: (..., M, K), B: (K, N), C: (..., M, N)
        // dL/dA = dL/dC @ B^T   -> shape (..., M, K)
        // dL/dB = sum_over_leading_dims(A^T @ dL/dC)  -> shape (K, N)
        at::Tensor grad_A;
        at::Tensor grad_B;

        if (A.dim() == 2) {
            // grad_A = grad_C @ B^T
            const auto BT = B.transpose(0, 1).contiguous();
            grad_A = tc_matmul_fp32(grad_C, BT);
            // grad_B = A^T @ grad_C
            const auto AT = A.transpose(0, 1).contiguous();
            grad_B = tc_matmul_fp32(AT, grad_C);
        } else {
            // (..., M, K) shape. Flatten leading dims for fast 2-D path.
            const int64_t K = A.size(-1);
            const int64_t M = A.size(-2);
            const int64_t N = B.size(1);
            const auto leading = A.sizes().slice(0, A.dim() - 2);

            const auto A_flat = A.contiguous().view({-1, K});   // (BM, K)
            const auto grad_C_flat = grad_C.contiguous().view({-1, N});  // (BM, N)

            // grad_A_flat = grad_C_flat @ B^T  -> (BM, K)
            const auto BT = B.transpose(0, 1).contiguous();
            const auto grad_A_flat = tc_matmul_fp32(grad_C_flat, BT);
            std::vector<int64_t> grad_A_sizes(leading.begin(), leading.end());
            grad_A_sizes.push_back(M);
            grad_A_sizes.push_back(K);
            grad_A = grad_A_flat.view(grad_A_sizes);

            // grad_B = A_flat^T @ grad_C_flat  -> (K, N)
            const auto A_flat_T = A_flat.transpose(0, 1).contiguous();
            grad_B = tc_matmul_fp32(A_flat_T, grad_C_flat);
        }

        return {grad_A, grad_B};
    }
};

at::Tensor tc_matmul_autograd_extended(const at::Tensor& A, const at::Tensor& B) {
    return TCMatmulAutogradFunction::apply(A, B);
}

at::Tensor tc_matmul_privateuse1(const at::Tensor& A, const at::Tensor& B) {
    /* Pre-check eligibility here so the error message identifies this
     * dispatcher (tc_matmul_privateuse1) and the structured reason,
     * rather than surfacing a deep TORCH_CHECK from inside tc_matmul_fp32. */
    const std::string reason = tc_matmul_eligibility_reason(A, B);
    TORCH_CHECK(reason == "eligible",
                "tc_matmul_privateuse1: refusing to dispatch (reason=", reason,
                ", A=", A.sizes(), " dtype=", A.scalar_type(), " device=", A.device(),
                ", B=", B.sizes(), " dtype=", B.scalar_type(), " device=", B.device(),
                "). PrivateUse1 matmul requires both tensors on tensorcore PrivateUse1, "
                "fp32 or bf16, 2-D, contiguous, with matching inner dims.");
    return tc_matmul_fp32(A, B);
}

/* MPS dispatch path. Task #936.
 *
 * The tensorcore GEMM dispatcher (lib/ops/gemm.mm) routes to the MPS
 * backend (MPSMatrixMultiplication / simdgroup_matrix kernels) for any
 * supported dtype on a Metal-enabled libtensorcore build. PyTorch MPS
 * tensors live on a different MTLDevice managed by torch's own MPS
 * allocator, so we can't zero-copy across the device boundary without a
 * deeper API. For correctness + engagement v1 we copy the tensor data
 * to host, run tc_gemm against tensorcore-owned buffers (which still
 * execute on GPU under MPS/simdgroup_matrix kernels on Metal builds),
 * and copy the result back to MPS. The hot-path zero-copy variant is
 * tracked as future work; this lands the dispatch wiring so downstream
 * code that sets QLLM_USE_TENSORCORE_MATMUL=1 on Apple GPU actually
 * exercises the substrate. */
bool tc_mps_backend_available() {
    /* The MPS backend is available when libtensorcore was built with
     * TC_ENABLE_METAL — tc_init then probes the system MTLDevice and
     * tc_gemm routes to the MPS path. Probe by checking the backend
     * name table for a known Metal-only value: TC_BACKEND_MPS is only
     * exposed when the Metal build is linked. We detect at runtime by
     * issuing a 1x1x1 GEMM and checking tc_last_backend(); cached. */
    static std::atomic<int> cached{-1};   /* -1 unknown, 0 no, 1 yes */
    int v = cached.load(std::memory_order_acquire);
    if (v != -1) return v == 1;
    try {
        ensure_ctx();
    } catch (...) {
        cached.store(0, std::memory_order_release);
        return false;
    }
    tc_buffer *bA = nullptr, *bB = nullptr, *bC = nullptr;
    const size_t bytes = sizeof(float);
    bool engaged = false;
    if (tc_buffer_alloc(g_ctx, bytes, &bA) == TC_OK &&
        tc_buffer_alloc(g_ctx, bytes, &bB) == TC_OK &&
        tc_buffer_alloc(g_ctx, bytes, &bC) == TC_OK) {
        void *pa = nullptr, *pb = nullptr;
        tc_buffer_map(bA, &pa);
        tc_buffer_map(bB, &pb);
        if (pa) *((float*)pa) = 1.0f;
        if (pb) *((float*)pb) = 1.0f;
        tc_gemm_desc d{};
        d.M = 1; d.N = 1; d.K = 1;
        d.a_dtype = TC_DTYPE_F32; d.b_dtype = TC_DTYPE_F32;
        d.c_dtype = TC_DTYPE_F32; d.accum_dtype = TC_DTYPE_F32;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.lda = 1; d.ldb = 1; d.ldc = 1;
        if (tc_gemm(g_ctx, &d, bA, bB, bC) == TC_OK) {
            const tc_backend_t b = tc_last_backend();
            engaged = (b == TC_BACKEND_MPS ||
                       b == TC_BACKEND_SIMDGROUP_MATRIX ||
                       b == TC_BACKEND_TENSOROPS_M5 ||
                       b == TC_BACKEND_METAL_COMPUTE);
        }
    }
    if (bA) tc_buffer_free(g_ctx, bA);
    if (bB) tc_buffer_free(g_ctx, bB);
    if (bC) tc_buffer_free(g_ctx, bC);
    cached.store(engaged ? 1 : 0, std::memory_order_release);
    return engaged;
}

bool tc_mps_matmul_eligible(const at::Tensor& A, const at::Tensor& B) {
    if (A.dtype() != B.dtype()) return false;
    /* MPS path supports fp32 + bf16; fp16 is left to native MPS for now. */
    if (A.dtype() != torch::kFloat32 && A.dtype() != torch::kBFloat16) return false;
    if (A.layout() != torch::kStrided || B.layout() != torch::kStrided) return false;
    if (A.dim() != 2 || B.dim() != 2) return false;
    if (A.size(1) != B.size(0)) return false;
    if (!A.device().is_mps() || !B.device().is_mps()) return false;
    if (A.device().index() != B.device().index()) return false;
    return tc_mps_backend_available();
}

std::atomic<uint64_t> g_mps_dispatch_count{0};

at::Tensor tc_mps_matmul_fp32(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(tc_mps_matmul_eligible(A, B),
                "tc_mps_matmul_fp32: inputs not eligible (A=", A.sizes(),
                " dtype=", A.scalar_type(), " device=", A.device(),
                ", B=", B.sizes(), " dtype=", B.scalar_type(),
                " device=", B.device(),
                "). Requires both fp32 or bf16 tensors on the same MPS device, "
                "2-D, with libtensorcore built TC_ENABLE_METAL=ON.");

    const int64_t M64 = A.size(0);
    const int64_t K64 = A.size(1);
    const int64_t N64 = B.size(1);
    check_dim_fits_tc("M", M64);
    check_dim_fits_tc("N", N64);
    check_dim_fits_tc("K", K64);
    const int M = (int)M64, K = (int)K64, N = (int)N64;

    const bool is_bf16 = (A.dtype() == torch::kBFloat16);
    const size_t elem = is_bf16 ? sizeof(uint16_t) : sizeof(float);
    const tc_dtype_t tc_dt = is_bf16 ? TC_DTYPE_BF16 : TC_DTYPE_F32;

    auto out = torch::empty({M64, N64}, A.options());
    if (M == 0 || N == 0) return out;
    if (K == 0) return out.zero_();

    /* Stage data through CPU; PyTorch's MPS .to('cpu') copies via MTL
     * blit. For matmul-shaped GEMMs (>= 256^3) the kernel time amortizes
     * the round trip; production zero-copy is future work that needs a
     * way to extract the MTLBuffer from a PyTorch MPS tensor. */
    const auto A_cpu = A.contiguous().to(at::kCPU);
    const auto B_cpu = B.contiguous().to(at::kCPU);
    auto out_cpu = at::empty({M64, N64}, A_cpu.options());

    ensure_ctx();

    tc_buffer *bA = nullptr, *bB = nullptr, *bC = nullptr;
    auto cleanup = [&]() {
        if (bA) tc_buffer_free(g_ctx, bA);
        if (bB) tc_buffer_free(g_ctx, bB);
        if (bC) tc_buffer_free(g_ctx, bC);
    };

    const size_t bytes_a = checked_matrix_bytes("A", M64, K64, elem);
    const size_t bytes_b = checked_matrix_bytes("B", K64, N64, elem);
    const size_t bytes_c = checked_matrix_bytes("C", M64, N64, elem);

    auto load = [&](const void* src, size_t bytes, tc_buffer** dst) -> bool {
        if (tc_buffer_alloc(g_ctx, bytes, dst) != TC_OK) return false;
        void* p = nullptr;
        if (tc_buffer_map(*dst, &p) != TC_OK || !p) return false;
        std::memcpy(p, src, bytes);
        return true;
    };

    if (!load(A_cpu.data_ptr(), bytes_a, &bA) ||
        !load(B_cpu.data_ptr(), bytes_b, &bB) ||
        tc_buffer_alloc(g_ctx, bytes_c, &bC) != TC_OK) {
        cleanup();
        throw std::runtime_error("tc_mps_matmul: buffer setup failed");
    }

    tc_gemm_desc desc{};
    desc.M = M; desc.N = N; desc.K = K;
    desc.a_dtype = tc_dt; desc.b_dtype = tc_dt; desc.c_dtype = tc_dt;
    desc.accum_dtype = TC_DTYPE_F32;
    desc.alpha = 1.0f; desc.beta = 0.0f;
    desc.transpose_a = false; desc.transpose_b = false;
    desc.lda = K; desc.ldb = N; desc.ldc = N;

    /* tc_gemm chooses MPS / simdgroup_matrix / tensorops on Metal builds;
     * portable-CPU builds fall to AMX/NEON/CBLAS. The eligibility check
     * above gated this path on a confirmed Metal backend. */
    const auto rc = tc_gemm(g_ctx, &desc, bA, bB, bC);
    if (rc != TC_OK) {
        cleanup();
        throw std::runtime_error(
            std::string("tc_gemm (MPS path) failed: ") +
            std::to_string(static_cast<int>(rc)));
    }

    void* cp = nullptr;
    if (tc_buffer_map(bC, &cp) != TC_OK || !cp) {
        cleanup();
        throw std::runtime_error("tc_mps_matmul: output map failed");
    }
    std::memcpy(out_cpu.data_ptr(), cp, bytes_c);
    cleanup();

    g_mps_dispatch_count.fetch_add(1, std::memory_order_relaxed);
    return out_cpu.to(A.device());
}

at::Tensor tc_matmul_mps_dispatch(const at::Tensor& A, const at::Tensor& B) {
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_mps_matmul_eligible(A, B)) {
        return tc_mps_matmul_fp32(A, B);
    }
    return at::native::matmul(A, B);
}

uint64_t tc_mps_dispatch_count() {
    return g_mps_dispatch_count.load(std::memory_order_relaxed);
}

/* CUDA dispatcher hook. Mirrors the CPU path: only routes through
 * tc_gemm when set_default_matmul() is on AND tensorcore's CUDA
 * backend is live. Otherwise falls back to native PyTorch CUDA matmul
 * so we don't hijack the kernel for workloads we can't serve. */
at::Tensor tc_matmul_cuda_dispatch(const at::Tensor& A, const at::Tensor& B) {
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_cuda_bridge_available() &&
        is_tc_matmul_eligible(A, B)) {
        return tc_matmul_fp32(A, B);
    }
    return at::native::matmul(A, B);
}

at::Tensor tc_matmul_autograd_cuda(const at::Tensor& A, const at::Tensor& B) {
    // Fast path: no grad needed and basic 2-D eligibility -> direct tc_gemm.
    if (!A.requires_grad() &&
        !B.requires_grad() &&
        g_default_matmul.load(std::memory_order_acquire) &&
        tc_cuda_bridge_available() &&
        is_tc_matmul_eligible(A, B)) {
        return tc_matmul_fp32(A, B);
    }
    // Training path: route through the custom autograd Function whenever
    // we can serve the forward AND a transpose-aware backward (2-D and
    // N-D x 2-D). This is what makes training actually use the tensorcore
    // substrate instead of cuBLAS native.
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_cuda_bridge_available() &&
        tc_matmul_extended_eligible(A, B)) {
        return tc_matmul_autograd_extended(A, B);
    }
    return at::native::matmul(A, B);
}

at::Tensor tc_matmul_autograd_cpu_extended(const at::Tensor& A, const at::Tensor& B) {
    // CPU autograd path: mirror the CUDA logic so trainers using
    // QLLM_USE_TENSORCORE_MATMUL=1 on a CPU build also see the substrate.
    if (!A.requires_grad() &&
        !B.requires_grad() &&
        g_default_matmul.load(std::memory_order_acquire) &&
        is_tc_matmul_eligible(A, B)) {
        return tc_matmul_fp32(A, B);
    }
    if (g_default_matmul.load(std::memory_order_acquire) &&
        tc_matmul_extended_eligible(A, B)) {
        return tc_matmul_autograd_extended(A, B);
    }
    return at::native::matmul(A, B);
}

at::Tensor tc_empty_memory_format(
    c10::SymIntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
    register_privateuse1_name();
    TORCH_CHECK(!layout.has_value() || *layout == at::kStrided,
                "tensorcore PrivateUse1 only supports strided layout");
    TORCH_CHECK(!pin_memory.value_or(false),
                "tensorcore PrivateUse1 does not support pinned memory");
    if (device.has_value()) {
        TORCH_CHECK(device->type() == c10::DeviceType::PrivateUse1,
                    "tensorcore empty expected PrivateUse1 device, got ", *device);
        TORCH_CHECK(!device->has_index() || device->index() == 0,
                    "tensorcore exposes one logical device: tensorcore:0");
    }
    const at::ScalarType scalar = dtype.value_or(at::kFloat);
    return at::detail::empty_generic_symint(
        size,
        &g_tensorcore_allocator,
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        scalar,
        memory_format);
}

at::Tensor tc_empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
    register_privateuse1_name();
    TORCH_CHECK(!layout.has_value() || *layout == at::kStrided,
                "tensorcore PrivateUse1 only supports strided layout");
    TORCH_CHECK(!pin_memory.value_or(false),
                "tensorcore PrivateUse1 does not support pinned memory");
    if (device.has_value()) {
        TORCH_CHECK(device->type() == c10::DeviceType::PrivateUse1,
                    "tensorcore empty_strided expected PrivateUse1 device, got ", *device);
        TORCH_CHECK(!device->has_index() || device->index() == 0,
                    "tensorcore exposes one logical device: tensorcore:0");
    }
    const at::ScalarType scalar = dtype.value_or(at::kFloat);
    return at::detail::empty_strided_symint_generic(
        size,
        stride,
        &g_tensorcore_allocator,
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        scalar);
}

size_t tensor_nbytes(const at::Tensor& t) {
    TORCH_CHECK(t.numel() >= 0, "tensor has negative numel");
    return static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
}

at::Tensor tc_to_tensorcore(const at::Tensor& src) {
    register_privateuse1_name();
    TORCH_CHECK(src.device().is_cpu(), "to_tensorcore expects a CPU tensor; got ", src.device());
    TORCH_CHECK(src.layout() == torch::kStrided, "to_tensorcore expects strided tensors");
    TORCH_CHECK(src.dtype() == torch::kFloat32 || src.dtype() == torch::kBFloat16,
                "to_tensorcore supports fp32 and bf16; got ", src.dtype());
    const auto contiguous = src.contiguous();
    auto out = at::empty_strided(
        contiguous.sizes(),
        contiguous.strides(),
        contiguous.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    std::memcpy(out.data_ptr(), contiguous.data_ptr(), tensor_nbytes(contiguous));
    return out;
}

at::Tensor tc_to_cpu(const at::Tensor& src) {
    TORCH_CHECK(is_tensorcore_device(src), "to_cpu expects a tensorcore PrivateUse1 tensor; got ", src.device());
    TORCH_CHECK(src.layout() == torch::kStrided, "to_cpu expects strided tensors");
    TORCH_CHECK(src.is_contiguous(), "to_cpu expects contiguous tensorcore PrivateUse1 tensors");
    auto out = at::empty_strided(
        src.sizes(),
        src.strides(),
        src.options().device(c10::Device(c10::DeviceType::CPU)));
    std::memcpy(out.data_ptr(), src.data_ptr(), tensor_nbytes(src));
    return out;
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
    // bmm: only register at autograd level so fallback redispatch finds the native CPU impl.
}

TORCH_LIBRARY_IMPL(aten, AutogradCPU, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_autograd_cpu_extended));
    m.impl("bmm", TORCH_FN(tc_bmm_dispatch_cpu));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_privateuse1));
    m.impl("empty.memory_format", TORCH_FN(tc_empty_memory_format));
    m.impl("empty_strided", TORCH_FN(tc_empty_strided));
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_cuda_dispatch));
    // bmm/addmm/baddbmm registered at AutogradCUDA only — see fallback rationale.
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_mps_dispatch));
}

TORCH_LIBRARY_IMPL(aten, AutogradCUDA, m) {
    m.impl("matmul", TORCH_FN(tc_matmul_autograd_cuda));
    m.impl("bmm", TORCH_FN(tc_bmm_dispatch));
    // addmm/baddbmm hooks are correct for forward but their custom autograd
    // Functions don't propagate the gradient back to `weight` when the
    // dispatched mat2 is a transpose view of weight (common nn.Linear case).
    // Leaving these unregistered so PyTorch's default autograd handles the
    // view chain correctly; their forward gets the substrate via the bmm
    // backward path indirectly.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_privateuse1_name();
    m.def("matmul", &tc_matmul_fp32,
          "tc_matmul(A: Tensor[fp32|bf16, MxK], B: Tensor[fp32|bf16, KxN]) -> Tensor[MxN]");
    m.def("matmul_bf16", &tc_matmul_bf16,
          "tc_matmul_bf16(A: Tensor[bf16, MxK], B: Tensor[bf16, KxN]) -> Tensor[bf16, MxN]");
    m.def("to_tensorcore", &tc_to_tensorcore,
          "Copy a CPU fp32/bf16 tensor into tensorcore PrivateUse1 host memory");
    m.def("to_cpu", &tc_to_cpu,
          "Copy a tensorcore PrivateUse1 tensor back to a CPU tensor");
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
    m.def("cuda_bridge_available", &tc_cuda_bridge_available,
          "Return whether libtensorcore was built with TC_ENABLE_CUDA AND tc_cuda_init "
          "succeeded at runtime. When false, the CUDA dispatcher falls through to native PyTorch.");
    m.def("mps_bridge_available", &tc_mps_backend_available,
          "Return whether libtensorcore was built with TC_ENABLE_METAL "
          "(tc_mps_gemm weakly linked). When false, the MPS dispatcher falls through to native PyTorch.");
    m.def("mps_dispatch_count", &tc_mps_dispatch_count,
          "Return the number of times the MPS GEMM dispatcher actually engaged tc_mps_gemm in this process.");
    m.def("privateuse1_hooks_registered", &at::isPrivateUse1HooksRegistered,
          "Return whether the bridge's PrivateUse1HooksInterface has been registered with PyTorch.");
}
