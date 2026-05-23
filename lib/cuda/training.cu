/*
 * tensorcore - CUDA training kernels.
 *
 * Pointwise + row-reduction ops on managed memory. These run on the
 * NVIDIA GPU when CUDA is active and the input buffer is CUDA-managed,
 * matching the existing CPU implementations in lib/ops/training_cpu.cpp.
 *
 * Kernels here:
 *   - rmsnorm_forward      : row-wise rsqrt(mean(x^2) + eps) * gamma
 *   - rmsnorm_backward     : row-wise dX plus fp32 dgamma reduction
 *   - layernorm_forward    : row-wise mean/variance normalization
 *   - adamw_step_fp32/fp16 : AdamW optimizer update + master weights
 *   - swiglu_forward       : elementwise x * sigmoid(x) * up
 *   - swiglu_backward      : elementwise dgate/dup
 *   - softmax_forward      : row-wise max-stabilized softmax
 *   - softmax_backward     : row-wise softmax Jacobian-vector product
 *
 * All take device pointers (CUDA-managed memory satisfies this) and run
 * with a configurable thread/block layout. Calls cudaDeviceSynchronize
 * at end so the host sees the writes.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

extern "C" TC_CUDA_INTERNAL void tc_cuda_set_last_kernel(const char* name);

namespace {

/* Block-wide sum reduction using shuffle within warp + shared memory across warps. */
__device__ float block_reduce_sum_f32(float v) {
    __shared__ float warp_sums[32];   /* up to 32 warps per block = 1024 threads */
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    /* Warp shuffle reduce. */
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    /* First warp reduces the warp sums. */
    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        }
    }
    /* Broadcast the result from thread 0 of warp 0 via shared memory. */
    __shared__ float total;
    if (threadIdx.x == 0) total = v;
    __syncthreads();
    return total;
}

__global__ void rmsnorm_forward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        __half* __restrict__ Y,
        float* __restrict__ rstd_out,
        int D,
        float eps) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    /* Phase 1: sum of squares over the row. */
    float local_ss = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        const float x = __half2float(X[n * D + d]);
        local_ss += x * x;
    }
    const float row_sum = block_reduce_sum_f32(local_ss);
    const float rstd = rsqrtf(row_sum / (float)D + eps);
    if (tid == 0) rstd_out[n] = rstd;

    /* Phase 2: scale + gamma. */
    for (int d = tid; d < D; d += block_size) {
        const float x  = __half2float(X[n * D + d]);
        const float g  = __half2float(gamma[d]);
        Y[n * D + d] = __float2half(x * rstd * g);
    }
}

__global__ void rmsnorm_backward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ dY,
        const float* __restrict__ rstd,
        __half* __restrict__ dX,
        float* __restrict__ dgamma,
        int D) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float rs = rstd[n];

    float local_dot = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        const float x = __half2float(X[n * D + d]);
        const float g = __half2float(gamma[d]);
        const float dy = __half2float(dY[n * D + d]);
        local_dot += dy * g * x * rs;
    }
    const float dot = block_reduce_sum_f32(local_dot) / (float)D;

    for (int d = tid; d < D; d += block_size) {
        const float x = __half2float(X[n * D + d]);
        const float g = __half2float(gamma[d]);
        const float dy = __half2float(dY[n * D + d]);
        const float xhat = x * rs;
        dX[n * D + d] = __float2half_rn(rs * (g * dy - xhat * dot));
        atomicAdd(dgamma + d, dy * xhat);
    }
}

__global__ void adamw_step_fp32_kernel(
        float* __restrict__ params,
        const float* __restrict__ grads,
        float* __restrict__ m,
        float* __restrict__ v,
        int n_elements,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bias_correction1,
        float bias_correction2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    const float g = grads[i];
    const float new_m = beta1 * m[i] + (1.0f - beta1) * g;
    const float new_v = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = new_m;
    v[i] = new_v;
    const float m_hat = new_m / bias_correction1;
    const float v_hat = new_v / bias_correction2;
    const float update = m_hat / (sqrtf(v_hat) + eps);
    /* AdamW: decoupled weight decay. */
    params[i] = (params[i] - lr * weight_decay * params[i]) - lr * update;
}

__global__ void adamw_step_fp16_kernel(
        float* __restrict__ params,
        const __half* __restrict__ grads,
        float* __restrict__ m,
        float* __restrict__ v,
        int n_elements,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bias_correction1,
        float bias_correction2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    const float g = __half2float(grads[i]);
    const float new_m = beta1 * m[i] + (1.0f - beta1) * g;
    const float new_v = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = new_m;
    v[i] = new_v;
    const float m_hat = new_m / bias_correction1;
    const float v_hat = new_v / bias_correction2;
    const float update = m_hat / (sqrtf(v_hat) + eps);
    params[i] = (params[i] - lr * weight_decay * params[i]) - lr * update;
}

__global__ void swiglu_forward_kernel(
        const __half* __restrict__ gate,
        const __half* __restrict__ up,
        __half* __restrict__ out,
        int n_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    /* SiLU clamped at exp tails. */
    double silu;
    const double gd = (double)g;
    if (gd < -50.0) silu = 0.0;
    else if (gd > 50.0) silu = gd;
    else silu = gd / (1.0 + exp(-gd));
    out[i] = __float2half_rn((float)(silu * (double)u));
}

__global__ void swiglu_backward_kernel(
        const __half* __restrict__ gate,
        const __half* __restrict__ up,
        const __half* __restrict__ dout,
        __half* __restrict__ dgate,
        __half* __restrict__ dup,
        int n_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float dy = __half2float(dout[i]);
    float silu;
    float dsilu;
    if (g < -50.0f) {
        silu = 0.0f;
        dsilu = 0.0f;
    } else if (g > 50.0f) {
        silu = g;
        dsilu = 1.0f;
    } else {
        const float sig = 1.0f / (1.0f + expf(-g));
        silu = g * sig;
        dsilu = sig + g * sig * (1.0f - sig);
    }
    dgate[i] = __float2half_rn(dy * u * dsilu);
    dup[i] = __float2half_rn(dy * silu);
}

/* LayerNorm forward: y = (x - mean) / sqrt(var + eps) * gamma + beta. */
__global__ void layernorm_forward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ beta,
        __half* __restrict__ Y,
        float* __restrict__ mean_out,
        float* __restrict__ rstd_out,
        int D,
        float eps) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        local_sum += __half2float(X[n * D + d]);
    }
    const float mean = block_reduce_sum_f32(local_sum) / (float)D;
    if (tid == 0 && mean_out) mean_out[n] = mean;

    float local_var = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        const float xc = __half2float(X[n * D + d]) - mean;
        local_var += xc * xc;
    }
    const float var = block_reduce_sum_f32(local_var) / (float)D;
    const float rstd = rsqrtf(var + eps);
    if (tid == 0 && rstd_out) rstd_out[n] = rstd;

    for (int d = tid; d < D; d += block_size) {
        const float x = __half2float(X[n * D + d]);
        const float g = __half2float(gamma[d]);
        const float b = beta ? __half2float(beta[d]) : 0.0f;
        Y[n * D + d] = __float2half((x - mean) * rstd * g + b);
    }
}

/* Block-wide max reduction (parallel to block_reduce_sum_f32 above). */
__device__ float block_reduce_max_f32(float v) {
    __shared__ float warp_maxs[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
    }
    if (lane == 0) warp_maxs[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        v = (lane < n_warps) ? warp_maxs[lane] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
        }
    }
    __shared__ float total_max;
    if (threadIdx.x == 0) total_max = v;
    __syncthreads();
    return total_max;
}

/* Softmax forward, row-wise, max-stabilized. */
__global__ void softmax_forward_kernel(
        const __half* __restrict__ X,
        __half* __restrict__ Y,
        int D) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float local_max = -INFINITY;
    for (int d = tid; d < D; d += block_size) {
        local_max = fmaxf(local_max, __half2float(X[n * D + d]));
    }
    const float m = block_reduce_max_f32(local_max);

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        local_sum += expf(__half2float(X[n * D + d]) - m);
    }
    const float total = block_reduce_sum_f32(local_sum);
    const float inv = 1.0f / total;

    for (int d = tid; d < D; d += block_size) {
        Y[n * D + d] = __float2half(expf(__half2float(X[n * D + d]) - m) * inv);
    }
}

__global__ void softmax_backward_kernel(
        const __half* __restrict__ Y,
        const __half* __restrict__ dY,
        __half* __restrict__ dX,
        int D) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float local_dot = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        const float y = __half2float(Y[n * D + d]);
        const float dy = __half2float(dY[n * D + d]);
        local_dot += y * dy;
    }
    const float dot = block_reduce_sum_f32(local_dot);

    for (int d = tid; d < D; d += block_size) {
        const float y = __half2float(Y[n * D + d]);
        const float dy = __half2float(dY[n * D + d]);
        dX[n * D + d] = __float2half_rn(y * (dy - dot));
    }
}

}  /* namespace */

/* C-linkage entry points called from lib/ops/training_cpu.cpp when the
 * buffer is CUDA-managed.
 *
 * Return contract:
 *   0  -> CUDA kernel completed
 *   1  -> unsupported for these pointers; caller should fall back to CPU
 *  -1  -> CUDA was selected but failed; caller should surface an error
 */

namespace {

constexpr int kCudaTrainingOk = 0;
constexpr int kCudaTrainingUnsupported = 1;
constexpr int kCudaTrainingError = -1;

bool is_managed_cuda_ptr(const void* ptr) {
    if (!ptr) return false;
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeManaged;
#else
    return attr.memoryType == cudaMemoryTypeManaged;
#endif
}

bool all_managed2(const void* a, const void* b) {
    return is_managed_cuda_ptr(a) && is_managed_cuda_ptr(b);
}

bool all_managed4(const void* a, const void* b, const void* c, const void* d) {
    return is_managed_cuda_ptr(a) && is_managed_cuda_ptr(b) &&
           is_managed_cuda_ptr(c) && is_managed_cuda_ptr(d);
}

bool all_managed5(const void* a, const void* b, const void* c,
                  const void* d, const void* e) {
    return is_managed_cuda_ptr(a) && is_managed_cuda_ptr(b) &&
           is_managed_cuda_ptr(c) && is_managed_cuda_ptr(d) &&
           is_managed_cuda_ptr(e);
}

bool all_managed6(const void* a, const void* b, const void* c,
                  const void* d, const void* e, const void* f) {
    return is_managed_cuda_ptr(a) && is_managed_cuda_ptr(b) &&
           is_managed_cuda_ptr(c) && is_managed_cuda_ptr(d) &&
           is_managed_cuda_ptr(e) && is_managed_cuda_ptr(f);
}

}  /* namespace */

extern "C" TC_CUDA_INTERNAL int tc_cuda_rmsnorm_forward(
        const void* X, const void* gamma, void* Y, void* rstd,
        int N, int D, float eps) {
    if (!X || !gamma || !Y || !rstd || N <= 0 || D <= 0) return kCudaTrainingError;
    if (!all_managed4(X, gamma, Y, rstd)) return kCudaTrainingUnsupported;

    /* Pick block size: power of 2 close to D, capped at 1024. */
    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;

    rmsnorm_forward_kernel<<<N, block_size>>>(
        (const __half*)X, (const __half*)gamma, (__half*)Y,
        (float*)rstd, D, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_rmsnorm_forward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_rmsnorm_backward(
        const void* X, const void* gamma, const void* dY, const void* rstd,
        void* dX, void* dgamma, int N, int D) {
    if (!X || !gamma || !dY || !rstd || !dX || !dgamma || N <= 0 || D <= 0) {
        return kCudaTrainingError;
    }
    if (!all_managed6(X, gamma, dY, rstd, dX, dgamma)) {
        return kCudaTrainingUnsupported;
    }

    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;

    if (cudaMemset(dgamma, 0, (size_t)D * sizeof(float)) != cudaSuccess) {
        return kCudaTrainingError;
    }
    rmsnorm_backward_kernel<<<N, block_size>>>(
        (const __half*)X, (const __half*)gamma, (const __half*)dY,
        (const float*)rstd, (__half*)dX, (float*)dgamma, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_rmsnorm_backward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_adamw_step_fp32(
        void* params, const void* grads, void* m, void* v,
        int n_elements, float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    if (!params || !grads || !m || !v || n_elements <= 0) return kCudaTrainingError;
    if (!all_managed4(params, grads, m, v)) return kCudaTrainingUnsupported;
    const int block_size = 256;
    const int blocks = (n_elements + block_size - 1) / block_size;
    adamw_step_fp32_kernel<<<blocks, block_size>>>(
        (float*)params, (const float*)grads, (float*)m, (float*)v,
        n_elements, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_adamw_step_fp32");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_adamw_step_fp16(
        void* params, const void* grads, void* m, void* v,
        int n_elements, float lr, float beta1, float beta2,
        float eps, float weight_decay,
        float bias_correction1, float bias_correction2) {
    if (!params || !grads || !m || !v || n_elements <= 0) return kCudaTrainingError;
    if (!all_managed4(params, grads, m, v)) return kCudaTrainingUnsupported;
    const int block_size = 256;
    const int blocks = (n_elements + block_size - 1) / block_size;
    adamw_step_fp16_kernel<<<blocks, block_size>>>(
        (float*)params, (const __half*)grads, (float*)m, (float*)v,
        n_elements, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_adamw_step_fp16");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_swiglu_forward(
        const void* gate, const void* up, void* out, int n_elements) {
    if (!gate || !up || !out || n_elements <= 0) return kCudaTrainingError;
    if (!all_managed2(gate, up) || !is_managed_cuda_ptr(out)) {
        return kCudaTrainingUnsupported;
    }
    const int block_size = 256;
    const int blocks = (n_elements + block_size - 1) / block_size;
    swiglu_forward_kernel<<<blocks, block_size>>>(
        (const __half*)gate, (const __half*)up, (__half*)out, n_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_swiglu_forward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_swiglu_backward(
        const void* gate, const void* up, const void* dout,
        void* dgate, void* dup, int n_elements) {
    if (!gate || !up || !dout || !dgate || !dup || n_elements <= 0) {
        return kCudaTrainingError;
    }
    if (!all_managed5(gate, up, dout, dgate, dup)) {
        return kCudaTrainingUnsupported;
    }
    const int block_size = 256;
    const int blocks = (n_elements + block_size - 1) / block_size;
    swiglu_backward_kernel<<<blocks, block_size>>>(
        (const __half*)gate, (const __half*)up, (const __half*)dout,
        (__half*)dgate, (__half*)dup, n_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_swiglu_backward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_layernorm_forward(
        const void* X, const void* gamma, const void* beta,
        void* Y, void* mean, void* rstd,
        int N, int D, float eps) {
    if (!X || !gamma || !Y || !mean || !rstd || N <= 0 || D <= 0) {
        return kCudaTrainingError;
    }
    if (!is_managed_cuda_ptr(X) || !is_managed_cuda_ptr(gamma) ||
        !is_managed_cuda_ptr(Y) || !is_managed_cuda_ptr(mean) ||
        !is_managed_cuda_ptr(rstd)) {
        return kCudaTrainingUnsupported;
    }
    if (beta && !is_managed_cuda_ptr(beta)) return kCudaTrainingUnsupported;
    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;
    layernorm_forward_kernel<<<N, block_size>>>(
        (const __half*)X, (const __half*)gamma, (const __half*)beta,
        (__half*)Y, (float*)mean, (float*)rstd, D, eps);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_layernorm_forward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_softmax_forward(
        const void* X, void* Y, int N, int D) {
    if (!X || !Y || N <= 0 || D <= 0) return kCudaTrainingError;
    if (!is_managed_cuda_ptr(X) || !is_managed_cuda_ptr(Y)) {
        return kCudaTrainingUnsupported;
    }
    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;
    softmax_forward_kernel<<<N, block_size>>>(
        (const __half*)X, (__half*)Y, D);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_softmax_forward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_softmax_backward(
        const void* Y, const void* dY, void* dX, int N, int D) {
    if (!Y || !dY || !dX || N <= 0 || D <= 0) return kCudaTrainingError;
    if (!is_managed_cuda_ptr(Y) || !is_managed_cuda_ptr(dY) ||
        !is_managed_cuda_ptr(dX)) {
        return kCudaTrainingUnsupported;
    }
    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;
    softmax_backward_kernel<<<N, block_size>>>(
        (const __half*)Y, (const __half*)dY, (__half*)dX, D);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_softmax_backward");
    return kCudaTrainingOk;
}
