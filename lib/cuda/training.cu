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
#include <mma.h>
#include <cstddef>
#include <cstdint>
#include <climits>

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
    float silu;
    if (g < -50.0f) silu = 0.0f;
    else if (g > 50.0f) silu = g;
    else silu = g / (1.0f + __expf(-g));
    out[i] = __float2half_rn(silu * u);
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

__global__ void layernorm_backward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ dY,
        const float* __restrict__ mean,
        const float* __restrict__ rstd,
        __half* __restrict__ dX,
        int D) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float me = mean[n];
    const float rs = rstd[n];

    float local_sum = 0.0f;
    float local_dot = 0.0f;
    for (int d = tid; d < D; d += block_size) {
        const float dyp = __half2float(dY[n * D + d]) * __half2float(gamma[d]);
        const float xhat = (__half2float(X[n * D + d]) - me) * rs;
        local_sum += dyp;
        local_dot += dyp * xhat;
    }
    const float sum_dyp = block_reduce_sum_f32(local_sum);
    const float dot_dyp_xhat = block_reduce_sum_f32(local_dot);
    const float inv_d = 1.0f / (float)D;

    for (int d = tid; d < D; d += block_size) {
        const float dyp = __half2float(dY[n * D + d]) * __half2float(gamma[d]);
        const float xhat = (__half2float(X[n * D + d]) - me) * rs;
        dX[n * D + d] = __float2half_rn(
            rs * (dyp - sum_dyp * inv_d - xhat * dot_dyp_xhat * inv_d));
    }
}

__global__ void rope_kernel(
        __half* __restrict__ X,
        const float* __restrict__ cos_t,
        const float* __restrict__ sin_t,
        int seq,
        int head_dim,
        size_t total_pairs,
        int inverse) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pairs) return;

    const int half = head_dim >> 1;
    const int k = (int)(i % (size_t)half);
    const size_t token = i / (size_t)half;
    const int s_idx = (int)(token % (size_t)seq);
    __half* row = X + token * (size_t)head_dim;
    const float c = cos_t[(size_t)s_idx * (size_t)half + (size_t)k];
    const float si = sin_t[(size_t)s_idx * (size_t)half + (size_t)k];
    const float x0 = __half2float(row[k]);
    const float x1 = __half2float(row[k + half]);

    if (inverse) {
        row[k]        = __float2half_rn(x0 * c + x1 * si);
        row[k + half] = __float2half_rn(-x0 * si + x1 * c);
    } else {
        row[k]        = __float2half_rn(x0 * c - x1 * si);
        row[k + half] = __float2half_rn(x0 * si + x1 * c);
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

bool all_managed3(const void* a, const void* b, const void* c) {
    return is_managed_cuda_ptr(a) && is_managed_cuda_ptr(b) &&
           is_managed_cuda_ptr(c);
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

extern "C" TC_CUDA_INTERNAL int tc_cuda_layernorm_backward(
        const void* X, const void* gamma, const void* dY,
        const void* mean, const void* rstd, void* dX,
        int N, int D) {
    if (!X || !gamma || !dY || !mean || !rstd || !dX || N <= 0 || D <= 0) {
        return kCudaTrainingError;
    }
    if (!all_managed6(X, gamma, dY, mean, rstd, dX)) {
        return kCudaTrainingUnsupported;
    }
    int block_size = 32;
    while (block_size < D && block_size < 1024) block_size <<= 1;
    layernorm_backward_kernel<<<N, block_size>>>(
        (const __half*)X, (const __half*)gamma, (const __half*)dY,
        (const float*)mean, (const float*)rstd, (__half*)dX, D);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_layernorm_backward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_rope_forward(
        void* X, const void* cos_t, const void* sin_t,
        int batch, int heads, int seq, int head_dim) {
    if (!X || !cos_t || !sin_t || batch <= 0 || heads <= 0 || seq <= 0 ||
        head_dim <= 0 || (head_dim & 1)) {
        return kCudaTrainingError;
    }
    if (!all_managed3(X, cos_t, sin_t)) return kCudaTrainingUnsupported;
    const size_t total_pairs =
        (size_t)batch * (size_t)heads * (size_t)seq * (size_t)(head_dim / 2);
    const int block_size = 256;
    if (total_pairs > (size_t)INT_MAX * (size_t)block_size) {
        return kCudaTrainingError;
    }
    const int blocks = (int)((total_pairs + (size_t)block_size - 1) /
                             (size_t)block_size);
    rope_kernel<<<blocks, block_size>>>(
        (__half*)X, (const float*)cos_t, (const float*)sin_t,
        seq, head_dim, total_pairs, 0);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_rope_forward");
    return kCudaTrainingOk;
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_rope_backward(
        void* dX, const void* cos_t, const void* sin_t,
        int batch, int heads, int seq, int head_dim) {
    if (!dX || !cos_t || !sin_t || batch <= 0 || heads <= 0 || seq <= 0 ||
        head_dim <= 0 || (head_dim & 1)) {
        return kCudaTrainingError;
    }
    if (!all_managed3(dX, cos_t, sin_t)) return kCudaTrainingUnsupported;
    const size_t total_pairs =
        (size_t)batch * (size_t)heads * (size_t)seq * (size_t)(head_dim / 2);
    const int block_size = 256;
    if (total_pairs > (size_t)INT_MAX * (size_t)block_size) {
        return kCudaTrainingError;
    }
    const int blocks = (int)((total_pairs + (size_t)block_size - 1) /
                             (size_t)block_size);
    rope_kernel<<<blocks, block_size>>>(
        (__half*)dX, (const float*)cos_t, (const float*)sin_t,
        seq, head_dim, total_pairs, 1);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_rope_backward");
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

/* ----------------------------------------------------------------------- *
 * FlashAttention-2 forward (CUDA).
 *
 * One thread block per (qtile, batch*Hq). Block size = kBr threads, one
 * thread per Q row in the tile. Q is held in per-thread registers; K and
 * V tiles stream through shared memory in chunks of kBc. Online softmax
 * maintains running max + sum-of-exp per Q row, and the output accumulator
 * lives in per-thread registers. Constraints: head_dim ≤ 128, no ALiBi
 * support (falls back to CPU when alibi_slopes != NULL).
 *
 * Targets the same contract as the CPU port in lib/ops/attention_cpu.cpp:
 * causal masking and sliding window are supported; LSE = m + log(l) is
 * written when LSE pointer is non-NULL.
 * ----------------------------------------------------------------------- */

constexpr int kFA_Br = 64;
constexpr int kFA_Bc = 64;
constexpr int kFA_MaxD = 128;

/* The forward kernel uses half2 vectorized loads + __hfma2 in the hot dot
 * product loop. This single change moves the inner loop from scalar
 * fp16→fp32 conversion + scalar FMA to packed fp16 FMAs, doubling the
 * arithmetic throughput per cycle. K/V loads are emitted as 32-bit
 * transactions, halving global → shared memory traffic. P @ V accumulation
 * still uses fp32 accumulators (matches the CPU and Metal kernels).
 *
 * Tile shape: Br=64 Q rows per block (one thread per row), Bc=64 K cols per
 * inner tile, head_dim ≤ 128 (must be a multiple of 2 for the half2 path).
 * For non-2-aligned D we fall through to a scalar inner loop. */
__global__ void flash_attention_forward_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ O,
        float* __restrict__ LSE,
        int B, int Hq, int Hkv, int Sq, int Sk, int D,
        float scale, int causal, int window_size) {
    const int qtile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / Hq;
    const int h = bh % Hq;
    const int Hq_per_Hkv = Hq / Hkv;
    const int hkv = h / Hq_per_Hkv;
    const int tid = threadIdx.x;
    const int q_row = qtile * kFA_Br + tid;
    const bool active = (q_row < Sq);
    const bool aligned = ((D & 1) == 0);
    const int D_half2 = D >> 1;

    extern __shared__ __half smem[];
    __half* sK = smem;
    __half* sV = smem + (size_t)kFA_Bc * D;

    const __half* Qbh = Q + ((size_t)b * Hq + h) * Sq * D;
    const __half* Kbh = K + ((size_t)b * Hkv + hkv) * Sk * D;
    const __half* Vbh = V + ((size_t)b * Hkv + hkv) * Sk * D;
    __half* Obh = O + ((size_t)b * Hq + h) * Sq * D;
    float* LSEbh = LSE ? (LSE + ((size_t)b * Hq + h) * Sq) : nullptr;

    __half Qrow[kFA_MaxD];
    float O_acc[kFA_MaxD];
    #pragma unroll
    for (int d = 0; d < kFA_MaxD; ++d) O_acc[d] = 0.0f;
    if (active) {
        const __half* Qsrc = Qbh + (size_t)q_row * D;
        if (aligned) {
            const __half2* Qsrc2 = reinterpret_cast<const __half2*>(Qsrc);
            __half2* Qrow2 = reinterpret_cast<__half2*>(Qrow);
            #pragma unroll
            for (int d = 0; d < D_half2; ++d) Qrow2[d] = Qsrc2[d];
        } else {
            for (int d = 0; d < D; ++d) Qrow[d] = Qsrc[d];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;

    for (int kb = 0; kb < Sk; kb += kFA_Bc) {
        const int bc_w = (kb + kFA_Bc <= Sk) ? kFA_Bc : (Sk - kb);

        /* Cooperative K/V load into shared. Use 32-bit (half2) loads when
         * aligned: blockDim.x threads each pull D_half2 elements per row,
         * coalesced over the half2 stride. */
        if (aligned) {
            const int total2 = bc_w * D_half2;
            __half2* sK2 = reinterpret_cast<__half2*>(sK);
            __half2* sV2 = reinterpret_cast<__half2*>(sV);
            for (int idx = tid; idx < total2; idx += blockDim.x) {
                const int c = idx / D_half2;
                const int d = idx - c * D_half2;
                const __half2* Krow2 = reinterpret_cast<const __half2*>(
                    Kbh + (size_t)(kb + c) * D);
                const __half2* Vrow2 = reinterpret_cast<const __half2*>(
                    Vbh + (size_t)(kb + c) * D);
                sK2[idx] = Krow2[d];
                sV2[idx] = Vrow2[d];
            }
        } else {
            const int total = bc_w * D;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                const int c = idx / D;
                const int d = idx - c * D;
                sK[idx] = Kbh[(size_t)(kb + c) * D + d];
                sV[idx] = Vbh[(size_t)(kb + c) * D + d];
            }
        }
        __syncthreads();

        if (active) {
            float S[kFA_Bc];
            float m_new = m;
            const __half2* Qrow2 = reinterpret_cast<const __half2*>(Qrow);
            for (int c = 0; c < bc_w; ++c) {
                float s = 0.0f;
                if (aligned) {
                    const __half2* Kc2 = reinterpret_cast<const __half2*>(sK + c * D);
                    __half2 acc2 = __float2half2_rn(0.0f);
                    #pragma unroll
                    for (int d = 0; d < kFA_MaxD / 2; ++d) {
                        if (d < D_half2) acc2 = __hfma2(Qrow2[d], Kc2[d], acc2);
                    }
                    s = __half2float(acc2.x) + __half2float(acc2.y);
                } else {
                    const __half* Kc = sK + c * D;
                    #pragma unroll 32
                    for (int d = 0; d < D; ++d) {
                        s += __half2float(Qrow[d]) * __half2float(Kc[d]);
                    }
                }
                s *= scale;
                const int k_idx = kb + c;
                if (causal && k_idx > q_row) s = -INFINITY;
                if (window_size > 0 && (q_row - k_idx) > window_size) s = -INFINITY;
                S[c] = s;
                if (s > m_new) m_new = s;
            }

            const float scale_old = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
            float l_new = l * scale_old;
            #pragma unroll
            for (int d = 0; d < kFA_MaxD; ++d) {
                if (d < D) O_acc[d] *= scale_old;
            }

            for (int c = 0; c < bc_w; ++c) {
                const float s = S[c];
                if (!isfinite(s)) continue;
                const float p = __expf(s - m_new);
                l_new += p;
                if (aligned) {
                    const __half2* Vc2 = reinterpret_cast<const __half2*>(sV + c * D);
                    const __half2 p2 = __float2half2_rn(p);
                    #pragma unroll
                    for (int d = 0; d < kFA_MaxD / 2; ++d) {
                        if (d < D_half2) {
                            const __half2 prod = __hmul2(p2, Vc2[d]);
                            O_acc[2 * d]     += __half2float(prod.x);
                            O_acc[2 * d + 1] += __half2float(prod.y);
                        }
                    }
                } else {
                    const __half* Vc = sV + c * D;
                    #pragma unroll 32
                    for (int d = 0; d < D; ++d) {
                        O_acc[d] += p * __half2float(Vc[d]);
                    }
                }
            }
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    if (active) {
        const float inv_l = (l != 0.0f) ? (1.0f / l) : 0.0f;
        __half* Orow = Obh + (size_t)q_row * D;
        for (int d = 0; d < D; ++d) {
            Orow[d] = __float2half_rn(O_acc[d] * inv_l);
        }
        if (LSEbh) {
            LSEbh[q_row] = m + __logf(fmaxf(l, 1e-30f));
        }
    }
}

/* ----------------------------------------------------------------------- *
 * FlashAttention-2 forward (CUDA, tensor-core / wmma variant).
 *
 * One block per (qtile, batch*Hq). Two warps per block; each warp owns 16 Q
 * rows. Tile shape: Br=32 Q rows, Bc=32 K cols. Templated on head_dim D ∈
 * {64, 128}; other D's fall through to the scalar/half2 kernel.
 *
 * Pipeline per K-tile:
 *   1. S[Br][Bc] = Q[Br][D] @ K^T[D][Bc] via wmma 16×16×16 fp16→fp32 mma.sync
 *   2. Apply scale + causal/swin mask + online-softmax row update; emit
 *      P[Br][Bc] fp16 to shared, rescale O accumulator (kept in fp32 shared)
 *      by the per-row scale_old, then track running m, l.
 *   3. O[Br][D] += P[Br][Bc] @ V[Bc][D] via wmma mma.sync, loading O frags
 *      from the rescaled shared accumulator and storing them back in-place.
 *
 * Memory budget per block (D=128):
 *   sQ  = 32 × 128 × 2 = 8 KB     sP = 32 × 32 × 2 = 2 KB
 *   sK  = 32 × 128 × 2 = 8 KB     sS = 32 × 32 × 4 = 4 KB
 *   sV  = 32 × 128 × 2 = 8 KB     sO = 32 × 128 × 4 = 16 KB
 *   sm + sl = 256 B
 *   --------------------------------------------------------
 *   total ≈ 46 KB — fits the 48 KB default shared per block on sm_70+.
 * ----------------------------------------------------------------------- */

namespace fa_wmma_constants {
    constexpr int kBr            = 32;
    constexpr int kBc            = 32;
    constexpr int kRowsPerWarp   = 16;
    constexpr int kNumWarps      = kBr / kRowsPerWarp;  /* 2 */
    constexpr int kBlockThreads  = kNumWarps * 32;
    constexpr int kBcFrags       = kBc / 16;             /* 2 */
}

template<int D>
__global__ void flash_attention_forward_wmma_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ O,
        float* __restrict__ LSE,
        int B, int Hq, int Hkv, int Sq, int Sk,
        float scale, int causal, int window_size) {
    using namespace nvcuda;
    using namespace fa_wmma_constants;
    static_assert(D % 16 == 0, "wmma kernel needs D multiple of 16");
    static_assert(D <= 128, "wmma kernel needs D <= 128");
    constexpr int kDFrags = D / 16;

    const int qtile = blockIdx.x;
    const int bh    = blockIdx.y;
    const int b     = bh / Hq;
    const int h     = bh % Hq;
    const int Hq_per_Hkv = Hq / Hkv;
    const int hkv   = h / Hq_per_Hkv;
    const int tid   = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int q_row_block = qtile * kBr;

    extern __shared__ __half smem_fa_wmma[];
    __half* sQ = smem_fa_wmma;
    __half* sK = sQ + (size_t)kBr * D;
    __half* sV = sK + (size_t)kBc * D;
    __half* sP = sV + (size_t)kBc * D;
    float*  sS = reinterpret_cast<float*>(sP + (size_t)kBr * kBc);
    float*  sO = sS + (size_t)kBr * kBc;
    float*  sm = sO + (size_t)kBr * D;
    float*  sl = sm + kBr;

    const __half* Qbh = Q + ((size_t)b * Hq + h) * Sq * D;
    const __half* Kbh = K + ((size_t)b * Hkv + hkv) * Sk * D;
    const __half* Vbh = V + ((size_t)b * Hkv + hkv) * Sk * D;
    __half* Obh       = O + ((size_t)b * Hq + h) * Sq * D;
    float*  LSEbh     = LSE ? (LSE + ((size_t)b * Hq + h) * Sq) : nullptr;

    /* Zero sO accumulator, init sm = -inf, sl = 0. */
    for (int i = tid; i < kBr * D; i += kBlockThreads) sO[i] = 0.0f;
    if (tid < kBr) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    /* Load Q tile cooperatively; OOB rows zero-pad. */
    for (int i = tid; i < kBr * D; i += kBlockThreads) {
        const int r = i / D;
        const int d = i - r * D;
        const int q_g = q_row_block + r;
        sQ[i] = (q_g < Sq) ? Qbh[(size_t)q_g * D + d] : __float2half(0.0f);
    }
    __syncthreads();

    for (int kb = 0; kb < Sk; kb += kBc) {
        /* Cooperative load K, V tiles. */
        for (int i = tid; i < kBc * D; i += kBlockThreads) {
            const int c = i / D;
            const int d = i - c * D;
            const int k_g = kb + c;
            sK[i] = (k_g < Sk) ? Kbh[(size_t)k_g * D + d] : __float2half(0.0f);
            sV[i] = (k_g < Sk) ? Vbh[(size_t)k_g * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        /* S = Q @ K^T via wmma. Per warp: 16 × Bc result, decomposed into kBcFrags acc frags. */
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[kBcFrags];
        #pragma unroll
        for (int n = 0; n < kBcFrags; ++n) wmma::fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
            wmma::load_matrix_sync(q_frag, sQ + warp_id * 16 * D + kk, D);
            #pragma unroll
            for (int n = 0; n < kBcFrags; ++n) {
                /* K[Bc][D] row-major viewed as K^T[D][Bc] col-major with ldm=D. */
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag, sK + n * 16 * D + kk, D);
                wmma::mma_sync(s_frag[n], q_frag, k_frag, s_frag[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < kBcFrags; ++n) {
            wmma::store_matrix_sync(
                sS + warp_id * 16 * kBc + n * 16,
                s_frag[n], kBc, wmma::mem_row_major);
        }
        __syncthreads();

        /* Per-row mask + online softmax + sO rescale + P emit.
         * 16 rows per warp; lanes 0..15 do per-row work, others idle this phase. */
        if (lane < kRowsPerWarp) {
            const int row = warp_id * 16 + lane;
            const int q_g = q_row_block + row;
            float* S_row = sS + row * kBc;

            /* Mask + scale (in place). */
            #pragma unroll
            for (int c = 0; c < kBc; ++c) {
                const int k_g = kb + c;
                float s = S_row[c] * scale;
                if (k_g >= Sk) s = -INFINITY;
                if (causal && k_g > q_g) s = -INFINITY;
                if (window_size > 0 && (q_g - k_g) > window_size) s = -INFINITY;
                S_row[c] = s;
            }

            /* Row max. */
            const float m_prev = sm[row];
            float m_new = m_prev;
            #pragma unroll
            for (int c = 0; c < kBc; ++c) {
                const float s = S_row[c];
                if (s > m_new) m_new = s;
            }

            /* scale_old: rescaling factor for prior O / l. If m_prev is -inf
             * (no prior iterations contributed) we use 0 (sO/sl are already 0
             * so the result is unchanged either way). */
            const float scale_old = (m_prev == -INFINITY)
                                  ? 0.0f
                                  : __expf(m_prev - m_new);

            /* Rescale sO row by scale_old. */
            float* O_row = sO + row * D;
            #pragma unroll
            for (int d = 0; d < D; ++d) O_row[d] *= scale_old;

            /* P emit + accumulate sum (l_new). */
            __half* P_row = sP + row * kBc;
            float l_new = sl[row] * scale_old;
            #pragma unroll
            for (int c = 0; c < kBc; ++c) {
                const float s = S_row[c];
                const float p = isfinite(s) ? __expf(s - m_new) : 0.0f;
                P_row[c] = __float2half(p);
                l_new += p;
            }
            sm[row] = m_new;
            sl[row] = l_new;
        }
        __syncthreads();

        /* O += P @ V via wmma. Each warp owns 16 rows; output is 16 × D
         * (kDFrags acc frags). Load o_frag from sO (already rescaled),
         * accumulate, store back. Inner k loop over Bc in steps of 16. */
        #pragma unroll
        for (int n = 0; n < kDFrags; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
            wmma::load_matrix_sync(
                o_frag, sO + warp_id * 16 * D + n * 16, D, wmma::mem_row_major);
            #pragma unroll
            for (int kk = 0; kk < kBc; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, sP + warp_id * 16 * kBc + kk, kBc);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag, sV + kk * D + n * 16, D);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(
                sO + warp_id * 16 * D + n * 16, o_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    /* Finalize: O /= sl; write fp16 output and LSE. */
    if (lane < kRowsPerWarp) {
        const int row = warp_id * 16 + lane;
        const int q_g = q_row_block + row;
        if (q_g < Sq) {
            const float l = sl[row];
            const float inv_l = (l != 0.0f) ? (1.0f / l) : 0.0f;
            __half* O_row = Obh + (size_t)q_g * D;
            const float* sO_row = sO + row * D;
            for (int d = 0; d < D; ++d) {
                O_row[d] = __float2half_rn(sO_row[d] * inv_l);
            }
            if (LSEbh) {
                LSEbh[q_g] = sm[row] + __logf(fmaxf(l, 1e-30f));
            }
        }
    }
}

/* Shared-memory footprint of the wmma kernel for a given D. */
template<int D>
constexpr size_t fa_wmma_shmem() {
    using namespace fa_wmma_constants;
    return (size_t)(kBr * D + kBc * D + kBc * D + kBr * kBc) * sizeof(__half)
         + (size_t)(kBr * kBc + kBr * D + kBr + kBr)         * sizeof(float);
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_attention_forward(
        const void* Q, const void* K, const void* V, void* O, void* LSE,
        int B, int Hq, int Hkv, int Sq, int Sk, int D,
        float scale, int causal, int window_size,
        const float* alibi_slopes) {
    if (!Q || !K || !V || !O) return kCudaTrainingError;
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Sq <= 0 || Sk <= 0 || D <= 0) return kCudaTrainingError;
    if (D > kFA_MaxD) return kCudaTrainingUnsupported;
    if (Hq % Hkv != 0) return kCudaTrainingError;
    if (alibi_slopes) return kCudaTrainingUnsupported;
    if (!is_managed_cuda_ptr(Q) || !is_managed_cuda_ptr(K) ||
        !is_managed_cuda_ptr(V) || !is_managed_cuda_ptr(O)) {
        return kCudaTrainingUnsupported;
    }
    if (LSE && !is_managed_cuda_ptr(LSE)) return kCudaTrainingUnsupported;

    /* Fast path: tensor-core wmma kernel for D ∈ {64, 128}. */
    if (D == 128 || D == 64) {
        using namespace fa_wmma_constants;
        dim3 grid((Sq + kBr - 1) / kBr, (unsigned)B * Hq);
        dim3 block(kBlockThreads);
        cudaError_t e = cudaSuccess;
        if (D == 128) {
            const size_t shmem = fa_wmma_shmem<128>();
            /* Opt in to extended dynamic shared memory (Ampere/Ada/Hopper:
             * default per-block ceiling is 48 KB; tell the driver this kernel
             * needs up to `shmem` bytes). Silently skipped on older arches. */
            cudaFuncSetAttribute(
                (const void*)flash_attention_forward_wmma_kernel<128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shmem);
            flash_attention_forward_wmma_kernel<128><<<grid, block, shmem>>>(
                (const __half*)Q, (const __half*)K, (const __half*)V,
                (__half*)O, (float*)LSE,
                B, Hq, Hkv, Sq, Sk, scale, causal, window_size);
        } else {
            const size_t shmem = fa_wmma_shmem<64>();
            cudaFuncSetAttribute(
                (const void*)flash_attention_forward_wmma_kernel<64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shmem);
            flash_attention_forward_wmma_kernel<64><<<grid, block, shmem>>>(
                (const __half*)Q, (const __half*)K, (const __half*)V,
                (__half*)O, (float*)LSE,
                B, Hq, Hkv, Sq, Sk, scale, causal, window_size);
        }
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            /* Surface the launch error name to last_kernel so the caller
             * can see which path failed (visible via tc.cuda_last_kernel_name). */
            tc_cuda_set_last_kernel(cudaGetErrorName(e));
            return kCudaTrainingError;
        }
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            tc_cuda_set_last_kernel(cudaGetErrorName(e));
            return kCudaTrainingError;
        }
        tc_cuda_set_last_kernel("cuda_attention_forward_wmma");
        return kCudaTrainingOk;
    }

    /* Fallback: scalar/half2 kernel for other D's. */
    dim3 grid((Sq + kFA_Br - 1) / kFA_Br, (unsigned)B * Hq);
    dim3 block(kFA_Br);
    const size_t shmem = (size_t)2 * kFA_Bc * D * sizeof(__half);
    flash_attention_forward_kernel<<<grid, block, shmem>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V,
        (__half*)O, (float*)LSE,
        B, Hq, Hkv, Sq, Sk, D, scale, causal, window_size);
    if (cudaGetLastError() != cudaSuccess) return kCudaTrainingError;
    if (cudaDeviceSynchronize() != cudaSuccess) return kCudaTrainingError;
    tc_cuda_set_last_kernel("cuda_attention_forward");
    return kCudaTrainingOk;
}

/* ----------------------------------------------------------------------- *
 * FlashAttention-2 backward (CUDA).
 *
 *   Inputs: Q, K, V (fp16); O, dO (fp16); LSE (fp32).
 *   Outputs: dQ, dK, dV (fp16).
 *   Algorithm (matches lib/ops/attention_cpu.cpp:tc_attention_backward):
 *     D_row = rowsum(dO * O)                    (precomputed per Q row)
 *     S[i,j] = (Q_i · K_j) * scale              (recompute, fwd LSE saved)
 *     P[i,j] = exp(S[i,j] - LSE[i])
 *     dV[j] += sum_i P[i,j] * dO[i]
 *     dP[i,j] = dO[i] · V[j]
 *     dS[i,j] = P[i,j] * (dP[i,j] - D_row[i])
 *     dK[j] += sum_i dS[i,j] * Q[i] * scale
 *     dQ[i] += sum_j dS[i,j] * K[j] * scale
 *
 *   Dispatch: grid = (Sk_tiles, B * Hq), block = kFA_Bc threads.
 *   Each block owns one K-column tile for one (b, h_q) pair; iterates over
 *   Q rows sequentially. dK/dV partial accumulators live in per-thread
 *   registers (no atomics needed — exclusive per K-tile). dQ contributions
 *   span K tiles, so use atomicAdd into a fp32 scratch (`dQ_fp32`) that
 *   the wrapper converts to fp16 after the kernel returns.
 *
 *   GQA: when Hq != Hkv, multiple Q heads map to the same KV head; dK/dV
 *   for the KV head needs contributions from all Q heads in the family.
 *   Handled by atomicAdd into the fp32 dK/dV scratch, same pattern as dQ.
 * ----------------------------------------------------------------------- */

__global__ void flash_attention_backward_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        const __half* __restrict__ O,
        const __half* __restrict__ dO,
        const float* __restrict__ LSE,
        float* __restrict__ dQ_fp32,         /* [B, Hq, Sq, D]   */
        float* __restrict__ dK_fp32,         /* [B, Hkv, Sk, D]  */
        float* __restrict__ dV_fp32,         /* [B, Hkv, Sk, D]  */
        int B, int Hq, int Hkv, int Sq, int Sk, int D,
        float scale, int causal, int window_size) {
    const int ktile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / Hq;
    const int h = bh % Hq;
    const int Hq_per_Hkv = Hq / Hkv;
    const int hkv = h / Hq_per_Hkv;
    const int tid = threadIdx.x;
    const int k_col = ktile * kFA_Bc + tid;
    const bool active = (k_col < Sk);
    const bool aligned = ((D & 1) == 0);
    const int D_half2 = D >> 1;

    extern __shared__ __half smem_bwd[];
    __half* sQ  = smem_bwd;                                  /* [Bc][D] (Q rows in this k-iter) */
    __half* sdO = sQ + (size_t)kFA_Bc * D;                   /* [Bc][D] */

    const __half* Qbh = Q + ((size_t)b * Hq + h) * Sq * D;
    const __half* Kbh = K + ((size_t)b * Hkv + hkv) * Sk * D;
    const __half* Vbh = V + ((size_t)b * Hkv + hkv) * Sk * D;
    const __half* Obh = O + ((size_t)b * Hq + h) * Sq * D;
    const __half* dObh = dO + ((size_t)b * Hq + h) * Sq * D;
    const float*  LSEbh = LSE + ((size_t)b * Hq + h) * Sq;
    float* dQbh = dQ_fp32 + ((size_t)b * Hq + h) * Sq * D;
    float* dKbh = dK_fp32 + ((size_t)b * Hkv + hkv) * Sk * D;
    float* dVbh = dV_fp32 + ((size_t)b * Hkv + hkv) * Sk * D;

    /* Per-thread K, V row (this block handles one K-tile). */
    __half Krow[kFA_MaxD];
    __half Vrow[kFA_MaxD];
    float dK_acc[kFA_MaxD];
    float dV_acc[kFA_MaxD];
    #pragma unroll
    for (int d = 0; d < kFA_MaxD; ++d) { dK_acc[d] = 0.0f; dV_acc[d] = 0.0f; }
    if (active) {
        const __half* Ksrc = Kbh + (size_t)k_col * D;
        const __half* Vsrc = Vbh + (size_t)k_col * D;
        if (aligned) {
            const __half2* Ksrc2 = reinterpret_cast<const __half2*>(Ksrc);
            const __half2* Vsrc2 = reinterpret_cast<const __half2*>(Vsrc);
            __half2* Krow2 = reinterpret_cast<__half2*>(Krow);
            __half2* Vrow2 = reinterpret_cast<__half2*>(Vrow);
            #pragma unroll
            for (int d = 0; d < kFA_MaxD / 2; ++d) {
                if (d < D_half2) { Krow2[d] = Ksrc2[d]; Vrow2[d] = Vsrc2[d]; }
            }
        } else {
            for (int d = 0; d < D; ++d) { Krow[d] = Ksrc[d]; Vrow[d] = Vsrc[d]; }
        }
    }

    /* Iterate Q rows in tiles of Br=Bc (so we reuse the same per-block thread layout). */
    for (int qb = 0; qb < Sq; qb += kFA_Bc) {
        const int qrows = (qb + kFA_Bc <= Sq) ? kFA_Bc : (Sq - qb);

        /* Cooperative load: Q tile [qrows][D] and dO tile [qrows][D] into shared. */
        if (aligned) {
            const int total2 = qrows * D_half2;
            __half2* sQ2 = reinterpret_cast<__half2*>(sQ);
            __half2* sdO2 = reinterpret_cast<__half2*>(sdO);
            for (int idx = tid; idx < total2; idx += blockDim.x) {
                const int r = idx / D_half2;
                const int d = idx - r * D_half2;
                const __half2* Qrow2 = reinterpret_cast<const __half2*>(Qbh + (size_t)(qb + r) * D);
                const __half2* dOrow2 = reinterpret_cast<const __half2*>(dObh + (size_t)(qb + r) * D);
                sQ2[idx] = Qrow2[d];
                sdO2[idx] = dOrow2[d];
            }
        } else {
            const int total = qrows * D;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                const int r = idx / D;
                const int d = idx - r * D;
                sQ[idx] = Qbh[(size_t)(qb + r) * D + d];
                sdO[idx] = dObh[(size_t)(qb + r) * D + d];
            }
        }
        __syncthreads();

        if (!active) {
            __syncthreads();
            continue;
        }

        /* For each Q row in this tile, compute its contribution to my K column. */
        for (int qr = 0; qr < qrows; ++qr) {
            const int q_g = qb + qr;
            const __half* Qrow_s = sQ + (size_t)qr * D;
            const __half* dOrow_s = sdO + (size_t)qr * D;

            /* causal / sliding-window mask */
            bool masked = false;
            if (causal && k_col > q_g) masked = true;
            if (window_size > 0 && (q_g - k_col) > window_size) masked = true;

            float P = 0.0f;
            float dP = 0.0f;
            float dS = 0.0f;
            float D_row = 0.0f;

            if (!masked) {
                /* S = Q · K * scale via half2 hfma2 (vectorized inner). */
                float s = 0.0f;
                if (aligned) {
                    const __half2* Qrow2 = reinterpret_cast<const __half2*>(Qrow_s);
                    const __half2* Krow2 = reinterpret_cast<const __half2*>(Krow);
                    __half2 acc2 = __float2half2_rn(0.0f);
                    #pragma unroll
                    for (int d = 0; d < kFA_MaxD / 2; ++d) {
                        if (d < D_half2) acc2 = __hfma2(Qrow2[d], Krow2[d], acc2);
                    }
                    s = __half2float(acc2.x) + __half2float(acc2.y);
                } else {
                    for (int d = 0; d < D; ++d) {
                        s += __half2float(Qrow_s[d]) * __half2float(Krow[d]);
                    }
                }
                s *= scale;
                P = __expf(s - LSEbh[q_g]);

                /* dP = dO · V */
                float dp = 0.0f;
                if (aligned) {
                    const __half2* dOrow2 = reinterpret_cast<const __half2*>(dOrow_s);
                    const __half2* Vrow2 = reinterpret_cast<const __half2*>(Vrow);
                    __half2 acc2 = __float2half2_rn(0.0f);
                    #pragma unroll
                    for (int d = 0; d < kFA_MaxD / 2; ++d) {
                        if (d < D_half2) acc2 = __hfma2(dOrow2[d], Vrow2[d], acc2);
                    }
                    dp = __half2float(acc2.x) + __half2float(acc2.y);
                } else {
                    for (int d = 0; d < D; ++d) {
                        dp += __half2float(dOrow_s[d]) * __half2float(Vrow[d]);
                    }
                }
                dP = dp;

                /* D_row[q_g] = sum_d O[q_g, d] * dO[q_g, d] — recompute here so we
                 * don't need an extra precomputation pass / extra global memory. */
                const __half* Orow = Obh + (size_t)q_g * D;
                float drow = 0.0f;
                if (aligned) {
                    const __half2* Orow2 = reinterpret_cast<const __half2*>(Orow);
                    const __half2* dOrow2 = reinterpret_cast<const __half2*>(dObh + (size_t)q_g * D);
                    __half2 acc2 = __float2half2_rn(0.0f);
                    #pragma unroll
                    for (int d = 0; d < kFA_MaxD / 2; ++d) {
                        if (d < D_half2) acc2 = __hfma2(Orow2[d], dOrow2[d], acc2);
                    }
                    drow = __half2float(acc2.x) + __half2float(acc2.y);
                } else {
                    const __half* dOg = dObh + (size_t)q_g * D;
                    for (int d = 0; d < D; ++d) {
                        drow += __half2float(Orow[d]) * __half2float(dOg[d]);
                    }
                }
                D_row = drow;
                dS = P * (dP - D_row);

                /* Local dV += P * dO[q_g] ; local dK += dS * Q[q_g] * scale */
                #pragma unroll
                for (int d = 0; d < kFA_MaxD; ++d) {
                    if (d < D) {
                        dV_acc[d] += P * __half2float(dOrow_s[d]);
                        dK_acc[d] += dS * __half2float(Qrow_s[d]) * scale;
                    }
                }

            }
            /* Warp-cooperative dQ update: rather than each thread doing D
             * atomicAdds (one per d), warp-reduce dS * K[d] * scale across
             * the 32 lanes per d, and have lane 0 do a single atomicAdd.
             * This cuts the dQ atomic count by 32× without changing math.
             * Per warp: contributes (sum over its 32 K cols) to dQ[q_g, d].
             * Total dQ atomics: 2 warps × D × 1 per Q row instead of
             * Bc × D = 64 × D per Q row → 32× fewer global atomics. */
            const unsigned mask_full = 0xffffffffu;
            const float dS_val = masked ? 0.0f : dS;
            float* dQrow = dQbh + (size_t)q_g * D;
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            #pragma unroll
            for (int d = 0; d < kFA_MaxD; ++d) {
                if (d < D) {
                    float val = dS_val * __half2float(Krow[d]) * scale;
                    /* Butterfly reduction across the 32 lanes of this warp. */
                    val += __shfl_xor_sync(mask_full, val, 16);
                    val += __shfl_xor_sync(mask_full, val,  8);
                    val += __shfl_xor_sync(mask_full, val,  4);
                    val += __shfl_xor_sync(mask_full, val,  2);
                    val += __shfl_xor_sync(mask_full, val,  1);
                    if (lane == 0 && val != 0.0f) atomicAdd(dQrow + d, val);
                    (void)warp_id; /* silence -Wunused on this branch */
                }
            }
        }
        __syncthreads();
    }

    /* Write per-thread dK/dV accumulators. With GQA, multiple Q heads map to
     * the same KV head and contribute to the same dK[hkv, k] / dV[hkv, k];
     * use atomicAdd for that case. Without GQA (Hq == Hkv) we still atomic to
     * keep the dispatch single-code-path. */
    if (active) {
        float* dKrow = dKbh + (size_t)k_col * D;
        float* dVrow = dVbh + (size_t)k_col * D;
        for (int d = 0; d < D; ++d) {
            atomicAdd(dKrow + d, dK_acc[d]);
            atomicAdd(dVrow + d, dV_acc[d]);
        }
    }
}

/* Convert fp32 grad scratch → fp16 output buffers. One thread per element. */
__global__ void fa_fp32_to_fp16_kernel(const float* __restrict__ src,
                                        __half* __restrict__ dst, size_t N) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = __float2half_rn(src[i]);
}

extern "C" TC_CUDA_INTERNAL int tc_cuda_attention_backward(
        const void* Q, const void* K, const void* V, const void* O,
        const void* dO, const void* LSE,
        void* dQ, void* dK, void* dV,
        int B, int Hq, int Hkv, int Sq, int Sk, int D,
        float scale, int causal, int window_size,
        const float* alibi_slopes) {
    if (!Q || !K || !V || !O || !dO || !LSE || !dQ || !dK || !dV)
        return kCudaTrainingError;
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Sq <= 0 || Sk <= 0 || D <= 0)
        return kCudaTrainingError;
    if (D > kFA_MaxD) return kCudaTrainingUnsupported;
    if (Hq % Hkv != 0) return kCudaTrainingError;
    if (alibi_slopes) return kCudaTrainingUnsupported;
    if (!is_managed_cuda_ptr(Q) || !is_managed_cuda_ptr(K) ||
        !is_managed_cuda_ptr(V) || !is_managed_cuda_ptr(O) ||
        !is_managed_cuda_ptr(dO) || !is_managed_cuda_ptr(LSE) ||
        !is_managed_cuda_ptr(dQ) || !is_managed_cuda_ptr(dK) ||
        !is_managed_cuda_ptr(dV)) {
        return kCudaTrainingUnsupported;
    }

    /* Allocate fp32 scratch for dQ, dK, dV accumulators (managed memory so
     * the conversion kernel can read them too). */
    const size_t dq_elts = (size_t)B * Hq * Sq * D;
    const size_t kv_elts = (size_t)B * Hkv * Sk * D;
    float* dQ_fp32 = nullptr;
    float* dK_fp32 = nullptr;
    float* dV_fp32 = nullptr;
    if (cudaMallocManaged((void**)&dQ_fp32, dq_elts * sizeof(float)) != cudaSuccess) return kCudaTrainingError;
    if (cudaMallocManaged((void**)&dK_fp32, kv_elts * sizeof(float)) != cudaSuccess) {
        cudaFree(dQ_fp32); return kCudaTrainingError;
    }
    if (cudaMallocManaged((void**)&dV_fp32, kv_elts * sizeof(float)) != cudaSuccess) {
        cudaFree(dQ_fp32); cudaFree(dK_fp32); return kCudaTrainingError;
    }
    if (cudaMemset(dQ_fp32, 0, dq_elts * sizeof(float)) != cudaSuccess ||
        cudaMemset(dK_fp32, 0, kv_elts * sizeof(float)) != cudaSuccess ||
        cudaMemset(dV_fp32, 0, kv_elts * sizeof(float)) != cudaSuccess) {
        cudaFree(dQ_fp32); cudaFree(dK_fp32); cudaFree(dV_fp32);
        return kCudaTrainingError;
    }

    dim3 grid((Sk + kFA_Bc - 1) / kFA_Bc, (unsigned)B * Hq);
    dim3 block(kFA_Bc);
    const size_t shmem = (size_t)2 * kFA_Bc * D * sizeof(__half);
    flash_attention_backward_kernel<<<grid, block, shmem>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V,
        (const __half*)O, (const __half*)dO, (const float*)LSE,
        dQ_fp32, dK_fp32, dV_fp32,
        B, Hq, Hkv, Sq, Sk, D, scale, causal, window_size);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(dQ_fp32); cudaFree(dK_fp32); cudaFree(dV_fp32);
        return kCudaTrainingError;
    }

    /* fp32 → fp16 conversion for the three outputs. */
    const int conv_block = 256;
    fa_fp32_to_fp16_kernel<<<(unsigned)((dq_elts + conv_block - 1) / conv_block),
                              conv_block>>>(dQ_fp32, (__half*)dQ, dq_elts);
    fa_fp32_to_fp16_kernel<<<(unsigned)((kv_elts + conv_block - 1) / conv_block),
                              conv_block>>>(dK_fp32, (__half*)dK, kv_elts);
    fa_fp32_to_fp16_kernel<<<(unsigned)((kv_elts + conv_block - 1) / conv_block),
                              conv_block>>>(dV_fp32, (__half*)dV, kv_elts);
    cudaDeviceSynchronize();

    cudaFree(dQ_fp32); cudaFree(dK_fp32); cudaFree(dV_fp32);
    tc_cuda_set_last_kernel("cuda_attention_backward");
    return kCudaTrainingOk;
}
