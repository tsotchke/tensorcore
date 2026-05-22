# tensorcore — C API reference

The complete public surface, grouped by header. Every symbol below appears
in `include/tensorcore/*.h` and is part of the stable ABI. Internal helpers
in `lib/core/internal.h` are out of scope.

Every function returns `tc_status_t` unless noted. `TC_OK == 0` on success;
negative values are errors. Use `tc_status_string(s)` to render an error.

## Umbrella

`include/tensorcore/tensorcore.h` includes every header below and exposes:

```c
#define TENSORCORE_VERSION_MAJOR 0
#define TENSORCORE_VERSION_MINOR 1
#define TENSORCORE_VERSION_PATCH 22  /* tracks include/tensorcore/tensorcore.h */

const char* tc_version(void);   /* "0.1.x" */
```

The `VERSION` field in `CMakeLists.txt` is authoritative for builds and
the `pkg-config` file; the header constants exist for downstream C code
that wants the version at compile time. They are kept in sync at release
time.

## Status — `status.h`

```c
typedef enum {
    TC_OK                       =   0,
    TC_ERR_NOT_INITIALIZED      =  -1,
    TC_ERR_ALREADY_INITIALIZED  =  -2,
    TC_ERR_NO_DEVICE            =  -3,
    TC_ERR_UNSUPPORTED_FAMILY   =  -4,  /* kernel needs newer Apple family */
    TC_ERR_UNSUPPORTED_DTYPE    =  -5,
    TC_ERR_INVALID_SHAPE        =  -6,
    TC_ERR_INVALID_ARG          =  -7,
    TC_ERR_ALLOC                =  -8,
    TC_ERR_KERNEL_NOT_FOUND     =  -9,  /* metallib missing the function */
    TC_ERR_PIPELINE             = -10,  /* MTLComputePipelineState build failed */
    TC_ERR_DISPATCH             = -11,
    TC_ERR_INTERNAL             = -99,
} tc_status_t;

const char* tc_status_string(tc_status_t s);
```

## Dtype — `dtype.h`

```c
typedef enum {
    TC_DTYPE_F16  = 0,   /* IEEE 754 binary16; simdgroup_matrix Apple7+        */
    TC_DTYPE_BF16 = 1,   /* bfloat16; simdgroup_matrix Apple9+; FP32 fallback  */
    TC_DTYPE_F32  = 2,   /* IEEE 754 binary32; simdgroup_matrix Apple7+        */
    TC_DTYPE_I8   = 3,   /* int8; simdgroup_matrix Apple10+; FP32 fallback     */
    TC_DTYPE_I32  = 4,   /* int32; i8 accumulator or index dtype               */
    TC_DTYPE_F64  = 5,   /* IEEE 754 binary64; emulated on GPU                 */
    TC_DTYPE_SF64 = 6,   /* SoftFloat-64 storage (uint2)                       */
    TC_DTYPE_DF64 = 7,   /* double-float (f32+f32 unevaluated sum)             */
    TC_DTYPE_FP24 = 8,   /* 24-bit ML format from eshkol-platform              */
    TC_DTYPE_FP53 = 9,   /* 53-bit format from eshkol-platform                 */
} tc_dtype_t;

static inline size_t tc_dtype_size(tc_dtype_t d);   /* bytes per element       */
const char*           tc_dtype_name(tc_dtype_t d);  /* "F16", "BF16", ...       */
```

Ordering is fixed; new dtypes append. The high-order entries (`SF64`,
`DF64`, `FP24`, `FP53`) are eshkol-platform precision modes that the v0.4
consolidation moves into tensorcore proper.

## Device, buffer, stream — `device.h`

### Opaque handles

```c
typedef struct tc_context tc_context;   /* MTLDevice + MTLCommandQueue + caches */
typedef struct tc_buffer  tc_buffer;    /* MTLBuffer (shared storage)           */
typedef struct tc_stream  tc_stream;    /* command-buffer lane                  */
```

### Family classification

```c
typedef enum {
    TC_FAMILY_UNKNOWN = 0,
    TC_FAMILY_APPLE7  = 7,    /* M1               */
    TC_FAMILY_APPLE8  = 8,    /* M2               */
    TC_FAMILY_APPLE9  = 9,    /* M3, A17 Pro       — bf16 simdgroup_matrix */
    TC_FAMILY_APPLE10 = 10,   /* M4                — int8 simdgroup_matrix */
    TC_FAMILY_APPLE11 = 11,   /* M5                — mpp::tensor_ops      */
} tc_family_t;

typedef struct {
    tc_family_t family;
    char        name[128];                    /* MTLDevice.name              */
    uint64_t    max_buffer_bytes;
    uint64_t    recommended_working_set_bytes;
    uint32_t    max_threadgroup_memory;       /* 32 KB on M-series           */
    uint32_t    max_threads_per_threadgroup;
    uint32_t    thread_execution_width;       /* 32 (simdgroup width)        */
    bool        unified_memory;
    bool        supports_bf16_simdgroup;
    bool        supports_i8_simdgroup;
    bool        supports_tensorops_m5;
    bool        supports_fp64_native;         /* false on Apple GPU           */
} tc_device_info;
```

### Lifecycle and query

```c
tc_status_t tc_init           (tc_context** out_ctx);
tc_status_t tc_shutdown       (tc_context*  ctx);
tc_status_t tc_device_info_get(tc_context*  ctx, tc_device_info* out_info);
```

`tc_init` is idempotent — second call returns `TC_ERR_ALREADY_INITIALIZED`.

### Buffers (unified memory)

```c
tc_status_t tc_buffer_alloc(tc_context* ctx, size_t bytes, tc_buffer** out);
tc_status_t tc_buffer_free (tc_context* ctx, tc_buffer*  buf);
tc_status_t tc_buffer_map  (tc_buffer*  buf, void**      out_ptr);
size_t      tc_buffer_size (const tc_buffer* buf);
```

`tc_buffer_map` returns a CPU-addressable pointer with no copy. The buffer
pool uses power-of-2 buckets and LIFO recycling.

### Streams

```c
tc_status_t tc_stream_create (tc_context* ctx, tc_stream** out);
tc_status_t tc_stream_destroy(tc_context* ctx, tc_stream*  s);
tc_status_t tc_stream_sync   (tc_stream*  s);
```

NULL stream means "use the default stream" — sync-commit on every call.

## Memory Tiering — `memory_tier.h`

The memory-tier ABI lets higher-level runtimes hint which buffers should
stay hot and which can eventually move to slower storage. The current
runtime ships an L0-only stub baseline: hints are accepted, queried buffers
report `TC_TIER_L0_DEVICE`, and L1-L4 hosting lands with the heterogeneous
mesh runtime.

```c
typedef enum {
    TC_TIER_L0_DEVICE      = 0,
    TC_TIER_L1_HOST_RAM    = 1,
    TC_TIER_L2_REMOTE_RAM  = 2,
    TC_TIER_L3_LOCAL_NVME  = 3,
    TC_TIER_L4_REMOTE_NVME = 4,
} tc_memory_tier_t;

typedef enum {
    TC_TIER_HINT_HOT  = 0,
    TC_TIER_HINT_WARM = 1,
    TC_TIER_HINT_COLD = 2,
    TC_TIER_HINT_ICE  = 3,
} tc_tier_hint_t;

tc_status_t tc_buffer_set_tier_hint(tc_buffer* b, tc_tier_hint_t hint);
tc_status_t tc_buffer_get_tier(const tc_buffer* b,
                               tc_memory_tier_t* out_tier);
tc_status_t tc_buffer_promote_async(tc_buffer* b,
                                    tc_memory_tier_t target_tier,
                                    tc_stream* stream);
tc_status_t tc_buffer_demote_async(tc_buffer* b,
                                   tc_memory_tier_t target_tier,
                                   tc_stream* stream);
tc_status_t tc_buffer_tier_sync(tc_buffer* b);
tc_status_t tc_memory_tier_usage(tc_context* ctx,
                                 tc_memory_tier_t tier,
                                 uint64_t* out_bytes_resident,
                                 uint64_t* out_bytes_capacity);
```

## Activation Checkpointing — `checkpoint.h`

The checkpoint ABI provides buffer-level hooks for frameworks that trade
activation memory for recompute. The current runtime ships resident-only
weak stubs: `discard` updates observability counters, and `realize` invokes
the registered recompute callback without freeing/reallocating the buffer.

```c
typedef uint64_t tc_checkpoint_id;
typedef tc_status_t(*tc_checkpoint_recompute_fn)(void* user_data);

tc_status_t tc_checkpoint_register(tc_buffer* buf,
                                   tc_checkpoint_recompute_fn recompute_fn,
                                   void* user_data,
                                   tc_checkpoint_id* out_id);
tc_status_t tc_checkpoint_discard(tc_checkpoint_id id);
tc_status_t tc_checkpoint_realize(tc_checkpoint_id id);
int tc_checkpoint_is_resident(tc_checkpoint_id id);
tc_status_t tc_checkpoint_unregister(tc_checkpoint_id id);
uint64_t tc_checkpoint_total_bytes_discarded(void);
uint64_t tc_checkpoint_count_resident(void);
uint64_t tc_checkpoint_count_discarded(void);
```

## GEMM — `gemm.h`

### Descriptor

```c
typedef struct {
    int32_t   M, N, K;

    tc_dtype_t a_dtype;
    tc_dtype_t b_dtype;
    tc_dtype_t c_dtype;
    tc_dtype_t accum_dtype;     /* normally F32 (or I32 for I8 inputs)     */

    bool transpose_a;
    bool transpose_b;

    float alpha;                /* C = alpha * A @ B + beta * C            */
    float beta;

    int32_t lda, ldb, ldc;      /* 0 → row-major contiguous default        */
} tc_gemm_desc;
```

### Calls

```c
tc_status_t tc_gemm        (tc_context* ctx, const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C);

tc_status_t tc_gemm_async  (tc_context* ctx, const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C, tc_stream* stream);

typedef struct {
    tc_gemm_desc base;
    int32_t      batch;
    int64_t      stride_a;
    int64_t      stride_b;
    int64_t      stride_c;
} tc_gemm_batched_desc;

tc_status_t tc_gemm_batched(tc_context* ctx, const tc_gemm_batched_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C);
```

### Diagnostics

```c
typedef enum {
    TC_BACKEND_NONE             = 0,
    TC_BACKEND_SIMDGROUP_MATRIX = 1,
    TC_BACKEND_TENSOROPS_M5     = 2,
    TC_BACKEND_MPS              = 3,
    TC_BACKEND_ACCELERATE_CPU   = 4,
    TC_BACKEND_SF64_EMULATED    = 5,
    TC_BACKEND_OZAKI_II         = 6,
    TC_BACKEND_PORTABLE_CPU     = 7,   /* portable C CPU backend (TC_ENABLE_METAL=OFF) */
} tc_backend_t;

tc_backend_t tc_last_backend(void);            /* thread-local                */
const char*  tc_backend_name(tc_backend_t b);  /* "simdgroup_matrix", ...     */
```

Use `tc_last_backend()` immediately after a GEMM or attention call to learn
which path served it. Useful both for debugging "why is this slow?" and for
adapting policy in higher-level code.

**Scope:** `tc_set_last_backend` is currently written from
`lib/ops/gemm.mm`, `lib/ops/attention.mm`, and `lib/tensorops/tensorops_m5.mm`.
Calls into training, conv, and quantized kernels do **not** update the
diagnostic; the value reflects whichever GEMM/attention/tensorops path
ran most recently on the calling thread. Treat the symbol as "last-GEMM-like"
rather than "last-call." (Widening this to every dispatch is a v0.2 polish
item.)

See [gemm.md](gemm.md) for kernel choices, tile sizes, and env flags.

## Attention — `attention.h`

### Descriptor

```c
typedef struct {
    int32_t batch;
    int32_t heads;
    int32_t seq_q;
    int32_t seq_kv;
    int32_t head_dim;          /* ≤ 128 for on-chip tile                    */

    tc_dtype_t io_dtype;       /* F16 or BF16                                */
    tc_dtype_t accum_dtype;    /* F32                                        */

    float    softmax_scale;    /* commonly 1 / sqrt(head_dim)                */
    bool     causal;
    bool     return_lse;       /* write log-sum-exp for backward             */

    int32_t  kv_heads;         /* 0 → heads (no GQA)                         */
    int32_t  window_size;      /* 0 → no sliding window                      */
    const float* alibi_slopes; /* NULL → no ALiBi; else host fp32 [heads]    */
} tc_attention_desc;
```

### Calls

```c
tc_status_t tc_attention_forward      (tc_context* ctx,
                                       const tc_attention_desc* d,
                                       const tc_buffer* Q,
                                       const tc_buffer* K,
                                       const tc_buffer* V,
                                       tc_buffer*       O,
                                       tc_buffer*       LSE);

tc_status_t tc_attention_forward_async(tc_context* ctx,
                                       const tc_attention_desc* d,
                                       const tc_buffer* Q,
                                       const tc_buffer* K,
                                       const tc_buffer* V,
                                       tc_buffer*       O,
                                       tc_buffer*       LSE,
                                       tc_stream*       stream);

tc_status_t tc_attention_backward     (tc_context* ctx,
                                       const tc_attention_desc* d,
                                       const tc_buffer* Q,
                                       const tc_buffer* K,
                                       const tc_buffer* V,
                                       const tc_buffer* O,
                                       const tc_buffer* dO,
                                       const tc_buffer* LSE,
                                       tc_buffer*       dQ,
                                       tc_buffer*       dK,
                                       tc_buffer*       dV);
```

See [attention.md](attention.md) for the FlashAttention design, the D=64 /
D=128 kernels, causal / GQA / window / ALiBi semantics, and the backward
pass.

## Training kernels — `training.h`

### RMSnorm

```c
tc_status_t tc_rmsnorm_forward (tc_context* ctx,
                                const tc_buffer* X,       /* [N, D] fp16    */
                                const tc_buffer* gamma,   /* [D]    fp16    */
                                tc_buffer*       Y,       /* [N, D] fp16    */
                                tc_buffer*       rstd_out,/* [N]    fp32    */
                                int N, int D, float eps);

tc_status_t tc_rmsnorm_backward(tc_context* ctx,
                                const tc_buffer* X,        /* [N, D] fp16   */
                                const tc_buffer* gamma,    /* [D]    fp16   */
                                const tc_buffer* dY,       /* [N, D] fp16   */
                                const tc_buffer* rstd,     /* [N]    fp32   */
                                tc_buffer*       dX,       /* [N, D] fp16   */
                                tc_buffer*       dgamma,   /* [D]    fp32   */
                                int N, int D);
```

`dgamma` is **fp32** — the kernel accumulates the cross-batch sum in fp32
for numerical stability. Allocate as `D * sizeof(float)` bytes; pass to
`tc_adamw_step` with `grad_dtype = TC_DTYPE_F32`.

### LayerNorm

```c
tc_status_t tc_layernorm_forward (tc_context* ctx,
                                  const tc_buffer* X, const tc_buffer* gamma,
                                  const tc_buffer* beta,
                                  tc_buffer* Y,
                                  tc_buffer* mean_out, tc_buffer* rstd_out,
                                  int N, int D, float eps);

tc_status_t tc_layernorm_backward(tc_context* ctx,
                                  const tc_buffer* X, const tc_buffer* gamma,
                                  const tc_buffer* dY,
                                  const tc_buffer* mean, const tc_buffer* rstd,
                                  tc_buffer* dX,
                                  int N, int D);
```

### RoPE

```c
tc_status_t tc_rope_forward(tc_context* ctx,
                            tc_buffer*       X,           /* [B, H, S, D] in-place */
                            const tc_buffer* cos_t,       /* [S, D/2] fp32          */
                            const tc_buffer* sin_t,       /* [S, D/2] fp32          */
                            int batch, int heads, int seq, int head_dim);
```

### SwiGLU

```c
tc_status_t tc_swiglu_forward (tc_context* ctx,
                               const tc_buffer* gate,
                               const tc_buffer* up,
                               tc_buffer*       out,
                               int n);

tc_status_t tc_swiglu_backward(tc_context* ctx,
                               const tc_buffer* gate,
                               const tc_buffer* up,
                               const tc_buffer* dout,
                               tc_buffer*       dgate,
                               tc_buffer*       dup,
                               int n);
```

### Softmax

```c
tc_status_t tc_softmax_forward (tc_context* ctx,
                                const tc_buffer* X, tc_buffer* Y,
                                int N, int D);

tc_status_t tc_softmax_backward(tc_context* ctx,
                                const tc_buffer* Y, const tc_buffer* dY,
                                tc_buffer* dX,
                                int N, int D);
```

### Fused AdamW

```c
tc_status_t tc_adamw_step(tc_context* ctx,
                          tc_buffer*       params_fp32,  /* in/out */
                          tc_buffer*       m_fp32,       /* in/out */
                          tc_buffer*       v_fp32,       /* in/out */
                          const tc_buffer* grads,
                          tc_dtype_t       grad_dtype,   /* F16 or F32     */
                          int n,
                          float lr, float beta1, float beta2, float eps,
                          float wd, float bc1, float bc2);
```

`bc1`, `bc2` are bias corrections (`1 - beta^t`) precomputed on the host.

### Fused RMSnorm + GEMV

```c
tc_status_t tc_fused_rmsnorm_gemv(tc_context* ctx,
                                  const tc_buffer* X,      /* [M, K] fp16 */
                                  const tc_buffer* gamma,  /* [K]    fp16 */
                                  const tc_buffer* W,      /* [K, N] fp16 */
                                  tc_buffer*       Y,      /* [M, N] fp16 */
                                  int M, int N, int K, float eps);
```

Inference primitive (M ≤ 4 is the design target). Eliminates the
intermediate write-back of the normalized vector. See
[training_kernels.md](training_kernels.md) for the kernel design.

## Conv2D — `conv.h`

```c
tc_status_t tc_conv2d_forward(tc_context* ctx,
                              const tc_buffer* X,        /* [N, IC, H, W]    fp16 */
                              const tc_buffer* weight,   /* [OC, IC, kH, kW] fp16 */
                              const tc_buffer* bias,     /* [OC] or NULL          */
                              tc_buffer*       Y,        /* [N, OC, oH, oW]  fp16 */
                              tc_buffer*       scratch_col,
                              int batch, int in_channels, int out_channels,
                              int H, int W, int kH, int kW,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int out_H, int out_W);

tc_status_t tc_conv2d_backward_input (tc_context* ctx,
                                      const tc_buffer* dY,
                                      const tc_buffer* weight,
                                      tc_buffer*       dX,
                                      tc_buffer*       scratch_col,
                                      tc_buffer*       scratch_dX_f32,
                                      /* same shape params as forward */ ...);

tc_status_t tc_conv2d_backward_weight(tc_context* ctx,
                                      const tc_buffer* X,
                                      const tc_buffer* dY,
                                      tc_buffer*       dW,
                                      tc_buffer*       scratch_col,
                                      /* same shape params as forward */ ...);
```

Strategy: `im2col` → `tc_gemm` → (forward outputs directly to Y;
backward-input uses `col2im` with fp32 atomic accumulation). See
[conv2d.md](conv2d.md) for scratch sizing.

## Quantized — `quantized.h`

```c
typedef enum {
    TC_QUANT_Q4_0 = 0,        /* 32 weights/block, fp16 scale, 4-bit GGML  */
    TC_QUANT_Q8_0 = 1,        /* 32 weights/block, fp16 scale, int8        */
} tc_quant_t;

tc_status_t tc_quantize_weights    (tc_context* ctx,
                                    const tc_buffer* W_fp16,
                                    tc_buffer*       W_quant,
                                    tc_quant_t       fmt,
                                    int N, int K);

tc_status_t tc_gemv_quantized      (tc_context* ctx,
                                    const tc_buffer* X,
                                    const tc_buffer* W_quant,
                                    tc_buffer*       Y,
                                    tc_quant_t       fmt,
                                    int M, int N, int K);

tc_status_t tc_gemv_quantized_async(tc_context* ctx,
                                    const tc_buffer* X,
                                    const tc_buffer* W_quant,
                                    tc_buffer*       Y,
                                    tc_quant_t       fmt,
                                    int M, int N, int K,
                                    tc_stream*       stream);

size_t      tc_quantized_size      (tc_quant_t fmt, int N, int K);
```

`Y[M, N] = X[M, K] @ W^T` where `W` is `[N, K]` quantized. The kernel is
tuned for M ≤ 4 (the LLM-inference path); larger M routes through
dequant + `tc_gemm` in a future pass. See [quantized.md](quantized.md).

## GGUF — `gguf.h`

### Types

```c
typedef enum {
    TC_GGUF_TYPE_F32  = 0,
    TC_GGUF_TYPE_F16  = 1,
    TC_GGUF_TYPE_Q4_0 = 2,
    TC_GGUF_TYPE_Q4_1 = 3,
    TC_GGUF_TYPE_Q8_0 = 8,
    TC_GGUF_TYPE_BF16 = 30,
    TC_GGUF_TYPE_UNSUPPORTED = -1,
} tc_gguf_type_t;

typedef struct tc_gguf_file        tc_gguf_file;
typedef struct tc_gguf_loaded_model tc_gguf_loaded_model;

typedef struct {
    const char*   name;
    int32_t       n_dims;
    uint64_t      dims[4];
    tc_gguf_type_t type;
    uint64_t      offset;     /* into tensor data region                 */
    size_t        n_bytes;
    const void*   data;       /* mmap pointer                            */
} tc_gguf_tensor_info;

typedef struct {
    const char*   name;
    int32_t       n_dims;
    uint64_t      dims[4];
    tc_gguf_type_t type;
    uint64_t      offset;
    size_t        n_bytes;
    tc_buffer*    buffer;     /* owned by the loaded model               */
} tc_gguf_loaded_tensor_info;

typedef struct {
    int64_t context_length;
    int64_t embedding_length;
    int64_t feed_forward_length;
    int64_t block_count;
    int64_t attention_head_count;
    int64_t attention_head_count_kv;
    int64_t rope_dimension_count;
    int64_t vocab_size;
    double  rms_norm_epsilon;
    double  rope_freq_base;
    double  rope_freq_scale;
} tc_gguf_llama_config;

typedef struct {
    int            N;
    int            K;
    tc_gguf_type_t gguf_type;
    tc_quant_t     quant_type;
    size_t         n_bytes;
    tc_buffer*     buffer;
} tc_gguf_quantized_matrix_info;
```

### File and tensor access

```c
tc_status_t tc_gguf_open    (const char* path, tc_gguf_file** out);
void        tc_gguf_close   (tc_gguf_file* f);

uint64_t    tc_gguf_tensor_count  (const tc_gguf_file* f);
uint64_t    tc_gguf_metadata_count(const tc_gguf_file* f);

tc_status_t tc_gguf_get_tensor(const tc_gguf_file* f, const char* name,
                               tc_gguf_tensor_info* out);
tc_status_t tc_gguf_tensor_at (const tc_gguf_file* f, uint64_t i,
                               tc_gguf_tensor_info* out);
```

### Metadata helpers

```c
const char* tc_gguf_meta_get_str(const tc_gguf_file* f, const char* key);
int64_t     tc_gguf_meta_get_i64(const tc_gguf_file* f, const char* key,
                                 int64_t def);
double      tc_gguf_meta_get_f64(const tc_gguf_file* f, const char* key,
                                 double  def);

uint64_t    tc_gguf_meta_array_count   (const tc_gguf_file* f, const char* key);
tc_status_t tc_gguf_meta_array_get_str (const tc_gguf_file* f, const char* key,
                                        uint64_t index,
                                        const char** out_ptr, size_t* out_len);
int64_t     tc_gguf_meta_array_get_i64 (const tc_gguf_file* f, const char* key,
                                        uint64_t index, int64_t def);
double      tc_gguf_meta_array_get_f64 (const tc_gguf_file* f, const char* key,
                                        uint64_t index, double  def);

tc_status_t tc_gguf_get_llama_config(const tc_gguf_file* f,
                                     tc_gguf_llama_config* out);
```

### Bulk copy and matrix info

```c
tc_status_t tc_gguf_tensor_to_buffer(tc_context* ctx,
                                     const tc_gguf_file* f,
                                     const char* name,
                                     tc_buffer** out_buffer);

tc_status_t tc_gguf_tensor_quantized_matrix_info(
    const tc_gguf_tensor_info*    tensor,
    tc_gguf_quantized_matrix_info* out);

tc_status_t tc_gguf_loaded_tensor_quantized_matrix_info(
    const tc_gguf_loaded_tensor_info* tensor,
    tc_gguf_quantized_matrix_info*    out);

tc_status_t tc_gguf_load_supported_tensors(tc_context* ctx,
                                           const tc_gguf_file* f,
                                           tc_gguf_loaded_model** out_model);
void        tc_gguf_loaded_model_free     (tc_context* ctx,
                                           tc_gguf_loaded_model* model);

uint64_t    tc_gguf_loaded_tensor_count        (const tc_gguf_loaded_model* m);
uint64_t    tc_gguf_loaded_skipped_tensor_count(const tc_gguf_loaded_model* m);

tc_status_t tc_gguf_loaded_tensor_at (const tc_gguf_loaded_model* m, uint64_t i,
                                      tc_gguf_loaded_tensor_info* out);
tc_status_t tc_gguf_loaded_get_tensor(const tc_gguf_loaded_model* m,
                                      const char* name,
                                      tc_gguf_loaded_tensor_info* out);
```

See [gguf.md](gguf.md) for the loading patterns and skip semantics.

## Distributed — `distributed.h`

### Types

```c
typedef enum {
    TC_DIST_SINGLE = 0,    /* no-op all-reduce; world_size=1 always succeeds */
    TC_DIST_RING   = 1,    /* TB5 ring (v0.5)                                */
    TC_DIST_GLOO   = 2,    /* CPU/Ethernet (v0.5)                            */
} tc_dist_backend_t;

typedef enum {
    TC_REDUCE_SUM = 0,
    TC_REDUCE_AVG = 1,
    TC_REDUCE_MAX = 2,
    TC_REDUCE_MIN = 3,
} tc_reduce_op_t;

typedef struct tc_dist_ctx tc_dist_ctx;
```

### Calls

```c
tc_status_t tc_dist_init   (tc_context* tc,
                            tc_dist_backend_t backend,
                            int world_size, int rank,
                            const char* rendezvous_url,
                            tc_dist_ctx** out);

tc_status_t tc_dist_finalize(tc_dist_ctx* d);

int tc_dist_world_size(const tc_dist_ctx* d);
int tc_dist_rank      (const tc_dist_ctx* d);

tc_status_t tc_allreduce(tc_dist_ctx* d,
                         tc_buffer*    buf,
                         size_t        num_elements,
                         tc_dtype_t    dtype,
                         tc_reduce_op_t op);

tc_status_t tc_broadcast(tc_dist_ctx* d,
                         tc_buffer*    buf,
                         size_t        num_elements,
                         tc_dtype_t    dtype,
                         int           root);

tc_status_t tc_allgather(tc_dist_ctx*    d,
                         const tc_buffer* in,
                         tc_buffer*       out,
                         size_t           num_elements_per_rank,
                         tc_dtype_t       dtype);

tc_status_t tc_barrier  (tc_dist_ctx* d);
```

See [distributed.md](distributed.md) for the single-host ring (threads
and fork transports) and the v0.5 TB5/Gloo plan.

## DiLoCo — `diloco.h`

DiLoCo is layered above an existing `tc_dist_ctx`. The current runtime
implements the single-rank/local outer-step path; multi-rank WAN
transport and compressed sparse all-reduce return explicit unsupported
statuses until the distributed substrate lands.

```c
typedef struct tc_diloco_ctx tc_diloco_ctx;

typedef enum {
    TC_DILOCO_COMPRESS_NONE = 0,
    TC_DILOCO_COMPRESS_FP16 = 1,
    TC_DILOCO_COMPRESS_FP8 = 2,
    TC_DILOCO_COMPRESS_TOPK_1PCT = 3,
    TC_DILOCO_COMPRESS_TOPK_01PCT = 4,
    TC_DILOCO_COMPRESS_LOWRANK = 5,
    TC_DILOCO_COMPRESS_SIGNSGD = 6,
} tc_diloco_compress_t;

typedef enum {
    TC_DILOCO_OUTER_SGD = 0,
    TC_DILOCO_OUTER_NESTEROV = 1,
    TC_DILOCO_OUTER_ADAM = 2,
} tc_diloco_outer_optimizer_t;

typedef struct {
    int inner_steps;
    float outer_lr;
    float outer_momentum;
    float outer_beta2;
    float outer_eps;
    tc_diloco_outer_optimizer_t outer_optimizer;
    tc_diloco_compress_t compress;
    bool async_overlap;
    bool tolerate_dropouts;
} tc_diloco_config;

tc_status_t tc_diloco_init(tc_dist_ctx* dist_ctx,
                           const tc_diloco_config* cfg,
                           tc_diloco_ctx** out);
tc_status_t tc_diloco_finalize(tc_diloco_ctx* d);
tc_status_t tc_diloco_add_parameter(tc_diloco_ctx* d,
                                    const char* name,
                                    tc_buffer* theta_local,
                                    size_t num_elements,
                                    tc_dtype_t dtype);
tc_status_t tc_diloco_step(tc_diloco_ctx* d,
                           bool* out_outer_step_pending);
tc_status_t tc_diloco_apply_outer(tc_diloco_ctx* d);

uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d);
uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d);
double tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d);
double tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d);
```

See [diloco.md](diloco.md) for the algorithm, topology model, and staged
transport work.

## HIP — `hip.h`

HIP/chipStar is the staged non-Apple GPU backend. The public symbols are
exported today as deterministic unsupported stubs so SDK consumers and FFI
generators can bind the future surface.

```c
typedef enum {
    TC_HIP_VENDOR_UNKNOWN = 0,
    TC_HIP_VENDOR_INTEL = 1,
    TC_HIP_VENDOR_NVIDIA = 2,
    TC_HIP_VENDOR_AMD = 3,
    TC_HIP_VENDOR_ARM_MALI = 4,
} tc_hip_vendor_t;

typedef struct {
    tc_hip_vendor_t vendor;
    char device_name[128];
    char driver_version[64];
    char opencl_version[64];
    uint64_t global_memory_bytes;
    uint64_t local_memory_bytes;
    uint32_t compute_units;
    uint32_t max_workgroup_size;
    uint32_t preferred_subgroup_size;
    bool supports_fp16;
    bool supports_fp64;
    bool supports_int8_dot;
    bool unified_memory;
} tc_hip_device_info;

tc_status_t tc_hip_init(tc_context* ctx);
tc_status_t tc_hip_device_info_get(tc_context* ctx,
                                   tc_hip_device_info* out_info);
int tc_hip_device_count(void);
tc_status_t tc_hip_device_at(int index, tc_hip_device_info* out_info);
tc_status_t tc_hip_select_device(tc_context* ctx, int index);
const char* tc_hip_last_kernel_name(void);
```

See [../lib/hip/README.md](../lib/hip/README.md) for the chipStar porting
plan.

## Notes for ABI consumers

- All entry points are `extern "C"`. C++ headers reverse-include cleanly.
- All opaque handles are pointer-stable for the lifetime of the context
  (or the loaded model, for GGUF).
- `tc_buffer_map` returns the same `void*` across calls for the same
  buffer — feel free to cache it.
- The library is thread-safe for distinct contexts; a single context may
  be used from multiple threads provided callers serialize access to the
  same `tc_buffer`. The pipeline cache and the buffer pool are internally
  guarded.
- All structs are passed by `const` pointer; do not retain pointers into
  descriptor structs beyond the call.
- ALiBi: `alibi_slopes` is read at the dispatch call and copied via
  `setBytes:`; the host array does not need to outlive the call.
