/*
 * tensorcore - Apple Silicon AMX backend for fp32 GEMM.
 *
 * AMX (Apple Matrix eXtension) is the matrix coprocessor that gives
 * Accelerate / vDSP_MMul / cblas_sgemm their ~10x throughput advantage over
 * NEON for GEMM on Apple Silicon. The ISA is **not publicly documented**.
 * This file uses encodings reverse-engineered by Dougall Johnson and Peter
 * Cawley (@corsix) - see https://github.com/corsix/amx for the canonical
 * reference. We emit raw instruction words via `.word` because Clang's
 * assembler does not natively recognize AMX mnemonics.
 *
 * Risk: Apple has never sanctioned third-party AMX use. The hardware has
 * remained stable across macOS 12/13/14/15 + Apple7/Apple8/Apple9/Apple10
 * silicon, but Apple can change semantics at any point. This backend is
 * opt-in via `TC_USE_AMX_GEMM=1`; the default fp32 path remains NEON ->
 * CBLAS -> reference. Production deployments wanting peak Apple-Silicon
 * throughput should still prefer Accelerate's cblas_sgemm, which Apple
 * keeps in sync with new silicon.
 *
 * Scope of this file (v0.2 - session 1.5):
 *   - fp32 only; one AMX worker for small shapes, two persistent pthread
 *     workers for M >= 256 unless TC_AMX_THREADS=1 is set.
 *   - K-tiled mega-pack inside the (i, j) loop with KC=256 -> ~0.37 TFLOPS
 *     at 4096^3 on M2 Ultra before multi-worker experiments (vs 0.08 TFLOPS
 *     for the single-thread NEON kernel).
 *   - alpha = 1, beta = 0 only (other alpha/beta combos fall through to NEON/CBLAS).
 *   - Non-transposed A, B only (transposed falls through).
 *   - M and N must be multiples of 16 (the AMX tile size for fp32 outer
 *     product); other sizes fall through.
 *   - Apple Silicon only (__APPLE__ && __aarch64__).
 *
 * Roadmap for sessions 2+:
 *   - **Multi-cluster scheduling that scales past two workers.** A naive
 *     pthread + thread_affinity_policy_data split regressed perf (workers
 *     either co-located on one cluster and serialized on its AMX unit, or
 *     split across clusters but paid heavy UltraFusion-fabric reads for the
 *     shared mega-pack). The current persistent-worker path uses
 *     per-worker local packs; broader worker counts still need
 *     topology-aware scheduling.
 *   - **fp16 / bf16 paths** via the corresponding AMX opcodes (FMA16,
 *     MAC16 with the proper accumulator-precision flags).
 *   - **Edge tiles** for M, N not divisible by 16 - pack with zero
 *     padding into 16-tile buffers.
 *   - **alpha != 1 / beta != 0** support - currently we always overwrite C; full
 *     `C = alphaAB + betaC` requires loading C into Z accumulators first.
 *   - **ISA version dispatch.** AMX1 (M1) vs AMX2 (M2, A14+) vs AMX3
 *     (M3, A16+) have subtle encoding differences for newer ops. fp32 FMA
 *     is stable across all generations; fp16/bf16 paths diverge.
 *   - **Memory layout.** Investigate whether AMX prefers col-major A or
 *     row-major A for the LDX path on different silicon revisions.
 *
 * Reverse-engineering references this file relies on:
 *   - corsix/amx README - instruction layout, opcode table
 *   - Dougall Johnson "Apple's M1 matrix coprocessor" (2021)
 *   - Apple's published patent US20180074824A1 (general architecture)
 */

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#  define TC_AMX_GEMM_BUILD 1
#endif

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

#if defined(TC_AMX_GEMM_BUILD)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <pthread.h>

#include <atomic>
#include <dlfcn.h>

/* Private libsystem_pthread hook: asks the kernel to schedule the calling
 * thread on the cluster OTHER than the one its parent / siblings are on.
 * Apple's Accelerate uses this to spread its two AMX-using threads across
 * the two clusters of M-series Ultra chips, doubling AMX throughput.
 *
 * The symbol lives in /usr/lib/system/libsystem_pthread.dylib at the same
 * address as `_pthread_prefer_alternate_amx_self` — they're aliases of one
 * function that tail-calls `__pthread_set_properties_self(0x20, 0, 0)`.
 *
 * Not declared in any public header, and not in the Mach-O export table of
 * libSystem so static linking fails to resolve it. We fetch via dlsym at
 * runtime — null pointer is treated as a graceful no-op (older OS, or some
 * stripped variant). */
using pthread_prefer_alternate_cluster_self_fn = void (*)(void);
static pthread_prefer_alternate_cluster_self_fn load_prefer_alternate_cluster() {
    /* Note: pass the C symbol name without the leading underscore. dlsym
     * prepends one when searching the Mach-O symbol table — passing
     * "_pthread_..." here would resolve to "__pthread_..." which doesn't
     * exist. Verified the unprefixed lookup returns a non-null address on
     * macOS 15.1. */
    static pthread_prefer_alternate_cluster_self_fn fn = []() {
        return reinterpret_cast<pthread_prefer_alternate_cluster_self_fn>(
            dlsym(RTLD_DEFAULT, "pthread_prefer_alternate_cluster_self"));
    }();
    return fn;
}

/* Public os_workgroup API. Joining a workgroup gives the scheduler a
 * topology hint: members of the same workgroup are treated as a coordinated
 * compute group and given stronger P-cluster stickiness than QoS alone
 * provides. Combined with `pthread_prefer_alternate_cluster_self` on one
 * member, this is the recipe Accelerate uses to keep its two AMX workers
 * locked to two distinct P-clusters.
 *
 * The workload_id is opaque to us but registered with the kernel's scheduler
 * policy table. Apple ships specific IDs for audio / video / etc.;
 * unrecognized IDs degrade to a generic "real-time compute group" hint. */
struct alignas(8) tc_os_workgroup_join_token_buf { unsigned char buf[64]; };
using tc_os_workgroup_t = void*;
using os_workgroup_create_with_workload_id_fn =
    tc_os_workgroup_t (*)(const char* name, const char* workload_id, void* attr);
using os_workgroup_join_fn = int (*)(tc_os_workgroup_t wg, void* token);
using os_workgroup_leave_fn = void (*)(tc_os_workgroup_t wg, void* token);

struct workgroup_api {
    os_workgroup_create_with_workload_id_fn create;
    os_workgroup_join_fn join;
    os_workgroup_leave_fn leave;
    tc_os_workgroup_t shared_wg;
};
static workgroup_api& workgroup_api_instance() {
    static workgroup_api api = []() {
        workgroup_api a = {};
        a.create = reinterpret_cast<os_workgroup_create_with_workload_id_fn>(
            dlsym(RTLD_DEFAULT, "os_workgroup_create_with_workload_id"));
        a.join = reinterpret_cast<os_workgroup_join_fn>(
            dlsym(RTLD_DEFAULT, "os_workgroup_join"));
        a.leave = reinterpret_cast<os_workgroup_leave_fn>(
            dlsym(RTLD_DEFAULT, "os_workgroup_leave"));
        if (a.create) {
            /* Try AMX-flavored workload IDs first, fall back to a generic
             * matrix-compute label. The kernel accepts any non-null ID;
             * unknown IDs get a generic compute-group policy. */
            const char* ids[] = {"com.apple.amx", "com.apple.compute.matrix",
                                 "com.apple.workgroup.compute"};
            for (const char* id : ids) {
                a.shared_wg = a.create("tensorcore.amx", id, nullptr);
                if (a.shared_wg) break;
            }
        }
        return a;
    }();
    return api;
}

/* ----------------------------------------------------------------------------
 * AMX instruction encoding
 *
 * All AMX instructions take the 32-bit form:
 *   0x00201000 | (op << 5) | reg_idx
 * where `op` is the AMX opcode (0..23 currently used) and `reg_idx` is the
 * AArch64 GPR (0..31) holding the AMX operand. For ops that don't take a
 * runtime operand (SET, CLR), reg_idx is unused but still part of the encoding.
 *
 * We pin all AMX operands into x10 via the `register __asm__("x10")` clobber
 * pattern. This lets us emit fully-resolved hex instruction words at compile
 * time without runtime instruction-word patching.
 *
 * Hex encodings used here:
 *   SET   = 0x00201000 | (17 << 5) | 0  = 0x00201220
 *   CLR   = 0x00201000 | (18 << 5) | 0  = 0x00201240
 *   LDX   = 0x00201000 | (0  << 5) | 10 = 0x0020100A
 *   LDY   = 0x00201000 | (1  << 5) | 10 = 0x0020102A
 *   LDZ   = 0x00201000 | (4  << 5) | 10 = 0x0020108A
 *   STZ   = 0x00201000 | (5  << 5) | 10 = 0x002010AA
 *   FMA32 = 0x00201000 | (12 << 5) | 10 = 0x0020118A
 * --------------------------------------------------------------------------*/

namespace {

#define AMX_NOP(opcode_word) \
    __asm__ volatile(".word " #opcode_word ::: "memory")

#define AMX_GPR(opcode_word, val) do { \
    register uint64_t _amx_v __asm__("x10") = (val); \
    __asm__ volatile(".word " #opcode_word \
                     : "+r"(_amx_v) :: "memory"); \
} while (0)

#define AMX_SET()       AMX_NOP(0x00201220)
#define AMX_CLR()       AMX_NOP(0x00201240)
#define AMX_LDX(val)    AMX_GPR(0x0020100A, val)
#define AMX_LDY(val)    AMX_GPR(0x0020102A, val)
#define AMX_LDZ(val)    AMX_GPR(0x0020108A, val)
#define AMX_STZ(val)    AMX_GPR(0x002010AA, val)
#define AMX_FMA32(val)  AMX_GPR(0x0020118A, val)

/* Operand encodings for memory ops:
 *   LDX / LDY:
 *     bits[55: 0]  virtual address (64-byte aligned)
 *     bits[58:56]  X/Y register slot (0..7)
 *     bit [62]     pair load (loads 128 bytes into slot N and N+1)
 *   LDZ / STZ:
 *     bits[55: 0]  virtual address
 *     bits[61:56]  Z register slot (0..63)
 *   FMA32:
 *     bits[5:0]    x_offset (bytes within X register)
 *     bits[10:6]   y_offset (bytes within Y register)
 *     bits[20:16]  z_row_start (0..15 for fp32; Z has 64 rows of 64 bytes
 *                  but a 16x16 fp32 outer product writes into 16 consecutive)
 *     bit [27]     1 = vector mode (broadcast X), 0 = matrix mode
 *     bit [29]     1 = vector mode (broadcast Y), 0 = matrix mode
 *   For a full 16x16 outer product Z[0..15] = X[0] outer Y[0], operand = 0. */

inline uint64_t amx_xy_op(const void* addr, int slot) {
    return (uint64_t)(uintptr_t)addr | ((uint64_t)slot << 56);
}

inline uint64_t amx_z_op(const void* addr, int slot) {
    return (uint64_t)(uintptr_t)addr | ((uint64_t)slot << 56);
}

/* ----------------------------------------------------------------------------
 * 16 x 16 x K fp32 AMX inner kernel.
 *
 * Inputs:
 *   A_pack: [K x 16] fp32, layout "A_pack[k*16 + m] = A[m, k]" (A's k-th
 *           column laid out contiguously for the 16-row panel). 64-byte
 *           aligned. Each K row = one A column = goes into Y.
 *   B_pack: [K x 16] fp32, layout "B_pack[k*16 + n] = B[k, n]" (B's k-th
 *           row laid out contiguously for the 16-col panel). 64-byte
 *           aligned. Each K row = one B row = goes into X.
 *   C_out:  [16 x 16] fp32 row-major output. 64-byte aligned.
 *
 * Computes:  C_out[m, n] = sum_k A[m, k] * B[k, n]   for 0 <= m, n < 16
 *
 * Why X = B-row and Y = A-column:
 *   AMX FMA32 with default operand computes Z[i, j] += X[j] * Y[i] (the
 *   convention has X varying across *columns* of Z, Y across *rows*). For
 *   C[m, n] = sum_k A[m, k] * B[k, n] we map m -> i, n -> j, so Y[i] = A[m, k]
 *   (the k-th A column, 16 rows) and X[j] = B[k, n] (the k-th B row, 16
 *   cols). Z[i] becomes a row of C and is stored via STZ. Getting this
 *   wrong silently transposes the result - verified bit-exact at 16x16x16
 *   against a scalar reference.
 *
 * Algorithm:
 *   AMX_SET
 *   zero Z[0..15]  (16 rows of 64 bytes = 16 x 16 fp32 accumulator)
 *   for k in [0, K):
 *     LDX  B_pack[k*16..k*16+15] into X[0]      (B's k-th row -> X)
 *     LDY  A_pack[k*16..k*16+15] into Y[0]      (A's k-th col -> Y)
 *     FMA32 0                                    (Z[i, j] += X[j] * Y[i])
 *   for r in [0, 16):  STZ C_out + r*16, Z[r]   (write 16 rows back)
 *   AMX_CLR
 *
 * The AMX unit retires one FMA32 per cycle = 256 fp32 FMA = 512 fp32 ops/cycle.
 * At ~3.2 GHz cluster frequency that's ~1.6 TFLOPS per cluster. M2 Ultra has
 * two clusters -> ~3.2 TFLOPS aggregate ceiling for fp32 GEMM (matches what
 * Accelerate measures). The small-shape path exercises one cluster; the
 * large-shape persistent-worker path attempts to use both clusters with worker-local
 * packing. */
/* The AMX Z register file is partitioned into 4 banks of 16 slots each
 * (slots 0..15, 16..31, 32..47, 48..63). For fp32 outer products, row r of
 * the result lands at Z slot r * 4 - so 16 result rows occupy positions
 * 0, 4, 8, ..., 60 (one row per bank). Indexing slots 0..15 silently mixes
 * up the rows; verified at 16x16x16 vs scalar reference.
 *
 * fp64 outer product uses every 8th slot, fp16 every 2nd. */
static constexpr int kZStrideF32 = 4;

inline void amx_zero_z_fp32() {
    alignas(64) static const float zero[16] = {0};
    for (int r = 0; r < 16; ++r) {
        AMX_LDZ(amx_z_op(zero, r * kZStrideF32));
    }
}

inline void amx_store_z_fp32(float* C, int ldc) {
    for (int r = 0; r < 16; ++r) {
        AMX_STZ(amx_z_op(C + r * ldc, r * kZStrideF32));
    }
}

/* Pure FMA loop - accumulates kc steps into the live Z accumulator without
 * zeroing or storing. Caller is responsible for AMX_SET, zeroing Z before
 * the first call for a given (i, j) tile, and store_z_fp32 after the last
 * call. Splitting the kernel this way lets us walk K in cache-friendly KC
 * chunks while keeping Z live across the inner loop. */
__attribute__((noinline))
static void amx_fma32_accumulate(int kc,
                                 const float* __restrict A_pack,
                                 const float* __restrict B_pack) {
    for (int k = 0; k < kc; ++k) {
        AMX_LDX(amx_xy_op(B_pack + k * 16, /*slot=*/ 0));   /* B row -> X */
        AMX_LDY(amx_xy_op(A_pack + k * 16, /*slot=*/ 0));   /* A col -> Y */
        AMX_FMA32(0);
    }
}

/* ----------------------------------------------------------------------------
 * Pack functions
 *
 * For the AMX kernel, A and B must each appear as 16-fp32-wide panels
 * (64-byte aligned) with K-stride 16 elements. The caller's A and B are
 * arbitrary row-major matrices with strides lda and ldb. We pack a 16-row
 * panel of A (transposed during pack - A's K column becomes the row of
 * A_pack) and a 16-col panel of B (no transpose needed).
 * --------------------------------------------------------------------------*/

/* Pack the full MxK matrix A into M/16 panels of Kx16 fp32 layout.
 *
 *   A_pack_mega[ (i/16) * K * 16 + k * 16 + m ] = A[i + m, k]
 *
 * After this single pass, the inner GEMM loop can iterate over (i, j, kp)
 * and pull contiguous K-tile-sized panels straight from the mega buffer,
 * with no further re-packing. Pack cost amortizes over N/16 tile columns.
 *
 * Strided reads (A's stride is lda over rows) are the bandwidth-limiting
 * step here - typical 10-20 GB/s single-thread on Apple Silicon. */
inline void pack_A_mega(const float* A, int lda, int M, int K,
                        float* A_pack_mega) {
    for (int i = 0; i < M; i += 16) {
        float* panel = A_pack_mega + (size_t)(i / 16) * K * 16;
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < 16; ++m) {
                panel[(size_t)k * 16 + m] = A[(size_t)(i + m) * lda + k];
            }
        }
    }
}

/* Pack the full KxN matrix B into N/16 panels of Kx16 fp32 layout.
 *
 *   B_pack_mega[ (j/16) * K * 16 + k * 16 + n ] = B[k, j + n]
 *
 * For non-transposed B, each "K row" of a panel is just 16 contiguous
 * fp32 from B's k-th row - a clean 64-byte memcpy with good prefetch.
 * Pack cost is dominated by linear sequential reads, ~25-50 GB/s. */
inline void pack_B_mega(const float* B, int ldb, int K, int N,
                        float* B_pack_mega) {
    for (int j = 0; j < N; j += 16) {
        float* panel = B_pack_mega + (size_t)(j / 16) * K * 16;
        for (int k = 0; k < K; ++k) {
            const float* src = B + (size_t)k * ldb + j;
            std::memcpy(panel + (size_t)k * 16, src, 16 * sizeof(float));
        }
    }
}

/* ----------------------------------------------------------------------------
 * Aligned heap allocator. AMX LDX/LDY/LDZ/STZ require 64-byte alignment.
 * --------------------------------------------------------------------------*/
inline float* aligned_alloc_fp32(size_t n_floats) {
    void* p = nullptr;
    const size_t bytes = ((n_floats * sizeof(float) + 63) / 64) * 64;
#if defined(__APPLE__) || defined(_GNU_SOURCE)
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
#else
    p = std::aligned_alloc(64, bytes);
#endif
    return static_cast<float*>(p);
}

static void amx_process_tile_strip(int i_start, int i_end,
                                   int N, int K, int ldc,
                                   const float* A_pack_mega,
                                   const float* B_pack_mega,
                                   float* C);

/* Worker-local pack of A's i-strip and the full B, then process its tile
 * strip. Cluster-local: the pack writes go into whichever cluster the
 * worker landed on, so the AMX kernel's LDX/LDY reads stay local. Halves
 * the inter-cluster traffic at large M compared to a shared mega-pack. */
static bool amx_worker_local(int i_start, int i_end,
                             int N, int K, int lda, int ldb, int ldc,
                             const float* A, const float* B, float* C) {
    static thread_local float* tls_A_pack = nullptr;
    static thread_local size_t tls_A_pack_cap = 0;
    static thread_local float* tls_B_pack = nullptr;
    static thread_local size_t tls_B_pack_cap = 0;

    const int M_strip = i_end - i_start;
    const size_t A_pack_needed = (size_t)M_strip * K;
    const size_t B_pack_needed = (size_t)K * N;
    if (tls_A_pack_cap < A_pack_needed) {
        std::free(tls_A_pack);
        tls_A_pack = aligned_alloc_fp32(A_pack_needed);
        tls_A_pack_cap = A_pack_needed;
    }
    if (tls_B_pack_cap < B_pack_needed) {
        std::free(tls_B_pack);
        tls_B_pack = aligned_alloc_fp32(B_pack_needed);
        tls_B_pack_cap = B_pack_needed;
    }
    if (!tls_A_pack || !tls_B_pack) return false;

    /* Pack the worker's A strip (M_strip rows starting at i_start) and the
     * full B. Each worker writes pack output cluster-locally. */
    pack_A_mega(A + (size_t)i_start * lda, lda, M_strip, K, tls_A_pack);
    pack_B_mega(B, ldb, K, N, tls_B_pack);

    /* Process tiles into C at the absolute (i_start + relative_i) offset.
     * amx_process_tile_strip walks [0, M_strip) inside the worker's local
     * pack, and writes into C base-offset by i_start*ldc. */
    amx_process_tile_strip(0, M_strip, N, K, ldc,
                           tls_A_pack, tls_B_pack,
                           C + (size_t)i_start * ldc);
    return true;
}

/* ----------------------------------------------------------------------------
 * Per-thread tile processor.
 *
 * Walks output tiles [i_start, i_end) × [0, N), reading from the shared
 * mega-packed A and B buffers. AMX state is armed by the caller; C tile
 * scratch is on the stack so each thread has its own.
 *
 * Z accumulator is live for the duration of a single (i, j) tile and walks
 * K in KC chunks. */
static void amx_process_tile_strip(int i_start, int i_end,
                                   int N, int K, int ldc,
                                   const float* A_pack_mega,
                                   const float* B_pack_mega,
                                   float* C) {
    /* Caller is responsible for AMX_SET. Doing it here would risk a
     * double-SET on workers (which arm on entry) — empirically that pattern
     * SIGILLs on macOS 15.1 / M2 Ultra at the second SET. */

    alignas(64) float C_tile[16 * 16];
    constexpr int KC = 256;

    for (int i = i_start; i < i_end; i += 16) {
        const float* A_panel = A_pack_mega + (size_t)(i / 16) * K * 16;
        for (int j = 0; j < N; j += 16) {
            const float* B_panel = B_pack_mega + (size_t)(j / 16) * K * 16;

            amx_zero_z_fp32();
            for (int kp = 0; kp < K; kp += KC) {
                const int kc = (kp + KC <= K) ? KC : (K - kp);
                amx_fma32_accumulate(kc, A_panel + (size_t)kp * 16,
                                          B_panel + (size_t)kp * 16);
            }
            amx_store_z_fp32(C_tile, 16);

            for (int mm = 0; mm < 16; ++mm) {
                std::memcpy(C + (size_t)(i + mm) * ldc + j,
                            C_tile + (size_t)mm * 16,
                            16 * sizeof(float));
            }
        }
    }
}

/* ----------------------------------------------------------------------------
 * Persistent AMX worker pool.
 *
 * Per-call GCD dispatch_apply creates fresh threads. macOS often places those
 * threads on E-cores (no AMX) even with USER_INTERACTIVE QoS. A persistent
 * pool of 2 long-lived pthreads, each pre-armed and held with
 * `pthread_prefer_alternate_cluster_self` on worker 1, lets the kernel
 * settle their cluster placement once and reuse the threads on every call.
 *
 * Synchronization: each worker waits on its own start-semaphore; the main
 * thread posts both, workers cover all M, and waits on the done-semaphores.
 * Worker queue depth = 1 (one outstanding job per worker), which is all we
 * need for a two-worker GEMM split. */
struct amx_work_unit {
    int i_start, i_end;
    int N, K, lda, ldb, ldc;
    const float* A;
    const float* B;
    float* C;
};

struct amx_worker_pool_t {
    pthread_t threads[2];
    dispatch_semaphore_t start[2];
    dispatch_semaphore_t done[2];
    amx_work_unit work[2];
    std::atomic<bool> shutdown{false};
    std::atomic<bool> ready{false};
    std::atomic<int> failed{0};
};

static amx_worker_pool_t g_pool;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_pool_dispatch_lock = PTHREAD_MUTEX_INITIALIZER;

extern "C" int pthread_set_qos_class_self_np(qos_class_t qc, int rel_prio);

static void* amx_worker_thread_entry(void* arg) {
    const int t = (int)(intptr_t)arg;

    /* Pin to a P-core; ask the kernel to push worker 1 to the alternate
     * cluster. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    /* Join the shared os_workgroup — provides P-cluster stickiness across
     * the worker's lifetime. Both workers join the same group; the
     * scheduler then sees them as a coordinated compute set. */
    tc_os_workgroup_join_token_buf join_token{};
    auto& wg_api = workgroup_api_instance();
    int workgroup_joined = 0;
    if (wg_api.shared_wg && wg_api.join &&
        wg_api.join(wg_api.shared_wg, &join_token) == 0) {
        workgroup_joined = 1;
    }

    if (t == 1) {
        auto prefer_alt = load_prefer_alternate_cluster();
        if (prefer_alt) prefer_alt();
    }

    /* AMX is per-thread state. Arm once for this persistent worker. Calling
     * SET again on an already-armed worker has SIGILL'd on some macOS 15.x
     * machines, so the tile processor intentionally never self-arms. */
    AMX_SET();

    while (!g_pool.shutdown.load(std::memory_order_acquire)) {
        dispatch_semaphore_wait(g_pool.start[t], DISPATCH_TIME_FOREVER);
        if (g_pool.shutdown.load(std::memory_order_acquire)) break;
        const amx_work_unit& w = g_pool.work[t];
        if (!amx_worker_local(w.i_start, w.i_end, w.N, w.K, w.lda, w.ldb, w.ldc,
                              w.A, w.B, w.C)) {
            g_pool.failed.store(1, std::memory_order_release);
        }
        dispatch_semaphore_signal(g_pool.done[t]);
    }
    if (workgroup_joined && wg_api.leave) {
        wg_api.leave(wg_api.shared_wg, &join_token);
    }
    return nullptr;
}

static void amx_pool_init_once() {
    g_pool.start[0] = dispatch_semaphore_create(0);
    g_pool.start[1] = dispatch_semaphore_create(0);
    g_pool.done[0] = dispatch_semaphore_create(0);
    g_pool.done[1] = dispatch_semaphore_create(0);
    if (!g_pool.start[0] || !g_pool.start[1] || !g_pool.done[0] || !g_pool.done[1]) {
        return;
    }

    int created = 0;
    for (int t = 0; t < 2; ++t) {
        if (pthread_create(&g_pool.threads[t], nullptr, amx_worker_thread_entry,
                           (void*)(intptr_t)t) != 0) {
            g_pool.shutdown.store(true, std::memory_order_release);
            for (int i = 0; i < created; ++i) {
                dispatch_semaphore_signal(g_pool.start[i]);
            }
            return;
        }
        pthread_detach(g_pool.threads[t]);
        ++created;
    }
    g_pool.ready.store(true, std::memory_order_release);
}

static bool amx_pool_dispatch_pair(const amx_work_unit& w0, const amx_work_unit& w1) {
    pthread_once(&g_pool_once, amx_pool_init_once);
    if (!g_pool.ready.load(std::memory_order_acquire)) return false;

    pthread_mutex_lock(&g_pool_dispatch_lock);
    g_pool.failed.store(0, std::memory_order_release);
    g_pool.work[0] = w0;
    g_pool.work[1] = w1;
    dispatch_semaphore_signal(g_pool.start[0]);
    dispatch_semaphore_signal(g_pool.start[1]);
    dispatch_semaphore_wait(g_pool.done[0], DISPATCH_TIME_FOREVER);
    dispatch_semaphore_wait(g_pool.done[1], DISPATCH_TIME_FOREVER);
    const bool ok = (g_pool.failed.load(std::memory_order_acquire) == 0);
    pthread_mutex_unlock(&g_pool_dispatch_lock);
    return ok;
}

}  // namespace

/* ----------------------------------------------------------------------------
 * Entry point
 *
 * Contract (this session):
 *   - returns 0 on success, -1 on (unsupported config | runtime failure)
 *   - supported: fp32, M%16==0, N%16==0, !transpose_a, !transpose_b,
 *                alpha==1, beta==0
 *   - other configs return -1 - caller (gemm_cpu.cpp) falls through to NEON
 *                                or CBLAS.
 * --------------------------------------------------------------------------*/
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda, int transpose_a,
                                                   const float* B, int ldb, int transpose_b,
                                                   float beta,
                                                   float* C, int ldc) {
    /* Session-1 capability gate. Sessions 2-3 will lift these. */
    if (alpha != 1.0f || beta != 0.0f) return -1;
    if (transpose_a || transpose_b) return -1;
    if ((M & 15) != 0 || (N & 15) != 0) return -1;

    /* Threading policy:
     *   - For M < 256: too small for parallel overhead to amortize. Pack
     *     once on the main thread (cluster-local for that thread) and run
     *     the single-thread strip processor.
     *   - For M >= 256: use the persistent two-thread pool. Each worker
     *     packs ITS OWN A strip + full B locally, so the AMX kernel reads
     *     stay in the worker's cluster (no UltraFusion-fabric round-trip on
     *     every K iter). Memory cost doubles vs shared pack (~2× M·K +
     *     2× K·N fp32) but pack throughput parallelizes too.
     *   - TC_AMX_THREADS=1 forces single-thread (for A/B measurement). */
    const char* threads_env = std::getenv("TC_AMX_THREADS");
    const bool single_thread = (threads_env && threads_env[0] == '1');
    const bool use_multi = !single_thread && M >= 256;

    if (use_multi) {
        /* Persistent pool: two long-lived pthreads, each USER_INTERACTIVE +
         * pre-armed for AMX, with worker 1 pushed to the alternate cluster
         * via the private hook. Across calls they stay warm — kernel learns
         * their P-cluster placement instead of re-deciding per dispatch. */
        const int strips_total = M / 16;
        const int strips_per_worker = strips_total / 2;
        amx_work_unit w0 = {0, strips_per_worker * 16,
                            N, K, lda, ldb, ldc, A, B, C};
        amx_work_unit w1 = {strips_per_worker * 16, M,
                            N, K, lda, ldb, ldc, A, B, C};
        if (!amx_pool_dispatch_pair(w0, w1)) return -1;
    } else {
        /* Mega pack buffers: pack ALL of A and B once each, then iterate
         * (i, j, kp) with no re-packing. Memory cost is M*K + K*N fp32
         * (e.g. 64 MB + 64 MB at 4096^3), but each (i, j, kp) iteration
         * only touches KC x 16 = 16 KB of packed data.
         *
         * The multi-worker path uses worker-local pack buffers instead, so
         * keep these main-thread buffers out of the large-shape path. */
        static thread_local float* tls_A_pack = nullptr;
        static thread_local size_t tls_A_pack_cap = 0;
        static thread_local float* tls_B_pack = nullptr;
        static thread_local size_t tls_B_pack_cap = 0;

        const size_t A_pack_needed = (size_t)M * K;
        const size_t B_pack_needed = (size_t)K * N;
        if (tls_A_pack_cap < A_pack_needed) {
            std::free(tls_A_pack);
            tls_A_pack = aligned_alloc_fp32(A_pack_needed);
            tls_A_pack_cap = A_pack_needed;
        }
        if (tls_B_pack_cap < B_pack_needed) {
            std::free(tls_B_pack);
            tls_B_pack = aligned_alloc_fp32(B_pack_needed);
            tls_B_pack_cap = B_pack_needed;
        }
        if (!tls_A_pack || !tls_B_pack) return -1;
        /* Arm AMX once for the calling thread. amx_process_tile_strip
         * no longer self-arms (workers handle their own to avoid the
         * double-SET trap), so callers must AMX_SET here. Subsequent calls
         * on the same thread skip SET through the thread-local guard. */
        static thread_local bool main_amx_armed = false;
        if (!main_amx_armed) {
            AMX_SET();
            main_amx_armed = true;
        }
        pack_A_mega(A, lda, M, K, tls_A_pack);
        pack_B_mega(B, ldb, K, N, tls_B_pack);
        amx_process_tile_strip(0, M, N, K, ldc, tls_A_pack, tls_B_pack, C);
    }

    /* Do not AMX_CLR - see comment in amx_process_tile_strip. */
    return 0;
}

extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32_available(void) {
    return 1;
}

#else  /* !TC_AMX_GEMM_BUILD */

#include <cstdint>

extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda, int transpose_a,
                                                   const float* B, int ldb, int transpose_b,
                                                   float beta,
                                                   float* C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha;
    (void)A; (void)lda; (void)transpose_a;
    (void)B; (void)ldb; (void)transpose_b;
    (void)beta; (void)C; (void)ldc;
    return -1;
}

extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32_available(void) {
    return 0;
}

#endif  /* TC_AMX_GEMM_BUILD */
