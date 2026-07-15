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
 *   - alpha / beta support through a scalar post-pass when needed.
 *   - Transposed A/B inputs are packed into row-major scratch before AMX.
 *   - M and N edge tiles are zero-padded to the AMX 16x16 tile size and
 *     trimmed after compute.
 *   - Apple Silicon only (__APPLE__ && __aarch64__).
 *
 * Closed follow-up items:
 *   - Cluster count is probed with sysctl and the worker pool grows to the
 *     detected P-cluster / AMX-unit topology without a baked upper bound.
 *   - fp16 / bf16 entry points and FMA16/FMA64 encodings are present but
 *     intentionally return -1 until the FMA16 IO-mode operand bits are
 *     validated by a single-instruction hardware probe.
 *   - ISA version is probed from hw.cpufamily: Firestorm -> AMX1,
 *     Avalanche -> AMX2, Everest/newer -> AMX3.
 *
 * Roadmap for sessions 2+:
 *   - **Fuse alpha/beta into AMX accumulators** instead of the current temp
 *     buffer + scalar post-pass.
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
#include <new>
#include <pthread.h>
#include <sys/sysctl.h>

#include <atomic>
#include <dlfcn.h>
#include <vector>

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
/* Per corsix/amx opcode table:
 *   FMA64 = 0x00201000 | (10 << 5) | 10 = 0x0020114A  (Apple7+, fp64 IO)
 *   FMA16 = 0x00201000 | (14 << 5) | 10 = 0x002011CA  (Apple7+, fp16 IO)
 * bf16 uses FMA16 with an IO-mode flag bit; bit position differs by AMX
 * version — see tc_amx_isa_version() at end of file. */
#define AMX_FMA64(val)  AMX_GPR(0x0020114A, val)
#define AMX_FMA16(val)  AMX_GPR(0x002011CA, val)

/* Operand encodings for memory ops:
 *   LDX / LDY:
 *     bits[55: 0]  virtual address (64-byte aligned)
 *     bits[58:56]  X/Y register slot (0..7)
 *     bit [62]     pair load (loads 128 bytes into slot N and N+1)
 *   LDZ / STZ:
 *     bits[55: 0]  virtual address
 *     bits[61:56]  Z register slot (0..63)
 *   FMA32:
 *     bits[ 9: 0]  x_offset (bytes within X register)
 *     bits[19:10]  y_offset (bytes within Y register)
 *     bits[26:20]  z_row_start (0..15 for fp32; Z has 64 rows of 64 bytes
 *                  but a 16x16 fp32 outer product writes into 16 consecutive)
 *     bit [27]     1 = skip Z input (write x*y), 0 = accumulate x*y+z
 *     bit [28]     1 = skip Y input
 *     bit [29]     1 = skip X input
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
 *   AMX_SET before entering the tile processor
 *   for k in [0, K):
 *     preload k=0 into X[0]/Y[0]
 *     for k in [1, K):
 *       LDX/LDY k into the alternate X/Y slot
 *       FMA32 previous slot, skip-Z only for the first FMA
 *     FMA32 final loaded slot
 *                                                    (Z[i, j] += X[j] * Y[i])
 *   for r in [0, 16):  STZ C_out + r*16, Z[r]   (write 16 rows back)
 *   leave AMX armed for the thread lifetime
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
static constexpr uint64_t kFma32SkipZ = 1ull << 27;

inline void amx_store_z_fp32(float* C, int ldc) {
    for (int r = 0; r < 16; ++r) {
        AMX_STZ(amx_z_op(C + r * ldc, r * kZStrideF32));
    }
}

inline uint64_t amx_fma32_op(int xy_slot, bool skip_z) {
    const uint64_t offset = (uint64_t)xy_slot * 64ull;
    return (skip_z ? kFma32SkipZ : 0ull) | (offset << 10) | offset;
}

/* Pure FMA loop - accumulates kc steps into the live Z accumulator without
 * storing. Caller is responsible for AMX_SET and store_z_fp32 after the last
 * call. The first FMA for a given (i, j) tile sets FMA32's skip-Z bit, which
 * writes x*y and avoids 16 LDZ-zero instructions per tile. Splitting the
 * kernel this way lets us walk K in cache-friendly KC chunks while keeping Z
 * live across the inner loop. The loop alternates X/Y slots 0 and 1, loading
 * the next k before issuing FMA on the previous slot so the AMX coprocessor can
 * overlap the load with the prior outer product. */
__attribute__((noinline))
static void amx_fma32_accumulate(int kc,
                                 const float* __restrict A_pack,
                                 const float* __restrict B_pack,
                                 bool skip_z_on_first_fma) {
    if (kc <= 0) return;

    AMX_LDX(amx_xy_op(B_pack, 0));   /* B row -> X */
    AMX_LDY(amx_xy_op(A_pack, 0));   /* A col -> Y */

    for (int k = 1; k < kc; ++k) {
        const int load_slot = k & 1;
        const int fma_slot = (k - 1) & 1;
        AMX_LDX(amx_xy_op(B_pack + k * 16, load_slot));
        AMX_LDY(amx_xy_op(A_pack + k * 16, load_slot));
        AMX_FMA32(amx_fma32_op(fma_slot, skip_z_on_first_fma && k == 1));
    }
    AMX_FMA32(amx_fma32_op((kc - 1) & 1, skip_z_on_first_fma && kc == 1));
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
 * Aligned heap allocator. AMX LDX/LDY/STZ require 64-byte alignment.
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

    /* Note: direct STZ-to-C (when C base + ldc are aligned) was measured
     * here and REGRESSED throughput by ~30 % at 4096³ on M2 Ultra. AMX_STZ
     * to non-cached memory waits at each hierarchy level; STZ to a hot L1
     * stack buffer then memcpy-ing wins despite the extra write. Leaving
     * the indirect path as the default until we have a benchmark showing
     * the direct path winning. */

    for (int i = i_start; i < i_end; i += 16) {
        const float* A_panel = A_pack_mega + (size_t)(i / 16) * K * 16;
        for (int j = 0; j < N; j += 16) {
            const float* B_panel = B_pack_mega + (size_t)(j / 16) * K * 16;

            bool first_k = true;
            for (int kp = 0; kp < K; kp += KC) {
                const int kc = (kp + KC <= K) ? KC : (K - kp);
                amx_fma32_accumulate(kc, A_panel + (size_t)kp * 16,
                                          B_panel + (size_t)kp * 16,
                                          first_k);
                first_k = false;
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
 * pool of long-lived pthreads, each pre-armed and spread with
 * `pthread_prefer_alternate_cluster_self`, lets the kernel settle their
 * cluster placement once and reuse the threads on every call.
 *
 * Synchronization: each worker waits on its own start-semaphore; the main
 * thread posts every worker, workers cover all M, and waits on their
 * done-semaphores.
 * Worker queue depth = 1 (one outstanding job per worker), which is all we
 * need for one GEMM split. */
struct amx_work_unit {
    int i_start, i_end;
    int N, K, lda, ldb, ldc;
    const float* A;
    const float* B;
    float* C;
};

/* AMX worker pool. Size is determined at runtime from the silicon's
 * actual cluster topology via tc_amx_cluster_count(), NOT a compile-time
 * constant. Storage is std::vector so the pool exactly matches the
 * spawned worker count and we have no MAX_AMX_WORKERS hardcode to drift
 * out of date as Apple ships more P-clusters per Ultra/Extreme tier. */

/* Forward-declare so amx_pool_init_once() below can call it before the
 * extern "C" definition further down. */
extern "C" int tc_amx_cluster_count(void);

struct amx_worker_pool_t {
    std::vector<pthread_t>            threads;
    std::vector<dispatch_semaphore_t> start;
    std::vector<dispatch_semaphore_t> done;
    std::vector<amx_work_unit>        work;
    std::atomic<bool> shutdown{false};
    std::atomic<bool> ready{false};
    std::atomic<int> failed{0};
    int n_workers{0};
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

    /* Workers 1..N-1 ask the scheduler to push them to a P-cluster
     * other than the one their siblings are on. On 2-cluster chips this
     * cleanly splits worker 0 vs worker 1; on 4-cluster Ultra chips the
     * scheduler cycles each successive worker across clusters. Worker 0
     * stays on its parent's home cluster so we don't lose the cache
     * affinity that the dispatching thread already has. */
    if (t > 0) {
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
    /* Detect actual cluster count from sysctl. The result is the spawned
     * worker count — no hardcoded ceiling. tc_amx_cluster_count() is the
     * single source of truth and already clamps to >= 1. */
    int n = tc_amx_cluster_count();
    if (n < 1) n = 1;
    g_pool.n_workers = n;
    g_pool.threads.resize((size_t)n);
    g_pool.start.resize((size_t)n, nullptr);
    g_pool.done.resize((size_t)n, nullptr);
    g_pool.work.resize((size_t)n);

    for (int t = 0; t < n; ++t) {
        g_pool.start[t] = dispatch_semaphore_create(0);
        g_pool.done[t]  = dispatch_semaphore_create(0);
        if (!g_pool.start[t] || !g_pool.done[t]) {
            g_pool.n_workers = 0;
            return;
        }
    }

    int created = 0;
    for (int t = 0; t < n; ++t) {
        if (pthread_create(&g_pool.threads[t], nullptr, amx_worker_thread_entry,
                           (void*)(intptr_t)t) != 0) {
            g_pool.shutdown.store(true, std::memory_order_release);
            for (int i = 0; i < created; ++i) {
                dispatch_semaphore_signal(g_pool.start[i]);
            }
            g_pool.n_workers = 0;
            return;
        }
        pthread_detach(g_pool.threads[t]);
        ++created;
    }
    g_pool.ready.store(true, std::memory_order_release);
}

/* Dispatch N work units across the pool. The pool must already be
 * sized at >= n_units (typically n_units == g_pool.n_workers). Slots
 * past n_units are not signaled.
 *
 * Held under g_pool_dispatch_lock for the duration to serialize
 * concurrent callers — the pool has a single work-array per worker so
 * concurrent dispatchers would clobber. The lock is the rate-limiter
 * for cross-thread AMX use; a higher-throughput design would use
 * per-worker MPSC queues, deferred. */
static bool amx_pool_dispatch_n(const amx_work_unit* units, int n_units) {
    pthread_once(&g_pool_once, amx_pool_init_once);
    if (!g_pool.ready.load(std::memory_order_acquire)) return false;
    if (n_units < 1 || n_units > g_pool.n_workers) return false;

    pthread_mutex_lock(&g_pool_dispatch_lock);
    g_pool.failed.store(0, std::memory_order_release);
    for (int t = 0; t < n_units; ++t) g_pool.work[t] = units[t];
    for (int t = 0; t < n_units; ++t) dispatch_semaphore_signal(g_pool.start[t]);
    for (int t = 0; t < n_units; ++t) {
        dispatch_semaphore_wait(g_pool.done[t], DISPATCH_TIME_FOREVER);
    }
    const bool ok = (g_pool.failed.load(std::memory_order_acquire) == 0);
    pthread_mutex_unlock(&g_pool_dispatch_lock);
    return ok;
}

/* Expose pool worker count for callers that partition M. Returns the
 * actual spawned worker count (>= 1 if pool initialized successfully),
 * or 0 if init failed. */
static int amx_pool_worker_count(void) {
    pthread_once(&g_pool_once, amx_pool_init_once);
    if (!g_pool.ready.load(std::memory_order_acquire)) return 0;
    return g_pool.n_workers;
}

}  // namespace

extern "C" TC_INTERNAL_SYMBOL int tc_amx_cluster_count(void);

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
/* Inner kernel: requires the rigid contract (M%16==0, N%16==0, alpha=1,
 * beta=0, no transpose). The outer wrapper tc_amx_gemm_f32 below relaxes
 * the contract via pad-and-trim + alpha/beta post-pass. */
static int tc_amx_gemm_f32_core(int M, int N, int K,
                                 const float* A, int lda,
                                 const float* B, int ldb,
                                 float* C, int ldc) {
    if (K == 0) {
        for (int i = 0; i < M; ++i) {
            std::memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
        }
        return 0;
    }

    /* Threading policy:
     *   - For M < 256: too small for parallel overhead to amortize. Pack
     *     once on the main thread (cluster-local for that thread) and run
     *     the single-thread strip processor.
     *   - For M >= 256: use the persistent topology-sized pool. Each worker
     *     packs ITS OWN A strip + full B locally, so the AMX kernel reads
     *     stay in the worker's cluster (no UltraFusion-fabric round-trip on
     *     every K iter). Memory cost scales with worker count vs shared pack,
     *     but pack throughput parallelizes too.
     *   - TC_AMX_THREADS=1 forces single-thread (for A/B measurement). */
    const char* threads_env = std::getenv("TC_AMX_THREADS");
    const bool single_thread = (threads_env && threads_env[0] == '1');
    const bool use_multi = !single_thread && M >= 256 && tc_amx_cluster_count() > 1;

    if (use_multi) {
        /* Persistent pool: N long-lived pthreads (N = detected P-clusters).
         * Each is pre-armed for
         * AMX, workers 1..N-1 pushed to alternate P-clusters via the
         * private hook. M is split into N strips, each 16-aligned.
         * Remainder rows fold into the last worker. */
        int n_workers = amx_pool_worker_count();
        if (n_workers < 2) n_workers = 2;  /* shouldn't happen given use_multi */
        const int strips_total = M / 16;
        const int strips_per_worker = strips_total / n_workers;
        /* Stack-allocate per-call; n_workers is bounded by P-cluster count
         * which any shipping silicon caps well below stack limits. */
        std::vector<amx_work_unit> units((size_t)n_workers);
        for (int t = 0; t < n_workers; ++t) {
            const int i_start = strips_per_worker * 16 * t;
            const int i_end = (t == n_workers - 1)
                                  ? M
                                  : strips_per_worker * 16 * (t + 1);
            units[t] = {i_start, i_end, N, K, lda, ldb, ldc, A, B, C};
        }
        if (!amx_pool_dispatch_n(units.data(), n_workers)) return -1;
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

/* Public wrapper: relaxes the rigid contract via pad-and-trim + alpha/beta
 * post-pass. Handles:
 *   - M, N not divisible by 16: zero-pad to next multiple, run core, trim
 *   - alpha != 1 or beta != 0: compute T = A*B in a temp buffer, then
 *     C = alpha*T + beta*C in a scalar fixup pass
 *
 * Transposed A/B are copied into the same row-major scratch layout used by
 * the pad-and-trim path.
 *
 * The pad-and-trim only kicks in for edge tiles, so well-aligned shapes
 * (the common case for transformer hidden dims) skip the overhead. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda, int transpose_a,
                                                   const float* B, int ldb, int transpose_b,
                                                   float beta,
                                                   float* C, int ldc) {
    if (M < 0 || N < 0 || K < 0) return -1;
    if (M == 0 || N == 0) return 0;
    if (!C || ldc < N) return -1;
    /* Validate strides per transpose flag. transpose_a=true means A is stored
     * as K×M (so leading dim is M, not K); same for B. */
    if (K > 0) {
        if (!A || !B) return -1;
        const int lda_min = transpose_a ? M : K;
        const int ldb_min = transpose_b ? K : N;
        if (lda < lda_min || ldb < ldb_min) return -1;
    }
    if (K == 0) {
        /* C = beta * C (alpha*A*B is zero); fixup pass only. */
        for (int i = 0; i < M; ++i) {
            float* row = C + (size_t)i * ldc;
            if (beta == 0.0f) {
                std::memset(row, 0, (size_t)N * sizeof(float));
            } else {
                for (int j = 0; j < N; ++j) row[j] *= beta;
            }
        }
        return 0;
    }

    const bool aligned = ((M & 15) == 0 && (N & 15) == 0);
    const bool plain = (alpha == 1.0f && beta == 0.0f);
    const bool no_transpose = !transpose_a && !transpose_b;

    if (aligned && plain && no_transpose) {
        return tc_amx_gemm_f32_core(M, N, K, A, lda, B, ldb, C, ldc);
    }

    /* Slow path: pad up to 16-multiple, compute into temp T, then post-pass.
     *   M_pad = ceil_16(M), N_pad = ceil_16(N)
     *   Temp T is M_pad x N_pad. Pack only the in-bounds rows/cols; pad is zero.
     *   Run core on T, then C[i,j] = alpha * T[i,j] + beta * C[i,j] for in-bounds.
     */
    const int M_pad = (M + 15) & ~15;
    const int N_pad = (N + 15) & ~15;
    const bool need_a_pad = (M_pad != M);
    const bool need_b_pad = (N_pad != N);

    try {
        /* Materialize A in M_pad × K row-major (un-transposed, zero-padded).
         * For transpose_a=true, source A is K×M with leading dim `lda` —
         * read A_orig[k,i] and write A_packed[i,k]. */
        const float* A_use = A;
        int lda_use = lda;
        std::vector<float> A_packed;
        const bool need_a_repack = transpose_a || need_a_pad;
        if (need_a_repack) {
            A_packed.assign((size_t)M_pad * K, 0.0f);
            if (transpose_a) {
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        A_packed[(size_t)i * K + k] = A[(size_t)k * lda + i];
                    }
                }
            } else {
                for (int i = 0; i < M; ++i) {
                    std::memcpy(A_packed.data() + (size_t)i * K,
                                A + (size_t)i * lda,
                                (size_t)K * sizeof(float));
                }
            }
            A_use = A_packed.data();
            lda_use = K;
        }

        /* Materialize B in K × N_pad row-major (un-transposed, zero-padded).
         * For transpose_b=true, source B is N×K with leading dim `ldb` —
         * read B_orig[j,k] and write B_packed[k,j]. */
        const float* B_use = B;
        int ldb_use = ldb;
        std::vector<float> B_packed;
        const bool need_b_repack = transpose_b || need_b_pad;
        if (need_b_repack) {
            B_packed.assign((size_t)K * N_pad, 0.0f);
            if (transpose_b) {
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        B_packed[(size_t)k * N_pad + j] = B[(size_t)j * ldb + k];
                    }
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    std::memcpy(B_packed.data() + (size_t)k * N_pad,
                                B + (size_t)k * ldb,
                                (size_t)N * sizeof(float));
                }
            }
            B_use = B_packed.data();
            ldb_use = N_pad;
        }

        std::vector<float> T((size_t)M_pad * N_pad, 0.0f);
        const int rc = tc_amx_gemm_f32_core(M_pad, N_pad, K,
                                            A_use, lda_use,
                                            B_use, ldb_use,
                                            T.data(), N_pad);
        if (rc != 0) return rc;

        /* Post-pass: C[i,j] = alpha * T[i,j] + beta * C[i,j] over the
         * in-bounds region. Cheap relative to the GEMM. */
        for (int i = 0; i < M; ++i) {
            const float* trow = T.data() + (size_t)i * N_pad;
            float* crow = C + (size_t)i * ldc;
            if (beta == 0.0f) {
                for (int j = 0; j < N; ++j) crow[j] = alpha * trow[j];
            } else {
                for (int j = 0; j < N; ++j) crow[j] = alpha * trow[j] + beta * crow[j];
            }
        }
    } catch (const std::bad_alloc&) {
        return -1;
    }
    return 0;
}

extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32_available(void) {
    return 1;
}

/* ----------------------------------------------------------------------- *
 * AMX ISA version + cluster count probes.
 *
 *   AMX1: M1/A14 (Apple7)    — fp64/fp32 FMA stable
 *   AMX2: M2/A15 (Apple8)    — adds quantized + 256-bit accumulator modes
 *   AMX3: M3+/A17+ (Apple9+) — refined fp16/bf16 IO-mode flag bits
 *
 * fp32 FMA is encoding-stable across all three. fp16/bf16 paths (when
 * we ship them) use isa_version to select the correct IO-mode bits. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_isa_version(void) {
    uint32_t fam = 0;
    size_t fam_sz = sizeof(fam);
    if (sysctlbyname("hw.cpufamily", &fam, &fam_sz, nullptr, 0) != 0) return 1;
    enum : uint32_t {
        FIRESTORM_ICESTORM = 0x1b588bb3u,   /* M1 / A14 */
        AVALANCHE_BLIZZARD = 0xda33d83du,   /* M2 / A15 */
        EVEREST_SAWTOOTH   = 0x8765edeau,   /* M3 / A17 */
    };
    switch (fam) {
        case FIRESTORM_ICESTORM: return 1;
        case AVALANCHE_BLIZZARD: return 2;
        case EVEREST_SAWTOOTH:   return 3;
        default: return (fam == 0u) ? 1 : 3;   /* unknown new chip → AMX3 */
    }
}

/* Number of P-clusters (= independent AMX units) on this silicon.
 * Derived from sysctl, never a baked constant. Apple's cluster size
 * is consistently 4 P-cores but the number of clusters scales with
 * the tier:
 *
 *   M base:             4 P-cores  / 4 = 1 cluster
 *   M Pro/Max:          8 P-cores  / 4 = 2 clusters
 *   M Ultra (Fusion):  16 P-cores  / 4 = 4 clusters
 *   future Extreme:    32+ P-cores / 4 = 8+ clusters
 *
 * Falls back to the older "8 P-cores => 2 clusters" heuristic only
 * when cpusperl2 sysctl is unavailable (very old macOS / non-Apple
 * silicon). Returns at least 1. No upper ceiling — the pool resizes
 * to match whatever this returns. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_cluster_count(void) {
    uint32_t p_cpus = 0;
    uint32_t per_cluster = 0;
    size_t sz1 = sizeof(p_cpus);
    size_t sz2 = sizeof(per_cluster);
    sysctlbyname("hw.perflevel0.physicalcpu", &p_cpus, &sz1, nullptr, 0);
    sysctlbyname("hw.perflevel0.cpusperl2",   &per_cluster, &sz2, nullptr, 0);
    if (p_cpus == 0) return 1;
    int n;
    if (per_cluster > 0) {
        n = (int)((p_cpus + per_cluster - 1) / per_cluster);
    } else {
        n = (p_cpus > 4) ? 2 : 1;
    }
    return (n < 1) ? 1 : n;
}

/* fp16 AMX GEMM entry. Returns -1 (unsupported) until the FMA16 operand
 * encoding for fp16-input mode is validated on hardware. Skeleton present;
 * actual dispatch would mirror tc_amx_gemm_f32_core's blocking with FMA16
 * (opcode 14) replacing FMA32 and 32-row Z layout (vs fp32's 16 rows).
 *
 * Callers (gemm_cpu.cpp) should fall through to NEON fp16 or convert to
 * fp32 + use tc_amx_gemm_f32. Two-rate fp16 (2× fp32 throughput on AMX)
 * remains the v0.3 prize. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f16(int M, int N, int K,
                                                   float alpha,
                                                   const void* A, int lda, int transpose_a,
                                                   const void* B, int ldb, int transpose_b,
                                                   float beta,
                                                   void* C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha;
    (void)A; (void)lda; (void)transpose_a;
    (void)B; (void)ldb; (void)transpose_b;
    (void)beta; (void)C; (void)ldc;
    return -1;
}

/* bf16 AMX GEMM entry. Same opcode as FMA16 (0x002011CA) with an
 * IO-mode flag bit per corsix's reference. The flag bit position differs
 * subtly between AMX2 and AMX3 — tc_amx_isa_version() above gates that.
 * Currently returns -1; callers convert bf16 to fp32 and use the fp32 AMX
 * path. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_bf16(int M, int N, int K,
                                                    float alpha,
                                                    const void* A, int lda, int transpose_a,
                                                    const void* B, int ldb, int transpose_b,
                                                    float beta,
                                                    void* C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha;
    (void)A; (void)lda; (void)transpose_a;
    (void)B; (void)ldb; (void)transpose_b;
    (void)beta; (void)C; (void)ldc;
    return -1;
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

extern "C" TC_INTERNAL_SYMBOL int tc_amx_isa_version(void) { return 0; }
extern "C" TC_INTERNAL_SYMBOL int tc_amx_cluster_count(void) { return 0; }
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f16(int, int, int, float,
                                                   const void*, int, int,
                                                   const void*, int, int,
                                                   float, void*, int) { return -1; }
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_bf16(int, int, int, float,
                                                    const void*, int, int,
                                                    const void*, int, int,
                                                    float, void*, int) { return -1; }

#endif  /* TC_AMX_GEMM_BUILD */
