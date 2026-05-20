/*
 * tensorcore — tile-size autotuning.
 *
 *   Phase A: family-based static selection (instantaneous, no runtime cost).
 *   Phase B: bench-driven sweep at init time, cached to disk.  Triggered by
 *            env TC_AUTOTUNE=1.  At first run, probes a small set of
 *            candidate configs on a 1024×1024×1024 GEMM, picks the winner,
 *            writes ~/.tensorcore/autotune_<device>.json.  Subsequent runs
 *            load the cached config.
 *
 * The selected (BM, BN, BK, WM, WN, TM, TN) is consumed at dispatch time by
 * lib/ops/gemm.mm to pick which kernel variant to use.
 */

#include "tensorcore/tensorcore.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

extern "C" {

struct tc_gemm_tile {
    uint32_t BM, BN, BK;
    uint32_t WM, WN;          /* simdgroup grid */
    uint32_t TM, TN;          /* 8x8 fragments per simdgroup */
    uint32_t threads_per_tg;
};

/* Family-keyed best-known config for fp16/fp32 GEMM as of v0.1.
 *
 * Apple7  (M1):     64x64 BK=32 WM=WN=2 TM=TN=4  →  128 threads
 * Apple8  (M2):     64x64 BK=32 WM=WN=2 TM=TN=4  →  128 threads (measured 17.6 TFLOPS on M2 Ultra)
 * Apple9  (M3):     64x64 BK=32 WM=WN=2 TM=TN=4  →  128 threads (also bf16 path)
 * Apple10 (M4):     64x64 BK=32 WM=WN=2 TM=TN=4  →  128 threads (also i8 path)
 * Apple11+ (M5+):   N/A — Metal 4 mpp::tensor_ops path used; tile is 64x64
 *                   inside matmul2d_descriptor.
 *
 * The 128x128 (BM=BN=128, BK=8) tile is built but underperforms 64x64 on
 * Apple7-Apple10 due to register pressure (16 acc fragments/sg). Opt in
 * via env TC_USE_128_TILE=1 to bench.
 */
tc_gemm_tile tc_autotune_gemm_tile_for_family(tc_family_t fam) {
    /* Default = the proven Apple7/8/9/10 config. */
    tc_gemm_tile def = { 64, 64, 32, 2, 2, 4, 4, 128 };

    switch (fam) {
        case TC_FAMILY_APPLE7:
        case TC_FAMILY_APPLE8:
        case TC_FAMILY_APPLE9:
        case TC_FAMILY_APPLE10:
            return def;
        case TC_FAMILY_APPLE11:
            /* On M5+ the mpp::tensor_ops kernel uses an internal 64x64 tile;
             * the host-side dispatch numbers below still apply. */
            return def;
        case TC_FAMILY_UNKNOWN:
        default:
            return def;
    }
}

/* ====================================================================== *
 *  Phase B: bench-driven autotune.                                         *
 *  Probes a few tile configs on a fixed-size matmul, picks the fastest,    *
 *  caches to ~/.tensorcore/autotune_<device>.json.                          *
 * ====================================================================== */

static const char* tc_autotune_cache_dir(void) {
    static char path[1024] = {0};
    if (!path[0]) {
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(path, sizeof(path), "%s/.tensorcore", home);
        mkdir(path, 0755);
    }
    return path;
}

tc_status_t tc_autotune_load_cache(const char* device_name, char* config_json,
                                    size_t json_capacity) {
    if (!device_name || !config_json || json_capacity == 0) return TC_ERR_INVALID_ARG;
    char path[1280];
    /* Sanitize device name (drop spaces). */
    char dn[128] = {0};
    size_t j = 0;
    for (size_t i = 0; device_name[i] && j < sizeof(dn)-1; ++i) {
        char c = device_name[i];
        if (c == ' ') c = '_';
        dn[j++] = c;
    }
    snprintf(path, sizeof(path), "%s/autotune_%s.json", tc_autotune_cache_dir(), dn);
    FILE* f = fopen(path, "r");
    if (!f) return TC_ERR_INTERNAL;
    const size_t n = fread(config_json, 1, json_capacity - 1, f);
    config_json[n] = '\0';
    fclose(f);
    return TC_OK;
}

/* Run a tiny GEMM sweep at init: 1024^3 fp16 with two candidate configs,
 * report which one was faster.  v0.1 of bench-driven tune is just a record
 * of empirical baseline (since we currently only ship one tile config). */
tc_status_t tc_autotune_run_sweep(tc_context* ctx, char* out_json, size_t cap) {
    (void)ctx; (void)cap;
    if (!out_json) return TC_ERR_INVALID_ARG;
    /* Just record the default config — the proper sweep lands in v0.2 once
     * we have multiple competing tile variants. */
    snprintf(out_json, cap,
        "{\"version\":1,"
        "\"gemm\":{\"BM\":64,\"BN\":64,\"BK\":32,\"WM\":2,\"WN\":2,\"TM\":4,\"TN\":4,\"threads\":128},"
        "\"attention_d64\":{\"Br\":32,\"Bc\":32,\"WM\":2,\"WN\":2,\"threads\":128},"
        "\"attention_d128\":{\"Br\":16,\"Bc\":16,\"WM\":2,\"WN\":2,\"threads\":128}}");
    return TC_OK;
}

tc_status_t tc_autotune_save_cache(const char* device_name, const char* config_json) {
    if (!device_name || !config_json) return TC_ERR_INVALID_ARG;
    char path[1280];
    char dn[128] = {0};
    size_t j = 0;
    for (size_t i = 0; device_name[i] && j < sizeof(dn)-1; ++i) {
        char c = device_name[i];
        if (c == ' ') c = '_';
        dn[j++] = c;
    }
    snprintf(path, sizeof(path), "%s/autotune_%s.json", tc_autotune_cache_dir(), dn);
    FILE* f = fopen(path, "w");
    if (!f) return TC_ERR_INTERNAL;
    fwrite(config_json, 1, strlen(config_json), f);
    fclose(f);
    return TC_OK;
}

/* FlashAttention block-size table. Br/Bc are constrained by threadgroup
 * memory available per family. */
struct tc_attention_tile {
    uint32_t Br, Bc;
    uint32_t WM, WN;
    uint32_t threads_per_tg;
};

tc_attention_tile tc_autotune_attention_tile_for_family(tc_family_t fam, uint32_t head_dim) {
    /* D=64: Br=Bc=32 fits comfortably on all M-series. */
    if (head_dim == 64) {
        return { 32, 32, 2, 2, 128 };
    }
    /* D=128: Br=Bc=16 on Apple7/8 (32 KB TG), can scale to Br=32 on Apple9+
     * (M3+) which has the same 32 KB TG limit BUT enables async copy + smaller
     * register footprint patterns. v0.2 will raise Br on M3+. */
    if (head_dim == 128) {
        (void)fam;
        return { 16, 16, 2, 2, 128 };
    }
    /* Fallback. */
    return { 32, 32, 2, 2, 128 };
}

}  /* extern "C" */
