/*
 * tensorcore — sparse top-k compression / decompression for DiLoCo.
 *
 * The dense Δθ allreduce sends fp32 (4 bytes/elem) over the wire. For a
 * 70B-parameter model that's 280 GB per outer step, which is impossible
 * on a 1-10 MB/s WAN even with K=1000 inner steps amortizing the cost.
 *
 * Sparsification: keep only the top-k magnitudes; transport (index,
 * fp16-value) pairs. Compression ratio for top-k 0.1% with fp16 values:
 *
 *   dense fp32:  4 bytes/elem × N
 *   sparse fp16: (4 bytes index + 2 bytes value) × 0.001 N = 0.006 N
 *   → 666× volume reduction
 *
 * Error feedback (residual carried to next outer step) is held in the
 * Parameter struct in diloco.cpp; this file only provides the packing
 * and unpacking primitives. The compressed payload format:
 *
 *   uint32_t n_total       (number of original elements)
 *   uint32_t n_kept        (number of non-zero entries packed)
 *   then n_kept × { uint32_t index; uint16_t fp16_value; uint16_t pad; }
 *
 * Alignment: the (index, value, pad) triplet is 8 bytes each to keep
 * the on-wire layout naturally aligned. Pad bits are zero and ignored
 * by the receiver.
 *
 * Multi-rank merge: when ranks all send their sparse Δθ, the union of
 * their kept indices may overlap. The receiver merges by summing values
 * at duplicate indices, then divides by world_size for the AVG semantic.
 * Implementation: scatter into a dense fp32 accumulator, then read out.
 */

#include "tensorcore/diloco.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

namespace {

inline uint16_t f32_to_f16_compress(float v) {
    union { float f; uint32_t u; } x = {v};
    const uint32_t bits = x.u;
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

inline float f16_to_f32_compress(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { float r; std::memcpy(&r, &sign, 4); return r; }
        int e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
        mant &= 0x03ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &bits, 4); return r;
}

}  // namespace

/* Pack a dense Δθ vector into the sparse top-k payload.
 *
 * Input: delta_fp32 [n], keep_fraction in (0, 1].
 * Output: out_payload populated with header + entries.
 *
 * Returns the number of bytes written to out_payload. The caller is
 * responsible for sizing out_payload at least
 *   sizeof(header) + n_kept * 8
 * where n_kept = ceil(n * keep_fraction).
 *
 * Side effect: zeros entries below threshold in delta_fp32. This is the
 * "drop" half of error-feedback; the caller separately carries the
 * residual into the next outer step.
 *
 * Public symbol kept hidden from the export surface — used by DiLoCo
 * internally. */
extern "C" TC_INTERNAL_SYMBOL size_t tc_diloco_sparse_pack(float* delta_fp32, size_t n,
                                                            float keep_fraction,
                                                            void* out_payload, size_t out_cap) {
    const size_t n_kept = std::max<size_t>(1, (size_t)(n * keep_fraction));
    const size_t header_bytes = 8;   /* n_total + n_kept */
    const size_t entry_bytes = 8;    /* idx + val + pad */
    const size_t needed = header_bytes + n_kept * entry_bytes;
    if (out_cap < needed) return 0;

    /* Magnitude threshold via nth_element on a copy. */
    std::vector<float> mag(n);
    for (size_t i = 0; i < n; ++i) mag[i] = std::fabs(delta_fp32[i]);
    if (n_kept >= n) {
        /* keep everything */
    } else {
        std::nth_element(mag.begin(), mag.begin() + (mag.size() - n_kept), mag.end());
    }
    const float thresh = (n_kept >= n) ? 0.0f : mag[mag.size() - n_kept];

    uint8_t* out = (uint8_t*)out_payload;
    uint32_t n_total_u = (uint32_t)n;
    uint32_t n_kept_u = 0;   /* fill in after counting */
    std::memcpy(out, &n_total_u, 4);
    std::memcpy(out + 4, &n_kept_u, 4);

    uint8_t* entries = out + header_bytes;
    size_t written = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(delta_fp32[i]) >= thresh) {
            if (written >= n_kept) break;
            uint32_t idx = (uint32_t)i;
            uint16_t val = f32_to_f16_compress(delta_fp32[i]);
            uint16_t pad = 0;
            std::memcpy(entries + written * entry_bytes + 0, &idx, 4);
            std::memcpy(entries + written * entry_bytes + 4, &val, 2);
            std::memcpy(entries + written * entry_bytes + 6, &pad, 2);
            ++written;
        } else {
            delta_fp32[i] = 0.0f;    /* drop from local view; carry-over in error-feedback handled elsewhere */
        }
    }
    /* Backfill the actual count (may be less than n_kept if ties caused early break). */
    n_kept_u = (uint32_t)written;
    std::memcpy(out + 4, &n_kept_u, 4);
    return header_bytes + written * entry_bytes;
}

/* Unpack a sparse top-k payload into a dense fp32 destination.
 *
 * dst[n_total] is assumed to be zeroed by the caller. Multiple unpacks
 * onto the same dst sum into the same indices (the merge step for
 * multi-rank all-reduce).
 *
 * Returns TC_OK on success, TC_ERR_INVALID_ARG on bad header. */
extern "C" TC_INTERNAL_SYMBOL int tc_diloco_sparse_unpack_add(const void* payload, size_t payload_bytes,
                                                               float* dst, size_t dst_capacity) {
    if (payload_bytes < 8) return -1;
    const uint8_t* in = (const uint8_t*)payload;
    uint32_t n_total = 0, n_kept = 0;
    std::memcpy(&n_total, in + 0, 4);
    std::memcpy(&n_kept, in + 4, 4);
    if (n_total != dst_capacity) return -1;
    if (payload_bytes < 8 + (size_t)n_kept * 8) return -1;
    const uint8_t* entries = in + 8;
    for (uint32_t k = 0; k < n_kept; ++k) {
        uint32_t idx = 0;
        uint16_t val = 0;
        std::memcpy(&idx, entries + (size_t)k * 8 + 0, 4);
        std::memcpy(&val, entries + (size_t)k * 8 + 4, 2);
        if (idx >= n_total) return -1;
        dst[idx] += f16_to_f32_compress(val);
    }
    return 0;
}

/* Compute the on-wire size that pack would produce given (n, keep_fraction).
 * Used by the DiLoCo bandwidth-budget accounting. */
extern "C" TC_INTERNAL_SYMBOL size_t tc_diloco_sparse_packed_size(size_t n, float keep_fraction) {
    const size_t n_kept = std::max<size_t>(1, (size_t)(n * keep_fraction));
    return 8 + n_kept * 8;
}
