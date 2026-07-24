/*
 * tensorcore — DiLoCo runtime.
 *
 * Layered above tc_dist_*: every cross-site collective uses the
 * tc_dist_ctx passed by the caller. DiLoCo holds the inner-loop counter,
 * the anchor copy of θ (θ_global), the outer-optimizer state, and the
 * compression / error-feedback buffers.
 *
 * The hot loop on a worker:
 *
 *     for step = 0..total_inner_steps:
 *         <do normal training step against θ_local>
 *         tc_diloco_step(d, &outer_pending);
 *         if (outer_pending) tc_diloco_apply_outer(d);
 *
 * If async_overlap = true, apply_outer captures immutable parameter
 * snapshots and dispatches cross-site work against private state. The next
 * inner-step batch can continue against live buffers. The caller polls or
 * waits, then commits at a declared boundary; commit rebases post-snapshot
 * local updates onto the new anchor.
 *
 * Compression schemes:
 *
 *   NONE      — Δθ sent as fp32, full 1:1
 *   FP16      — convert Δθ to fp16 before send, convert back at receive
 *   FP8       — per-tensor scale, fp8 magnitude (E4M3 saturation)
 *   TOPK_*    — keep top-K magnitudes; error-feedback retains the residual
 *   LOWRANK   — PowerSGD: rank-r approximation of each parameter tensor
 *   SIGNSGD   — 1-bit per element, scaled at receive end
 *
 * For now, the in-tree implementation covers local/single-rank outer
 * steps plus portable-CPU GLOO multi-rank outer steps for NONE,
 * FP16-intent, and TOPK masking with error feedback. TOPK over GLOO
 * uses sparse (idx, fp16-val) payloads on the wire. FP8 / LOWRANK /
 * SIGNSGD and dropout-tolerant WAN recovery return explicit unsupported
 * statuses so downstream code gets a stable failure instead of incorrect
 * results.
 *
 * Memory cost: one anchor θ + one momentum buffer + (top-k) one
 * error-feedback buffer per parameter, all fp32. For a 70B model that's
 * ~3 × 280 GB = 840 GB if every parameter is fp32; in practice the
 * fp32 anchor is the optimizer master-weight, the momentum is the
 * outer-optimizer state (Nesterov), and the error-feedback piece only
 * exists when top-k is enabled. Plan for ~3-5× the model size in
 * fp32 working memory.
 */

#include "tensorcore/diloco.h"
#include "tensorcore/tensorcore.h"

/* Internal helper exported by lib/distributed/distributed_cpu.cpp + Metal's
 * lib/distributed/distributed.mm. Gives DiLoCo access to the parent
 * tc_context so it can allocate temporary buffers in the same arena as
 * the user's tc_dist_ctx. */
#if defined(_WIN32)
#define TC_DILOCO_INTERNAL_SYMBOL
#else
#define TC_DILOCO_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

extern "C" TC_DILOCO_INTERNAL_SYMBOL tc_context* tc_dist_get_context(tc_dist_ctx* d);

/* GLOO sparse-compressed allreduce hook. Returns the GlooState* if the
 * transport is TC_DIST_GLOO, nullptr otherwise. When present, DiLoCo
 * can invoke tc_gloo_sparse_allreduce to ship Δθ as (idx, fp16) pairs
 * instead of dense fp32 — 100-1000× less bandwidth at top-k 0.1%-1%. */
struct GlooState;
extern "C" TC_DILOCO_INTERNAL_SYMBOL GlooState* tc_dist_get_gloo_state(tc_dist_ctx* d);
extern "C" TC_DILOCO_INTERNAL_SYMBOL int tc_gloo_sparse_allreduce(GlooState* s, int world_size, int rank,
                                                                    const void* payload_in, size_t payload_in_bytes,
                                                                    float* dense_out, size_t n_total);

/* Sparse pack primitives from lib/distributed/sparse_compress.cpp. */
extern "C" TC_DILOCO_INTERNAL_SYMBOL size_t tc_diloco_sparse_pack(float* delta_fp32, size_t n,
                                                                   float keep_fraction,
                                                                   void* out_payload, size_t out_cap);
extern "C" TC_DILOCO_INTERNAL_SYMBOL size_t tc_diloco_sparse_packed_size(size_t n, float keep_fraction);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Parameter {
    std::string  name;
    tc_buffer*   theta_local;       /* model's working copy, fp16 or fp32 */
    size_t       num_elements;
    tc_dtype_t   dtype;

    std::vector<float> theta_anchor;     /* θ_global_anchor in fp32        */
    std::vector<float> outer_momentum;   /* outer-optimizer state          */
    std::vector<float> error_feedback;   /* residual for top-k (lazy alloc) */
};

inline uint16_t f32_to_f16_bits(float x) {
    union { float f; uint32_t u; } v = {x};
    const uint32_t f = v.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t round = (mant >> (shift - 1)) & 1u;
        return (uint16_t)(sign | ((mant >> shift) + round));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    const uint32_t round = (mant >> 12) & 1u;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant >> 13) + round));
}

inline float f16_to_f32_bits(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            ++exp;
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = {out};
    return v.f;
}

/* Read parameter from its tc_buffer into a fp32 host array. */
bool theta_to_fp32(const Parameter& p, std::vector<float>& dst) {
    void* mp = nullptr;
    if (tc_buffer_map(p.theta_local, &mp) != TC_OK || !mp) return false;
    dst.resize(p.num_elements);
    if (p.dtype == TC_DTYPE_F32) {
        std::memcpy(dst.data(), mp, p.num_elements * sizeof(float));
    } else if (p.dtype == TC_DTYPE_F16) {
        const uint16_t* src = (const uint16_t*)mp;
        for (size_t i = 0; i < p.num_elements; ++i) dst[i] = f16_to_f32_bits(src[i]);
    } else {
        return false;
    }
    return true;
}

/* Write fp32 host array back to the parameter's tc_buffer. */
bool theta_from_fp32(Parameter& p, const std::vector<float>& src) {
    void* mp = nullptr;
    if (tc_buffer_map(p.theta_local, &mp) != TC_OK || !mp) return false;
    if (p.dtype == TC_DTYPE_F32) {
        std::memcpy(mp, src.data(), p.num_elements * sizeof(float));
    } else if (p.dtype == TC_DTYPE_F16) {
        uint16_t* dst = (uint16_t*)mp;
        for (size_t i = 0; i < p.num_elements; ++i) dst[i] = f32_to_f16_bits(src[i]);
    } else {
        return false;
    }
    return true;
}

}  // namespace

struct tc_diloco_ctx {
    tc_dist_ctx*               dist;
    tc_diloco_config           cfg;
    std::vector<Parameter>     params;

    /* inner-loop counter */
    std::atomic<uint64_t>      inner_steps_total{0};
    std::atomic<uint64_t>      outer_steps_total{0};
    int                        inner_steps_since_outer = 0;

    /* timing instrumentation */
    std::atomic<double>        last_outer_seconds{0.0};
    std::atomic<double>        last_outer_bytes{0.0};

    /* async-overlap support */
    std::thread                outer_thread;
    mutable std::mutex         outer_mutex;
    std::condition_variable    outer_cv;
    tc_diloco_async_state_t    outer_state = TC_DILOCO_ASYNC_IDLE;
    tc_status_t                outer_status = TC_OK;
    uint64_t                   next_round_id = 1;
    uint64_t                   active_round_id = 0;
    std::vector<Parameter>     pending_params;
    std::vector<std::vector<float>> pending_snapshots;
    double                     pending_seconds = 0.0;
    double                     pending_bytes = 0.0;

    /* Caller-owned topology identities persisted in the checkpoint blob. */
    uint64_t                   topology_epoch = 0;
    uint64_t                   membership_epoch = 0;

    ~tc_diloco_ctx() {
        if (outer_thread.joinable()) outer_thread.join();
    }
};

/* ------------------------------------------------------------------------
 * Versioned checkpoint serialization
 * ------------------------------------------------------------------------ */

namespace {

/* Both wire versions share one payload layout. Only the header differs, so a
 * v1 and a v2 blob written from the same state have identical payload bytes. */
constexpr uint8_t kStateMagicV1[8] = {'T', 'C', 'D', 'L', 'S', 'T', 'A', '1'};
constexpr uint8_t kStateMagicV2[8] = {'T', 'C', 'D', 'L', 'S', 'T', 'A', '2'};
constexpr uint32_t kStateHeaderSizeV1 = TC_DILOCO_STATE_V1_HEADER_SIZE;
constexpr uint32_t kStateHeaderSizeV2 = TC_DILOCO_STATE_V2_HEADER_SIZE;
constexpr size_t kStateDigestBytes = TC_DILOCO_STATE_V2_DIGEST_BYTES;
constexpr size_t kStateTotalSizeOffset = 16;       /* both versions */
constexpr size_t kStateV1ChecksumOffset = 24;
constexpr size_t kStateV2PayloadSizeOffset = 24;
constexpr size_t kStateV2DigestOffset = 32;

/* Fixed prefix of the shared payload: config, identity, counters, async
 * result, and the parameter count. */
constexpr size_t kStatePayloadPrefixSize = 148;

/* The headers are written field by field, so freeze the documented byte
 * offsets here rather than trusting the write order to stay put. */
static_assert(kStateTotalSizeOffset == sizeof(kStateMagicV1) + 4 + 4,
              "total size follows magic, version, and header size");
static_assert(kStateV1ChecksumOffset == kStateTotalSizeOffset + 8,
              "v1 payload checksum follows total size");
static_assert(kStateV1ChecksumOffset + 8 == kStateHeaderSizeV1,
              "v1 header ends after the payload checksum");
static_assert(kStateV2PayloadSizeOffset == kStateTotalSizeOffset + 8,
              "v2 payload size follows total size");
static_assert(kStateV2DigestOffset == kStateV2PayloadSizeOffset + 8,
              "v2 payload digest follows payload size");
static_assert(kStateV2DigestOffset + kStateDigestBytes == kStateHeaderSizeV2,
              "v2 header ends after the payload digest");
static_assert(sizeof(kStateMagicV1) == sizeof(kStateMagicV2),
              "both wire versions use an 8-byte magic");

bool state_version_supported(uint32_t version) {
    return version == TC_DILOCO_STATE_ABI_VERSION_1 ||
           version == TC_DILOCO_STATE_ABI_VERSION_2;
}

uint32_t state_header_size(uint32_t version) {
    return version == TC_DILOCO_STATE_ABI_VERSION_2 ? kStateHeaderSizeV2
                                                    : kStateHeaderSizeV1;
}

/* SHA-256 over the serialized payload. This is an integrity check on a
 * caller-supplied buffer, not an authentication tag; tc_transport_auth owns
 * keyed message authentication. */
class Sha256 {
public:
    Sha256() { reset(); }

    void update(const void* input, size_t bytes) {
        const uint8_t* p = static_cast<const uint8_t*>(input);
        total_bytes_ += bytes;
        while (bytes > 0) {
            const size_t n = std::min(bytes, sizeof(block_) - block_used_);
            std::memcpy(block_ + block_used_, p, n);
            block_used_ += n;
            p += n;
            bytes -= n;
            if (block_used_ == sizeof(block_)) {
                transform(block_);
                block_used_ = 0;
            }
        }
    }

    void finish(uint8_t out[kStateDigestBytes]) {
        const uint64_t total_bits = total_bytes_ * UINT64_C(8);
        block_[block_used_++] = 0x80;
        if (block_used_ > 56) {
            std::memset(block_ + block_used_, 0, sizeof(block_) - block_used_);
            transform(block_);
            block_used_ = 0;
        }
        std::memset(block_ + block_used_, 0, 56 - block_used_);
        for (unsigned i = 0; i < 8; ++i) {
            block_[56 + i] = (uint8_t)(total_bits >> (56 - 8 * i));
        }
        transform(block_);
        for (unsigned i = 0; i < 8; ++i) {
            out[4 * i + 0] = (uint8_t)(state_[i] >> 24);
            out[4 * i + 1] = (uint8_t)(state_[i] >> 16);
            out[4 * i + 2] = (uint8_t)(state_[i] >> 8);
            out[4 * i + 3] = (uint8_t)(state_[i]);
        }
    }

private:
    static uint32_t rotr32(uint32_t v, unsigned n) {
        return (v >> n) | (v << (32 - n));
    }

    void reset() {
        state_[0] = UINT32_C(0x6a09e667); state_[1] = UINT32_C(0xbb67ae85);
        state_[2] = UINT32_C(0x3c6ef372); state_[3] = UINT32_C(0xa54ff53a);
        state_[4] = UINT32_C(0x510e527f); state_[5] = UINT32_C(0x9b05688c);
        state_[6] = UINT32_C(0x1f83d9ab); state_[7] = UINT32_C(0x5be0cd19);
        std::memset(block_, 0, sizeof(block_));
        block_used_ = 0;
        total_bytes_ = 0;
    }

    void transform(const uint8_t block[64]) {
        static constexpr uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        };
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = ((uint32_t)block[4 * i + 0] << 24) |
                   ((uint32_t)block[4 * i + 1] << 16) |
                   ((uint32_t)block[4 * i + 2] << 8) |
                   ((uint32_t)block[4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            const uint32_t s0 =
                rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 =
                rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint32_t state_[8];
    uint8_t block_[64];
    size_t block_used_ = 0;
    uint64_t total_bytes_ = 0;
};

void state_payload_digest(const uint8_t* payload, size_t size,
                          uint8_t out[kStateDigestBytes]) {
    Sha256 hash;
    hash.update(payload, size);
    hash.finish(out);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint64_t double_bits(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double bits_double(uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void store_u64_le(uint8_t* bytes, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        bytes[offset + i] = (uint8_t)((value >> (8 * i)) & UINT64_C(0xff));
    }
}

uint64_t state_checksum(const uint8_t* data, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

class StateWriter {
public:
    StateWriter(void* data, size_t capacity)
        : data_(static_cast<uint8_t*>(data)), capacity_(capacity) {}

    void raw(const void* data, size_t size) {
        if (!ok_ || size > capacity_ - offset_) {
            ok_ = false;
            return;
        }
        const uint8_t* begin = static_cast<const uint8_t*>(data);
        std::memcpy(data_ + offset_, begin, size);
        offset_ += size;
    }

    void u32(uint32_t value) {
        if (!ok_ || 4 > capacity_ - offset_) {
            ok_ = false;
            return;
        }
        for (size_t i = 0; i < 4; ++i) {
            data_[offset_ + i] =
                (uint8_t)((value >> (8 * i)) & UINT32_C(0xff));
        }
        offset_ += 4;
    }

    void u64(uint64_t value) {
        if (!ok_ || 8 > capacity_ - offset_) {
            ok_ = false;
            return;
        }
        for (size_t i = 0; i < 8; ++i) {
            data_[offset_ + i] =
                (uint8_t)((value >> (8 * i)) & UINT64_C(0xff));
        }
        offset_ += 8;
    }

    bool string(const std::string& value) {
        if (value.size() > UINT32_MAX) return false;
        u32((uint32_t)value.size());
        raw(value.data(), value.size());
        return ok_;
    }

    void floats(const std::vector<float>& values) {
        u64((uint64_t)values.size());
        for (float value : values) u32(float_bits(value));
    }

    bool ok() const { return ok_; }
    size_t size() const { return offset_; }

private:
    uint8_t* data_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    bool ok_ = true;
};

class StateReader {
public:
    StateReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool raw(void* out, size_t size) {
        if (size > size_ - offset_) return false;
        std::memcpy(out, data_ + offset_, size);
        offset_ += size;
        return true;
    }

    bool u32(uint32_t& value) {
        if (4 > size_ - offset_) return false;
        value = 0;
        for (size_t i = 0; i < 4; ++i) {
            value |= (uint32_t)data_[offset_ + i] << (8 * i);
        }
        offset_ += 4;
        return true;
    }

    bool u64(uint64_t& value) {
        if (8 > size_ - offset_) return false;
        value = 0;
        for (size_t i = 0; i < 8; ++i) {
            value |= (uint64_t)data_[offset_ + i] << (8 * i);
        }
        offset_ += 8;
        return true;
    }

    bool string(std::string& value) {
        uint32_t size = 0;
        if (!u32(size) || size > size_ - offset_) return false;
        value.assign(reinterpret_cast<const char*>(data_ + offset_), size);
        offset_ += size;
        return true;
    }

    bool floats(std::vector<float>& values) {
        uint64_t count = 0;
        if (!u64(count) || count > (size_ - offset_) / sizeof(uint32_t) ||
            count > (uint64_t)std::numeric_limits<size_t>::max()) {
            return false;
        }
        values.resize((size_t)count);
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t bits = 0;
            if (!u32(bits)) return false;
            values[i] = bits_float(bits);
        }
        return true;
    }

    bool at_end() const { return offset_ == size_; }
    size_t remaining() const { return size_ - offset_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
};

struct SerializedParameter {
    std::string name;
    uint32_t dtype = 0;
    uint64_t num_elements = 0;
    std::vector<float> theta_anchor;
    std::vector<float> outer_momentum;
    std::vector<float> error_feedback;
    std::vector<float> pending_theta_anchor;
    std::vector<float> pending_outer_momentum;
    std::vector<float> pending_error_feedback;
    std::vector<float> pending_snapshot;
};

struct SerializedState {
    uint32_t inner_steps = 0;
    uint32_t outer_lr = 0;
    uint32_t outer_momentum = 0;
    uint32_t outer_beta2 = 0;
    uint32_t outer_eps = 0;
    uint32_t outer_optimizer = 0;
    uint32_t compress = 0;
    uint32_t async_overlap = 0;
    uint32_t tolerate_dropouts = 0;
    uint32_t world_size = 0;
    uint32_t rank = 0;
    uint64_t topology_epoch = 0;
    uint64_t membership_epoch = 0;
    uint64_t inner_steps_total = 0;
    uint64_t outer_steps_total = 0;
    uint64_t inner_steps_since_outer = 0;
    uint64_t last_outer_seconds = 0;
    uint64_t last_outer_bytes = 0;
    uint64_t next_round_id = 0;
    uint64_t active_round_id = 0;
    uint32_t outer_state = 0;
    uint32_t outer_status = 0;
    uint64_t pending_seconds = 0;
    uint64_t pending_bytes = 0;
    std::vector<SerializedParameter> params;
};

bool valid_optimizer_state_size(const Parameter& p) {
    return p.outer_momentum.empty() ||
           p.outer_momentum.size() == p.num_elements ||
           (p.num_elements <= std::numeric_limits<size_t>::max() / 2 &&
            p.outer_momentum.size() == 2 * p.num_elements);
}

bool valid_parameter_state(const Parameter& p) {
    return p.theta_anchor.size() == p.num_elements &&
           valid_optimizer_state_size(p) &&
           (p.error_feedback.empty() ||
            p.error_feedback.size() == p.num_elements);
}

bool add_checkpoint_size(size_t& total, size_t amount) {
    if (amount > std::numeric_limits<size_t>::max() - total) return false;
    total += amount;
    return true;
}

bool add_checkpoint_vector_size(size_t& total,
                                const std::vector<float>& values) {
    if (values.size() >
        (std::numeric_limits<size_t>::max() - sizeof(uint64_t)) /
            sizeof(uint32_t)) {
        return false;
    }
    return add_checkpoint_size(
        total, sizeof(uint64_t) + values.size() * sizeof(uint32_t));
}

tc_status_t state_payload_size_locked(const tc_diloco_ctx* d,
                                      size_t& out_size) {
    if (d->outer_state == TC_DILOCO_ASYNC_RUNNING) return TC_ERR_BUSY;
    if (d->inner_steps_since_outer < 0) return TC_ERR_INTERNAL;
    if (d->outer_state != TC_DILOCO_ASYNC_IDLE &&
        d->outer_state != TC_DILOCO_ASYNC_READY &&
        d->outer_state != TC_DILOCO_ASYNC_FAILED) {
        return TC_ERR_INTERNAL;
    }
    if ((d->outer_state == TC_DILOCO_ASYNC_FAILED &&
         d->outer_status == TC_OK) ||
        (d->outer_state != TC_DILOCO_ASYNC_FAILED &&
         d->outer_status != TC_OK)) {
        return TC_ERR_INTERNAL;
    }

    const bool has_pending = d->outer_state == TC_DILOCO_ASYNC_READY;
    if (has_pending &&
        (d->pending_params.size() != d->params.size() ||
         d->pending_snapshots.size() != d->params.size())) {
        return TC_ERR_INTERNAL;
    }
    if (!has_pending &&
        (!d->pending_params.empty() || !d->pending_snapshots.empty())) {
        return TC_ERR_INTERNAL;
    }
    if (!has_pending &&
        (d->pending_seconds != 0.0 || d->pending_bytes != 0.0)) {
        return TC_ERR_INTERNAL;
    }

    /* Fixed config/identity/counter/state prefix, header excluded. */
    size_t size = kStatePayloadPrefixSize;
    static const std::vector<float> empty;
    for (size_t i = 0; i < d->params.size(); ++i) {
        const Parameter& p = d->params[i];
        if (!valid_parameter_state(p) || p.name.size() > UINT32_MAX ||
            !add_checkpoint_size(size, 16 + p.name.size()) ||
            !add_checkpoint_vector_size(size, p.theta_anchor) ||
            !add_checkpoint_vector_size(size, p.outer_momentum) ||
            !add_checkpoint_vector_size(size, p.error_feedback)) {
            return TC_ERR_INTERNAL;
        }

        if (has_pending) {
            const Parameter& pending = d->pending_params[i];
            if (pending.name != p.name || pending.dtype != p.dtype ||
                pending.num_elements != p.num_elements ||
                !valid_parameter_state(pending) ||
                d->pending_snapshots[i].size() != p.num_elements ||
                !add_checkpoint_vector_size(size, pending.theta_anchor) ||
                !add_checkpoint_vector_size(size, pending.outer_momentum) ||
                !add_checkpoint_vector_size(size, pending.error_feedback) ||
                !add_checkpoint_vector_size(size, d->pending_snapshots[i])) {
                return TC_ERR_INTERNAL;
            }
        } else if (!add_checkpoint_vector_size(size, empty) ||
                   !add_checkpoint_vector_size(size, empty) ||
                   !add_checkpoint_vector_size(size, empty) ||
                   !add_checkpoint_vector_size(size, empty)) {
            return TC_ERR_INTERNAL;
        }
    }
    out_size = size;
    return TC_OK;
}

/* Total blob size for a requested wire version. */
tc_status_t state_blob_size_locked(const tc_diloco_ctx* d,
                                   uint32_t version,
                                   size_t& out_size) {
    if (!state_version_supported(version)) return TC_ERR_ABI_MISMATCH;
    size_t payload_size = 0;
    const tc_status_t status = state_payload_size_locked(d, payload_size);
    if (status != TC_OK) return status;
    const size_t header_size = state_header_size(version);
    if (payload_size > std::numeric_limits<size_t>::max() - header_size) {
        return TC_ERR_INTERNAL;
    }
    out_size = header_size + payload_size;
    return TC_OK;
}

tc_status_t write_state_blob_locked(const tc_diloco_ctx* d,
                                    uint32_t version,
                                    void* out_data,
                                    size_t out_size,
                                    size_t& out_written) {
    if (!state_version_supported(version)) return TC_ERR_ABI_MISMATCH;
    size_t payload_size = 0;
    const tc_status_t size_status = state_payload_size_locked(d, payload_size);
    if (size_status != TC_OK) return size_status;
    const size_t header_size = state_header_size(version);
    if (payload_size > std::numeric_limits<size_t>::max() - header_size) {
        return TC_ERR_INTERNAL;
    }
    const size_t required_size = header_size + payload_size;
    if (!out_data || out_size < required_size) return TC_ERR_INVALID_ARG;

    StateWriter writer(out_data, out_size);
    if (version == TC_DILOCO_STATE_ABI_VERSION_2) {
        static const uint8_t zero_digest[kStateDigestBytes] = {0};
        writer.raw(kStateMagicV2, sizeof(kStateMagicV2));
        writer.u32(TC_DILOCO_STATE_ABI_VERSION_2);
        writer.u32(kStateHeaderSizeV2);
        writer.u64(0); /* total size, patched below */
        writer.u64((uint64_t)payload_size);
        writer.raw(zero_digest, sizeof(zero_digest)); /* patched below */
    } else {
        writer.raw(kStateMagicV1, sizeof(kStateMagicV1));
        writer.u32(TC_DILOCO_STATE_ABI_VERSION_1);
        writer.u32(kStateHeaderSizeV1);
        writer.u64(0); /* total size, patched below */
        writer.u64(0); /* payload checksum, patched below */
    }
    if (!writer.ok() || writer.size() != header_size) return TC_ERR_INTERNAL;

    writer.u32((uint32_t)d->cfg.inner_steps);
    writer.u32(float_bits(d->cfg.outer_lr));
    writer.u32(float_bits(d->cfg.outer_momentum));
    writer.u32(float_bits(d->cfg.outer_beta2));
    writer.u32(float_bits(d->cfg.outer_eps));
    writer.u32((uint32_t)d->cfg.outer_optimizer);
    writer.u32((uint32_t)d->cfg.compress);
    writer.u32(d->cfg.async_overlap ? 1u : 0u);
    writer.u32(d->cfg.tolerate_dropouts ? 1u : 0u);
    writer.u32((uint32_t)(d->dist ? tc_dist_world_size(d->dist) : 1));
    writer.u32((uint32_t)(d->dist ? tc_dist_rank(d->dist) : 0));
    writer.u64(d->topology_epoch);
    writer.u64(d->membership_epoch);
    writer.u64(d->inner_steps_total.load(std::memory_order_relaxed));
    writer.u64(d->outer_steps_total.load(std::memory_order_relaxed));
    writer.u64((uint64_t)d->inner_steps_since_outer);
    writer.u64(double_bits(d->last_outer_seconds.load(std::memory_order_relaxed)));
    writer.u64(double_bits(d->last_outer_bytes.load(std::memory_order_relaxed)));
    writer.u64(d->next_round_id);
    writer.u64(d->active_round_id);
    writer.u32((uint32_t)d->outer_state);
    writer.u32((uint32_t)(int32_t)d->outer_status);
    writer.u64(double_bits(d->pending_seconds));
    writer.u64(double_bits(d->pending_bytes));
    writer.u64((uint64_t)d->params.size());

    static const std::vector<float> empty;
    for (size_t i = 0; i < d->params.size(); ++i) {
        const Parameter& p = d->params[i];
        if (!writer.string(p.name)) return TC_ERR_INTERNAL;
        writer.u32((uint32_t)p.dtype);
        writer.u64((uint64_t)p.num_elements);
        writer.floats(p.theta_anchor);
        writer.floats(p.outer_momentum);
        writer.floats(p.error_feedback);

        if (d->outer_state == TC_DILOCO_ASYNC_READY) {
            const Parameter& pending = d->pending_params[i];
            writer.floats(pending.theta_anchor);
            writer.floats(pending.outer_momentum);
            writer.floats(pending.error_feedback);
            writer.floats(d->pending_snapshots[i]);
        } else {
            writer.floats(empty);
            writer.floats(empty);
            writer.floats(empty);
            writer.floats(empty);
        }
    }

    if (!writer.ok() || writer.size() != required_size) return TC_ERR_INTERNAL;
    uint8_t* bytes = static_cast<uint8_t*>(out_data);
    store_u64_le(bytes, kStateTotalSizeOffset, (uint64_t)required_size);
    if (version == TC_DILOCO_STATE_ABI_VERSION_2) {
        state_payload_digest(bytes + header_size, payload_size,
                             bytes + kStateV2DigestOffset);
    } else {
        store_u64_le(bytes, kStateV1ChecksumOffset,
                     state_checksum(bytes + header_size, payload_size));
    }
    out_written = required_size;
    return TC_OK;
}

/* Parse a blob of any wire version up to max_abi_version.
 *
 * Returns TC_ERR_ABI_MISMATCH when the blob announces a known but too-new
 * version, TC_ERR_INVALID_ARG for an unrecognized magic, an inconsistent
 * header, a digest mismatch, truncation, or trailing bytes. Nothing is
 * interpreted until the header and payload digest both verify. */
tc_status_t read_state_blob(const void* data, size_t data_size,
                            uint32_t max_abi_version,
                            SerializedState& state,
                            uint32_t& out_abi_version) {
    if (!data || data_size < kStateHeaderSizeV1) return TC_ERR_INVALID_ARG;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);

    uint32_t blob_version = 0;
    if (std::memcmp(bytes, kStateMagicV2, sizeof(kStateMagicV2)) == 0) {
        blob_version = TC_DILOCO_STATE_ABI_VERSION_2;
    } else if (std::memcmp(bytes, kStateMagicV1, sizeof(kStateMagicV1)) == 0) {
        blob_version = TC_DILOCO_STATE_ABI_VERSION_1;
    } else {
        return TC_ERR_INVALID_ARG;
    }
    if (blob_version > max_abi_version) return TC_ERR_ABI_MISMATCH;

    const size_t header_size = state_header_size(blob_version);
    if (data_size < header_size) return TC_ERR_INVALID_ARG;
    const size_t payload_size = data_size - header_size;

    StateReader reader(bytes, data_size);
    uint8_t magic[sizeof(kStateMagicV1)]{};
    uint32_t abi_version = 0;
    uint32_t declared_header_size = 0;
    uint64_t total_size = 0;
    if (!reader.raw(magic, sizeof(magic)) || !reader.u32(abi_version) ||
        !reader.u32(declared_header_size) || !reader.u64(total_size) ||
        abi_version != blob_version ||
        declared_header_size != (uint32_t)header_size ||
        total_size != data_size) {
        return TC_ERR_INVALID_ARG;
    }

    if (blob_version == TC_DILOCO_STATE_ABI_VERSION_2) {
        uint64_t declared_payload_size = 0;
        uint8_t declared_digest[kStateDigestBytes]{};
        uint8_t actual_digest[kStateDigestBytes]{};
        if (!reader.u64(declared_payload_size) ||
            !reader.raw(declared_digest, sizeof(declared_digest)) ||
            declared_payload_size != payload_size) {
            return TC_ERR_INVALID_ARG;
        }
        state_payload_digest(bytes + header_size, payload_size, actual_digest);
        if (std::memcmp(declared_digest, actual_digest,
                        sizeof(actual_digest)) != 0) {
            return TC_ERR_INVALID_ARG;
        }
    } else {
        uint64_t checksum = 0;
        if (!reader.u64(checksum) ||
            checksum != state_checksum(bytes + header_size, payload_size)) {
            return TC_ERR_INVALID_ARG;
        }
    }
    out_abi_version = blob_version;

    uint64_t param_count = 0;
    if (!reader.u32(state.inner_steps) ||
        !reader.u32(state.outer_lr) ||
        !reader.u32(state.outer_momentum) ||
        !reader.u32(state.outer_beta2) ||
        !reader.u32(state.outer_eps) ||
        !reader.u32(state.outer_optimizer) ||
        !reader.u32(state.compress) ||
        !reader.u32(state.async_overlap) ||
        !reader.u32(state.tolerate_dropouts) ||
        !reader.u32(state.world_size) || !reader.u32(state.rank) ||
        !reader.u64(state.topology_epoch) ||
        !reader.u64(state.membership_epoch) ||
        !reader.u64(state.inner_steps_total) ||
        !reader.u64(state.outer_steps_total) ||
        !reader.u64(state.inner_steps_since_outer) ||
        !reader.u64(state.last_outer_seconds) ||
        !reader.u64(state.last_outer_bytes) ||
        !reader.u64(state.next_round_id) ||
        !reader.u64(state.active_round_id) ||
        !reader.u32(state.outer_state) || !reader.u32(state.outer_status) ||
        !reader.u64(state.pending_seconds) ||
        !reader.u64(state.pending_bytes) || !reader.u64(param_count) ||
        param_count > (uint64_t)std::numeric_limits<size_t>::max()) {
        return TC_ERR_INVALID_ARG;
    }

    /* Every parameter consumes at least name/dtype/count plus seven vector
     * length fields. Bound allocation by the bytes actually present. */
    constexpr size_t kMinimumSerializedParameterBytes = 72;
    if (param_count > reader.remaining() / kMinimumSerializedParameterBytes) {
        return TC_ERR_INVALID_ARG;
    }
    state.params.resize((size_t)param_count);
    for (SerializedParameter& p : state.params) {
        if (!reader.string(p.name) || !reader.u32(p.dtype) ||
            !reader.u64(p.num_elements) ||
            !reader.floats(p.theta_anchor) ||
            !reader.floats(p.outer_momentum) ||
            !reader.floats(p.error_feedback) ||
            !reader.floats(p.pending_theta_anchor) ||
            !reader.floats(p.pending_outer_momentum) ||
            !reader.floats(p.pending_error_feedback) ||
            !reader.floats(p.pending_snapshot)) {
            return TC_ERR_INVALID_ARG;
        }
    }
    return reader.at_end() ? TC_OK : TC_ERR_INVALID_ARG;
}

bool state_config_matches(const tc_diloco_ctx* d, const SerializedState& s) {
    return s.inner_steps == (uint32_t)d->cfg.inner_steps &&
           s.outer_lr == float_bits(d->cfg.outer_lr) &&
           s.outer_momentum == float_bits(d->cfg.outer_momentum) &&
           s.outer_beta2 == float_bits(d->cfg.outer_beta2) &&
           s.outer_eps == float_bits(d->cfg.outer_eps) &&
           s.outer_optimizer == (uint32_t)d->cfg.outer_optimizer &&
           s.compress == (uint32_t)d->cfg.compress &&
           s.async_overlap == (d->cfg.async_overlap ? 1u : 0u) &&
           s.tolerate_dropouts == (d->cfg.tolerate_dropouts ? 1u : 0u);
}

bool serialized_vector_size_valid(const std::vector<float>& values,
                                  size_t num_elements,
                                  bool allow_double) {
    return values.empty() || values.size() == num_elements ||
           (allow_double &&
            num_elements <= std::numeric_limits<size_t>::max() / 2 &&
            values.size() == 2 * num_elements);
}

tc_status_t validate_serialized_state(const tc_diloco_ctx* d,
                                      const SerializedState& s) {
    if (!state_config_matches(d, s) || s.params.size() != d->params.size() ||
        s.inner_steps_since_outer > (uint64_t)std::numeric_limits<int>::max()) {
        return TC_ERR_INVALID_ARG;
    }
    const uint32_t world = (uint32_t)(d->dist ? tc_dist_world_size(d->dist) : 1);
    const uint32_t rank = (uint32_t)(d->dist ? tc_dist_rank(d->dist) : 0);
    if (s.world_size != world || s.rank != rank ||
        (d->topology_epoch != 0 && d->topology_epoch != s.topology_epoch) ||
        (d->membership_epoch != 0 &&
         d->membership_epoch != s.membership_epoch) ||
        s.next_round_id == 0 || s.next_round_id <= s.active_round_id) {
        return TC_ERR_INVALID_ARG;
    }

    const bool ready = s.outer_state == TC_DILOCO_ASYNC_READY;
    const bool failed = s.outer_state == TC_DILOCO_ASYNC_FAILED;
    if (s.outer_state != TC_DILOCO_ASYNC_IDLE && !ready && !failed) {
        return TC_ERR_INVALID_ARG;
    }
    if ((ready && (!d->cfg.async_overlap || (int32_t)s.outer_status != TC_OK)) ||
        (failed && (!d->cfg.async_overlap || (int32_t)s.outer_status == TC_OK)) ||
        (s.outer_state == TC_DILOCO_ASYNC_IDLE &&
         (int32_t)s.outer_status != TC_OK) ||
        (!ready && (s.pending_seconds != 0 || s.pending_bytes != 0))) {
        return TC_ERR_INVALID_ARG;
    }

    for (size_t i = 0; i < s.params.size(); ++i) {
        const Parameter& live = d->params[i];
        const SerializedParameter& saved = s.params[i];
        if (saved.name != live.name || saved.dtype != (uint32_t)live.dtype ||
            saved.num_elements != live.num_elements ||
            saved.theta_anchor.size() != live.num_elements ||
            !serialized_vector_size_valid(
                saved.outer_momentum, live.num_elements, true) ||
            !serialized_vector_size_valid(
                saved.error_feedback, live.num_elements, false)) {
            return TC_ERR_INVALID_ARG;
        }
        if (ready) {
            if (saved.pending_theta_anchor.size() != live.num_elements ||
                !serialized_vector_size_valid(
                    saved.pending_outer_momentum, live.num_elements, true) ||
                !serialized_vector_size_valid(
                    saved.pending_error_feedback, live.num_elements, false) ||
                saved.pending_snapshot.size() != live.num_elements) {
                return TC_ERR_INVALID_ARG;
            }
        } else if (!saved.pending_theta_anchor.empty() ||
                   !saved.pending_outer_momentum.empty() ||
                   !saved.pending_error_feedback.empty() ||
                   !saved.pending_snapshot.empty()) {
            return TC_ERR_INVALID_ARG;
        }
    }
    return TC_OK;
}

}  // namespace

/* ------------------------------------------------------------------------
 * Compression / decompression
 * ------------------------------------------------------------------------ */

namespace {

/* Convert Δθ into the per-rank send buffer (still fp32 here; compression
 * shrinks the byte size before tc_allreduce). For the simple NONE / FP16
 * path, the on-wire format is fp32 (allreduce-friendly); we cast at the
 * boundary on the receive side. */
void compute_delta(const Parameter& p, const std::vector<float>& theta_now,
                   std::vector<float>& delta_out) {
    delta_out.resize(p.num_elements);
    for (size_t i = 0; i < p.num_elements; ++i) {
        delta_out[i] = theta_now[i] - p.theta_anchor[i];
    }
}

/* Top-k sparsification with error feedback. Keep the K largest |Δθ_i|
 * entries; the rest go into the error-feedback buffer for next outer step.
 *
 * Compressed payload is encoded as (index, value) pairs in two flat
 * arrays. For tc_allreduce we send a dense fp32 vector — the all-reduce
 * sums per-rank top-k contributions. The receive side averages and the
 * outer-optimizer absorbs the result.
 *
 * For simplicity v1: top-k still sends a *dense* fp32 vector after
 * masking; the bandwidth savings come from a separate "sparse_pack" path
 * not yet wired here. v2 wires the sparse pack into tc_allreduce. */
void compute_delta_topk(Parameter& p, const std::vector<float>& theta_now,
                        std::vector<float>& delta_out, float keep_fraction) {
    delta_out.resize(p.num_elements);
    if (p.error_feedback.size() != p.num_elements) {
        p.error_feedback.assign(p.num_elements, 0.0f);
    }
    /* Δθ + error_feedback (carry-over residual from prior outer step). */
    std::vector<float> magnitude(p.num_elements);
    for (size_t i = 0; i < p.num_elements; ++i) {
        delta_out[i] = theta_now[i] - p.theta_anchor[i] + p.error_feedback[i];
        magnitude[i] = std::fabs(delta_out[i]);
    }
    /* Find the |Δ| threshold for top-K. Approximate via nth_element on a
     * copy — exact within rounding. */
    const size_t K = std::max<size_t>(1, (size_t)(keep_fraction * p.num_elements));
    std::vector<float> mag_copy = magnitude;
    std::nth_element(mag_copy.begin(), mag_copy.begin() + (mag_copy.size() - K),
                     mag_copy.end());
    const float thresh = mag_copy[mag_copy.size() - K];

    /* Mask Δθ; entries below threshold contribute to error_feedback. */
    for (size_t i = 0; i < p.num_elements; ++i) {
        if (magnitude[i] >= thresh) {
            p.error_feedback[i] = 0.0f;
        } else {
            p.error_feedback[i] = delta_out[i];   /* carry over for next time */
            delta_out[i] = 0.0f;
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------------
 * Outer-optimizer step
 * ------------------------------------------------------------------------ */

namespace {

void apply_outer_optimizer(Parameter& p, const std::vector<float>& delta_avg,
                           const tc_diloco_config& cfg) {
    /* delta_avg has been all-reduced and averaged across ranks. */
    if (p.outer_momentum.size() != p.num_elements) {
        p.outer_momentum.assign(p.num_elements, 0.0f);
    }
    const float lr = cfg.outer_lr;
    const float mu = cfg.outer_momentum;

    if (cfg.outer_optimizer == TC_DILOCO_OUTER_SGD) {
        for (size_t i = 0; i < p.num_elements; ++i) {
            p.theta_anchor[i] += lr * delta_avg[i];
        }
    } else if (cfg.outer_optimizer == TC_DILOCO_OUTER_NESTEROV) {
        /* Standard Nesterov on the Δθ "gradient" (treating Δ̄θ as -∇L). */
        for (size_t i = 0; i < p.num_elements; ++i) {
            const float prev = p.outer_momentum[i];
            const float v = mu * prev + delta_avg[i];
            p.outer_momentum[i] = v;
            /* look-ahead step */
            p.theta_anchor[i] += lr * (delta_avg[i] + mu * v);
        }
    } else if (cfg.outer_optimizer == TC_DILOCO_OUTER_ADAM) {
        /* Outer Adam: m = β1 m + (1 - β1) Δ; v = β2 v + (1 - β2) Δ^2;
         * θ ← θ + lr * m̂ / (√v̂ + eps). We treat Δ as +∇ here (param goes
         * toward Δ̄θ, not away from it). */
        std::vector<float>& m = p.outer_momentum;
        /* second-moment piggybacks in the back half of momentum if we
         * resize it to 2× length. */
        if (m.size() < 2 * p.num_elements) m.resize(2 * p.num_elements, 0.0f);
        float* m1 = m.data();
        float* m2 = m.data() + p.num_elements;
        const float b1 = cfg.outer_momentum > 0.0f ? cfg.outer_momentum : 0.9f;
        const float b2 = cfg.outer_beta2 > 0.0f ? cfg.outer_beta2 : 0.999f;
        const float eps = cfg.outer_eps > 0.0f ? cfg.outer_eps : 1e-8f;
        for (size_t i = 0; i < p.num_elements; ++i) {
            m1[i] = b1 * m1[i] + (1.0f - b1) * delta_avg[i];
            m2[i] = b2 * m2[i] + (1.0f - b2) * delta_avg[i] * delta_avg[i];
            p.theta_anchor[i] += lr * m1[i] / (std::sqrt(m2[i]) + eps);
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------ */

extern "C" tc_status_t tc_diloco_init(tc_dist_ctx* dist_ctx,
                                       const tc_diloco_config* cfg,
                                       tc_diloco_ctx** out) {
    if (!cfg || !out) return TC_ERR_INVALID_ARG;
    *out = nullptr;
    if (cfg->inner_steps <= 0) return TC_ERR_INVALID_ARG;
    if (cfg->outer_optimizer != TC_DILOCO_OUTER_SGD &&
        cfg->outer_optimizer != TC_DILOCO_OUTER_NESTEROV &&
        cfg->outer_optimizer != TC_DILOCO_OUTER_ADAM) {
        return TC_ERR_INVALID_ARG;
    }
    if (cfg->tolerate_dropouts) {
        /* Elastic membership and checkpoint-backed rank recovery are not yet
         * implemented. Fail closed instead of silently weakening the flag. */
        return TC_ERR_INVALID_ARG;
    }
    if (cfg->compress != TC_DILOCO_COMPRESS_NONE &&
        cfg->compress != TC_DILOCO_COMPRESS_FP16 &&
        cfg->compress != TC_DILOCO_COMPRESS_TOPK_1PCT &&
        cfg->compress != TC_DILOCO_COMPRESS_TOPK_01PCT) {
        /* fp8 / lowrank / signsgd not yet implemented */
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    auto* d = new (std::nothrow) tc_diloco_ctx();
    if (!d) return TC_ERR_ALLOC;
    d->dist = dist_ctx;
    d->cfg = *cfg;
    *out = d;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_finalize(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    if (d->outer_thread.joinable()) d->outer_thread.join();
    {
        std::lock_guard<std::mutex> lk(d->outer_mutex);
        if (d->outer_state == TC_DILOCO_ASYNC_FAILED) {
            return d->outer_status;
        }
        if (d->outer_state != TC_DILOCO_ASYNC_IDLE) {
            return TC_ERR_BUSY;
        }
    }
    delete d;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_add_parameter(tc_diloco_ctx* d,
                                                const char* name,
                                                tc_buffer* theta_local,
                                                size_t num_elements,
                                                tc_dtype_t dtype) {
    if (!d || !theta_local || num_elements == 0) return TC_ERR_INVALID_ARG;
    if (dtype != TC_DTYPE_F16 && dtype != TC_DTYPE_F32) return TC_ERR_UNSUPPORTED_DTYPE;
    const size_t element_size = tc_dtype_size(dtype);
    if (element_size == 0 ||
        num_elements > std::numeric_limits<size_t>::max() / element_size ||
        num_elements * element_size > tc_buffer_size(theta_local)) {
        return TC_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lk(d->outer_mutex);
    if (d->outer_state != TC_DILOCO_ASYNC_IDLE) return TC_ERR_BUSY;
    try {
        Parameter p;
        p.name = name ? name : "";
        p.theta_local = theta_local;
        p.num_elements = num_elements;
        p.dtype = dtype;
        /* Snapshot θ_anchor from the current θ_local at registration time. */
        if (!theta_to_fp32(p, p.theta_anchor)) return TC_ERR_INVALID_ARG;
        d->params.push_back(std::move(p));
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_step(tc_diloco_ctx* d,
                                       bool* out_outer_step_pending) {
    if (!d || !out_outer_step_pending) return TC_ERR_INVALID_ARG;
    d->inner_steps_total.fetch_add(1, std::memory_order_relaxed);
    d->inner_steps_since_outer += 1;
    *out_outer_step_pending = (d->inner_steps_since_outer >= d->cfg.inner_steps);
    return TC_OK;
}

namespace {

struct OuterMetrics {
    double seconds = 0.0;
    double bytes = 0.0;
};

tc_status_t capture_snapshots(
    const std::vector<Parameter>& params,
    std::vector<std::vector<float>>& snapshots) {
    try {
        snapshots.clear();
        snapshots.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            if (!theta_to_fp32(params[i], snapshots[i])) {
                snapshots.clear();
                return TC_ERR_INVALID_ARG;
            }
        }
        return TC_OK;
    } catch (const std::bad_alloc&) {
        snapshots.clear();
        return TC_ERR_ALLOC;
    } catch (...) {
        snapshots.clear();
        return TC_ERR_INTERNAL;
    }
}

/* Compute a complete outer round against immutable snapshots and private
 * anchor/optimizer/error-feedback state. The worker never maps or writes a
 * registered theta_local buffer. */
tc_status_t compute_outer_round(
    tc_dist_ctx* dist,
    const tc_diloco_config& cfg,
    std::vector<Parameter>& params,
    const std::vector<std::vector<float>>& snapshots,
    OuterMetrics& metrics) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    size_t total_bytes_sent = 0;

    if (snapshots.size() != params.size()) return TC_ERR_INVALID_ARG;

    /* For each registered parameter:
     *   1. Read the immutable fp32 round snapshot.
     *   2. Compute Δθ against the private anchor copy.
     *   3. all-reduce Δθ across the dist_ctx.
     *   4. Update only the private anchor/optimizer state. */
    std::vector<float> delta;

    for (size_t param_index = 0; param_index < params.size(); ++param_index) {
        auto& p = params[param_index];
        const std::vector<float>& theta_now = snapshots[param_index];
        if (theta_now.size() != p.num_elements) return TC_ERR_INVALID_ARG;

        switch (cfg.compress) {
        case TC_DILOCO_COMPRESS_NONE:
        case TC_DILOCO_COMPRESS_FP16:
            compute_delta(p, theta_now, delta);
            break;
        case TC_DILOCO_COMPRESS_TOPK_1PCT:
            compute_delta_topk(p, theta_now, delta, 0.01f);
            break;
        case TC_DILOCO_COMPRESS_TOPK_01PCT:
            compute_delta_topk(p, theta_now, delta, 0.001f);
            break;
        default:
            return TC_ERR_UNSUPPORTED_DTYPE;
        }

        /* Multi-rank: cross-site all-reduce-AVG of Δθ. Dense compression
         * modes use fp32 AVG allreduce; TOPK over portable GLOO takes the
         * sparse hook below so the transport ships only non-zero
         * (idx, fp16-val) pairs. */
        const int world = dist ? tc_dist_world_size(dist) : 1;
        if (world > 1) {
            tc_context* parent_ctx = tc_dist_get_context(dist);
            if (!parent_ctx) return TC_ERR_INTERNAL;
            const int rank = tc_dist_rank(dist);

            /* Sparse-compressed path: when compression is TOPK and the
             * underlying transport is GLOO, ship only the (idx, fp16-val)
             * pairs instead of the dense fp32 delta. Bandwidth dropped from
             * 4N bytes per rank to ~8 * keep_fraction * N bytes — for top-k
             * 0.1% on a 70B model that's 140 MB instead of 280 GB on the
             * wire per outer step. */
            GlooState* gloo_state = tc_dist_get_gloo_state(dist);
            const bool sparse_path =
                gloo_state &&
                (cfg.compress == TC_DILOCO_COMPRESS_TOPK_1PCT ||
                 cfg.compress == TC_DILOCO_COMPRESS_TOPK_01PCT);

            if (sparse_path) {
                const float keep = (cfg.compress == TC_DILOCO_COMPRESS_TOPK_1PCT)
                                       ? 0.01f : 0.001f;
                const size_t pack_cap = tc_diloco_sparse_packed_size(p.num_elements, keep);
                std::vector<uint8_t> payload(pack_cap);
                /* Note: compute_delta_topk already ran and zeroed sub-threshold
                 * entries; pack here just emits the non-zero (idx, val) pairs.
                 * Calling pack again on the masked delta is correct — the
                 * threshold logic is idempotent on an already-sparse input. */
                const size_t written = tc_diloco_sparse_pack(
                    delta.data(), p.num_elements, keep, payload.data(), pack_cap);
                if (written == 0) return TC_ERR_INTERNAL;

                std::vector<float> merged(p.num_elements, 0.0f);
                const int rc = tc_gloo_sparse_allreduce(
                    gloo_state, world, rank,
                    payload.data(), written, merged.data(), p.num_elements);
                if (rc != 0) return TC_ERR_INTERNAL;

                /* AVG: divide by world_size. */
                const float inv = 1.0f / (float)world;
                for (size_t i = 0; i < p.num_elements; ++i) {
                    delta[i] = merged[i] * inv;
                }
                total_bytes_sent += written;
            } else {
                /* Dense fp32 allreduce — used for NONE/FP16 compression
                 * and for transports without sparse support. */
                const size_t bytes = p.num_elements * sizeof(float);
                tc_buffer* delta_buf = nullptr;
                if (tc_buffer_alloc(parent_ctx, bytes, &delta_buf) != TC_OK) {
                    return TC_ERR_ALLOC;
                }
                void* mp = nullptr;
                if (tc_buffer_map(delta_buf, &mp) != TC_OK) {
                    tc_buffer_free(parent_ctx, delta_buf);
                    return TC_ERR_INTERNAL;
                }
                std::memcpy(mp, delta.data(), bytes);
                tc_status_t s = tc_allreduce(dist, delta_buf, p.num_elements,
                                              TC_DTYPE_F32, TC_REDUCE_AVG);
                if (s == TC_OK) {
                    std::memcpy(delta.data(), mp, bytes);
                    total_bytes_sent += bytes;
                }
                tc_buffer_free(parent_ctx, delta_buf);
                if (s != TC_OK) return s;
            }
        }

        /* Outer-optimizer step on θ_anchor. */
        apply_outer_optimizer(p, delta, cfg);
    }

    metrics.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    metrics.bytes = (double)total_bytes_sent;
    return TC_OK;
}

/* Apply private round state on the caller thread. For async commit, preserve
 * inner-loop work performed after the snapshot:
 *
 *   committed_live = new_anchor + (current_live - round_snapshot)
 *
 * The anchor remains new_anchor, so the preserved local drift contributes to
 * the next outer delta. Prepare every output before writing and roll back any
 * earlier parameter if a later write fails. */
tc_status_t commit_round(
    std::vector<Parameter>& live_params,
    std::vector<Parameter>& next_params,
    const std::vector<std::vector<float>>& snapshots,
    bool rebase_overlap) {
    if (live_params.size() != next_params.size() ||
        snapshots.size() != live_params.size()) {
        return TC_ERR_INVALID_ARG;
    }

    try {
        std::vector<std::vector<float>> live_before(live_params.size());
        std::vector<std::vector<float>> outputs(live_params.size());
        for (size_t i = 0; i < live_params.size(); ++i) {
            if (!theta_to_fp32(live_params[i], live_before[i])) {
                return TC_ERR_INVALID_ARG;
            }
            if (next_params[i].theta_anchor.size() != live_params[i].num_elements ||
                snapshots[i].size() != live_params[i].num_elements) {
                return TC_ERR_INVALID_ARG;
            }
            outputs[i] = next_params[i].theta_anchor;
            if (rebase_overlap) {
                for (size_t j = 0; j < live_params[i].num_elements; ++j) {
                    outputs[i][j] += live_before[i][j] - snapshots[i][j];
                }
            }
        }

        size_t written = 0;
        for (; written < live_params.size(); ++written) {
            if (!theta_from_fp32(live_params[written], outputs[written])) {
                for (size_t rollback = 0; rollback < written; ++rollback) {
                    (void)theta_from_fp32(live_params[rollback], live_before[rollback]);
                }
                return TC_ERR_INVALID_ARG;
            }
        }

        for (size_t i = 0; i < live_params.size(); ++i) {
            live_params[i].theta_anchor = std::move(next_params[i].theta_anchor);
            live_params[i].outer_momentum = std::move(next_params[i].outer_momentum);
            live_params[i].error_feedback = std::move(next_params[i].error_feedback);
        }
        return TC_OK;
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
}

void publish_committed_metrics(tc_diloco_ctx* d, const OuterMetrics& metrics) {
    d->last_outer_seconds.store(metrics.seconds, std::memory_order_relaxed);
    d->last_outer_bytes.store(metrics.bytes, std::memory_order_relaxed);
    d->outer_steps_total.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

extern "C" tc_status_t tc_diloco_apply_outer(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;

    if (!d->cfg.async_overlap) {
        std::vector<std::vector<float>> snapshots;
        tc_status_t status = capture_snapshots(d->params, snapshots);
        if (status != TC_OK) return status;
        try {
            std::vector<Parameter> next_params = d->params;
            OuterMetrics metrics;
            status = compute_outer_round(d->dist, d->cfg, next_params,
                                         snapshots, metrics);
            if (status != TC_OK) return status;
            status = commit_round(d->params, next_params, snapshots, false);
            if (status != TC_OK) return status;
            publish_committed_metrics(d, metrics);
            d->inner_steps_since_outer = 0;
            return TC_OK;
        } catch (const std::bad_alloc&) {
            return TC_ERR_ALLOC;
        } catch (...) {
            return TC_ERR_INTERNAL;
        }
    }

    std::unique_lock<std::mutex> lk(d->outer_mutex);
    if (d->outer_state == TC_DILOCO_ASYNC_FAILED) return d->outer_status;
    if (d->outer_state != TC_DILOCO_ASYNC_IDLE) return TC_ERR_BUSY;
    if (d->outer_thread.joinable()) {
        lk.unlock();
        d->outer_thread.join();
        lk.lock();
    }

    std::vector<std::vector<float>> snapshots;
    tc_status_t status = capture_snapshots(d->params, snapshots);
    if (status != TC_OK) return status;

    try {
        std::vector<Parameter> round_params = d->params;
        const uint64_t round_id = d->next_round_id++;
        d->active_round_id = round_id;
        d->outer_status = TC_OK;
        d->outer_state = TC_DILOCO_ASYNC_RUNNING;

        d->outer_thread = std::thread(
            [d, round_id, round_params = std::move(round_params),
             snapshots = std::move(snapshots)]() mutable {
                OuterMetrics metrics;
                tc_status_t worker_status = TC_OK;
                try {
                    worker_status = compute_outer_round(
                        d->dist, d->cfg, round_params, snapshots, metrics);
                } catch (const std::bad_alloc&) {
                    worker_status = TC_ERR_ALLOC;
                } catch (...) {
                    worker_status = TC_ERR_INTERNAL;
                }
                {
                    std::lock_guard<std::mutex> worker_lk(d->outer_mutex);
                    d->outer_status = worker_status;
                    d->active_round_id = round_id;
                    if (worker_status == TC_OK) {
                        d->pending_params = std::move(round_params);
                        d->pending_snapshots = std::move(snapshots);
                        d->pending_seconds = metrics.seconds;
                        d->pending_bytes = metrics.bytes;
                        d->outer_state = TC_DILOCO_ASYNC_READY;
                    } else {
                        d->pending_params.clear();
                        d->pending_snapshots.clear();
                        d->outer_state = TC_DILOCO_ASYNC_FAILED;
                    }
                }
                d->outer_cv.notify_all();
            });
    } catch (const std::bad_alloc&) {
        d->outer_state = TC_DILOCO_ASYNC_IDLE;
        d->outer_status = TC_OK;
        return TC_ERR_ALLOC;
    } catch (...) {
        d->outer_state = TC_DILOCO_ASYNC_IDLE;
        d->outer_status = TC_OK;
        return TC_ERR_INTERNAL;
    }

    d->inner_steps_since_outer = 0;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_async_poll(
    const tc_diloco_ctx* d,
    tc_diloco_async_state_t* out_state,
    tc_status_t* out_worker_status,
    uint64_t* out_round_id) {
    if (!d || !out_state || !out_worker_status || !out_round_id) {
        return TC_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lk(d->outer_mutex);
    *out_state = d->outer_state;
    *out_worker_status = d->outer_status;
    *out_round_id = d->active_round_id;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_async_wait(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;
    std::unique_lock<std::mutex> lk(d->outer_mutex);
    d->outer_cv.wait(lk, [d] {
        return d->outer_state != TC_DILOCO_ASYNC_RUNNING;
    });
    return d->outer_state == TC_DILOCO_ASYNC_FAILED ? d->outer_status : TC_OK;
}

extern "C" tc_status_t tc_diloco_async_commit(tc_diloco_ctx* d) {
    if (!d) return TC_ERR_INVALID_ARG;

    std::unique_lock<std::mutex> lk(d->outer_mutex);
    if (d->outer_state == TC_DILOCO_ASYNC_RUNNING) return TC_ERR_BUSY;
    if (d->outer_state == TC_DILOCO_ASYNC_IDLE) return TC_OK;

    lk.unlock();
    if (d->outer_thread.joinable()) d->outer_thread.join();
    lk.lock();

    if (d->outer_state == TC_DILOCO_ASYNC_FAILED) {
        const tc_status_t failure = d->outer_status;
        d->pending_seconds = 0.0;
        d->pending_bytes = 0.0;
        d->outer_status = TC_OK;
        d->outer_state = TC_DILOCO_ASYNC_IDLE;
        return failure;
    }
    if (d->outer_state != TC_DILOCO_ASYNC_READY) return TC_ERR_INTERNAL;

    OuterMetrics metrics;
    metrics.seconds = d->pending_seconds;
    metrics.bytes = d->pending_bytes;
    const tc_status_t status = commit_round(
        d->params, d->pending_params, d->pending_snapshots, true);
    if (status == TC_OK) publish_committed_metrics(d, metrics);

    d->pending_params.clear();
    d->pending_snapshots.clear();
    d->pending_seconds = 0.0;
    d->pending_bytes = 0.0;
    d->outer_status = TC_OK;
    d->outer_state = TC_DILOCO_ASYNC_IDLE;
    return status;
}

extern "C" tc_status_t tc_diloco_state_set_epochs(
    tc_diloco_ctx* d,
    uint64_t topology_epoch,
    uint64_t membership_epoch) {
    if (!d) return TC_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lk(d->outer_mutex);
    if (d->outer_state != TC_DILOCO_ASYNC_IDLE) return TC_ERR_BUSY;
    d->topology_epoch = topology_epoch;
    d->membership_epoch = membership_epoch;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_state_get_epochs(
    const tc_diloco_ctx* d,
    uint64_t* out_topology_epoch,
    uint64_t* out_membership_epoch) {
    if (!d || !out_topology_epoch || !out_membership_epoch) {
        return TC_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lk(d->outer_mutex);
    *out_topology_epoch = d->topology_epoch;
    *out_membership_epoch = d->membership_epoch;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_state_size(
    const tc_diloco_ctx* d,
    uint32_t requested_abi_version,
    size_t* out_size) {
    if (!d || !out_size) return TC_ERR_INVALID_ARG;
    if (!state_version_supported(requested_abi_version)) {
        return TC_ERR_ABI_MISMATCH;
    }

    std::lock_guard<std::mutex> lk(d->outer_mutex);
    size_t size = 0;
    const tc_status_t status =
        state_blob_size_locked(d, requested_abi_version, size);
    if (status != TC_OK) return status;
    *out_size = size;
    return TC_OK;
}

extern "C" tc_status_t tc_diloco_state_serialize(
    const tc_diloco_ctx* d,
    uint32_t requested_abi_version,
    void* out_data,
    size_t out_size,
    size_t* out_written) {
    if (!d || !out_data || !out_written) return TC_ERR_INVALID_ARG;
    if (!state_version_supported(requested_abi_version)) {
        return TC_ERR_ABI_MISMATCH;
    }

    std::lock_guard<std::mutex> lk(d->outer_mutex);
    size_t written = 0;
    const tc_status_t status = write_state_blob_locked(
        d, requested_abi_version, out_data, out_size, written);
    if (status == TC_OK) *out_written = written;
    return status;
}

extern "C" tc_status_t tc_diloco_state_deserialize(
    tc_diloco_ctx* d,
    uint32_t requested_abi_version,
    const void* data,
    size_t data_size) {
    if (!d || !data) return TC_ERR_INVALID_ARG;
    if (!state_version_supported(requested_abi_version)) {
        return TC_ERR_ABI_MISMATCH;
    }

    std::lock_guard<std::mutex> lk(d->outer_mutex);
    if (d->outer_state != TC_DILOCO_ASYNC_IDLE ||
        d->outer_thread.joinable()) {
        return TC_ERR_BUSY;
    }

    try {
        SerializedState saved;
        uint32_t blob_abi_version = 0;
        const tc_status_t parse = read_state_blob(
            data, data_size, requested_abi_version, saved, blob_abi_version);
        if (parse != TC_OK) return parse;
        const tc_status_t validation = validate_serialized_state(d, saved);
        if (validation != TC_OK) return validation;

        /* Prepare every allocation before mutating d so malformed or
         * allocation-failed restores leave the live context untouched. */
        std::vector<Parameter> restored_params = d->params;
        for (size_t i = 0; i < restored_params.size(); ++i) {
            restored_params[i].theta_anchor = saved.params[i].theta_anchor;
            restored_params[i].outer_momentum = saved.params[i].outer_momentum;
            restored_params[i].error_feedback = saved.params[i].error_feedback;
        }

        std::vector<Parameter> restored_pending;
        std::vector<std::vector<float>> restored_snapshots;
        if (saved.outer_state == TC_DILOCO_ASYNC_READY) {
            restored_pending = restored_params;
            restored_snapshots.resize(restored_params.size());
            for (size_t i = 0; i < restored_pending.size(); ++i) {
                restored_pending[i].theta_anchor =
                    saved.params[i].pending_theta_anchor;
                restored_pending[i].outer_momentum =
                    saved.params[i].pending_outer_momentum;
                restored_pending[i].error_feedback =
                    saved.params[i].pending_error_feedback;
                restored_snapshots[i] = saved.params[i].pending_snapshot;
            }
        }

        d->params = std::move(restored_params);
        d->pending_params = std::move(restored_pending);
        d->pending_snapshots = std::move(restored_snapshots);
        d->topology_epoch = saved.topology_epoch;
        d->membership_epoch = saved.membership_epoch;
        d->inner_steps_total.store(
            saved.inner_steps_total, std::memory_order_relaxed);
        d->outer_steps_total.store(
            saved.outer_steps_total, std::memory_order_relaxed);
        d->inner_steps_since_outer = (int)saved.inner_steps_since_outer;
        d->last_outer_seconds.store(
            bits_double(saved.last_outer_seconds), std::memory_order_relaxed);
        d->last_outer_bytes.store(
            bits_double(saved.last_outer_bytes), std::memory_order_relaxed);
        d->next_round_id = saved.next_round_id;
        d->active_round_id = saved.active_round_id;
        d->outer_state = (tc_diloco_async_state_t)saved.outer_state;
        d->outer_status = (tc_status_t)(int32_t)saved.outer_status;
        d->pending_seconds = bits_double(saved.pending_seconds);
        d->pending_bytes = bits_double(saved.pending_bytes);
        return TC_OK;
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
}

extern "C" tc_status_t tc_diloco_capability_query(
    const tc_diloco_ctx* d,
    uint32_t requested_abi_version,
    tc_diloco_capabilities* out,
    size_t out_size) {
    if (!out) return TC_ERR_INVALID_ARG;
    if (requested_abi_version != TC_DILOCO_CAPABILITIES_ABI_VERSION_1) {
        return TC_ERR_ABI_MISMATCH;
    }
    if (out_size < TC_DILOCO_CAPABILITIES_V1_MIN_SIZE) {
        return TC_ERR_INVALID_ARG;
    }

    /* Exactly the configurations tc_diloco_init accepts. */
    const uint64_t optimizers_supported =
        TC_DILOCO_OUTER_OPTIMIZER_BIT(TC_DILOCO_OUTER_SGD) |
        TC_DILOCO_OUTER_OPTIMIZER_BIT(TC_DILOCO_OUTER_NESTEROV) |
        TC_DILOCO_OUTER_OPTIMIZER_BIT(TC_DILOCO_OUTER_ADAM);
    const uint64_t topk_modes =
        TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_TOPK_1PCT) |
        TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_TOPK_01PCT);
    const uint64_t compress_supported =
        TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_NONE) |
        TC_DILOCO_COMPRESS_BIT(TC_DILOCO_COMPRESS_FP16) | topk_modes;

    /* Every outer optimizer keeps all of its moments in Parameter::
     * outer_momentum (Adam packs m and v into one 2N vector) and the outer
     * step count lives in outer_steps_total. Both are in the blob, and no
     * outer update depends on state outside them, so all three round-trip
     * exactly. Top-k error feedback is likewise carried per parameter. */
    const uint64_t optimizers_serializable = optimizers_supported;
    const uint64_t compress_serializable = compress_supported;

    /* Sparse payloads require a Gloo transport and more than one rank; every
     * other accepted mode places dense fp32 on the wire. Without a bound
     * context we cannot promise a transport, so report the dense answer. */
    uint64_t sparse_wire = 0;
    uint64_t dense_fp32_wire = compress_supported;
    if (d) {
        std::lock_guard<std::mutex> lk(d->outer_mutex);
        tc_dist_ctx* dist = d->dist;
        const int world = dist ? tc_dist_world_size(dist) : 1;
        if (dist && world > 1 && tc_dist_get_gloo_state(dist) != nullptr) {
            sparse_wire = topk_modes;
            dense_fp32_wire = compress_supported & ~topk_modes;
        }
    }

    tc_diloco_capabilities capabilities{};
    capabilities.struct_size = (uint32_t)sizeof(capabilities);
    capabilities.abi_version = TC_DILOCO_CAPABILITIES_ABI_VERSION_1;
    capabilities.runtime_version_major = TENSORCORE_VERSION_MAJOR;
    capabilities.runtime_version_minor = TENSORCORE_VERSION_MINOR;
    capabilities.runtime_version_patch = TENSORCORE_VERSION_PATCH;
    capabilities.state_abi_version_min = TC_DILOCO_STATE_ABI_VERSION_MIN;
    capabilities.state_abi_version_max = TC_DILOCO_STATE_ABI_VERSION_CURRENT;
    capabilities.state_abi_version_current =
        TC_DILOCO_STATE_ABI_VERSION_CURRENT;
    capabilities.state_header_size =
        state_header_size(TC_DILOCO_STATE_ABI_VERSION_CURRENT);
    capabilities.state_payload_digest_bytes =
        TC_DILOCO_STATE_ABI_VERSION_CURRENT == TC_DILOCO_STATE_ABI_VERSION_2
            ? (uint32_t)kStateDigestBytes
            : 0u;
    capabilities.outer_optimizer_supported_mask = optimizers_supported;
    capabilities.outer_optimizer_serializable_mask = optimizers_serializable;
    capabilities.compress_supported_mask = compress_supported;
    capabilities.compress_serializable_mask = compress_serializable;
    capabilities.compress_sparse_wire_mask = sparse_wire;
    capabilities.compress_dense_fp32_wire_mask = dense_fp32_wire;
    capabilities.state_feature_mask =
        TC_DILOCO_STATE_FEATURE_OUTER_ANCHOR |
        TC_DILOCO_STATE_FEATURE_OUTER_MOMENTS |
        TC_DILOCO_STATE_FEATURE_ERROR_FEEDBACK |
        TC_DILOCO_STATE_FEATURE_COUNTERS |
        TC_DILOCO_STATE_FEATURE_PENDING_ROUND |
        TC_DILOCO_STATE_FEATURE_TOPOLOGY_EPOCH |
        TC_DILOCO_STATE_FEATURE_MEMBERSHIP_EPOCH |
        TC_DILOCO_STATE_FEATURE_PAYLOAD_SHA256 |
        TC_DILOCO_STATE_FEATURE_DETERMINISTIC_LAYOUT |
        TC_DILOCO_STATE_FEATURE_ATOMIC_RESTORE;
    capabilities.async_overlap_supported = 1;
    capabilities.async_state_serializable = 1;
    /* tc_diloco_init still rejects tolerate_dropouts; elastic membership and
     * rank recovery are not implemented. */
    capabilities.tolerate_dropouts_supported = 0;

    const size_t copy_size = std::min(out_size, sizeof(capabilities));
    std::memcpy(out, &capabilities, copy_size);
    return TC_OK;
}

extern "C" uint64_t tc_diloco_outer_steps_completed(const tc_diloco_ctx* d) {
    return d ? d->outer_steps_total.load(std::memory_order_relaxed) : 0;
}

extern "C" uint64_t tc_diloco_inner_steps_completed(const tc_diloco_ctx* d) {
    return d ? d->inner_steps_total.load(std::memory_order_relaxed) : 0;
}

extern "C" double tc_diloco_last_outer_step_seconds(const tc_diloco_ctx* d) {
    return d ? d->last_outer_seconds.load(std::memory_order_relaxed) : 0.0;
}

extern "C" double tc_diloco_last_outer_bytes_sent(const tc_diloco_ctx* d) {
    return d ? d->last_outer_bytes.load(std::memory_order_relaxed) : 0.0;
}
