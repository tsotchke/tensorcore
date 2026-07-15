#if defined(_WIN32)
#  define _CRT_RAND_S
#endif

#include "transport_auth_internal.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/random.h>
#  include <unistd.h>
#endif

namespace tc_transport_auth_internal {
namespace {

constexpr size_t kDigestBytes = 32;
constexpr size_t kNonceBytes = 32;
constexpr size_t kWirePrefixBytes = 80;
constexpr size_t kMaxKeys = 256;
constexpr size_t kMaxSecretBytes = 4096;
constexpr uint8_t kWireVersion = 1;
constexpr uint8_t kClientHello = 1;
constexpr uint8_t kServerHello = 2;
constexpr uint8_t kClientFinish = 3;
constexpr uint8_t kMagic[4] = {'T', 'C', 'A', 'H'};
constexpr char kDomain[] = "tensorcore-transport-auth-v1";

void secure_zero(void* data, size_t bytes) noexcept {
    volatile uint8_t* p = static_cast<volatile uint8_t*>(data);
    while (bytes--) *p++ = 0;
}

uint32_t rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

uint32_t load_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

void store_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);
    p[3] = static_cast<uint8_t>(v);
}

void store_le16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

uint16_t load_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

void store_le64(uint8_t* p, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

uint64_t load_le64(const uint8_t* p) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const void* input, size_t bytes) {
        const uint8_t* p = static_cast<const uint8_t*>(input);
        total_bytes_ += bytes;
        while (bytes > 0) {
            const size_t n = std::min(bytes, block_.size() - block_used_);
            std::memcpy(block_.data() + block_used_, p, n);
            block_used_ += n;
            p += n;
            bytes -= n;
            if (block_used_ == block_.size()) {
                transform(block_.data());
                block_used_ = 0;
            }
        }
    }

    void finish(uint8_t out[kDigestBytes]) {
        const uint64_t total_bits = total_bytes_ * UINT64_C(8);
        block_[block_used_++] = 0x80;
        if (block_used_ > 56) {
            std::fill(block_.begin() + static_cast<ptrdiff_t>(block_used_),
                      block_.end(), uint8_t{0});
            transform(block_.data());
            block_used_ = 0;
        }
        std::fill(block_.begin() + static_cast<ptrdiff_t>(block_used_),
                  block_.begin() + 56, uint8_t{0});
        for (unsigned i = 0; i < 8; ++i) {
            block_[56 + i] = static_cast<uint8_t>(total_bits >> (56 - 8 * i));
        }
        transform(block_.data());
        for (size_t i = 0; i < state_.size(); ++i) store_be32(out + 4 * i, state_[i]);
        secure_zero(block_.data(), block_.size());
        secure_zero(state_.data(), state_.size() * sizeof(state_[0]));
    }

private:
    void reset() {
        state_ = {UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
                  UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
                  UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
                  UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)};
        block_.fill(0);
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
        for (size_t i = 0; i < 16; ++i) w[i] = load_be32(block + 4 * i);
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
        secure_zero(w, sizeof(w));
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    size_t block_used_ = 0;
    uint64_t total_bytes_ = 0;
};

void hmac_sha256(const uint8_t* key, size_t key_bytes,
                 const std::vector<std::pair<const void*, size_t>>& pieces,
                 uint8_t out[kDigestBytes]) {
    uint8_t normalized[64] = {};
    if (key_bytes > sizeof(normalized)) {
        Sha256 hash;
        hash.update(key, key_bytes);
        hash.finish(normalized);
    } else {
        std::memcpy(normalized, key, key_bytes);
    }
    uint8_t inner_pad[64];
    uint8_t outer_pad[64];
    for (size_t i = 0; i < 64; ++i) {
        inner_pad[i] = static_cast<uint8_t>(normalized[i] ^ 0x36u);
        outer_pad[i] = static_cast<uint8_t>(normalized[i] ^ 0x5cu);
    }
    uint8_t inner[kDigestBytes];
    Sha256 inner_hash;
    inner_hash.update(inner_pad, sizeof(inner_pad));
    for (const auto& piece : pieces) {
        if (piece.second != 0) inner_hash.update(piece.first, piece.second);
    }
    inner_hash.finish(inner);
    Sha256 outer_hash;
    outer_hash.update(outer_pad, sizeof(outer_pad));
    outer_hash.update(inner, sizeof(inner));
    outer_hash.finish(out);
    secure_zero(normalized, sizeof(normalized));
    secure_zero(inner_pad, sizeof(inner_pad));
    secure_zero(outer_pad, sizeof(outer_pad));
    secure_zero(inner, sizeof(inner));
}

bool constant_time_equal(const uint8_t* a, const uint8_t* b, size_t bytes) {
    uint8_t difference = 0;
    for (size_t i = 0; i < bytes; ++i) difference |= static_cast<uint8_t>(a[i] ^ b[i]);
    return difference == 0;
}

bool random_bytes(uint8_t* out, size_t bytes) {
#if defined(_WIN32)
    size_t offset = 0;
    while (offset < bytes) {
        unsigned int word = 0;
        if (rand_s(&word) != 0) return false;
        const size_t n = std::min(bytes - offset, sizeof(word));
        std::memcpy(out + offset, &word, n);
        offset += n;
    }
    return true;
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    arc4random_buf(out, bytes);
    return true;
#elif defined(__linux__)
    size_t offset = 0;
    while (offset < bytes) {
        const ssize_t n = ::getrandom(out + offset, bytes - offset, 0);
        if (n > 0) { offset += static_cast<size_t>(n); continue; }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
    if (offset == bytes) return true;
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    while (offset < bytes) {
        const ssize_t n = ::read(fd, out + offset, bytes - offset);
        if (n > 0) { offset += static_cast<size_t>(n); continue; }
        if (n < 0 && errno == EINTR) continue;
        ::close(fd);
        return false;
    }
    ::close(fd);
    return true;
#else
    (void)out; (void)bytes;
    return false;
#endif
}

bool valid_identity(const char* identity) {
    if (!identity) return false;
    const size_t n = std::strlen(identity);
    if (n == 0 || n > TC_TRANSPORT_AUTH_MAX_IDENTITY_BYTES) return false;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(identity[i]);
        if (c < 0x21 || c > 0x7e) return false;
    }
    return true;
}

struct Message {
    uint8_t kind = 0;
    uint64_t key_id = 0;
    std::array<uint8_t, kNonceBytes> nonce{};
    std::array<uint8_t, kNonceBytes> peer_nonce{};
    std::string identity;
    std::array<uint8_t, kDigestBytes> mac{};
};

void encode_prefix(const Message& message, uint8_t out[kWirePrefixBytes]) {
    std::memset(out, 0, kWirePrefixBytes);
    std::memcpy(out, kMagic, sizeof(kMagic));
    out[4] = kWireVersion;
    out[5] = message.kind;
    store_le16(out + 6, static_cast<uint16_t>(message.identity.size()));
    store_le64(out + 8, message.key_id);
    std::memcpy(out + 16, message.nonce.data(), message.nonce.size());
    std::memcpy(out + 48, message.peer_nonce.data(), message.peer_nonce.size());
}

void compute_message_mac(const Key& key, const Message& message,
                         const char* channel, const void* context,
                         size_t context_bytes, uint8_t out[kDigestBytes]) {
    uint8_t prefix[kWirePrefixBytes];
    encode_prefix(message, prefix);
    const size_t channel_bytes = channel ? std::strlen(channel) : 0;
    const std::vector<std::pair<const void*, size_t>> pieces = {
        {kDomain, sizeof(kDomain) - 1},
        {channel, channel_bytes},
        {context, context_bytes},
        {prefix, sizeof(prefix)},
        {message.identity.data(), message.identity.size()},
    };
    hmac_sha256(key.secret.data(), key.secret.size(), pieces, out);
    secure_zero(prefix, sizeof(prefix));
}

bool send_message(const Io& io, const Key& key, Message* message,
                  const char* channel, const void* context,
                  size_t context_bytes) {
    uint8_t prefix[kWirePrefixBytes];
    compute_message_mac(key, *message, channel, context, context_bytes,
                        message->mac.data());
    encode_prefix(*message, prefix);
    const bool ok = io.write_all(io.user, prefix, sizeof(prefix)) &&
                    (message->identity.empty() ||
                     io.write_all(io.user, message->identity.data(), message->identity.size())) &&
                    io.write_all(io.user, message->mac.data(), message->mac.size());
    secure_zero(prefix, sizeof(prefix));
    return ok;
}

bool receive_message(const Io& io, Message* message) {
    uint8_t prefix[kWirePrefixBytes];
    if (!io.read_all(io.user, prefix, sizeof(prefix))) return false;
    if (std::memcmp(prefix, kMagic, sizeof(kMagic)) != 0 ||
        prefix[4] != kWireVersion) {
        secure_zero(prefix, sizeof(prefix));
        return false;
    }
    const uint16_t identity_bytes = load_le16(prefix + 6);
    if (identity_bytes > TC_TRANSPORT_AUTH_MAX_IDENTITY_BYTES) {
        secure_zero(prefix, sizeof(prefix));
        return false;
    }
    message->kind = prefix[5];
    message->key_id = load_le64(prefix + 8);
    std::memcpy(message->nonce.data(), prefix + 16, message->nonce.size());
    std::memcpy(message->peer_nonce.data(), prefix + 48, message->peer_nonce.size());
    message->identity.resize(identity_bytes);
    const bool ok = (identity_bytes == 0 ||
                     io.read_all(io.user, message->identity.data(), identity_bytes)) &&
                    io.read_all(io.user, message->mac.data(), message->mac.size());
    secure_zero(prefix, sizeof(prefix));
    return ok;
}

bool verify_message(const Key& key, const Message& message,
                    const char* channel, const void* context,
                    size_t context_bytes) {
    uint8_t expected[kDigestBytes];
    compute_message_mac(key, message, channel, context, context_bytes, expected);
    const bool ok = constant_time_equal(expected, message.mac.data(), sizeof(expected));
    secure_zero(expected, sizeof(expected));
    return ok;
}

bool all_zero(const std::array<uint8_t, kNonceBytes>& nonce) {
    uint8_t value = 0;
    for (uint8_t byte : nonce) value |= byte;
    return value == 0;
}

bool crypto_self_test() {
    /* RFC 4231 test case 1: HMAC-SHA-256("Hi There", 0x0b * 20). */
    static constexpr uint8_t expected[kDigestBytes] = {
        0xb0,0x34,0x4c,0x61,0xd8,0xdb,0x38,0x53,
        0x5c,0xa8,0xaf,0xce,0xaf,0x0b,0xf1,0x2b,
        0x88,0x1d,0xc2,0x00,0xc9,0x83,0x3d,0xa7,
        0x26,0xe9,0x37,0x6c,0x2e,0x32,0xcf,0xf7,
    };
    uint8_t test_key[20];
    std::memset(test_key, 0x0b, sizeof(test_key));
    static constexpr char data[] = "Hi There";
    uint8_t actual[kDigestBytes];
    const std::vector<std::pair<const void*, size_t>> pieces = {
        {data, sizeof(data) - 1},
    };
    hmac_sha256(test_key, sizeof(test_key), pieces, actual);
    const bool ok = constant_time_equal(actual, expected, sizeof(expected));
    secure_zero(test_key, sizeof(test_key));
    secure_zero(actual, sizeof(actual));
    return ok;
}

}  // namespace

Keyring& Keyring::operator=(const Keyring& other) {
    if (this != &other) {
        Keyring copy(other);
        swap(copy);
    }
    return *this;
}

Keyring& Keyring::operator=(Keyring&& other) noexcept {
    if (this != &other) {
        clear_secrets();
        local_identity_ = std::move(other.local_identity_);
        active_key_id_ = other.active_key_id_;
        keys_ = std::move(other.keys_);
        other.active_key_id_ = 0;
    }
    return *this;
}

Keyring::~Keyring() { clear_secrets(); }

void Keyring::clear_secrets() noexcept {
    for (Key& key : keys_) {
        if (!key.secret.empty()) secure_zero(key.secret.data(), key.secret.size());
    }
    keys_.clear();
    active_key_id_ = 0;
}

void Keyring::swap(Keyring& other) noexcept {
    local_identity_.swap(other.local_identity_);
    std::swap(active_key_id_, other.active_key_id_);
    keys_.swap(other.keys_);
}

tc_status_t Keyring::copy_from(const tc_transport_auth_config* config,
                               Keyring* out) {
    if (!config || !out) return TC_ERR_INVALID_ARG;
    if (config->abi_version != TC_TRANSPORT_AUTH_ABI_VERSION_1)
        return TC_ERR_ABI_MISMATCH;
    if (!valid_identity(config->local_identity) || config->active_key_id == 0 ||
        !config->keys || config->key_count == 0 || config->key_count > kMaxKeys)
        return TC_ERR_INVALID_ARG;
    try {
        static const bool crypto_ok = crypto_self_test();
        if (!crypto_ok) return TC_ERR_INTERNAL;
        Keyring result;
        result.local_identity_ = config->local_identity;
        result.active_key_id_ = config->active_key_id;
        result.keys_.reserve(config->key_count);
        for (size_t i = 0; i < config->key_count; ++i) {
            const tc_transport_auth_key& source = config->keys[i];
            if (!valid_identity(source.identity) || source.key_id == 0 ||
                !source.secret || source.secret_bytes < TC_TRANSPORT_AUTH_MIN_SECRET_BYTES ||
                source.secret_bytes > kMaxSecretBytes) {
                return TC_ERR_INVALID_ARG;
            }
            for (const Key& existing : result.keys_) {
                if (existing.key_id == source.key_id &&
                    existing.identity == source.identity) {
                    return TC_ERR_INVALID_ARG;
                }
            }
            Key key;
            key.identity = source.identity;
            key.key_id = source.key_id;
            const uint8_t* secret = static_cast<const uint8_t*>(source.secret);
            key.secret.assign(secret, secret + source.secret_bytes);
            result.keys_.push_back(std::move(key));
        }
        if (!result.active_key()) return TC_ERR_INVALID_ARG;
        out->swap(result);
        return TC_OK;
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
}

const Key* Keyring::active_key() const {
    return find(local_identity_, active_key_id_);
}

const Key* Keyring::find(const std::string& identity, uint64_t key_id) const {
    for (const Key& key : keys_) {
        if (key.key_id == key_id && key.identity == identity) return &key;
    }
    return nullptr;
}

tc_status_t client_handshake(const Io& io, const Keyring& keyring,
                             const char* expected_peer_identity,
                             const char* channel, const void* context,
                             size_t context_bytes, PeerIdentity* out_peer) {
    try {
        if (!io.read_all || !io.write_all || !valid_identity(expected_peer_identity) ||
            !channel || !*channel || (!context && context_bytes != 0))
            return TC_ERR_INVALID_ARG;
        const Key* local_key = keyring.active_key();
        if (!local_key) return TC_ERR_INVALID_ARG;

    Message client;
    client.kind = kClientHello;
    client.key_id = local_key->key_id;
    client.identity = keyring.local_identity();
    if (!random_bytes(client.nonce.data(), client.nonce.size())) return TC_ERR_INTERNAL;
    if (!send_message(io, *local_key, &client, channel, context, context_bytes))
        return TC_ERR_AUTH;

    Message server;
    if (!receive_message(io, &server) || server.kind != kServerHello ||
        server.identity != expected_peer_identity ||
        !constant_time_equal(server.peer_nonce.data(), client.nonce.data(), kNonceBytes))
        return TC_ERR_AUTH;
    const Key* server_key = keyring.find(server.identity, server.key_id);
    if (!server_key || !verify_message(*server_key, server, channel, context, context_bytes))
        return TC_ERR_AUTH;

    Message finish;
    finish.kind = kClientFinish;
    finish.key_id = local_key->key_id;
    finish.nonce = client.nonce;
    finish.peer_nonce = server.nonce;
    if (!send_message(io, *local_key, &finish, channel, context, context_bytes))
        return TC_ERR_AUTH;
    if (out_peer) {
        out_peer->identity = server.identity;
        out_peer->key_id = server.key_id;
    }
        return TC_OK;
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
}

tc_status_t server_handshake(const Io& io, const Keyring& keyring,
                             const char* expected_peer_identity,
                             const char* channel, const void* context,
                             size_t context_bytes, PeerIdentity* out_peer) {
    try {
        if (!io.read_all || !io.write_all ||
            (expected_peer_identity && !valid_identity(expected_peer_identity)) ||
            !channel || !*channel || (!context && context_bytes != 0))
            return TC_ERR_INVALID_ARG;
        const Key* local_key = keyring.active_key();
        if (!local_key) return TC_ERR_INVALID_ARG;

    Message client;
    if (!receive_message(io, &client) || client.kind != kClientHello ||
        client.identity.empty() || !all_zero(client.peer_nonce) ||
        (expected_peer_identity && client.identity != expected_peer_identity))
        return TC_ERR_AUTH;
    const Key* client_key = keyring.find(client.identity, client.key_id);
    if (!client_key || !verify_message(*client_key, client, channel, context, context_bytes))
        return TC_ERR_AUTH;

    Message server;
    server.kind = kServerHello;
    server.key_id = local_key->key_id;
    server.identity = keyring.local_identity();
    server.peer_nonce = client.nonce;
    if (!random_bytes(server.nonce.data(), server.nonce.size())) return TC_ERR_INTERNAL;
    if (!send_message(io, *local_key, &server, channel, context, context_bytes))
        return TC_ERR_AUTH;

    Message finish;
    if (!receive_message(io, &finish) || finish.kind != kClientFinish ||
        !finish.identity.empty() || finish.key_id != client.key_id ||
        !constant_time_equal(finish.nonce.data(), client.nonce.data(), kNonceBytes) ||
        !constant_time_equal(finish.peer_nonce.data(), server.nonce.data(), kNonceBytes) ||
        !verify_message(*client_key, finish, channel, context, context_bytes))
        return TC_ERR_AUTH;
    if (out_peer) {
        out_peer->identity = client.identity;
        out_peer->key_id = client.key_id;
    }
        return TC_OK;
    } catch (const std::bad_alloc&) {
        return TC_ERR_ALLOC;
    } catch (...) {
        return TC_ERR_INTERNAL;
    }
}

}  // namespace tc_transport_auth_internal
