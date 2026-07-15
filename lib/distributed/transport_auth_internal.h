#ifndef TC_LIB_DISTRIBUTED_TRANSPORT_AUTH_INTERNAL_H
#define TC_LIB_DISTRIBUTED_TRANSPORT_AUTH_INTERNAL_H

#include "tensorcore/status.h"
#include "tensorcore/transport_auth.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tc_transport_auth_internal {

struct Key {
    std::string identity;
    uint64_t key_id = 0;
    std::vector<uint8_t> secret;
};

class Keyring {
public:
    Keyring() = default;
    Keyring(const Keyring&) = default;
    Keyring(Keyring&&) noexcept = default;
    Keyring& operator=(const Keyring& other);
    Keyring& operator=(Keyring&& other) noexcept;
    ~Keyring();

    static tc_status_t copy_from(const tc_transport_auth_config* config,
                                 Keyring* out);

    const std::string& local_identity() const { return local_identity_; }
    uint64_t active_key_id() const { return active_key_id_; }
    const Key* active_key() const;
    const Key* find(const std::string& identity, uint64_t key_id) const;
    bool empty() const { return keys_.empty(); }
    void swap(Keyring& other) noexcept;

private:
    void clear_secrets() noexcept;

    std::string local_identity_;
    uint64_t active_key_id_ = 0;
    std::vector<Key> keys_;
};

struct Io {
    void* user = nullptr;
    bool (*read_all)(void* user, void* data, size_t bytes) = nullptr;
    bool (*write_all)(void* user, const void* data, size_t bytes) = nullptr;
};

struct PeerIdentity {
    std::string identity;
    uint64_t key_id = 0;
};

/* Mutual challenge-response. `context` is authenticated but not sent, so both
 * sides must supply the same rank/group binding. */
tc_status_t client_handshake(const Io& io,
                             const Keyring& keyring,
                             const char* expected_peer_identity,
                             const char* channel,
                             const void* context,
                             size_t context_bytes,
                             PeerIdentity* out_peer);

tc_status_t server_handshake(const Io& io,
                             const Keyring& keyring,
                             const char* expected_peer_identity,
                             const char* channel,
                             const void* context,
                             size_t context_bytes,
                             PeerIdentity* out_peer);

}  // namespace tc_transport_auth_internal

#endif
