#ifndef TENSORCORE_TRANSPORT_AUTH_H
#define TENSORCORE_TRANSPORT_AUTH_H

/*
 * Versioned pre-shared-key authentication for TensorCore TCP transports.
 *
 * Authentication is opt-in and additive: legacy constructors retain their
 * existing unauthenticated behavior, while the authenticated constructors in
 * remote_tensor.h, mesh_collective.h, and distributed.h require this config.
 *
 * Every key is bound to an application identity and an opaque non-zero key
 * ID. A node's keyring contains its active local key plus the peer keys it is
 * willing to trust. Multiple IDs for the same identity provide a bounded
 * overlap window during rotation. Secrets are copied by the runtime and must
 * contain at least 32 bytes of high-entropy key material.
 *
 * This is mutual HMAC-SHA-256 authentication, not payload encryption. Deploy
 * it on an encrypted/private transport (for example Tailscale) when payload
 * confidentiality is required.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TC_TRANSPORT_AUTH_ABI_VERSION_1 UINT32_C(1)
#define TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT \
    TC_TRANSPORT_AUTH_ABI_VERSION_1

#define TC_TRANSPORT_AUTH_MIN_SECRET_BYTES ((size_t)32)
#define TC_TRANSPORT_AUTH_MAX_IDENTITY_BYTES ((size_t)127)

typedef struct {
    /* Application-defined stable identity, for example "trainer/rank-3". */
    const char* identity;
    /* Opaque deployment key ID. Zero is reserved and rejected. */
    uint64_t key_id;
    /* High-entropy PSK bytes. The runtime never treats these as text. */
    const void* secret;
    size_t secret_bytes;
} tc_transport_auth_key;

typedef struct {
    uint32_t abi_version;
    /* Identity presented by this endpoint. */
    const char* local_identity;
    /* Key ID used for new handshakes; must select local_identity in keys. */
    uint64_t active_key_id;
    /* Local and trusted-peer keys. (identity, key_id) pairs must be unique. */
    const tc_transport_auth_key* keys;
    size_t key_count;
} tc_transport_auth_config;

#ifdef __cplusplus
}
#endif

#endif /* TENSORCORE_TRANSPORT_AUTH_H */
