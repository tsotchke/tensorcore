# Transport identity authentication

TensorCore provides opt-in mutual authentication for remote-tensor, mesh,
and Gloo TCP connections. It uses a versioned three-message HMAC-SHA-256
challenge/response protocol with fresh 256-bit nonces. The transcript binds
the transport kind, protocol version, claimed application identity, key ID,
and (for Gloo) world/rank direction.

Authentication proves possession of configured pre-shared keys. It does not
encrypt tensor or collective payloads. Use an encrypted private transport such
as Tailscale when confidentiality is required, and keep network ACLs as an
independent boundary.

## Keyring contract

`tc_transport_auth_config` names the local identity, selects the active local
key ID, and supplies local plus trusted-peer keys. Identities are printable
ASCII strings of at most 127 bytes. Key IDs are non-zero opaque deployment
values. Secrets must contain at least 32 bytes of high-entropy material.

The runtime validates and deep-copies a config before returning. Duplicate
`(identity, key_id)` pairs, unknown ABI versions, short secrets, and an active
key that does not select the local identity fail before a listener is exposed.
Copied secrets are overwritten when their keyring is retired.

Treat every peer PSK as impersonation authority for that peer identity. Give
an endpoint only its own secret and the peer secrets required by its transport
role; do not distribute a fleet-wide keyring to every node. Store and deliver
those secrets through the deployment's secret manager rather than config files
or command-line arguments.

```c
static const unsigned char server_key[32] = { /* secret-store bytes */ };
static const unsigned char client_key[32] = { /* secret-store bytes */ };

tc_transport_auth_key keys[] = {
    { "weights/server-a", 101, server_key, sizeof(server_key) },
    { "compute/client-a", 201, client_key, sizeof(client_key) },
};
tc_transport_auth_config auth = {
    .abi_version = TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT,
    .local_identity = "compute/client-a",
    .active_key_id = 201,
    .keys = keys,
    .key_count = 2,
};

tc_remote_ctx* client = NULL;
tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                             NULL, &auth, &client);
int peer = -1;
tc_remote_connect_authenticated(client, "tcp://100.64.0.10:9400",
                                "weights/server-a", &peer);
```

`tc_remote_peer_identity` and `tc_remote_peer_key_id` expose the identity that
was authenticated for a client peer. `tc_remote_auth_failure_count` counts
failed handshakes without counting ordinary network or tensor-name errors.

## Gloo rank binding

`tc_dist_init_authenticated` additionally takes `rank_identities[world_size]`.
Each rendezvous connection authenticates the claimed rank against the
corresponding identity. When the direct ring is enabled, every directional
neighbor socket receives a separate handshake bound to `(world, from_rank,
to_rank)`. Broker fallback remains safe because those rendezvous sockets were
already authenticated.

The authenticated side rejects missing or unknown handshake versions; it does
not silently retry the legacy protocol. A truly old unauthenticated peer may
only observe the mismatch as connection/collective failure because its wire
format predates negotiation. New deployments should switch an entire group to
`tc_dist_init_authenticated` together.

## Rotation procedure

Key IDs are append-only deployment identifiers. Rotate without a downgrade
window:

1. Distribute configs containing both old and new peer key IDs.
2. Switch each endpoint's `active_key_id` to its new local key.
3. Confirm new sessions report the new peer key IDs.
4. Remove old IDs after the maximum old-session drain period.

`tc_remote_auth_rotate` atomically replaces the keyring used by future
handshakes while existing authenticated connections continue. Gloo and mesh
contexts establish their sockets during initialization, so rolling launches
use overlapping keyrings and recreate the context to activate new local IDs.

## Replay and failure behavior

A captured client hello cannot complete a replay: the server returns a fresh
nonce and requires a final proof covering both nonces. Replaying an old final
proof fails constant-time verification. Wrong keys, identity mismatches,
unknown versions, and legacy/authenticated mixing close the candidate socket
before application frames are processed.

Active authentication I/O uses the existing remote timeout or
`TC_TRANSPORT_AUTH_TIMEOUT_MS` for Gloo (default 5000 ms; accepted range
100–600000). Idle authenticated application connections keep the existing
long-compute behavior after the handshake completes.
