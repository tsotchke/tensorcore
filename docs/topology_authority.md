# Topology authority

TensorCore schedules accelerators from a signed, drift-checked topology
snapshot. The authority reconciles four inputs by stable node identity:

| Source | Required input | Role |
|---|---|---|
| `computer_mesh` | `nodes.json` | Declared nodes, aliases, lifecycle, OS/architecture, and capabilities |
| Tailscale | One observation envelope for every policy-manifested profile | Online/offline connectivity and observed identities |
| Cloud provider | One complete observation envelope for every policy-manifested inventory | Instance state and accelerator observations |
| TensorCore | `configs/mesh_resources.json` | Scheduler resource identity, capacity, backend, and desired admission state |

The checked-in contract is `configs/topology_authority.json`. Its state model
keeps lifecycle, observation freshness, connectivity, and admission separate.
Consequently an offline node is not silently retired, a retired node cannot
become eligible merely by appearing online, and an active scheduler resource
is quarantined unless its declared node has a fresh online observation.

## Private observation capture

Raw Tailscale and cloud inventories contain private network and account data.
Store them outside the checkout, restrict access to the operator, and never
commit them. For example:

```sh
export TC_TOPOLOGY_STATE="$HOME/.local/state/tensorcore/topology"
mkdir -p "$TC_TOPOLOGY_STATE/raw" "$TC_TOPOLOGY_STATE/observations"
chmod 700 "$TC_TOPOLOGY_STATE" "$TC_TOPOLOGY_STATE/raw" \
  "$TC_TOPOLOGY_STATE/observations"
```

For each profile listed by `tailscale switch --list`, switch to that profile,
capture `tailscale status --json`, wrap it under the matching policy ID, and
restore the original profile when finished:

```sh
tailscale status --json > "$TC_TOPOLOGY_STATE/raw/tailscale-mesh.json"
python3 scripts/wrap_topology_observation.py tailscale \
  --profile-id mesh \
  --status-json "$TC_TOPOLOGY_STATE/raw/tailscale-mesh.json" \
  --output "$TC_TOPOLOGY_STATE/observations/tailscale-mesh.json"
```

Repeat for the `personal` policy ID. The reconciler rejects a missing profile
or an unmanifested profile, so a partial capture cannot masquerade as complete.
Switching profiles can interrupt overlay connections; capture from a local
console and restore the original profile before starting reconciliation.

Capture the complete cloud instance list and wrap it with the corresponding
policy ID. The current contract uses one aggregate `gcp` inventory:

```sh
gcloud compute instances list --format=json \
  > "$TC_TOPOLOGY_STATE/raw/gcp-instances.json"
python3 scripts/wrap_topology_observation.py cloud \
  --inventory-id gcp \
  --provider gcp \
  --instances-json "$TC_TOPOLOGY_STATE/raw/gcp-instances.json" \
  --output "$TC_TOPOLOGY_STATE/observations/gcp.json"
```

The wrapper records an RFC3339 observation time but does not redact its input.
Both raw files and private envelopes remain operator-only evidence.

## Reconcile and sign

Set a deployment secret in the operator and scheduler environments. The value
must not be committed or written to the public snapshot:

```sh
export TC_TOPOLOGY_SIGNING_KEY='deployment-secret-from-your-secret-store'
```

Generate the private scheduler artifact and the separately signed public
artifact:

```sh
python3 scripts/topology_authority.py reconcile \
  --policy configs/topology_authority.json \
  --declarations computer_mesh="$HOME/Desktop/computer_mesh/nodes.json" \
  --tailscale-profile mesh="$TC_TOPOLOGY_STATE/observations/tailscale-mesh.json" \
  --tailscale-profile personal="$TC_TOPOLOGY_STATE/observations/tailscale-personal.json" \
  --cloud-inventory gcp="$TC_TOPOLOGY_STATE/observations/gcp.json" \
  --scheduler-inventory tensorcore=configs/mesh_resources.json \
  --private-output "$TC_TOPOLOGY_STATE/private.json" \
  --public-output "$TC_TOPOLOGY_STATE/public.json" \
  --require-clean --require-signature
```

`private.json` uses schema `tensorcore.topology_snapshot.v1`. It retains
stable aliases, source hashes, and detailed drift diagnostics for the
scheduler operator. `public.json` uses
`tensorcore.public_topology_snapshot.v1`; it omits private aliases, raw source
hashes, network coordinates, cloud account fields, and detailed unrecognized
identities. The public artifact still requires review before publication.

Verify either artifact without rewriting it:

```sh
python3 scripts/topology_authority.py verify \
  "$TC_TOPOLOGY_STATE/private.json" --require-signature --json
```

The signature is HMAC-SHA256. It authenticates scheduler/operator evidence
that shares the deployment secret; it is not a third-party public-key
signature. The embedded SHA-256 digest continues to detect accidental changes
when a verifier does not have the secret.

## Fail-closed scheduler admission

Production scheduler loops, submissions, and audits require a private topology
snapshot by default. Configure:

```sh
export TC_TOPOLOGY_SNAPSHOT="$TC_TOPOLOGY_STATE/private.json"
export TC_TOPOLOGY_MAX_AGE_SEC=300
export TC_TOPOLOGY_SIGNING_KEY='deployment-secret-from-your-secret-store'
```

The scheduler verifies all of the following before probing, claiming, or
launching work:

- schema, SHA-256 digest, and HMAC signature;
- snapshot age and future clock skew;
- `drift.clean=true`;
- exact agreement between snapshot and scheduler resource IDs;
- backend, class, capacity, and scheduler-status equality;
- `eligible` authority admission for every active resource.

The accepted topology digest, authority node ID, and admission state are copied
into job and lease metadata. A production command without a snapshot fails
with `snapshot_required`. Development dry-runs may omit it; a supplied snapshot
is always validated. `--allow-unreconciled-topology` and
`--allow-unsigned-topology` are explicit emergency/development bypasses and
must not be used by the production service.

Regenerate the signed snapshot at least every 300 seconds for the default
scheduler policy. Source observations have their own, longer freshness bounds;
regenerating a snapshot does not make a stale Tailscale or cloud observation
fresh.

## Current Windows boundary

The `win11` guest on `old-donkey` remains CPU-only. Its QXL display is not GPU
passthrough, and the host's only RTX 3050 remains assigned to Linux. Therefore:

- `old-donkey:cuda3050` remains the Linux CUDA lane;
- `jack-blupc:cuda3060` remains blocked historical inventory;
- no Windows CUDA capacity may be inferred from `win11`;
- Windows CUDA admission requires a future dedicated GPU or external Windows
  CUDA runner with fresh evidence.

## Verification

The portable authority and observation-envelope tests run on Ubuntu, macOS,
and Windows in `.github/workflows/ci.yml`. The scheduler binding test remains a
Linux/macOS control-plane test because that scheduler uses POSIX file locking.

```sh
python3 scripts/wrap_topology_observation_selftest.py
python3 scripts/topology_authority_selftest.py
python3 scripts/topology_scheduler_binding_selftest.py
python3 scripts/check_topology_authority_docs.py
python3 scripts/check_topology_authority_sprint.py \
  --trace-output build/topology-authority-gates.jsonl \
  --require-tracked-clean
```

The tests cover complete source manifests, unknown identities, stale and
future observations, alias collisions, backend conflicts, missing profiles,
unadmitted scheduler resources, public redaction, signed round trips, and
tamper rejection.
