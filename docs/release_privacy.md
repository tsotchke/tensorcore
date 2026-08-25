# Public release privacy policy

TensorCore release content must be portable and safe to publish. Machine-local
coordinates are not documentation, configuration defaults, or acceptable test
fixtures.

## Prohibited in tracked release content

- Personal macOS, Linux, or Windows home directories.
- Account-qualified internal SSH targets.
- RFC 1918 or carrier-grade NAT addresses used by a real mesh.
- Live cloud project, zone, or instance identifiers.
- Private repository names or clone URLs.
- Tokens, credentials, private keys, signed URLs, or secret-bearing command
  lines.
- Evidence files that reveal local filesystem layout or network topology.

## Safe forms

Use portable environment variables and documentation-only values:

```text
$HOME/src/tensorcore
/srv/tensorcore/runs
C:/Users/example/src/tensorcore
builder@windows-host.example
ssh://git@code-host.example/organization/private-model-runtime.git
tcp://192.0.2.10:9000
```

The address blocks `192.0.2.0/24`, `198.51.100.0/24`, and
`203.0.113.0/24` are reserved for documentation. Loopback is appropriate for
single-host tests.

## Required gate

```sh
python3 scripts/check_release_privacy_selftest.py
python3 scripts/check_release_privacy.py
python3 scripts/check_release_privacy.py --history
python3 scripts/check_release_artifact_privacy_selftest.py
python3 scripts/check_release_artifact_privacy.py dist/*
python3 scripts/check_release_artifact_privacy.py build/release_smoke_runtime_evidence.json
```

The checker scans every tracked UTF-8 text file and rejects private home
paths, known account-qualified host forms, private infrastructure literals,
and addresses in RFC 1918 or carrier-grade NAT space. CI and the release
workflow both run it.

The release workflow fetches complete tag ancestry and runs `--history`. That
mode scans every reachable UTF-8 Git blob with the same home, account, private
infrastructure, and network rules used for the current tree; it is not limited
to a finite list of remembered incidents. A sanitized current tree therefore
cannot pass publication while a violating historical blob remains reachable.

The artifact checker reads directories, wheels/zip files, tar archives, and
binary payloads. It rejects embedded absolute user homes and private network
addresses, including compiler debug metadata, baked fallback paths, and JSON
receipts uploaded by release workflows.

Cloud export helpers must also:

- require the project, host, and zone explicitly;
- run the privacy gate before transfer;
- exclude untracked files by default;
- use a unique destination;
- delete the payload after evidence is retrieved.

## Git history

Cleaning the current tree does not remove a secret or coordinate from earlier
Git objects. Before first public publication, inspect every ref that will be
pushed. If private data ever existed in reachable history, either perform a
reviewed history rewrite or publish into a new repository from a sanitized
tree. Rotating exposed credentials remains mandatory even after history is
rewritten.

Do not push tags, backup refs, pull-request refs, or release branches until
that history review is complete.

## Runtime evidence

Public evidence should describe hardware classes and software versions, not
fleet topology. A useful receipt includes:

- source revision and dirty state;
- operating-system and architecture class;
- device model and compute capability;
- backend and kernel names;
- test counts and explicit skips;
- schema version and validator result.

It should omit usernames, home directories, IP addresses, SSH aliases, cloud
project names, instance names, and unrelated process lists.

Release-smoke evidence records only logical path labels and artifact basenames.
Windows host evidence schema v2 records the platform and architecture class,
but deliberately omits checkout paths, clone URLs, account names, and machine
names. `scripts/check_finish_campaign_evidence.py` accepts only this public-safe
schema and requires every hardware receipt to match the exact release commit.
