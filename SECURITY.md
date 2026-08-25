# Security Policy

## Supported versions

The `v0.1.x` series is the only supported branch. v1.0 will tighten this
guarantee; today the most recently released `v0.1.x` is the only
version receiving fixes.

## Threat model

`tensorcore` is primarily a kernel library, but its distributed backends do
handle network input. Authenticated remote-tensor, mesh, and GLOO constructors
perform mutual HMAC-SHA256 authentication with stable peer-identity and rank
binding. They do not provide transport encryption. Legacy constructors remain
available for compatibility and are unauthenticated, so they must be confined
to trusted networks. The practical attack surface includes:

1. **GGUF file parsing** — `lib/io/gguf.c` parses a memory-mapped GGUF
   file. A maliciously crafted GGUF could trigger an out-of-bounds read
   if the parser misses a bounds check.
2. **Tensor dimensions and offsets** — `tc_buffer_*` and the dispatch
   path trust the caller's shape descriptors. Passing oversized values
   could trigger out-of-bounds GPU access.
3. **Native library load** — `python/tensorcore/__init__.py` searches
   well-known prefixes and the `TENSORCORE_LIB` env var for the dylib.
   A user able to set environment variables can substitute a different
   dynamic library at load time — but they could do that for any
   ctypes-loaded library.
4. **Distributed transports** — `tc_remote_*`, mesh collectives built on it,
   and the GLOO TCP backend parse messages and move caller-owned tensor data
   over sockets. Bind services to a specific Tailscale/private address, apply
   tailnet ACLs and host firewall rules, use the authenticated constructors,
   rotate transport keys when membership changes, and never expose legacy
   listeners directly to an untrusted network. `0.0.0.0` and `*` deliberately
   listen on every available interface.

`tensorcore` is **not** intended to be hardened against an attacker who
can run arbitrary code in your process. It runs alongside model code as a
peer, not a security boundary.

## Reporting a vulnerability

If you discover what you believe is a security issue, please email the
maintainer at the address in the git history
(`git log --author=tsotchke -1 --format="%ae"`) rather than opening a
public issue.

Include:

- The vulnerability class (memory corruption, info leak, etc.).
- A minimal reproducer (input file, descriptor, call sequence).
- The version (`tc_version()` output), OS, architecture, compiler, and backend.

You should expect:

- An acknowledgement within 5 business days.
- A fix or mitigation plan within 30 days for high-severity issues.
- Public disclosure coordinated with you after the fix ships.

## Out of scope

- **Apple Silicon GPU side channels** (rowhammer, voltage attacks, etc.)
  — out of scope; report to Apple.
- **Side-channel timing of dispatched kernels** — out of scope; the
  library does not promise constant-time execution.
- **Behavior under deliberately corrupted `tc_buffer*` objects** — out
  of scope; opaque-handle API contract is "callers don't fabricate
  handles."

## Hardening posture

The library does not currently:

- Use stack canaries beyond what the toolchain provides.
- Run under `-fsanitize=address` by default in release builds. `cmake
  -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address"`
  works for local fuzzing.
- Bounds-check GGUF tensor data offsets against the mmap region size
  exhaustively. v0.2 adds explicit checks.
- Encrypt distributed payloads. Authenticated constructors prove peer identity
  and bind identities to ranks, but confidentiality still requires a secure
  network overlay such as Tailscale/WireGuard. Legacy constructors provide
  neither authentication nor encryption.

If you find an issue in these areas, please report it.
