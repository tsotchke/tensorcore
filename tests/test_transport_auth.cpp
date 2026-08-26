#include "tensorcore/tensorcore.h"
#include "tensorcore/remote_tensor.h"
#include "tensorcore/mesh_collective.h"
#include "../lib/distributed/remote_tensor_internal.h"

#if defined(_WIN32)
int main() { return 77; }
#else

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kAuthPrefixBytes = 80;
constexpr size_t kAuthMacBytes = 32;

int fail(const char* what) {
    std::fprintf(stderr, "transport_auth: FAIL: %s\n", what);
    return 1;
}

int reserve_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        ::close(fd);
        return -1;
    }
    const int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

int listen_loopback(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 8) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

int connect_loopback(int port) {
    for (int attempt = 0; attempt < 50; ++attempt) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
            return fd;
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return -1;
}

bool send_all(int fd, const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (bytes != 0) {
        const ssize_t n = ::send(fd, p, bytes, 0);
        if (n <= 0) return false;
        p += n;
        bytes -= static_cast<size_t>(n);
    }
    return true;
}

bool recv_all(int fd, void* data, size_t bytes) {
    uint8_t* p = static_cast<uint8_t*>(data);
    while (bytes != 0) {
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n <= 0) return false;
        p += n;
        bytes -= static_cast<size_t>(n);
    }
    return true;
}

bool receive_auth_message(int fd, std::vector<uint8_t>* message) {
    std::array<uint8_t, kAuthPrefixBytes> prefix{};
    if (!recv_all(fd, prefix.data(), prefix.size())) return false;
    const size_t identity_bytes = static_cast<size_t>(prefix[6]) |
                                  (static_cast<size_t>(prefix[7]) << 8);
    if (identity_bytes > TC_TRANSPORT_AUTH_MAX_IDENTITY_BYTES) return false;
    message->resize(prefix.size() + identity_bytes + kAuthMacBytes);
    std::memcpy(message->data(), prefix.data(), prefix.size());
    return recv_all(fd, message->data() + prefix.size(),
                    identity_bytes + kAuthMacBytes);
}

bool relay_auth_message(int from, int to, std::vector<uint8_t>* capture) {
    std::vector<uint8_t> message;
    if (!receive_auth_message(from, &message)) return false;
    if (capture) *capture = message;
    return send_all(to, message.data(), message.size());
}

std::array<uint8_t, 32> secret(uint8_t seed) {
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<uint8_t>(seed + 17 * i + (i >> 1));
    return out;
}

tc_transport_auth_key key(const char* identity, uint64_t key_id,
                          const std::array<uint8_t, 32>& bytes) {
    return tc_transport_auth_key{identity, key_id, bytes.data(), bytes.size()};
}

tc_transport_auth_config config(const char* identity, uint64_t active_key_id,
                                const tc_transport_auth_key* keys,
                                size_t key_count) {
    return tc_transport_auth_config{
        TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT,
        identity,
        active_key_id,
        keys,
        key_count,
    };
}

bool wait_for_failures(tc_remote_ctx* server, uint64_t count) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (tc_remote_auth_failure_count(server) >= count) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

int remote_auth_test() {
    int rc = 0;
    tc_context* ctx = nullptr;
    if (tc_init(&ctx) != TC_OK || !ctx) return fail("remote tc_init");

    const auto server_old_secret = secret(0x11);
    const auto server_new_secret = secret(0x31);
    const auto client_old_secret = secret(0x51);
    const auto client_new_secret = secret(0x71);
    const auto wrong_secret = secret(0x91);

    const tc_transport_auth_key initial_server_keys[] = {
        key("weights/server-a", 101, server_old_secret),
        key("compute/client-a", 201, client_old_secret),
    };
    const tc_transport_auth_key initial_client_keys[] = {
        key("compute/client-a", 201, client_old_secret),
        key("weights/server-a", 101, server_old_secret),
    };
    const auto initial_server = config("weights/server-a", 101,
                                       initial_server_keys, 2);
    const auto initial_client = config("compute/client-a", 201,
                                       initial_client_keys, 2);

    tc_remote_ctx* invalid = reinterpret_cast<tc_remote_ctx*>(1);
    auto bad_abi = initial_server;
    bad_abi.abi_version = TC_TRANSPORT_AUTH_ABI_VERSION_CURRENT + 1;
    if (tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                     nullptr, &bad_abi, &invalid) !=
            TC_ERR_ABI_MISMATCH || invalid != nullptr) {
        rc |= fail("auth ABI mismatch is not atomic");
    }

    const int server_port = reserve_port();
    if (server_port <= 0) return fail("reserve remote server port");
    char server_url[96];
    std::snprintf(server_url, sizeof(server_url), "tcp://127.0.0.1:%d",
                  server_port);
    tc_remote_ctx* server = nullptr;
    if (tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER,
                                     server_url, &initial_server, &server) != TC_OK)
        return fail("authenticated remote server init");
    const std::array<uint8_t, 256> payload = [] {
        std::array<uint8_t, 256> value{};
        for (size_t i = 0; i < value.size(); ++i) value[i] = static_cast<uint8_t>(i ^ 0xa5u);
        return value;
    }();
    if (tc_remote_register_tensor(server, "auth/tensor", payload.data(),
                                  payload.size()) != TC_OK)
        rc |= fail("register authenticated tensor");

    tc_remote_ctx* client = nullptr;
    if (tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                     nullptr, &initial_client, &client) != TC_OK)
        return fail("authenticated remote client init");
    int persistent_peer = -1;
    if (tc_remote_connect_authenticated(client, server_url, "weights/server-a",
                                        &persistent_peer) != TC_OK ||
        persistent_peer < 0 ||
        !tc_remote_peer_identity(client, persistent_peer) ||
        std::strcmp(tc_remote_peer_identity(client, persistent_peer),
                    "weights/server-a") != 0 ||
        tc_remote_peer_key_id(client, persistent_peer) != 101) {
        rc |= fail("mutual identity/key binding");
    }
    std::array<uint8_t, 256> fetched{};
    if (persistent_peer < 0 ||
        tc_remote_tensor_fetch(client, persistent_peer, "auth/tensor",
                               fetched.data(), fetched.size()) != TC_OK ||
        fetched != payload) {
        rc |= fail("authenticated remote fetch");
    }
    fetched.fill(0);
    if (persistent_peer < 0 ||
        tc_remote_internal_reconnect(
            client, persistent_peer, server_url, "weights/server-a") != TC_OK ||
        tc_remote_tensor_fetch(client, persistent_peer, "auth/tensor",
                               fetched.data(), fetched.size()) != TC_OK ||
        fetched != payload ||
        !tc_remote_peer_identity(client, persistent_peer) ||
        std::strcmp(tc_remote_peer_identity(client, persistent_peer),
                    "weights/server-a") != 0 ||
        tc_remote_peer_key_id(client, persistent_peer) != 101) {
        rc |= fail("authenticated in-place reconnect");
    }

    int rejected_peer = 99;
    if (tc_remote_connect_authenticated(client, server_url, "weights/server-b",
                                        &rejected_peer) != TC_ERR_AUTH ||
        rejected_peer != -1) {
        rc |= fail("expected server identity mismatch");
    }

    const tc_transport_auth_key wrong_client_keys[] = {
        key("compute/client-a", 201, wrong_secret),
        key("weights/server-a", 101, server_old_secret),
    };
    const auto wrong_client_config = config("compute/client-a", 201,
                                            wrong_client_keys, 2);
    tc_remote_ctx* wrong_client = nullptr;
    if (tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                     nullptr, &wrong_client_config,
                                     &wrong_client) != TC_OK) {
        rc |= fail("wrong-key client init");
    } else {
        rejected_peer = 99;
        if (tc_remote_connect_authenticated(wrong_client, server_url,
                                            "weights/server-a",
                                            &rejected_peer) != TC_ERR_AUTH ||
            rejected_peer != -1) {
            rc |= fail("wrong key was accepted");
        }
        tc_remote_shutdown(wrong_client);
    }

    /* Capture a valid client hello through a transparent handshake proxy,
     * then replay it. The server emits a fresh challenge, but closes the
     * replayed session when no valid finish proof follows. */
    const int proxy_port = reserve_port();
    const int proxy_listener = listen_loopback(proxy_port);
    if (proxy_port <= 0 || proxy_listener < 0) {
        rc |= fail("replay proxy listener");
    } else {
        std::vector<uint8_t> captured_hello;
        std::vector<uint8_t> original_server_hello;
        std::vector<uint8_t> captured_finish;
        std::atomic<bool> proxy_ok{false};
        std::thread proxy([&] {
            const int downstream = ::accept(proxy_listener, nullptr, nullptr);
            const int upstream = connect_loopback(server_port);
            const bool ok = downstream >= 0 && upstream >= 0 &&
                relay_auth_message(downstream, upstream, &captured_hello) &&
                relay_auth_message(upstream, downstream, &original_server_hello) &&
                relay_auth_message(downstream, upstream, &captured_finish);
            proxy_ok.store(ok, std::memory_order_release);
            if (downstream >= 0) { ::shutdown(downstream, SHUT_RDWR); ::close(downstream); }
            if (upstream >= 0) { ::shutdown(upstream, SHUT_RDWR); ::close(upstream); }
        });
        char proxy_url[96];
        std::snprintf(proxy_url, sizeof(proxy_url), "tcp://127.0.0.1:%d", proxy_port);
        tc_remote_ctx* capture_client = nullptr;
        int capture_peer = -1;
        const bool capture_connected =
            tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                         nullptr, &initial_client,
                                         &capture_client) == TC_OK &&
            tc_remote_connect_authenticated(capture_client, proxy_url,
                                            "weights/server-a",
                                            &capture_peer) == TC_OK;
        proxy.join();
        ::close(proxy_listener);
        if (capture_client) tc_remote_shutdown(capture_client);
        if (!capture_connected || !proxy_ok.load(std::memory_order_acquire) ||
            captured_hello.empty() || original_server_hello.empty()) {
            rc |= fail("capture valid auth transcript");
        } else {
            const int replay_fd = connect_loopback(server_port);
            std::vector<uint8_t> replay_server_hello;
            const bool replay_challenged = replay_fd >= 0 &&
                send_all(replay_fd, captured_hello.data(), captured_hello.size()) &&
                receive_auth_message(replay_fd, &replay_server_hello) &&
                !captured_finish.empty() &&
                send_all(replay_fd, captured_finish.data(), captured_finish.size());
            if (replay_fd >= 0) {
                ::shutdown(replay_fd, SHUT_RDWR);
                ::close(replay_fd);
            }
            const bool fresh_challenge = replay_challenged &&
                replay_server_hello.size() >= 48 && original_server_hello.size() >= 48 &&
                std::memcmp(replay_server_hello.data() + 16,
                            original_server_hello.data() + 16, 32) != 0;
            if (!fresh_challenge) rc |= fail("replay did not receive a fresh challenge");
        }
    }

    const tc_transport_auth_key overlap_server_keys[] = {
        key("weights/server-a", 101, server_old_secret),
        key("weights/server-a", 102, server_new_secret),
        key("compute/client-a", 201, client_old_secret),
        key("compute/client-a", 202, client_new_secret),
    };
    const auto overlap_server = config("weights/server-a", 102,
                                       overlap_server_keys, 4);
    if (tc_remote_auth_rotate(server, &overlap_server) != TC_OK)
        rc |= fail("server overlap rotation");
    fetched.fill(0);
    if (tc_remote_tensor_fetch(client, persistent_peer, "auth/tensor",
                               fetched.data(), fetched.size()) != TC_OK ||
        fetched != payload) {
        rc |= fail("existing authenticated session did not survive rotation");
    }

    const tc_transport_auth_key overlap_client_keys[] = {
        key("compute/client-a", 201, client_old_secret),
        key("compute/client-a", 202, client_new_secret),
        key("weights/server-a", 101, server_old_secret),
        key("weights/server-a", 102, server_new_secret),
    };
    const auto overlap_client_config = config("compute/client-a", 201,
                                              overlap_client_keys, 4);
    tc_remote_ctx* rotating_client = nullptr;
    int overlap_peer = -1;
    if (tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                     nullptr, &overlap_client_config,
                                     &rotating_client) != TC_OK ||
        tc_remote_connect_authenticated(rotating_client, server_url,
                                        "weights/server-a",
                                        &overlap_peer) != TC_OK ||
        tc_remote_peer_key_id(rotating_client, overlap_peer) != 102) {
        rc |= fail("rotation overlap connection");
    }

    const tc_transport_auth_key retired_server_keys[] = {
        key("weights/server-a", 102, server_new_secret),
        key("compute/client-a", 202, client_new_secret),
    };
    const auto retired_server = config("weights/server-a", 102,
                                       retired_server_keys, 2);
    if (tc_remote_auth_rotate(server, &retired_server) != TC_OK)
        rc |= fail("retire old server/client keys");
    rejected_peer = 99;
    if (!rotating_client ||
        tc_remote_connect_authenticated(rotating_client, server_url,
                                        "weights/server-a",
                                        &rejected_peer) != TC_ERR_AUTH ||
        rejected_peer != -1) {
        rc |= fail("retired client key was accepted");
    }

    const tc_transport_auth_key new_client_keys[] = {
        key("compute/client-a", 202, client_new_secret),
        key("weights/server-a", 102, server_new_secret),
    };
    const auto new_client_config = config("compute/client-a", 202,
                                          new_client_keys, 2);
    int new_peer = -1;
    if (!rotating_client ||
        tc_remote_auth_rotate(rotating_client, &new_client_config) != TC_OK ||
        tc_remote_connect_authenticated(rotating_client, server_url,
                                        "weights/server-a", &new_peer) != TC_OK ||
        tc_remote_peer_key_id(rotating_client, new_peer) != 102) {
        rc |= fail("client live rotation/new handshake");
    }

    if (!wait_for_failures(server, 4))
        rc |= fail("server auth failure observability");

    if (rotating_client) tc_remote_shutdown(rotating_client);
    tc_remote_shutdown(client);
    tc_remote_shutdown(server);

    /* Authenticated client against legacy server: the authenticated side
     * must reject the missing TCAH handshake without falling back. */
    const int legacy_port = reserve_port();
    char legacy_url[96];
    std::snprintf(legacy_url, sizeof(legacy_url), "tcp://127.0.0.1:%d", legacy_port);
    tc_remote_ctx* legacy_server = nullptr;
    tc_remote_ctx* auth_client = nullptr;
    if (tc_remote_init(ctx, TC_REMOTE_ROLE_WEIGHT_SERVER, legacy_url,
                       &legacy_server) != TC_OK ||
        tc_remote_init_authenticated(ctx, TC_REMOTE_ROLE_COMPUTE_CLIENT,
                                     nullptr, &initial_client,
                                     &auth_client) != TC_OK) {
        rc |= fail("legacy/auth setup");
    } else {
        rejected_peer = 99;
        if (tc_remote_connect_authenticated(auth_client, legacy_url,
                                            "weights/server-a",
                                            &rejected_peer) != TC_ERR_AUTH ||
            rejected_peer != -1) {
            rc |= fail("legacy remote server downgrade");
        }
    }
    if (auth_client) tc_remote_shutdown(auth_client);
    if (legacy_server) tc_remote_shutdown(legacy_server);
    tc_shutdown(ctx);
    return rc;
}

int run_mesh_rank(int rank, const char* const* urls) {
    static const char* identities[] = {"mesh/rank-0", "mesh/rank-1"};
    const auto rank0_secret = secret(0x42);
    const auto rank1_secret = secret(0x84);
    const tc_transport_auth_key keys[] = {
        key(identities[0], 301, rank0_secret),
        key(identities[1], 302, rank1_secret),
    };
    const auto auth = config(identities[rank], static_cast<uint64_t>(301 + rank),
                             keys, 2);
    tc_context* ctx = nullptr;
    const tc_status_t context_status = tc_init(&ctx);
    if (context_status != TC_OK) {
        std::fprintf(stderr, "transport_auth: mesh rank %d tc_init status=%d\n",
                     rank, static_cast<int>(context_status));
        return 1;
    }
    tc_mesh_group_t* group = nullptr;
    const tc_status_t init_status = tc_mesh_group_init_authenticated(
        ctx, 2, rank, urls, identities, &auth, &group);
    if (init_status != TC_OK || !group) {
        std::fprintf(stderr,
                     "transport_auth: mesh rank %d group init status=%d group=%p\n",
                     rank, static_cast<int>(init_status),
                     static_cast<void*>(group));
        tc_shutdown(ctx);
        return 1;
    }
    float values[16];
    for (float& value : values) value = static_cast<float>(rank + 1);
    const tc_status_t reduce_status = tc_mesh_allreduce(
        group, values, 16, TC_COLL_DTYPE_F32, TC_REDUCE_SUM);
    int rc = reduce_status == TC_OK ? 0 : 1;
    if (reduce_status != TC_OK) {
        std::fprintf(stderr,
                     "transport_auth: mesh rank %d allreduce status=%d\n",
                     rank, static_cast<int>(reduce_status));
    }
    for (float value : values) {
        if (value != 3.0f) rc = 1;
    }
    const tc_status_t shutdown_status = tc_mesh_group_shutdown(group);
    if (shutdown_status != TC_OK) {
        std::fprintf(stderr,
                     "transport_auth: mesh rank %d shutdown status=%d\n",
                     rank, static_cast<int>(shutdown_status));
        rc = 1;
    }
    tc_shutdown(ctx);
    return rc;
}

int mesh_auth_test() {
    const int port0 = reserve_port();
    const int port1 = reserve_port();
    if (port0 <= 0 || port1 <= 0 || port0 == port1) return 77;
    char url0[96];
    char url1[96];
    std::snprintf(url0, sizeof(url0), "tcp://127.0.0.1:%d", port0);
    std::snprintf(url1, sizeof(url1), "tcp://127.0.0.1:%d", port1);
    const char* urls[] = {url0, url1};
    pid_t children[2];
    for (int rank = 0; rank < 2; ++rank) {
        children[rank] = ::fork();
        if (children[rank] < 0) return 1;
        if (children[rank] == 0) {
            if (rank) ::usleep(50000);
            _exit(run_mesh_rank(rank, urls));
        }
    }
    int rc = 0;
    for (pid_t child : children) {
        int status = 0;
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            std::fprintf(stderr,
                         "transport_auth: mesh child %ld exit=%d signal=%d\n",
                         static_cast<long>(child),
                         WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                         WIFSIGNALED(status) ? WTERMSIG(status) : 0);
            rc = 1;
        }
    }
    return rc;
}

std::array<std::array<uint8_t, 32>, 8> gloo_secrets() {
    std::array<std::array<uint8_t, 32>, 8> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = secret(static_cast<uint8_t>(0x20 + 13 * i));
    return values;
}

int run_gloo_rank(int rank, int world, const char* url,
                  bool wrong_rank1_key, bool legacy_rank1) {
    static const char* identities[] = {
        "trainer/rank-0", "trainer/rank-1", "trainer/rank-2", "trainer/rank-3"
    };
    auto secrets = gloo_secrets();
    const auto wrong = secret(0xe1);
    tc_transport_auth_key keys[8];
    for (int r = 0; r < 4; ++r) {
        keys[2 * r] = key(identities[r], static_cast<uint64_t>(100 + r),
                          (wrong_rank1_key && rank == 1 && r == 1) ? wrong : secrets[2 * r]);
        keys[2 * r + 1] = key(identities[r], static_cast<uint64_t>(200 + r),
                              secrets[2 * r + 1]);
    }
    const uint64_t active = rank == 0 ? 200 : static_cast<uint64_t>(100 + rank);
    const auto auth = config(identities[rank], active, keys,
                             static_cast<size_t>(2 * world));

    tc_context* ctx = nullptr;
    if (tc_init(&ctx) != TC_OK) return 1;
    tc_dist_ctx* dist = nullptr;
    tc_status_t init_status;
    if (legacy_rank1 && rank == 1) {
        init_status = tc_dist_init(ctx, TC_DIST_GLOO, world, rank, url, &dist);
    } else {
        init_status = tc_dist_init_authenticated(
            ctx, TC_DIST_GLOO, world, rank, url, identities,
            static_cast<size_t>(world), &auth, &dist);
    }

    if (wrong_rank1_key) {
        const int ok = init_status == TC_ERR_AUTH && dist == nullptr;
        if (dist) tc_dist_finalize(dist);
        tc_shutdown(ctx);
        return ok ? 0 : 1;
    }
    if (legacy_rank1) {
        const bool ok = (rank == 1)
            ? (init_status == TC_OK && dist != nullptr)
            : (init_status == TC_ERR_AUTH && dist == nullptr);
        if (dist) tc_dist_finalize(dist);
        tc_shutdown(ctx);
        return ok ? 0 : 1;
    }
    if (init_status != TC_OK || !dist) {
        tc_shutdown(ctx);
        return 1;
    }
    tc_buffer* buffer = nullptr;
    float* values = nullptr;
    int rc = 0;
    if (tc_buffer_alloc(ctx, 64 * sizeof(float), &buffer) != TC_OK ||
        tc_buffer_map(buffer, reinterpret_cast<void**>(&values)) != TC_OK) {
        rc = 1;
    } else {
        for (size_t i = 0; i < 64; ++i) values[i] = static_cast<float>(rank + 1);
        if (tc_allreduce(dist, buffer, 64, TC_DTYPE_F32, TC_REDUCE_SUM) != TC_OK)
            rc = 1;
        const float expected = static_cast<float>(world * (world + 1) / 2);
        for (size_t i = 0; i < 64 && !rc; ++i) {
            if (values[i] != expected) rc = 1;
        }
    }
    if (buffer) tc_buffer_free(ctx, buffer);
    tc_dist_finalize(dist);
    tc_shutdown(ctx);
    return rc;
}

int run_gloo_case(int world, bool wrong_rank1_key, bool legacy_rank1,
                  bool ring) {
    const int port = reserve_port();
    if (port <= 0 || port + world + 64 >= 65535) return 77;
    char url[96];
    std::snprintf(url, sizeof(url), "gloo+tcp://127.0.0.1:%d", port);
    if (ring) {
        ::setenv("TC_GLOO_RING", "1", 1);
        ::setenv("TC_GLOO_ADVERTISE_HOSTS",
                 "127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1", 1);
    } else {
        ::unsetenv("TC_GLOO_RING");
        ::unsetenv("TC_GLOO_ADVERTISE_HOSTS");
    }
    if (legacy_rank1) ::setenv("TC_TRANSPORT_AUTH_TIMEOUT_MS", "500", 1);

    std::vector<pid_t> children;
    for (int rank = 0; rank < world; ++rank) {
        const pid_t pid = ::fork();
        if (pid < 0) return 1;
        if (pid == 0) {
            if (rank > 0) ::usleep(static_cast<useconds_t>(rank * 50000));
            _exit(run_gloo_rank(rank, world, url, wrong_rank1_key, legacy_rank1));
        }
        children.push_back(pid);
    }
    int rc = 0;
    for (pid_t child : children) {
        int status = 0;
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            rc = 1;
        }
    }
    ::unsetenv("TC_GLOO_RING");
    ::unsetenv("TC_GLOO_ADVERTISE_HOSTS");
    ::unsetenv("TC_TRANSPORT_AUTH_TIMEOUT_MS");
    return rc;
}

int gloo_auth_test() {
    int rc = 0;
    rc |= run_gloo_case(4, false, false, true);
    if (rc) rc |= fail("authenticated Gloo direct ring and rotation overlap");
    const int wrong = run_gloo_case(2, true, false, false);
    if (wrong != 0) rc |= fail("Gloo wrong-key rejection");
    const int mixed = run_gloo_case(2, false, true, false);
    if (mixed != 0) rc |= fail("Gloo legacy/auth mixed-version rejection");
    return rc;
}

}  // namespace

int main() {
    /* Fork every multi-rank process before initializing Metal in the parent.
     * Apple's GPU/runtime state is not safe to inherit across fork(). */
    int rc = 0;
    const int mesh = mesh_auth_test();
    if (mesh != 0) rc |= fail("authenticated mesh collective");
    if (!rc) std::puts("  mesh identity-bound collective: OK");
    rc |= gloo_auth_test();
    if (!rc) std::puts("  Gloo rank binding, ring auth, rotation, and downgrade rejection: OK");
    const int remote = remote_auth_test();
    if (remote != 0) rc |= remote;
    if (!remote) std::puts(
        "  remote mutual auth, replay defense, live rotation, and in-place reconnect: OK");
    std::puts(rc ? "FAIL" : "OK");
    return rc;
}

#endif
