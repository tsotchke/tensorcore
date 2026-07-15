/* Inject a Gloo peer loss and prove async DiLoCo failure is observable. */

#include "tensorcore/tensorcore.h"

#if defined(_WIN32)
int main(void) { return 77; }
#else

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define N_ELEMS 16

static int fail(const char* what) {
    fprintf(stderr, "diloco_async_failure: FAIL: %s\n", what);
    return 1;
}

static int reserve_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) != 0) {
        close(fd);
        return -1;
    }
    const int port = (int)ntohs(addr.sin_port);
    close(fd);
    return port;
}

static int child_rank(const char* url, int ready_fd, int close_fd) {
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    char token = 'R';
    int rc = 0;

    if (tc_init(&ctx) != TC_OK) rc = 1;
    if (!rc && tc_dist_init(ctx, TC_DIST_GLOO, 2, 1, url, &dist) != TC_OK) rc = 1;
    if (write(ready_fd, &token, 1) != 1) rc = 1;
    if (read(close_fd, &token, 1) != 1) rc = 1;
    if (dist && tc_dist_finalize(dist) != TC_OK) rc = 1;
    if (ctx && tc_shutdown(ctx) != TC_OK) rc = 1;
    return rc;
}

int main(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) {
        fprintf(stderr, "diloco_async_failure: SKIP: no loopback port\n");
        return 77;
    }

    char url[96];
    snprintf(url, sizeof(url), "gloo+tcp://127.0.0.1:%d", port);
    int ready_pipe[2];
    int close_pipe[2];
    if (pipe(ready_pipe) != 0 || pipe(close_pipe) != 0) return fail("pipe");

    alarm(45);
    pid_t child = fork();
    if (child < 0) return fail("fork");
    if (child == 0) {
        close(ready_pipe[0]);
        close(close_pipe[1]);
        const int child_rc = child_rank(url, ready_pipe[1], close_pipe[0]);
        close(ready_pipe[1]);
        close(close_pipe[0]);
        _exit(child_rc ? 1 : 0);
    }

    close(ready_pipe[1]);
    close(close_pipe[0]);
    int rc = 0;
    tc_context* ctx = NULL;
    tc_dist_ctx* dist = NULL;
    tc_diloco_ctx* d = NULL;
    tc_diloco_ctx* restored = NULL;
    tc_buffer* theta = NULL;
    tc_buffer* restored_theta = NULL;
    float* values = NULL;
    float* restored_values = NULL;
    char token = 0;

    if (tc_init(&ctx) != TC_OK) rc |= fail("tc_init");
    if (!rc && tc_dist_init(ctx, TC_DIST_GLOO, 2, 0, url, &dist) != TC_OK) {
        rc |= fail("rank 0 dist init");
    }
    if (!rc && read(ready_pipe[0], &token, 1) != 1) rc |= fail("peer ready");
    close(ready_pipe[0]);

    /* Close rank 1 cleanly before rank 0 launches the collective. */
    token = 'C';
    if (!rc && write(close_pipe[1], &token, 1) != 1) rc |= fail("peer close request");
    close(close_pipe[1]);
    int child_status = 0;
    if (waitpid(child, &child_status, 0) < 0 ||
        !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
        rc |= fail("peer shutdown");
    }

    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = 1;
    cfg.outer_lr = 1.0f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = TC_DILOCO_COMPRESS_NONE;
    cfg.async_overlap = true;

    if (!rc && tc_diloco_init(dist, &cfg, &d) != TC_OK) rc |= fail("diloco init");
    if (!rc && tc_buffer_alloc(ctx, N_ELEMS * sizeof(float), &theta) != TC_OK) {
        rc |= fail("buffer alloc");
    }
    if (!rc && tc_buffer_map(theta, (void**)&values) != TC_OK) rc |= fail("buffer map");
    if (!rc) {
        for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 1.0f;
        if (tc_diloco_add_parameter(d, "p", theta, N_ELEMS, TC_DTYPE_F32) != TC_OK) {
            rc |= fail("add parameter");
        }
    }
    if (!rc) {
        for (size_t i = 0; i < N_ELEMS; ++i) values[i] = 2.0f;
        if (tc_diloco_apply_outer(d) != TC_OK) rc |= fail("launch");
    }

    tc_status_t background = TC_OK;
    if (!rc) {
        background = tc_diloco_async_wait(d);
        if (background == TC_OK) rc |= fail("peer loss was reported as success");
    }

    tc_diloco_async_state_t state = TC_DILOCO_ASYNC_IDLE;
    tc_status_t polled_status = TC_OK;
    uint64_t round_id = 0;
    if (!rc && tc_diloco_async_poll(d, &state, &polled_status, &round_id) != TC_OK) {
        rc |= fail("poll");
    }
    if (!rc && (state != TC_DILOCO_ASYNC_FAILED ||
                polled_status != background || round_id != 1)) {
        rc |= fail("failed state/status/round identity");
    }
    if (!rc && tc_diloco_outer_steps_completed(d) != 0) {
        rc |= fail("failed round was counted");
    }
    if (!rc) {
        for (size_t i = 0; i < N_ELEMS; ++i) {
            if (values[i] != 2.0f) {
                rc |= fail("failed worker changed live parameters");
                break;
            }
        }
    }

    /* FAILED is a durable pending state: serialize it, restore it into a
     * fresh context, and prove acknowledgement returns the same error without
     * mutating the caller-restored live model. */
    unsigned char* state_blob = NULL;
    size_t state_size = 0;
    if (!rc && tc_diloco_state_size(
            d, TC_DILOCO_STATE_ABI_VERSION_CURRENT, &state_size) != TC_OK) {
        rc |= fail("failed state size");
    }
    if (!rc) {
        state_blob = (unsigned char*)malloc(state_size);
        if (!state_blob) {
            rc |= fail("failed state allocation");
        } else {
            size_t written = 0;
            if (tc_diloco_state_serialize(
                    d, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
                    state_blob, state_size, &written) != TC_OK ||
                written != state_size) {
                rc |= fail("failed state serialization");
            }
        }
    }
    if (!rc && tc_diloco_init(dist, &cfg, &restored) != TC_OK) {
        rc |= fail("restored diloco init");
    }
    if (!rc && tc_buffer_alloc(
            ctx, N_ELEMS * sizeof(float), &restored_theta) != TC_OK) {
        rc |= fail("restored buffer alloc");
    }
    if (!rc && tc_buffer_map(
            restored_theta, (void**)&restored_values) != TC_OK) {
        rc |= fail("restored buffer map");
    }
    if (!rc) {
        for (size_t i = 0; i < N_ELEMS; ++i) restored_values[i] = 2.0f;
        if (tc_diloco_add_parameter(
                restored, "p", restored_theta, N_ELEMS, TC_DTYPE_F32) != TC_OK) {
            rc |= fail("restored add parameter");
        }
    }
    if (!rc && tc_diloco_state_deserialize(
            restored, TC_DILOCO_STATE_ABI_VERSION_CURRENT,
            state_blob, state_size) != TC_OK) {
        rc |= fail("failed state restore");
    }
    if (!rc && tc_diloco_async_poll(
            restored, &state, &polled_status, &round_id) != TC_OK) {
        rc |= fail("restored failed poll");
    }
    if (!rc && (state != TC_DILOCO_ASYNC_FAILED ||
                polled_status != background || round_id != 1)) {
        rc |= fail("restored failed identity");
    }
    if (!rc && tc_diloco_async_commit(restored) != background) {
        rc |= fail("restored failure acknowledgement");
    }
    if (!rc) {
        for (size_t i = 0; i < N_ELEMS; ++i) {
            if (restored_values[i] != 2.0f) {
                rc |= fail("restored failed state changed live parameters");
                break;
            }
        }
    }
    free(state_blob);

    if (!rc && tc_diloco_async_commit(d) != background) {
        rc |= fail("commit did not acknowledge background failure");
    }
    if (!rc && tc_diloco_async_poll(d, &state, &polled_status, &round_id) != TC_OK) {
        rc |= fail("poll after acknowledgement");
    }
    if (!rc && (state != TC_DILOCO_ASYNC_IDLE || polled_status != TC_OK)) {
        rc |= fail("failure acknowledgement did not restore idle state");
    }

    if (restored && tc_diloco_finalize(restored) != TC_OK) {
        rc |= fail("restored diloco finalize");
    }
    if (d && tc_diloco_finalize(d) != TC_OK) rc |= fail("diloco finalize");
    if (restored_theta) tc_buffer_free(ctx, restored_theta);
    if (theta) tc_buffer_free(ctx, theta);
    if (dist) tc_dist_finalize(dist);
    if (ctx) tc_shutdown(ctx);

    printf("diloco_async_failure: %s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}

#endif
