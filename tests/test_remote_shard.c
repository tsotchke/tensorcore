/*
 * test_remote_shard.c — fork-based 2-peer tc_remote_shard smoke.
 *
 * Parent = rank 0, child = rank 1. Both bind tc_remote servers,
 * form a mesh group, build a row-shard plan over a synthetic
 * [rows=8, cols=4] fp32 tensor where the value at (r, c) is
 * 100*r + c. Each rank registers its locally-owned row block,
 * then both ranks fetch the FULL tensor via tc_remote_shard_get
 * and verify every cell matches the canonical formula. That
 * exercises register, owner-routing, local-fastpath, and
 * remote-fetch concatenation.
 */

/* tensorcore.h transitively pulls tc_init / tc_shutdown / tc_context. */
#include "tensorcore/tensorcore.h"
#include "tensorcore/mesh_collective.h"
#include "tensorcore/remote_shard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <math.h>

#define ROWS 8
#define COLS 4

static int run_peer(int my_rank, const char* my_url, const char* peer_url) {
    /* Unbuffered output so an early-exit rank's diagnostics still appear
     * in the parent's pipe before the fd gets reaped. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "rank %d: tc_init failed\n", my_rank);
        return 1;
    }
    const char* urls[2] = {
        (my_rank == 0) ? my_url : peer_url,
        (my_rank == 1) ? my_url : peer_url,
    };
    tc_mesh_group_t* g = NULL;
    if (tc_mesh_group_init(ctx, 2, my_rank, urls, &g) != TC_OK) {
        fprintf(stderr, "rank %d: group_init failed\n", my_rank);
        tc_shutdown(ctx);
        return 1;
    }

    int fails = 0;

    tc_shard_plan_t plan = {
        .n_peers = 2, .rows = ROWS, .cols = COLS,
        .dtype = TC_COLL_DTYPE_F32,
    };

    /* Each rank owns ⌈ROWS/N⌉ rows = 4. */
    int32_t lo, hi;
    tc_remote_shard_local_range(&plan, my_rank, &lo, &hi);
    const int local_rows = hi - lo;
    if (local_rows <= 0) {
        fprintf(stderr, "rank %d: empty local range\n", my_rank);
        fails++;
    }

    float local[ROWS * COLS];  /* over-sized; we only fill local rows */
    for (int r = 0; r < local_rows; ++r) {
        for (int c = 0; c < COLS; ++c) {
            const int gr = lo + r;
            local[r * COLS + c] = (float)(100 * gr + c);
        }
    }

    /* Sanity: balanced block-row owner assignment */
    if (my_rank == 0) {
        const int32_t exp0 = tc_remote_shard_owner(&plan, 3);
        const int32_t exp1 = tc_remote_shard_owner(&plan, 4);
        if (exp0 != 0 || exp1 != 1) {
            fprintf(stderr, "rank 0: owner(3)=%d (want 0); owner(4)=%d (want 1)\n",
                    exp0, exp1);
            fails++;
        } else {
            printf("  PASS rank 0 shard_owner balanced\n");
        }
    }

    if (tc_remote_shard_register(g, &plan, "smoke/weights", local) != TC_OK) {
        fprintf(stderr, "rank %d: shard_register failed\n", my_rank);
        fails++;
    }

    /* Now every rank fetches the entire tensor and validates it cell-by-cell. */
    float full[ROWS * COLS];
    memset(full, 0, sizeof(full));
    tc_status_t s = tc_remote_shard_get(g, &plan, "smoke/weights",
                                         0, ROWS, local, full);
    if (s != TC_OK) {
        fprintf(stderr, "rank %d: shard_get full failed status=%d\n", my_rank, s);
        fails++;
    } else {
        int mismatches = 0;
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                const float want = (float)(100 * r + c);
                const float got  = full[r * COLS + c];
                if (fabsf(got - want) > 1e-6f) {
                    if (mismatches < 4) {
                        fprintf(stderr, "rank %d: cell (%d,%d) got %g want %g\n",
                                my_rank, r, c, got, want);
                    }
                    mismatches++;
                }
            }
        }
        if (mismatches > 0) {
            fprintf(stderr, "rank %d: shard_get full had %d cell mismatches\n",
                    my_rank, mismatches);
            fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank 0 shard_get full_range\n");
        }
    }

    /* Partial fetch — rows 2..6 (crosses owner boundary at row 4). */
    float partial[4 * COLS];
    memset(partial, 0, sizeof(partial));
    s = tc_remote_shard_get(g, &plan, "smoke/weights", 2, 6, local, partial);
    if (s != TC_OK) {
        fprintf(stderr, "rank %d: shard_get partial failed status=%d\n", my_rank, s);
        fails++;
    } else {
        int mismatches = 0;
        for (int r = 0; r < 4; ++r) {
            const int gr = 2 + r;
            for (int c = 0; c < COLS; ++c) {
                const float want = (float)(100 * gr + c);
                if (fabsf(partial[r * COLS + c] - want) > 1e-6f) mismatches++;
            }
        }
        if (mismatches > 0) {
            fprintf(stderr, "rank %d: partial fetch had %d mismatches\n",
                    my_rank, mismatches);
            fails++;
        } else if (my_rank == 0) {
            printf("  PASS rank 0 shard_get cross_owner_range\n");
        }
    }

    /* ===== Push protocol: publish_put + drain_puts =====
     *
     * Non-owner (rank 0) publishes a one-row update for row 4 (owner=1)
     * setting every cell to 7777. Owner (rank 1) calls drain_puts;
     * verifies the cell was applied. Symmetric: rank 1 publishes a
     * put for row 1 (owner=0), rank 0 drains. */
    {
        float push_row[COLS];
        for (int c = 0; c < COLS; ++c) push_row[c] = 7777.0f + (float)c;
        const int target_row = (my_rank == 0) ? 4 : 1;
        const int target_owner = tc_remote_shard_owner(&plan, target_row);
        if (target_owner != my_rank) {
            /* I'm the publisher for this target row. */
            tc_status_t ps = tc_remote_shard_publish_put(g, &plan, "smoke/weights",
                                                          target_row, target_row + 1,
                                                          push_row);
            if (ps != TC_OK) {
                fprintf(stderr, "rank %d: publish_put failed status=%d\n", my_rank, ps);
                fails++;
            }
        }
        /* Both ranks try to drain. Owners will see one applied put;
         * non-owners will see zero (their drain just polls peers and
         * finds no put with the right header). */
        /* Allow publisher's registration to propagate. */
        sleep(1);
        int32_t applied = -1;
        tc_status_t ds = tc_remote_shard_drain_puts(g, &plan, "smoke/weights",
                                                      local, &applied);
        if (ds != TC_OK) {
            fprintf(stderr, "rank %d: drain_puts failed status=%d\n", my_rank, ds);
            fails++;
        }
        /* Owner must have applied exactly 1 put; non-owner applies 0. */
        const int my_local_row = (my_rank == 0) ? 1 : 4;
        const int my_lo = (my_rank == 0) ? 0 : 4;
        if (applied != 1) {
            fprintf(stderr, "rank %d: expected applied=1, got %d\n",
                    my_rank, applied);
            fails++;
        } else {
            /* Check our local copy now carries 7777+c at the relevant row. */
            int mism = 0;
            for (int c = 0; c < COLS; ++c) {
                const float got = local[(my_local_row - my_lo) * COLS + c];
                const float want = 7777.0f + (float)c;
                if (fabsf(got - want) > 1e-6f) mism++;
            }
            if (mism > 0) {
                fprintf(stderr, "rank %d: drain_puts applied wrong values "
                        "(%d cells mismatch on local row %d)\n",
                        my_rank, mism, my_local_row);
                fails++;
            } else if (my_rank == 0) {
                printf("  PASS rank 0 shard_drain_puts applied=1\n");
            }
        }
    }

    if (my_rank == 0) {
        printf("  total mesh bytes shipped through rank %d: %llu\n",
               my_rank, (unsigned long long)tc_mesh_total_bytes(g));
    }

    if (tc_mesh_group_shutdown(g) != TC_OK) {
        fprintf(stderr, "rank %d: group_shutdown failed\n", my_rank);
        fails++;
    }
    tc_shutdown(ctx);
    return fails;
}

int main(void) {
    const char* rank0_url = "tcp://127.0.0.1:54320";
    const char* rank1_url = "tcp://127.0.0.1:54321";

    pid_t child = fork();
    if (child < 0) { perror("fork"); return 1; }

    if (child == 0) {
        sleep(1);  /* let rank 0 bind first */
        int rc = run_peer(1, rank1_url, rank0_url);
        _exit(rc == 0 ? 0 : 1);
    }

    int rc0 = run_peer(0, rank0_url, rank1_url);
    int status = 0;
    waitpid(child, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

    const int total = rc0 + rc1;
    printf("\n  rank0 fails=%d  rank1 fails=%d  → %s\n",
           rc0, rc1, total == 0 ? "ALL PASS" : "SOME FAIL");
    return total == 0 ? 0 : 1;
}
