/* Three-rank decentralized mesh AllReduce smoke over production TCP fetches. */

#include "tensorcore/tensorcore.h"
#include "tensorcore/mesh_collective.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>


static int run_rank(int rank) {
    const char* urls[3] = {
        "tcp://127.0.0.1:54330",
        "tcp://127.0.0.1:54331",
        "tcp://127.0.0.1:54332",
    };
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK || !ctx) return 1;
    tc_mesh_group_t* group = NULL;
    if (tc_mesh_group_init(ctx, 3, rank, urls, &group) != TC_OK || !group) {
        tc_shutdown(ctx);
        return 1;
    }
    int failed = 0;
    if (strcmp(tc_mesh_allreduce_algorithm(group), "all_to_all") != 0)
        failed = 1;
    float values[8];
    for (size_t index = 0; index < 8; ++index)
        values[index] = (float)(rank + 1) * (float)(index + 1);
    if (tc_mesh_allreduce(
            group, values, 8, TC_COLL_DTYPE_F32,
            TC_REDUCE_SUM) != TC_OK) {
        failed = 1;
    }
    for (size_t index = 0; index < 8; ++index) {
        if (fabsf(values[index] - 6.0f * (float)(index + 1)) > 1e-5f)
            failed = 1;
    }
    if (tc_mesh_retained_snapshot_count(group) > 1) failed = 1;
    if (tc_mesh_group_shutdown(group) != TC_OK) failed = 1;
    if (tc_shutdown(ctx) != TC_OK) failed = 1;
    return failed;
}


int main(void) {
    pid_t rank1 = fork();
    if (rank1 < 0) return 1;
    if (rank1 == 0) {
        sleep(1);
        _exit(run_rank(1));
    }
    pid_t rank2 = fork();
    if (rank2 < 0) return 1;
    if (rank2 == 0) {
        sleep(2);
        _exit(run_rank(2));
    }
    const int rank0_status = run_rank(0);
    int status1 = 0, status2 = 0;
    waitpid(rank1, &status1, 0);
    waitpid(rank2, &status2, 0);
    const int failed = rank0_status || !WIFEXITED(status1) || WEXITSTATUS(status1) ||
        !WIFEXITED(status2) || WEXITSTATUS(status2);
    printf("three-rank decentralized AllReduce: %s\n", failed ? "FAIL" : "OK");
    return failed ? 1 : 0;
}
