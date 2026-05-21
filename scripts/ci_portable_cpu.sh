#!/usr/bin/env bash
set -euo pipefail

build_dir="${TC_CPU_BUILD_DIR:-build-portable-cpu}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
tmp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
install_dir="${TC_CPU_INSTALL_DIR:-${tmp_root}/tensorcore-portable-cpu-install}"
consumer_dir="${TC_CPU_CONSUMER_DIR:-${tmp_root}/tensorcore-portable-cpu-consumer}"

cmake -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DTC_ENABLE_METAL=OFF \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_BENCH=OFF \
  -DTC_BUILD_EXAMPLES=OFF
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure

cmake -E rm -rf "$install_dir" "$consumer_dir"
cmake --install "$build_dir" --prefix "$install_dir"
cmake -E make_directory "$consumer_dir"

cat > "$consumer_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(tensorcore_portable_cpu_consumer C)

find_package(tensorcore CONFIG REQUIRED)

add_executable(portable_cpu_consumer main.c)
target_link_libraries(portable_cpu_consumer PRIVATE tensorcore::tensorcore)
CMAKE

cat > "$consumer_dir/main.c" <<'C'
#include "tensorcore/tensorcore.h"

#include <math.h>
#include <stdio.h>

static int check_status(const char* label, tc_status_t got) {
    if (got == TC_OK) return 0;
    fprintf(stderr, "%s failed: %s\n", label, tc_status_string(got));
    return 1;
}

int main(void) {
    tc_context* ctx = 0;
    tc_buffer* A = 0;
    tc_buffer* B = 0;
    tc_buffer* C = 0;
    float* Ap = 0;
    float* Bp = 0;
    float* Cp = 0;
    int rc = 0;

    if (check_status("tc_init", tc_init(&ctx))) return 1;
    if (check_status("tc_buffer_alloc(A)", tc_buffer_alloc(ctx, 4 * sizeof(float), &A))) rc = 1;
    if (check_status("tc_buffer_alloc(B)", tc_buffer_alloc(ctx, 4 * sizeof(float), &B))) rc = 1;
    if (check_status("tc_buffer_alloc(C)", tc_buffer_alloc(ctx, 4 * sizeof(float), &C))) rc = 1;
    if (rc) goto cleanup;

    if (check_status("tc_buffer_map(A)", tc_buffer_map(A, (void**)&Ap))) rc = 1;
    if (check_status("tc_buffer_map(B)", tc_buffer_map(B, (void**)&Bp))) rc = 1;
    if (check_status("tc_buffer_map(C)", tc_buffer_map(C, (void**)&Cp))) rc = 1;
    if (rc) goto cleanup;

    Ap[0] = 1.0f; Ap[1] = 2.0f; Ap[2] = 3.0f; Ap[3] = 4.0f;
    Bp[0] = 5.0f; Bp[1] = 6.0f; Bp[2] = 7.0f; Bp[3] = 8.0f;
    Cp[0] = 0.0f; Cp[1] = 0.0f; Cp[2] = 0.0f; Cp[3] = 0.0f;

    tc_gemm_desc d = {0};
    d.M = 2;
    d.N = 2;
    d.K = 2;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    if (check_status("tc_gemm", tc_gemm(ctx, &d, A, B, C))) rc = 1;
    if (tc_last_backend() != TC_BACKEND_PORTABLE_CPU) {
        fprintf(stderr, "unexpected backend: %s\n", tc_backend_name(tc_last_backend()));
        rc = 1;
    }
    if (fabsf(Cp[0] - 19.0f) > 1e-5f || fabsf(Cp[1] - 22.0f) > 1e-5f ||
        fabsf(Cp[2] - 43.0f) > 1e-5f || fabsf(Cp[3] - 50.0f) > 1e-5f) {
        fprintf(stderr, "unexpected GEMM result\n");
        rc = 1;
    }

cleanup:
    if (C) tc_buffer_free(ctx, C);
    if (B) tc_buffer_free(ctx, B);
    if (A) tc_buffer_free(ctx, A);
    if (ctx) tc_shutdown(ctx);
    return rc;
}
C

cmake -S "$consumer_dir" -B "$consumer_dir/build" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_PREFIX_PATH="$install_dir"
cmake --build "$consumer_dir/build" --parallel
"$consumer_dir/build/portable_cpu_consumer"

if command -v pkg-config >/dev/null 2>&1; then
  PKG_CONFIG_PATH="$install_dir/lib/pkgconfig" pkg-config --modversion tensorcore
  PKG_CONFIG_PATH="$install_dir/lib/pkgconfig" pkg-config --libs --static tensorcore
fi
