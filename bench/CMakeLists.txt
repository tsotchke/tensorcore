set(TC_BENCH
    bench_gemm
    bench_attention
    bench_inference_7b
)
foreach(b IN LISTS TC_BENCH)
    add_executable(${b} ${b}.c)
    target_link_libraries(${b} PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
    target_include_directories(${b} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()

# Shared-library GEMM benchmark for runtime paths that intentionally keep
# optional dependencies internal to libtensorcore, such as AVX2 OpenMP fanout.
add_executable(bench_gemm_shared bench_gemm.c)
target_link_libraries(bench_gemm_shared PRIVATE tensorcore_shared ${ACCELERATE_FRAMEWORK})
target_include_directories(bench_gemm_shared PRIVATE ${CMAKE_SOURCE_DIR}/include)
