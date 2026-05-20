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
