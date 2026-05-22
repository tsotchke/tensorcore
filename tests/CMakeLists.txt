# tensorcore tests
#
# Metal tests link against tensorcore + Accelerate. CPU-only builds use the
# portable smoke test and return before the Apple-specific test list.

if(NOT TC_ENABLE_METAL)
    add_executable(test_portable_cpu test_portable_cpu.c)
    target_link_libraries(test_portable_cpu PRIVATE tensorcore)
    target_include_directories(test_portable_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_portable_cpu COMMAND test_portable_cpu)
    set_tests_properties(test_portable_cpu PROPERTIES TIMEOUT 120)

    add_executable(test_conv2d test_conv2d.c)
    target_link_libraries(test_conv2d PRIVATE tensorcore)
    target_include_directories(test_conv2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_conv2d COMMAND test_conv2d)
    set_tests_properties(test_conv2d PROPERTIES TIMEOUT 120)

    # DiLoCo single-rank runtime smoke; runs everywhere since the runtime
    # is platform-independent (lives in lib/distributed/diloco.cpp).
    add_executable(test_diloco test_diloco.c)
    target_link_libraries(test_diloco PRIVATE tensorcore)
    target_include_directories(test_diloco PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco COMMAND test_diloco)
    set_tests_properties(test_diloco PROPERTIES TIMEOUT 60)

    # Sparse top-k compression round-trip — exercises lib/distributed/sparse_compress.cpp.
    add_executable(test_sparse_compress test_sparse_compress.c)
    target_link_libraries(test_sparse_compress PRIVATE tensorcore)
    target_include_directories(test_sparse_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_sparse_compress COMMAND test_sparse_compress)
    set_tests_properties(test_sparse_compress PROPERTIES TIMEOUT 30)

    # GLOO multi-rank fork test: actual cross-process allreduce + broadcast
    # + barrier via TCP sockets on 127.0.0.1. Validates the public
    # TC_DIST_GLOO dispatch end-to-end at world_size=2.
    add_executable(test_gloo_fork test_gloo_fork.c)
    target_link_libraries(test_gloo_fork PRIVATE tensorcore)
    target_include_directories(test_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_gloo_fork COMMAND test_gloo_fork)
    set_tests_properties(test_gloo_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Multi-rank DiLoCo over GLOO: end-to-end cross-process DiLoCo training
    # with two ranks taking different inner-step updates and converging
    # at the outer-step boundary. Proves the cross-continent training story
    # locally without needing two machines.
    add_executable(test_diloco_gloo_fork test_diloco_gloo_fork.c)
    target_link_libraries(test_diloco_gloo_fork PRIVATE tensorcore)
    target_include_directories(test_diloco_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco_gloo_fork COMMAND test_diloco_gloo_fork)
    set_tests_properties(test_diloco_gloo_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Sparse-compressed multi-rank DiLoCo: exercises the TOPK path through
    # GLOO sparse_allreduce. Verifies bytes-on-the-wire is < dense.
    add_executable(test_diloco_sparse_fork test_diloco_sparse_fork.c)
    target_link_libraries(test_diloco_sparse_fork PRIVATE tensorcore)
    target_include_directories(test_diloco_sparse_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco_sparse_fork COMMAND test_diloco_sparse_fork)
    set_tests_properties(test_diloco_sparse_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
    return()
endif()

set(TC_TESTS
    test_device
    test_gemm_f32
    test_gemm_f16
    test_gemm_bf16
    test_gemm_i8
    test_attention_correctness
    test_attention_backward
    test_training_kernels
    test_transformer_block
    test_e2e_training
    test_conv2d
    test_distributed_ring
    test_quantized
    test_fused_norm_gemv
    test_distributed_ring_fork
    test_gguf
    test_tensorops_select
    test_tensorops_runtime
)

foreach(t IN LISTS TC_TESTS)
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 120)
    set_tests_properties(${t} PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    )
endforeach()

# DiLoCo single-rank runtime smoke — same source as the CPU build's test,
# runs on Apple too because the runtime is platform-independent.
add_executable(test_diloco test_diloco.c)
target_link_libraries(test_diloco PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco COMMAND test_diloco)
set_tests_properties(test_diloco PROPERTIES TIMEOUT 60)

# Sparse top-k compression round-trip — exercises lib/distributed/sparse_compress.cpp.
add_executable(test_sparse_compress test_sparse_compress.c)
target_link_libraries(test_sparse_compress PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_sparse_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_sparse_compress COMMAND test_sparse_compress)
set_tests_properties(test_sparse_compress PROPERTIES TIMEOUT 30)

# GLOO multi-rank fork test (CPU-only; Apple builds use distributed.mm which
# doesn't have the GLOO TCP transport wired yet). Only registered on the CPU
# branch via the earlier return().

add_executable(test_buffer_pool test_buffer_pool.mm)
target_link_libraries(test_buffer_pool PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_buffer_pool PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib/core
)
add_test(NAME test_buffer_pool COMMAND test_buffer_pool)
set_tests_properties(test_buffer_pool PROPERTIES
    ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    TIMEOUT 120
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE TC_PYTHON_NUMPY_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(TC_PYTHON_NUMPY_RESULT EQUAL 0)
        add_test(
            NAME python_basic
            COMMAND ${CMAKE_COMMAND} -E env
                "TENSORCORE_LIB=$<TARGET_FILE:tensorcore_shared>"
                "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/python/tests/test_basic.py
        )
        set_tests_properties(python_basic PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120
        )
    else()
        message(STATUS "tensorcore: Python tests disabled (numpy not importable)")
    endif()
else()
    message(STATUS "tensorcore: Python tests disabled (Python3 interpreter not found)")
endif()
