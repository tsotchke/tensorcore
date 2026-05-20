# tensorcore tests
#
# Each .c becomes one ctest case. Tests link against tensorcore + Accelerate.

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
)

foreach(t IN LISTS TC_TESTS)
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
        TIMEOUT 120
    )
endforeach()
