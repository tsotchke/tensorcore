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
    test_gguf
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
