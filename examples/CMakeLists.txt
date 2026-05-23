add_executable(hello_gemm hello_gemm.c)
target_link_libraries(hello_gemm PRIVATE tensorcore)
target_include_directories(hello_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(gguf_inspect gguf_inspect.c)
target_link_libraries(gguf_inspect PRIVATE tensorcore)
target_include_directories(gguf_inspect PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(decode_step decode_step.c)
target_link_libraries(decode_step PRIVATE tensorcore)
target_include_directories(decode_step PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(decode_step PRIVATE "-lm")

add_executable(training_step training_step.c)
target_link_libraries(training_step PRIVATE tensorcore)
target_include_directories(training_step PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(training_step PRIVATE "-lm")

add_executable(mesh_training_demo mesh_training_demo.c)
target_link_libraries(mesh_training_demo PRIVATE tensorcore)
target_include_directories(mesh_training_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mesh_training_demo PRIVATE "-lm")

# Register the assembly examples as CTest smoke tests so CI catches
# regressions in the inference + training primitive sequences.
if(TC_BUILD_TESTS AND TC_ENABLE_METAL)
    add_test(NAME example_decode_step COMMAND decode_step)
    set_tests_properties(example_decode_step PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
        TIMEOUT 60
    )

    add_test(NAME example_training_step COMMAND training_step)
    set_tests_properties(example_training_step PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
        TIMEOUT 60
    )
endif()

if(TC_BUILD_TESTS)
    add_test(NAME example_mesh_training_demo
        COMMAND mesh_training_demo --inner 2 --outer 1
    )
    set_tests_properties(example_mesh_training_demo PROPERTIES TIMEOUT 60)
    add_test(NAME example_mesh_training_demo_checkpoint
        COMMAND mesh_training_demo --inner 2 --outer 1 --checkpoint
    )
    set_tests_properties(example_mesh_training_demo_checkpoint PROPERTIES TIMEOUT 60)
    if(TC_ENABLE_METAL)
        set_tests_properties(example_mesh_training_demo PROPERTIES
            ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
        )
        set_tests_properties(example_mesh_training_demo_checkpoint PROPERTIES
            ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
        )
    endif()
endif()
