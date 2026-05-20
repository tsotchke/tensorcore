add_executable(hello_gemm hello_gemm.c)
target_link_libraries(hello_gemm PRIVATE tensorcore)
target_include_directories(hello_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(gguf_inspect gguf_inspect.c)
target_link_libraries(gguf_inspect PRIVATE tensorcore)
target_include_directories(gguf_inspect PRIVATE ${CMAKE_SOURCE_DIR}/include)
