add_executable(hello_gemm hello_gemm.c)
target_link_libraries(hello_gemm PRIVATE tensorcore)
target_include_directories(hello_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
