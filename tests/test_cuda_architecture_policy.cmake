include("${CMAKE_CURRENT_LIST_DIR}/../cmake/tensorcore_cuda_architectures.cmake")

function(expect_cuda_architectures version expected)
    tc_default_cuda_architectures("${version}" actual)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR
            "CUDA ${version} architecture policy mismatch: "
            "got '${actual}', expected '${expected}'")
    endif()
endfunction()

expect_cuda_architectures(
    "11.4.152"
    "70-real;72-real;75-real;80-real;86-real;86-virtual")
expect_cuda_architectures(
    "11.8.0"
    "70-real;72-real;75-real;80-real;86-real;89-real;90-real;90-virtual")
expect_cuda_architectures(
    "12.8.93"
    "70-real;72-real;75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual")
expect_cuda_architectures(
    "13.0.0"
    "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual")
