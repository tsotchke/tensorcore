cmake_minimum_required(VERSION 3.20)

project(tensorcore
    VERSION 0.1.0
    LANGUAGES C CXX OBJC OBJCXX
    DESCRIPTION "CUDA-equivalent tensor-core acceleration for Apple Silicon"
)

# ----------------------------------------------------------------------------
# Platform gate
# ----------------------------------------------------------------------------
if(NOT APPLE)
    message(FATAL_ERROR
        "tensorcore is Apple-Silicon-only. For NVIDIA, use the CUDA path in "
        "eshkol-platform/lib/backend/gpu/.")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(WARNING
        "tensorcore is built for arm64 (Apple Silicon). x86_64 build will "
        "succeed but Metal simdgroup_matrix kernels will not run.")
endif()

# ----------------------------------------------------------------------------
# Standards and flags
# ----------------------------------------------------------------------------
set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic -Wno-gnu-zero-variadic-macro-arguments
    -fobjc-arc
)
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(TC_BUILD_TESTS  "Build correctness tests" ON)
option(TC_BUILD_BENCH  "Build TFLOPS bench harness" ON)
option(TC_BUILD_EXAMPLES "Build example programs" ON)
option(TC_ENABLE_TENSOROPS "Compile the M5 / Metal 4 TensorOps path (macOS 26.2+)" OFF)

# ----------------------------------------------------------------------------
# Frameworks
# ----------------------------------------------------------------------------
find_library(METAL_FRAMEWORK            Metal                       REQUIRED)
find_library(FOUNDATION_FRAMEWORK       Foundation                  REQUIRED)
find_library(METALKIT_FRAMEWORK         MetalKit                    REQUIRED)
find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(ACCELERATE_FRAMEWORK       Accelerate                  REQUIRED)

# ----------------------------------------------------------------------------
# Metal kernel compilation
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(compile_metallib)

# ----------------------------------------------------------------------------
# SDK detection — Metal 4 / mpp::tensor_ops requires macOS 26.0+ SDK.
# We detect the SDK version and conditionally include the tensorops kernels.
# Build succeeds on any SDK >= 12.0; tensorops path only compiles on >= 26.0.
# ----------------------------------------------------------------------------
execute_process(
    COMMAND xcrun --show-sdk-version
    OUTPUT_VARIABLE TC_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT TC_SDK_VERSION)
    set(TC_SDK_VERSION "0.0")
endif()

if(TC_SDK_VERSION VERSION_GREATER_EQUAL "26.0")
    set(TC_HAVE_METAL4 ON)
    set(TC_METAL_STD "metal4.0")
    message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- ENABLING Metal 4 / mpp::tensor_ops path")
else()
    set(TC_HAVE_METAL4 OFF)
    set(TC_METAL_STD "metal3.1")
    message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- Metal 4 path requires SDK >= 26.0 (skipping)")
endif()

set(TC_METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_simdgroup.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_simdgroup_128.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/flash_attention.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/flash_attention_d128.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/flash_attention_backward.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/flash_attention_backward_d128.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/training_kernels.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/conv2d.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/conv2d_backward.metal
)

# The async_copy kernels use private AIR intrinsics via __asm. Xcode 16.x
# accepts them; Xcode 17+ / macOS 26+ rejects (per research). Gate the
# compilation by SDK version so the build succeeds either way; on newer
# SDKs the async-kernel name simply isn't present in the metallib and the
# host-side dispatch silently falls back to the sync path.
if(TC_SDK_VERSION VERSION_LESS "26.0")
    set(TC_HAVE_ASYNC_COPY ON)
    list(APPEND TC_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_async.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_async_128.metal
    )
    message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- ENABLING async_copy GEMM kernels")
else()
    set(TC_HAVE_ASYNC_COPY OFF)
    message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- DISABLING async_copy GEMM kernels (Xcode 17+ rejects __asm intrinsics)")
endif()

if(TC_HAVE_METAL4)
    list(APPEND TC_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/tensorops_gemm.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/tensorops_flash_attention.metal
    )
endif()

set(TC_METALLIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.metallib")

tc_compile_metallib(
    TARGET  tensorcore_metallib
    SOURCES ${TC_METAL_SOURCES}
    OUTPUT  ${TC_METALLIB_PATH}
    STD     ${TC_METAL_STD}
    FLAGS   -gline-tables-only
)

# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------
set(TC_SOURCES
    lib/core/status.c
    lib/core/dtype.c
    lib/core/device.mm
    lib/core/pipeline_cache.mm
    lib/core/buffer_pool.mm
    lib/core/autotune.cpp
    lib/ops/gemm.mm
    lib/ops/attention.mm
    lib/ops/training.mm
    lib/ops/conv.mm
    lib/distributed/distributed.mm
    lib/distributed/ring_local.mm
    lib/fallback/mps_gemm.mm
    lib/fallback/accelerate_gemm.c
    lib/c_api/c_api.mm
)

if(TC_HAVE_METAL4)
    list(APPEND TC_SOURCES lib/tensorops/tensorops_m5.mm)
    add_compile_definitions(TC_HAVE_METAL4_SDK=1)
endif()

add_library(tensorcore STATIC ${TC_SOURCES})
add_library(tensorcore::tensorcore ALIAS tensorcore)
add_dependencies(tensorcore tensorcore_metallib)

target_include_directories(tensorcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_compile_definitions(tensorcore PRIVATE
    TC_METALLIB_PATH="${TC_METALLIB_PATH}"
)

target_link_libraries(tensorcore PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${METALPERFORMANCESHADERS_FRAMEWORK}
    ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

set_target_properties(tensorcore PROPERTIES
    OUTPUT_NAME tensorcore
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(TC_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(TC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ----------------------------------------------------------------------------
# Install
# ----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS tensorcore
    EXPORT  tensorcoreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/tensorcore DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${TC_METALLIB_PATH} DESTINATION ${CMAKE_INSTALL_LIBDIR})

message(STATUS "tensorcore: configured (Metal4=${TC_HAVE_METAL4}, tests=${TC_BUILD_TESTS}, bench=${TC_BUILD_BENCH})")
message(STATUS "  metallib -> ${TC_METALLIB_PATH}")
