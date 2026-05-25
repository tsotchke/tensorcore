cmake_minimum_required(VERSION 3.20)

project(tensorcore
    VERSION 0.1.22
    LANGUAGES C CXX
    DESCRIPTION "CUDA-equivalent tensor-core acceleration for Apple Silicon"
)

# ----------------------------------------------------------------------------
# Platform gate
# ----------------------------------------------------------------------------
if(APPLE)
    set(TC_ENABLE_METAL_DEFAULT ON)
else()
    set(TC_ENABLE_METAL_DEFAULT OFF)
endif()
option(TC_ENABLE_METAL "Build the Apple Metal unified-memory GPU backend" ${TC_ENABLE_METAL_DEFAULT})

if(TC_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR "TC_ENABLE_METAL requires Apple platforms. Disable it for the portable CPU backend.")
endif()

if(TC_ENABLE_METAL)
    enable_language(OBJC OBJCXX)
endif()

if(TC_ENABLE_METAL AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
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

if(MSVC)
    set(TC_HOST_WARNING_FLAGS /W4)
else()
    set(TC_HOST_WARNING_FLAGS
        -Wall -Wextra -Wpedantic
    )
    if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND TC_HOST_WARNING_FLAGS -Wno-gnu-zero-variadic-macro-arguments)
    endif()
endif()
set(TC_HOST_COMPILE_LANGUAGES C CXX)
if(TC_ENABLE_METAL)
    list(APPEND TC_HOST_COMPILE_LANGUAGES OBJC OBJCXX)
endif()
foreach(_tc_lang IN LISTS TC_HOST_COMPILE_LANGUAGES)
    foreach(_tc_flag IN LISTS TC_HOST_WARNING_FLAGS)
        add_compile_options($<$<COMPILE_LANGUAGE:${_tc_lang}>:${_tc_flag}>)
    endforeach()
endforeach()
if(TC_ENABLE_METAL)
    add_compile_options($<$<COMPILE_LANGUAGE:OBJC>:-fobjc-arc>)
    add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
endif()
if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()
add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(TC_BUILD_TESTS  "Build correctness tests" ON)
option(TC_BUILD_BENCH  "Build TFLOPS bench harness" ON)
option(TC_BUILD_EXAMPLES "Build example programs" ON)
option(TC_ENABLE_TENSOROPS "Build Metal 4 mpp::tensor_ops kernels when the SDK supports them" ON)
option(TC_ENABLE_CUDA "Build direct CUDA backend for NVIDIA hardware (requires CUDA toolkit)" OFF)
option(TC_ENABLE_HIP "Build chipStar HIP backend (requires chipStar/LLVM 19/SPIRV)" OFF)

# ----------------------------------------------------------------------------
# CUDA / HIP backend detection
# ----------------------------------------------------------------------------
set(TC_CUDA_LINK_LIBS)
set(TC_HIP_LINK_LIBS)
set(TC_ENABLE_HIPBLAS OFF)

if(TC_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND TARGET CUDA::cudart AND TARGET CUDA::cublas)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        message(STATUS "tensorcore: CUDA backend enabled (CUDA ${CUDAToolkit_VERSION})")
        list(APPEND TC_CUDA_LINK_LIBS CUDA::cudart CUDA::cublas)
        add_compile_definitions(TC_ENABLE_CUDA=1)
    else()
        message(WARNING "tensorcore: TC_ENABLE_CUDA=ON but CUDA runtime/cuBLAS targets were not found; disabling")
        set(TC_ENABLE_CUDA OFF)
    endif()
endif()

if(TC_ENABLE_HIP)
    find_package(hip QUIET)
    set(TC_HIP_RUNTIME_TARGET)
    if(TARGET hip::host)
        set(TC_HIP_RUNTIME_TARGET hip::host)
    elseif(TARGET hip::device)
        set(TC_HIP_RUNTIME_TARGET hip::device)
    endif()
    if(hip_FOUND AND TC_HIP_RUNTIME_TARGET)
        message(STATUS "tensorcore: HIP runtime enabled")
        list(APPEND TC_HIP_LINK_LIBS ${TC_HIP_RUNTIME_TARGET})
        add_compile_definitions(TC_ENABLE_HIP=1)

        find_package(hipblas QUIET)
        set(TC_HIPBLAS_TARGET)
        if(TARGET roc::hipblas)
            set(TC_HIPBLAS_TARGET roc::hipblas)
        elseif(TARGET hipblas::hipblas)
            set(TC_HIPBLAS_TARGET hipblas::hipblas)
        endif()
        if(hipblas_FOUND AND TC_HIPBLAS_TARGET)
            set(TC_ENABLE_HIPBLAS ON)
            message(STATUS "tensorcore: HIP hipBLAS GEMM enabled")
            list(APPEND TC_HIP_LINK_LIBS ${TC_HIPBLAS_TARGET})
            add_compile_definitions(TC_ENABLE_HIPBLAS=1)
        else()
            message(WARNING "tensorcore: HIP runtime found but hipBLAS targets were not found; HIP GEMM disabled")
        endif()
    else()
        message(WARNING "tensorcore: TC_ENABLE_HIP=ON but HIP runtime targets were not found; disabling")
        set(TC_ENABLE_HIP OFF)
    endif()
endif()

set(TC_MACHO_COMPATIBILITY_VERSION
    "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.0"
)

set(TC_APPLE_FRAMEWORK_LINKS)
set(TC_PC_PRIVATE_LIBS)
set(TC_STATIC_RUNTIME_LINKS)
set(TC_CXX_RUNTIME_LIB)
if(MSVC)
    set(TC_CXX_RUNTIME_LIB)
elseif(APPLE)
    set(TC_CXX_RUNTIME_LIB c++)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(TC_CXX_RUNTIME_LIB stdc++)
elseif(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES MATCHES "(^|;)stdc\\+\\+(;|$)")
    set(TC_CXX_RUNTIME_LIB stdc++)
else()
    set(TC_CXX_RUNTIME_LIB c++)
endif()
if(TC_CXX_RUNTIME_LIB)
    list(APPEND TC_STATIC_RUNTIME_LINKS "$<$<LINK_LANGUAGE:C>:-l${TC_CXX_RUNTIME_LIB}>")
endif()
if(TC_ENABLE_METAL)
    # ------------------------------------------------------------------------
    # Frameworks
    # ------------------------------------------------------------------------
    find_library(METAL_FRAMEWORK            Metal                       REQUIRED)
    find_library(FOUNDATION_FRAMEWORK       Foundation                  REQUIRED)
    find_library(METALKIT_FRAMEWORK         MetalKit                    REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    find_library(ACCELERATE_FRAMEWORK       Accelerate                  REQUIRED)

    set(TC_APPLE_FRAMEWORK_LINKS
        "-framework Metal"
        "-framework Foundation"
        "-framework MetalKit"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph"
        "-framework Accelerate"
    )
    if(APPLE)
        list(APPEND TC_STATIC_RUNTIME_LINKS "$<$<LINK_LANGUAGE:C>:-lobjc>")
    endif()
    set(TC_PC_PRIVATE_LIBS "${TC_APPLE_FRAMEWORK_LINKS} -l${TC_CXX_RUNTIME_LIB} -lobjc")
elseif(TC_CXX_RUNTIME_LIB)
    set(TC_PC_PRIVATE_LIBS "-l${TC_CXX_RUNTIME_LIB}")
endif()

# ----------------------------------------------------------------------------
# OpenMP — used to parallelize the CPU dequant/requant passes in the BLAS
# bridge. Optional; missing OpenMP just leaves the loops serial.
# ----------------------------------------------------------------------------
find_package(OpenMP COMPONENTS C CXX QUIET)
if(OpenMP_FOUND)
    set(TC_HAS_OPENMP 1)
    message(STATUS "tensorcore: OpenMP available (${OpenMP_CXX_VERSION})")
else()
    set(TC_HAS_OPENMP 0)
endif()

# ----------------------------------------------------------------------------
# CBLAS detection for the portable CPU GEMM backend.
#
# When CBLAS is available, lib/ops/gemm_cpu.cpp delegates fp32 / fp16 GEMM to
# cblas_sgemm. Accelerate already ships cblas on macOS; on Linux/elsewhere we
# pick up OpenBLAS / MKL / Netlib via find_package(BLAS). This keeps the
# CPU-only backend correctness-first while avoiding the impractically slow
# triple-loop path for common fp32/fp16 GEMM calls.
# ----------------------------------------------------------------------------
set(TC_CBLAS_LIBRARIES "")
set(TC_CBLAS_PC_PRIVATE_LIBS "")
set(TC_HAS_CBLAS 0)
if(APPLE)
    if(NOT ACCELERATE_FRAMEWORK)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
    endif()
    if(ACCELERATE_FRAMEWORK)
        if(NOT TC_ENABLE_METAL)
            set(TC_CBLAS_LIBRARIES "-framework Accelerate")
            set(TC_CBLAS_PC_PRIVATE_LIBS "-framework Accelerate")
        endif()
        set(TC_HAS_CBLAS 1)
        message(STATUS "tensorcore: CBLAS via Accelerate")
    endif()
else()
    # Try Intel MKL first if installed (best perf on Intel + works on AMD).
    # Then OpenBLAS / Netlib via find_package(BLAS).
    find_path(MKL_INCLUDE_DIR mkl_cblas.h
        PATHS /usr/include /usr/include/mkl /opt/intel/oneapi/mkl/latest/include
              /opt/intel/mkl/include
    )
    find_library(MKL_RT_LIB NAMES mkl_rt
        PATHS /usr/lib/x86_64-linux-gnu /opt/intel/oneapi/mkl/latest/lib/intel64
              /opt/intel/mkl/lib/intel64
    )
    if(MKL_INCLUDE_DIR AND MKL_RT_LIB)
        set(TC_CBLAS_LIBRARIES "${MKL_RT_LIB}" pthread m dl)
        set(TC_CBLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
        set(TC_CBLAS_PC_PRIVATE_LIBS "-lmkl_rt -lpthread -lm -ldl")
        set(TC_HAS_CBLAS 1)
        message(STATUS "tensorcore: CBLAS via Intel MKL (${MKL_RT_LIB})")
    else()
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            find_path(CBLAS_INCLUDE_DIR cblas.h
                PATHS /usr/include /usr/local/include /opt/homebrew/include
                      /usr/include/x86_64-linux-gnu /usr/include/openblas
            )
            if(CBLAS_INCLUDE_DIR)
                set(TC_CBLAS_LIBRARIES ${BLAS_LIBRARIES})
                set(TC_CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})
                string(REPLACE ";" " " TC_CBLAS_PC_PRIVATE_LIBS "${BLAS_LIBRARIES}")
                set(TC_HAS_CBLAS 1)
                message(STATUS "tensorcore: CBLAS via system BLAS (${BLAS_LIBRARIES})")
            else()
                message(STATUS "tensorcore: BLAS found but cblas.h not located; CPU GEMM stays on the reference")
            endif()
        else()
            message(STATUS "tensorcore: no CBLAS found; CPU GEMM uses the triple-loop reference (<=1 GFLOPS)")
        endif()
    endif()
endif()
if(TC_CBLAS_PC_PRIVATE_LIBS)
    string(APPEND TC_PC_PRIVATE_LIBS " ${TC_CBLAS_PC_PRIVATE_LIBS}")
endif()

# ----------------------------------------------------------------------------
# Metal kernel compilation
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(TC_ENABLE_METAL)
    include(compile_metallib)
endif()

# ----------------------------------------------------------------------------
# SDK detection — Metal 4 / mpp::tensor_ops requires macOS 26.0+ SDK.
# We detect the SDK version and conditionally include the tensorops kernels.
# Build succeeds on any SDK >= 12.0; tensorops path only compiles on >= 26.0.
# ----------------------------------------------------------------------------
if(TC_ENABLE_METAL)
    execute_process(
        COMMAND xcrun --show-sdk-version
        OUTPUT_VARIABLE TC_SDK_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT TC_SDK_VERSION)
        set(TC_SDK_VERSION "0.0")
    endif()

    if(TC_SDK_VERSION VERSION_GREATER_EQUAL "26.0" AND TC_ENABLE_TENSOROPS)
        set(TC_HAVE_METAL4 ON)
        set(TC_METAL_STD "metal4.0")
        message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- ENABLING Metal 4 / mpp::tensor_ops path")
    elseif(TC_SDK_VERSION VERSION_GREATER_EQUAL "26.0")
        set(TC_HAVE_METAL4 OFF)
        set(TC_METAL_STD "metal3.1")
        message(STATUS "tensorcore: SDK ${TC_SDK_VERSION} -- Metal 4 / mpp::tensor_ops path disabled by TC_ENABLE_TENSOROPS=OFF")
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
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_quantized.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/gemm_quantized_v2.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/fused_norm_gemv.metal
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
else()
    set(TC_HAVE_METAL4 OFF)
    message(STATUS "tensorcore: Metal backend disabled; building portable CPU backend")
endif()

# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------
set(TC_COMMON_SOURCES
    lib/core/status.c
    lib/core/dtype.c
    lib/core/autotune.cpp
    lib/core/trace.cpp
    lib/io/gguf.c
    lib/tensorops/tensorops_select.c
    lib/distributed/diloco.cpp
    lib/distributed/sparse_compress.cpp
    lib/hip/device.cpp
    lib/hip/gemm.cpp
    lib/cuda/device.cpp
    lib/cuda/buffer.cpp
    lib/cuda/gemm.cpp
    lib/core/memory_tier_stub.cpp
    lib/core/checkpoint_stub.cpp
)
if(TC_ENABLE_CUDA)
    list(APPEND TC_COMMON_SOURCES lib/cuda/training.cu)
endif()

if(TC_ENABLE_METAL)
    set(TC_SOURCES
        ${TC_COMMON_SOURCES}
        lib/core/device.mm
        lib/core/pipeline_cache.mm
        lib/core/buffer_pool.mm
        lib/ops/gemm.mm
        lib/ops/attention.mm
        lib/ops/training.mm
        lib/ops/conv.mm
        lib/distributed/distributed.mm
        lib/distributed/ring_local.mm
        lib/distributed/gloo_tcp.cpp
        lib/ops/quantized.mm
        lib/fallback/mps_gemm.mm
        lib/fallback/accelerate_gemm.c
        lib/c_api/c_api.mm
    )
else()
    set(TC_SOURCES
        ${TC_COMMON_SOURCES}
        lib/core/device_cpu.cpp
        lib/ops/gemm_cpu.cpp
        lib/ops/gemm_cpu_avx2.cpp
        lib/ops/gemm_cpu_neon.cpp
        lib/ops/gemm_cpu_amx.cpp
        lib/ops/training_cpu.cpp
        lib/ops/attention_cpu.cpp
        lib/ops/conv2d_cpu.cpp
        lib/ops/quantized_cpu.cpp
        lib/ops/unsupported_cpu.cpp
        lib/distributed/distributed_cpu.cpp
        lib/distributed/gloo_tcp.cpp
        lib/c_api/c_api_cpu.c
    )
endif()

# AVX2 micro-kernel needs explicit -mavx2 -mfma flags on x86 so the source
# compiles even when the default arch is older than Haswell. On non-x86
# platforms the file builds a tiny stub and must not receive x86-only flags.
if(NOT TC_ENABLE_METAL)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
        include(CheckCXXCompilerFlag)
        if(MSVC)
            check_cxx_compiler_flag("/arch:AVX2" TC_COMPILER_SUPPORTS_ARCH_AVX2)
            if(TC_COMPILER_SUPPORTS_ARCH_AVX2)
                set_source_files_properties(lib/ops/gemm_cpu_avx2.cpp PROPERTIES
                    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>"
                )
            endif()
        else()
            check_cxx_compiler_flag("-mavx2" TC_COMPILER_SUPPORTS_MAVX2)
            check_cxx_compiler_flag("-mfma" TC_COMPILER_SUPPORTS_MFMA)
        endif()
        if(NOT MSVC AND TC_COMPILER_SUPPORTS_MAVX2 AND TC_COMPILER_SUPPORTS_MFMA)
            set_source_files_properties(lib/ops/gemm_cpu_avx2.cpp PROPERTIES
                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>"
            )
        endif()
    endif()

    # NEON micro-kernel: Apple Clang on arm64 defines __ARM_NEON by default,
    # so no extra flag is needed on Apple Silicon. On Linux ARM (Cortex-A78
    # class, Jetson Xavier / Orin), older GCC may need -march=armv8-a+simd.
    # On non-aarch64 the source file is a tiny stub and must not receive
    # arm-only flags.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=armv8-a+simd" TC_COMPILER_SUPPORTS_MARCH_NEON)
        if(TC_COMPILER_SUPPORTS_MARCH_NEON AND NOT APPLE)
            # Apple Clang already enables NEON for arm64 targets; on Linux ARM
            # this gate forces the SIMD extension to be on even when the host
            # default is older.
            set_source_files_properties(lib/ops/gemm_cpu_neon.cpp PROPERTIES
                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+simd>"
            )
        endif()
    endif()
endif()

if(TC_HAVE_METAL4)
    list(APPEND TC_SOURCES lib/tensorops/tensorops_m5.mm)
    add_compile_definitions(TC_HAVE_METAL4_SDK=1)
endif()

# Build BOTH static and shared; shared is what Python ctypes loads.
set(TC_WINDOWS_EXPORTS_DEF)
if(WIN32)
    set(TC_WINDOWS_EXPORTS_DEF "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.def")
    file(WRITE "${TC_WINDOWS_EXPORTS_DEF}" "EXPORTS\n")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorcore.exports"
        TC_WINDOWS_EXPORT_SYMBOLS
        REGEX "^_?tc_"
    )
    foreach(_tc_symbol IN LISTS TC_WINDOWS_EXPORT_SYMBOLS)
        string(STRIP "${_tc_symbol}" _tc_symbol)
        string(REGEX REPLACE "^_" "" _tc_symbol "${_tc_symbol}")
        file(APPEND "${TC_WINDOWS_EXPORTS_DEF}" "    ${_tc_symbol}\n")
    endforeach()
endif()
add_library(tensorcore STATIC ${TC_SOURCES})
add_library(tensorcore_shared SHARED ${TC_SOURCES} ${TC_WINDOWS_EXPORTS_DEF})
set_target_properties(tensorcore_shared PROPERTIES
    OUTPUT_NAME tensorcore
    MACHO_CURRENT_VERSION "${PROJECT_VERSION}"
    MACHO_COMPATIBILITY_VERSION "${TC_MACHO_COMPATIBILITY_VERSION}"
)
if(WIN32)
    set_target_properties(tensorcore_shared PROPERTIES PREFIX "")
endif()
add_library(tensorcore::tensorcore_shared ALIAS tensorcore_shared)
if(TC_ENABLE_METAL)
    add_dependencies(tensorcore_shared tensorcore_metallib)
endif()
target_include_directories(tensorcore_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
if(TC_ENABLE_METAL)
    target_compile_definitions(tensorcore_shared PRIVATE
        TC_METALLIB_PATH="${TC_METALLIB_PATH}"
    )
endif()
if(TC_HAS_CBLAS)
    target_compile_definitions(tensorcore_shared PRIVATE TC_HAS_CBLAS=1)
    if(TC_CBLAS_INCLUDE_DIRS)
        target_include_directories(tensorcore_shared PRIVATE ${TC_CBLAS_INCLUDE_DIRS})
    endif()
endif()
if(OpenMP_FOUND)
    # Keep OpenMP internal to the shared library. The static install target
    # remains dependency-light so downstream consumers do not need to discover
    # libomp just to use tensorcore's CMake package.
    target_link_libraries(tensorcore_shared PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(tensorcore_shared PUBLIC
    ${TC_APPLE_FRAMEWORK_LINKS}
    ${TC_CBLAS_LIBRARIES}
    ${TC_CUDA_LINK_LIBS}
    ${TC_HIP_LINK_LIBS}
)
if(TC_ENABLE_METAL)
    target_link_options(tensorcore_shared PRIVATE
        "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorcore.exports"
    )
endif()
add_library(tensorcore::tensorcore ALIAS tensorcore)
if(TC_ENABLE_METAL)
    add_dependencies(tensorcore tensorcore_metallib)
endif()

target_include_directories(tensorcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

if(TC_ENABLE_METAL)
    target_compile_definitions(tensorcore PRIVATE
        TC_METALLIB_PATH="${TC_METALLIB_PATH}"
    )
endif()
if(TC_HAS_CBLAS)
    target_compile_definitions(tensorcore PRIVATE TC_HAS_CBLAS=1)
    if(TC_CBLAS_INCLUDE_DIRS)
        target_include_directories(tensorcore PRIVATE ${TC_CBLAS_INCLUDE_DIRS})
    endif()
endif()
target_link_libraries(tensorcore PUBLIC
    ${TC_APPLE_FRAMEWORK_LINKS}
    ${TC_CBLAS_LIBRARIES}
    ${TC_CUDA_LINK_LIBS}
    ${TC_HIP_LINK_LIBS}
    ${TC_STATIC_RUNTIME_LINKS}
)

set_target_properties(tensorcore PROPERTIES
    OUTPUT_NAME tensorcore
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
if(WIN32)
    set_target_properties(tensorcore PROPERTIES
        OUTPUT_NAME tensorcore_static
    )
endif()

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
include(CMakePackageConfigHelpers)

install(TARGETS tensorcore tensorcore_shared
    EXPORT  tensorcoreTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/tensorcore DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(TC_ENABLE_METAL)
    install(FILES ${TC_METALLIB_PATH} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

set(TC_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/tensorcore")
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorcoreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcoreConfig.cmake"
    INSTALL_DESTINATION "${TC_CONFIG_INSTALL_DIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcoreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorcore.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.pc"
    @ONLY
)
install(EXPORT tensorcoreTargets
    NAMESPACE tensorcore::
    DESTINATION "${TC_CONFIG_INSTALL_DIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcoreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcoreConfigVersion.cmake"
    DESTINATION "${TC_CONFIG_INSTALL_DIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tensorcore.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

message(STATUS "tensorcore: configured (Metal=${TC_ENABLE_METAL}, Metal4=${TC_HAVE_METAL4}, TensorOps=${TC_ENABLE_TENSOROPS}, CUDA=${TC_ENABLE_CUDA}, HIP=${TC_ENABLE_HIP}, HIPBLAS=${TC_ENABLE_HIPBLAS}, tests=${TC_BUILD_TESTS}, bench=${TC_BUILD_BENCH})")
if(TC_ENABLE_METAL)
    message(STATUS "  metallib -> ${TC_METALLIB_PATH}")
endif()
