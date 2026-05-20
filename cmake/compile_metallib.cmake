# compile_metallib.cmake
#
# Compile a set of .metal source files into a single default.metallib via
#     .metal --[xcrun metal]--> .air --[xcrun metallib]--> default.metallib
#
# This is the qgt-style precompiled path: zero runtime compile cost, and the
# resulting .metallib loads via [MTLDevice newDefaultLibraryWithBundle:] or
# [MTLDevice newLibraryWithURL:].
#
# Usage:
#   include(compile_metallib)
#   tc_compile_metallib(
#       TARGET    tensorcore_metallib
#       SOURCES   ${TC_METAL_SOURCES}
#       OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
#       FLAGS     -ffast-math -mmacosx-version-min=12.0
#       STD       metal3.0
#   )

function(tc_compile_metallib)
    cmake_parse_arguments(TC_MLIB
        ""
        "TARGET;OUTPUT;STD"
        "SOURCES;FLAGS"
        ${ARGN}
    )

    if(NOT APPLE)
        message(FATAL_ERROR "tc_compile_metallib requires Apple platform")
    endif()
    if(NOT TC_MLIB_TARGET)
        message(FATAL_ERROR "tc_compile_metallib: TARGET is required")
    endif()
    if(NOT TC_MLIB_OUTPUT)
        message(FATAL_ERROR "tc_compile_metallib: OUTPUT is required")
    endif()
    if(NOT TC_MLIB_STD)
        # Metal 3.0 = macOS 13+. Metal 3.1 = macOS 14+ (bf16). Metal 3.2 = macOS 15+.
        set(TC_MLIB_STD "metal3.1")
    endif()

    set(_air_dir "${CMAKE_CURRENT_BINARY_DIR}/metal_air")
    file(MAKE_DIRECTORY "${_air_dir}")

    set(_air_files "")
    foreach(_metal_src IN LISTS TC_MLIB_SOURCES)
        get_filename_component(_name "${_metal_src}" NAME_WE)
        set(_air "${_air_dir}/${_name}.air")
        add_custom_command(
            OUTPUT  "${_air}"
            COMMAND xcrun -sdk macosx metal
                        -std=${TC_MLIB_STD}
                        ${TC_MLIB_FLAGS}
                        -c "${_metal_src}"
                        -o "${_air}"
            DEPENDS "${_metal_src}"
            COMMENT "Compiling Metal kernel: ${_name}.metal -> ${_name}.air"
            VERBATIM
        )
        list(APPEND _air_files "${_air}")
    endforeach()

    add_custom_command(
        OUTPUT  "${TC_MLIB_OUTPUT}"
        COMMAND xcrun -sdk macosx metallib
                    ${_air_files}
                    -o "${TC_MLIB_OUTPUT}"
        DEPENDS ${_air_files}
        COMMENT "Linking ${TC_MLIB_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${TC_MLIB_TARGET} ALL DEPENDS "${TC_MLIB_OUTPUT}")
endfunction()
