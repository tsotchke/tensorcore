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

find_program(TC_XCRUN_EXECUTABLE xcrun)

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
    if(NOT TC_MLIB_SOURCES)
        message(FATAL_ERROR "tc_compile_metallib: SOURCES is required")
    endif()
    if(NOT TC_XCRUN_EXECUTABLE)
        message(FATAL_ERROR "tc_compile_metallib: xcrun was not found; install Xcode command line tools")
    endif()
    if(NOT TC_MLIB_STD)
        # Metal 3.0 = macOS 13+. Metal 3.1 = macOS 14+ (bf16). Metal 3.2 = macOS 15+.
        set(TC_MLIB_STD "metal3.1")
    endif()

    execute_process(
        COMMAND "${TC_XCRUN_EXECUTABLE}" -sdk macosx -find metal
        OUTPUT_VARIABLE _tc_metal_tool
        ERROR_VARIABLE _tc_metal_tool_error
        RESULT_VARIABLE _tc_metal_tool_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _tc_metal_tool_result EQUAL 0 OR NOT _tc_metal_tool)
        message(FATAL_ERROR
            "tc_compile_metallib: xcrun could not locate the Metal compiler "
            "(xcrun -sdk macosx -find metal failed: ${_tc_metal_tool_error})")
    endif()

    execute_process(
        COMMAND "${TC_XCRUN_EXECUTABLE}" -sdk macosx -find metallib
        OUTPUT_VARIABLE _tc_metallib_tool
        ERROR_VARIABLE _tc_metallib_tool_error
        RESULT_VARIABLE _tc_metallib_tool_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _tc_metallib_tool_result EQUAL 0 OR NOT _tc_metallib_tool)
        message(FATAL_ERROR
            "tc_compile_metallib: xcrun could not locate the metallib linker "
            "(xcrun -sdk macosx -find metallib failed: ${_tc_metallib_tool_error})")
    endif()

    string(MAKE_C_IDENTIFIER "${TC_MLIB_TARGET}" _target_stem)
    set(_air_dir "${CMAKE_CURRENT_BINARY_DIR}/metal_air/${_target_stem}")
    file(MAKE_DIRECTORY "${_air_dir}")

    set(_air_files "")
    foreach(_metal_src IN LISTS TC_MLIB_SOURCES)
        get_filename_component(_metal_abs "${_metal_src}" ABSOLUTE)
        get_filename_component(_metal_dir "${_metal_abs}" DIRECTORY)
        file(RELATIVE_PATH _rel_metal_src "${CMAKE_CURRENT_SOURCE_DIR}" "${_metal_abs}")
        if(_rel_metal_src MATCHES "^\\.\\.")
            set(_rel_metal_src "${_metal_abs}")
        endif()
        string(MAKE_C_IDENTIFIER "${_rel_metal_src}" _air_stem)
        set(_variant_key "${TC_MLIB_TARGET}|${TC_MLIB_STD}|${TC_MLIB_FLAGS}|${_metal_abs}")
        string(SHA1 _air_hash "${_variant_key}")
        string(SUBSTRING "${_air_hash}" 0 12 _air_hash_short)
        set(_air "${_air_dir}/${_air_stem}-${_air_hash_short}.air")
        set(_depfile "${_air_dir}/${_air_stem}-${_air_hash_short}.d")

        add_custom_command(
            OUTPUT  "${_air}"
            COMMAND "${TC_XCRUN_EXECUTABLE}" -sdk macosx metal
                        -std=${TC_MLIB_STD}
                        -I "${_metal_dir}"
                        -MMD
                        -MP
                        -MF "${_depfile}"
                        -MT "${_air}"
                        -fdiagnostics-absolute-paths
                        -fdiagnostics-show-note-include-stack
                        ${TC_MLIB_FLAGS}
                        -c "${_metal_abs}"
                        -o "${_air}"
            DEPENDS "${_metal_abs}"
            DEPFILE "${_depfile}"
            BYPRODUCTS "${_depfile}"
            COMMENT "Compiling Metal kernel: ${_rel_metal_src} -> ${_air_stem}-${_air_hash_short}.air"
            VERBATIM
        )
        list(APPEND _air_files "${_air}")
    endforeach()

    list(LENGTH _air_files _air_count)
    add_custom_command(
        OUTPUT  "${TC_MLIB_OUTPUT}"
        COMMAND "${TC_XCRUN_EXECUTABLE}" -sdk macosx metallib
                    ${_air_files}
                    -o "${TC_MLIB_OUTPUT}"
        DEPENDS ${_air_files}
        COMMENT "Linking Metal library (${_air_count} AIR files): ${TC_MLIB_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${TC_MLIB_TARGET} ALL DEPENDS "${TC_MLIB_OUTPUT}")
endfunction()
