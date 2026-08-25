function(tc_default_cuda_architectures toolkit_version out_var)
    if(toolkit_version VERSION_GREATER_EQUAL "13.0")
        set(archs 75 80 86 89)          # Turing and newer
        set(newest 89)
    elseif(toolkit_version VERSION_GREATER_EQUAL "11.8")
        set(archs 70 72 75 80 86 89)    # Volta/Xavier through Ada
        set(newest 89)
    else()
        # CUDA 11.4, used by JetPack 5 / L4T R35, predates Ada sm_89.
        set(archs 70 72 75 80 86)
        set(newest 86)
    endif()

    if(toolkit_version VERSION_GREATER_EQUAL "11.8")
        list(APPEND archs 90)            # Hopper
        set(newest 90)
    endif()
    if(toolkit_version VERSION_GREATER_EQUAL "12.8")
        list(APPEND archs 100 120)       # Blackwell datacenter and client
        set(newest 120)
    endif()

    list(TRANSFORM archs APPEND "-real")
    list(APPEND archs "${newest}-virtual")
    set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()
