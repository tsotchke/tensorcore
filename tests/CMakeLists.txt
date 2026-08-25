# tensorcore tests
#
# Metal tests link against tensorcore + Accelerate. CPU-only builds use the
# portable smoke test and return before the Apple-specific test list.

find_package(Threads REQUIRED)

add_test(
    NAME test_cuda_architecture_policy
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_architecture_policy.cmake
)
set_tests_properties(test_cuda_architecture_policy PROPERTIES TIMEOUT 30)

# Family selection is a pure policy layer and must compile identically on
# macOS, Linux, and Windows even though only macOS performs MTLDevice probes.
add_executable(test_apple_family_policy test_apple_family_policy.cpp)
target_include_directories(test_apple_family_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib)
add_test(NAME test_apple_family_policy COMMAND test_apple_family_policy)
set_tests_properties(test_apple_family_policy PROPERTIES TIMEOUT 30)

add_executable(test_runtime_capabilities test_runtime_capabilities.c)
target_link_libraries(test_runtime_capabilities PRIVATE tensorcore)
target_include_directories(test_runtime_capabilities PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_runtime_capabilities COMMAND test_runtime_capabilities)
set_tests_properties(test_runtime_capabilities PROPERTIES TIMEOUT 30)
if(TC_ENABLE_METAL)
    set_tests_properties(test_runtime_capabilities PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    )
endif()

# Async DiLoCo is implemented in the platform-independent distributed layer.
# Keep its snapshot/rebase and peer-loss state-machine coverage enabled on
# both the portable CPU and Apple/Metal builds.
add_executable(test_diloco_async test_diloco_async.c)
target_link_libraries(test_diloco_async PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_async PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_async COMMAND test_diloco_async)
set_tests_properties(test_diloco_async PROPERTIES TIMEOUT 60)

add_executable(test_diloco_async_failure_fork test_diloco_async_failure_fork.c)
target_link_libraries(test_diloco_async_failure_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_async_failure_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_async_failure_fork COMMAND test_diloco_async_failure_fork)
set_tests_properties(test_diloco_async_failure_fork PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 60
)

add_executable(test_diloco_checkpoint test_diloco_checkpoint.c)
target_link_libraries(test_diloco_checkpoint PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_checkpoint PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_checkpoint COMMAND test_diloco_checkpoint)
set_tests_properties(test_diloco_checkpoint PROPERTIES TIMEOUT 60)

add_executable(test_diloco_state_abi test_diloco_state_abi.c)
target_link_libraries(test_diloco_state_abi PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_state_abi PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_state_abi COMMAND test_diloco_state_abi)
set_tests_properties(test_diloco_state_abi PROPERTIES TIMEOUT 60)

add_executable(test_transport_auth test_transport_auth.cpp)
target_link_libraries(test_transport_auth PRIVATE
    tensorcore ${ACCELERATE_FRAMEWORK} Threads::Threads)
target_include_directories(test_transport_auth PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_transport_auth COMMAND test_transport_auth)
set_tests_properties(test_transport_auth PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 90
)
if(TC_ENABLE_METAL)
    set_tests_properties(test_transport_auth PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    )
endif()

# ---------------------------------------------------------------------
# Backend-independent geometry + manifold tests.
#
# These are fp32 host code with no Metal or CUDA dependency, so they must
# register on EVERY build. They are defined here, above the backend split,
# because the non-Metal branch below ends in return() -- anything declared
# after it is silently Apple-only. That is how these seven went unregistered
# on every Linux/CUDA build: 30 of 53 tests ran there, and the 23 missing
# ones included the entire manifold family. Nothing failed; they were just
# never built, which is the quietest way for a test to stop being a test.
# ---------------------------------------------------------------------
set(TC_BACKEND_INDEPENDENT_TESTS
    test_poincare
    test_sphere
    test_product_manifold
    test_quantum_gates
    test_phase_attention
    test_riemannian_adam
    test_riemannian_offmanifold
    test_riemannian_adam_parity
)
foreach(t IN LISTS TC_BACKEND_INDEPENDENT_TESTS)
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 60)
endforeach()

if(NOT TC_ENABLE_METAL)
    add_executable(test_portable_cpu test_portable_cpu.c)
    target_link_libraries(test_portable_cpu PRIVATE tensorcore)
    target_include_directories(test_portable_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_portable_cpu COMMAND test_portable_cpu)
    set_tests_properties(test_portable_cpu PROPERTIES TIMEOUT 120)

    add_executable(test_amx_gemm test_amx_gemm.c)
    target_link_libraries(test_amx_gemm PRIVATE tensorcore)
    target_include_directories(test_amx_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_amx_gemm COMMAND test_amx_gemm)
    set_tests_properties(test_amx_gemm PROPERTIES
        TIMEOUT 120
        SKIP_RETURN_CODE 77
    )

    add_executable(test_amx_edge test_amx_edge.c)
    target_link_libraries(test_amx_edge PRIVATE tensorcore)
    target_include_directories(test_amx_edge PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_amx_edge COMMAND test_amx_edge)
    set_tests_properties(test_amx_edge PROPERTIES
        TIMEOUT 120
        SKIP_RETURN_CODE 77
    )

    add_executable(test_amx_probe test_amx_probe.c)
    target_link_libraries(test_amx_probe PRIVATE tensorcore)
    target_include_directories(test_amx_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_amx_probe COMMAND test_amx_probe)
    set_tests_properties(test_amx_probe PROPERTIES TIMEOUT 30)

    add_executable(test_conv2d test_conv2d.c)
    target_link_libraries(test_conv2d PRIVATE tensorcore)
    target_include_directories(test_conv2d PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_conv2d COMMAND test_conv2d)
    set_tests_properties(test_conv2d PROPERTIES TIMEOUT 120)

    add_executable(test_sparse_24 test_sparse_24.c)
    target_link_libraries(test_sparse_24 PRIVATE tensorcore)
    target_include_directories(test_sparse_24 PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_sparse_24 COMMAND test_sparse_24)
    set_tests_properties(test_sparse_24 PROPERTIES TIMEOUT 30)

    add_executable(test_training_kernels test_training_kernels.c)
    target_link_libraries(test_training_kernels PRIVATE tensorcore)
    target_include_directories(test_training_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_training_kernels COMMAND test_training_kernels)
    set_tests_properties(test_training_kernels PROPERTIES TIMEOUT 120)

    add_executable(test_e2e_training test_e2e_training.c)
    target_link_libraries(test_e2e_training PRIVATE tensorcore)
    target_include_directories(test_e2e_training PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_e2e_training COMMAND test_e2e_training)
    set_tests_properties(test_e2e_training PROPERTIES TIMEOUT 120)

    # DiLoCo single-rank runtime smoke; runs everywhere since the runtime
    # is platform-independent (lives in lib/distributed/diloco.cpp).
    add_executable(test_diloco test_diloco.c)
    target_link_libraries(test_diloco PRIVATE tensorcore)
    target_include_directories(test_diloco PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco COMMAND test_diloco)
    set_tests_properties(test_diloco PROPERTIES TIMEOUT 60)

    # Sparse top-k compression round-trip - exercises lib/distributed/sparse_compress.cpp.
    add_executable(test_sparse_compress test_sparse_compress.c)
    target_link_libraries(test_sparse_compress PRIVATE tensorcore)
    target_include_directories(test_sparse_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_sparse_compress COMMAND test_sparse_compress)
    set_tests_properties(test_sparse_compress PROPERTIES TIMEOUT 30)

    # GLOO multi-rank fork test: actual cross-process allreduce + broadcast
    # + barrier via TCP sockets on 127.0.0.1. Validates the public
    # TC_DIST_GLOO dispatch end-to-end at world_size=4.
    add_executable(test_gloo_fork test_gloo_fork.c)
    target_link_libraries(test_gloo_fork PRIVATE tensorcore)
    target_include_directories(test_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_gloo_fork COMMAND test_gloo_fork)
    set_tests_properties(test_gloo_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Multi-rank DiLoCo over GLOO: end-to-end cross-process DiLoCo training
    # with two ranks taking different inner-step updates and converging
    # at the outer-step boundary. Proves the cross-continent training story
    # locally without needing two machines.
    add_executable(test_diloco_gloo_fork test_diloco_gloo_fork.c)
    target_link_libraries(test_diloco_gloo_fork PRIVATE tensorcore)
    target_include_directories(test_diloco_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco_gloo_fork COMMAND test_diloco_gloo_fork)
    set_tests_properties(test_diloco_gloo_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Sparse-compressed multi-rank DiLoCo: exercises the TOPK path through
    # GLOO sparse_allreduce. Verifies bytes-on-the-wire is < dense.
    add_executable(test_diloco_sparse_fork test_diloco_sparse_fork.c)
    target_link_libraries(test_diloco_sparse_fork PRIVATE tensorcore)
    target_include_directories(test_diloco_sparse_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_diloco_sparse_fork COMMAND test_diloco_sparse_fork)
    set_tests_properties(test_diloco_sparse_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # 4-rank ring all-reduce fork test. Exercises ring topology setup +
    # reduce-scatter + all-gather over loopback (where Tailscale NAT
    # issues don't apply).
    add_executable(test_gloo_ring_fork test_gloo_ring_fork.c)
    target_link_libraries(test_gloo_ring_fork PRIVATE tensorcore)
    target_include_directories(test_gloo_ring_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_gloo_ring_fork COMMAND test_gloo_ring_fork)
    set_tests_properties(test_gloo_ring_fork PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Activation checkpointing real impl: discard frees buffer storage,
    # realize re-allocs + invokes user callback to refill.
    add_executable(test_checkpoint test_checkpoint.c)
    target_link_libraries(test_checkpoint PRIVATE tensorcore)
    target_include_directories(test_checkpoint PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_checkpoint COMMAND test_checkpoint)
    set_tests_properties(test_checkpoint PROPERTIES
        TIMEOUT 30
    )

    add_executable(test_checkpoint_concurrency test_checkpoint_concurrency.cpp)
    target_link_libraries(test_checkpoint_concurrency PRIVATE tensorcore Threads::Threads)
    target_include_directories(test_checkpoint_concurrency PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_checkpoint_concurrency COMMAND test_checkpoint_concurrency)
    set_tests_properties(test_checkpoint_concurrency PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 30
    )

    # Split-binary remote distributed test for real multi-machine deployment.
    # Run manually with --rank/--world/--url. On Windows, CTest runs a small
    # two-process loopback launcher because fork() tests are not available.
    add_executable(test_dist_remote test_dist_remote.c)
    target_link_libraries(test_dist_remote PRIVATE tensorcore)
    target_include_directories(test_dist_remote PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(WIN32)
        add_test(NAME test_dist_remote_local
            COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                    -File ${CMAKE_SOURCE_DIR}/scripts/run_windows_gloo_smoke.ps1
                    -Exe $<TARGET_FILE:test_dist_remote>
        )
        set_tests_properties(test_dist_remote_local PROPERTIES TIMEOUT 60)
    endif()

    # CUDA backend smoke + perf gate. Only compiled when TC_ENABLE_CUDA=ON.
    # At runtime, exits 77 (skip) when no NVIDIA device is present.
    if(TC_ENABLE_CUDA)
        add_executable(test_cuda_gemm test_cuda_gemm.c)
        target_link_libraries(test_cuda_gemm PRIVATE tensorcore)
        target_include_directories(test_cuda_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_cuda_gemm COMMAND test_cuda_gemm)
        set_tests_properties(test_cuda_gemm PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
        )
    endif()

    # HIP/chipStar runtime probe. Only compiled when TC_ENABLE_HIP=ON.
    # At runtime, exits 77 (skip) when no HIP/chipStar device is present.
    if(TC_ENABLE_HIP)
        add_executable(test_hip_device test_hip_device.c)
        target_link_libraries(test_hip_device PRIVATE tensorcore)
        target_include_directories(test_hip_device PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_hip_device COMMAND test_hip_device)
        set_tests_properties(test_hip_device PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
        )
    endif()

    # HIP/chipStar GEMM smoke. Only compiled when hipBLAS was found.
    # At runtime, exits 77 (skip) when no HIP/chipStar device is present.
    if(TC_ENABLE_HIPBLAS)
        add_executable(test_hip_gemm test_hip_gemm.c)
        target_link_libraries(test_hip_gemm PRIVATE tensorcore)
        target_include_directories(test_hip_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_hip_gemm COMMAND test_hip_gemm)
        set_tests_properties(test_hip_gemm PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
        )
    endif()

    # The remote/mesh transport is backend-independent. Exercise its legacy
    # framing and lifecycle coverage on portable Unix builds as well as Metal.
    if(NOT WIN32)
        add_executable(test_mesh_collective test_mesh_collective.c)
        target_link_libraries(test_mesh_collective PRIVATE tensorcore Threads::Threads)
        target_include_directories(test_mesh_collective PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_mesh_collective COMMAND test_mesh_collective)
        set_tests_properties(test_mesh_collective PROPERTIES TIMEOUT 30)

        add_executable(test_remote_tensor_fork test_remote_tensor_fork.c)
        target_link_libraries(test_remote_tensor_fork PRIVATE tensorcore Threads::Threads)
        target_include_directories(test_remote_tensor_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_remote_tensor_fork COMMAND test_remote_tensor_fork)
        set_tests_properties(test_remote_tensor_fork PROPERTIES
            TIMEOUT 60
            SKIP_RETURN_CODE 77
        )

        add_executable(test_remote_tensor_adversarial test_remote_tensor_adversarial.cpp)
        target_link_libraries(test_remote_tensor_adversarial PRIVATE tensorcore Threads::Threads)
        target_include_directories(test_remote_tensor_adversarial PRIVATE ${CMAKE_SOURCE_DIR}/include)
        add_test(NAME test_remote_tensor_adversarial COMMAND test_remote_tensor_adversarial)
        set_tests_properties(test_remote_tensor_adversarial PROPERTIES TIMEOUT 15)
    endif()
    return()
endif()

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
    test_tensorops_select
    test_tensorops_runtime
)

foreach(t IN LISTS TC_TESTS)
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 120)
    set_tests_properties(${t} PROPERTIES
        ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    )
endforeach()

# DiLoCo single-rank runtime smoke - same source as the CPU build's test,
# runs on Apple too because the runtime is platform-independent.
add_executable(test_diloco test_diloco.c)
target_link_libraries(test_diloco PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco COMMAND test_diloco)
set_tests_properties(test_diloco PROPERTIES TIMEOUT 60)

# Poincaré ball ops on the Apple/Metal build (the CPU primitives compile + run
# on both backends since they're fp32 host code).




# Fork-based 2-peer mesh collective: AllReduce/Broadcast/AllGather over
# the production tc_remote_tensor_fetch transport. Bound to localhost.
add_executable(test_mesh_collective test_mesh_collective.c)
target_link_libraries(test_mesh_collective PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_mesh_collective PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_mesh_collective COMMAND test_mesh_collective)
set_tests_properties(test_mesh_collective PROPERTIES TIMEOUT 30)

# Fork-based 2-peer remote-shard smoke: register + owner-routed get over
# the same tc_remote transport as the mesh collectives.
add_executable(test_remote_shard test_remote_shard.c)
target_link_libraries(test_remote_shard PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_remote_shard PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_remote_shard COMMAND test_remote_shard)
set_tests_properties(test_remote_shard PROPERTIES TIMEOUT 30)




add_executable(test_sparse_24 test_sparse_24.c)
target_link_libraries(test_sparse_24 PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_sparse_24 PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_sparse_24 COMMAND test_sparse_24)
set_tests_properties(test_sparse_24 PROPERTIES TIMEOUT 30)

add_executable(test_remote_tensor_fork test_remote_tensor_fork.c)
target_link_libraries(test_remote_tensor_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_remote_tensor_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_remote_tensor_fork COMMAND test_remote_tensor_fork)
set_tests_properties(test_remote_tensor_fork PROPERTIES
    TIMEOUT 60
    SKIP_RETURN_CODE 77
)

add_executable(test_remote_tensor_adversarial test_remote_tensor_adversarial.cpp)
target_link_libraries(test_remote_tensor_adversarial PRIVATE
    tensorcore ${ACCELERATE_FRAMEWORK} Threads::Threads)
target_include_directories(test_remote_tensor_adversarial PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_remote_tensor_adversarial COMMAND test_remote_tensor_adversarial)
set_tests_properties(test_remote_tensor_adversarial PROPERTIES TIMEOUT 15)

# Split-binary remote distributed test. Launch manually with
# --rank/--world/--url against another machine.
add_executable(test_dist_remote test_dist_remote.c)
target_link_libraries(test_dist_remote PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_dist_remote PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Sparse top-k compression round-trip - exercises lib/distributed/sparse_compress.cpp.
add_executable(test_sparse_compress test_sparse_compress.c)
target_link_libraries(test_sparse_compress PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_sparse_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_sparse_compress COMMAND test_sparse_compress)
set_tests_properties(test_sparse_compress PROPERTIES TIMEOUT 30)

# GLOO multi-rank fork tests. Apple builds use distributed.mm, portable CPU
# builds use distributed_cpu.cpp, and both share lib/distributed/gloo_tcp.cpp.
add_executable(test_gloo_fork test_gloo_fork.c)
target_link_libraries(test_gloo_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_gloo_fork COMMAND test_gloo_fork)
set_tests_properties(test_gloo_fork PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 60
)

add_executable(test_diloco_gloo_fork test_diloco_gloo_fork.c)
target_link_libraries(test_diloco_gloo_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_gloo_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_gloo_fork COMMAND test_diloco_gloo_fork)
set_tests_properties(test_diloco_gloo_fork PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 60
)

add_executable(test_diloco_sparse_fork test_diloco_sparse_fork.c)
target_link_libraries(test_diloco_sparse_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_diloco_sparse_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_diloco_sparse_fork COMMAND test_diloco_sparse_fork)
set_tests_properties(test_diloco_sparse_fork PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 60
)

add_executable(test_gloo_ring_fork test_gloo_ring_fork.c)
target_link_libraries(test_gloo_ring_fork PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_gloo_ring_fork PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(test_gloo_ring_fork PRIVATE TC_TEST_METAL_BUILD=1)
add_test(NAME test_gloo_ring_fork COMMAND test_gloo_ring_fork)
set_tests_properties(test_gloo_ring_fork PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 60
)

# Activation checkpointing: discard frees storage while preserving the
# tc_buffer handle, then realize re-allocs + invokes user callback.
add_executable(test_checkpoint test_checkpoint.c)
target_link_libraries(test_checkpoint PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_checkpoint PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_checkpoint COMMAND test_checkpoint)
set_tests_properties(test_checkpoint PROPERTIES
    TIMEOUT 30
)

add_executable(test_checkpoint_concurrency test_checkpoint_concurrency.cpp)
target_link_libraries(test_checkpoint_concurrency PRIVATE
    tensorcore
    ${ACCELERATE_FRAMEWORK}
    Threads::Threads
)
target_include_directories(test_checkpoint_concurrency PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_checkpoint_concurrency COMMAND test_checkpoint_concurrency)
set_tests_properties(test_checkpoint_concurrency PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 30
)

add_executable(test_buffer_pool test_buffer_pool.mm)
target_link_libraries(test_buffer_pool PRIVATE tensorcore ${ACCELERATE_FRAMEWORK})
target_include_directories(test_buffer_pool PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib/core
)
add_test(NAME test_buffer_pool COMMAND test_buffer_pool)
set_tests_properties(test_buffer_pool PROPERTIES
    ENVIRONMENT "TC_METALLIB=${TC_METALLIB_PATH}"
    TIMEOUT 120
)

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
