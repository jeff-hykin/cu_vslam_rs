# Build cuVSLAM's CUDA device code against CuMetal (https://github.com/Lulzx/cuda-metal) so it runs
# on Apple Silicon GPUs through Metal. There is no nvcc and no CUDA Toolkit on macOS, so this file
# stands in for both: it replaces the CUDA::* imported targets that FindCUDAToolkit would have
# produced, and it compiles .cu files with clang's own CUDA front end instead of using CMake's CUDA
# language (which cannot be enabled without a working nvcc).
#
# The compile line mirrors what cumetalc's executable driver does
# (cuda-metal/compiler/cumetalc/main.cpp, run_executable_driver): clang emits host code using the
# standard CUDA registration ABI and drives the device side through CuMetal's ptxas/fatbinary
# shims, which pass PTX into a fatbinary envelope rather than assembling SASS. libcumetal then JITs
# that PTX to a metallib on first launch. cumetalc itself is not reusable here because it only
# emits whole executables, not the relocatable objects a static library needs.

set(CUMETAL_ROOT "$ENV{HOME}/.local/cumetal" CACHE PATH "Path to the CuMetal installation prefix")
set(CUMETAL_CUDA_CLANG "/opt/homebrew/opt/llvm/bin/clang++" CACHE FILEPATH
    "CUDA-capable clang++ used to compile .cu files for CuMetal (Apple clang cannot do this)")
set(CUMETAL_CUDA_ARCH "sm_80" CACHE STRING "Virtual CUDA architecture clang targets before PTX handoff")

set(CUMETAL_INCLUDE_DIR ${CUMETAL_ROOT}/include)
set(CUMETAL_LIB_DIR ${CUMETAL_ROOT}/lib)
set(CUMETAL_TOOLCHAIN_DIR ${CUMETAL_ROOT}/libexec/cumetal/cuda_toolchain)
set(CUMETAL_COMPAT_DIR ${CMAKE_SOURCE_DIR}/cmake/cumetal_compat)

foreach(_required
        ${CUMETAL_INCLUDE_DIR}/cuda_runtime.h
        ${CUMETAL_LIB_DIR}/libcumetal.dylib
        ${CUMETAL_TOOLCHAIN_DIR}/ptxas
        ${CUMETAL_TOOLCHAIN_DIR}/fatbinary
        ${CUMETAL_CUDA_CLANG})
    if(NOT EXISTS ${_required})
        message(FATAL_ERROR "CuMetal build requires ${_required}, which does not exist. "
                            "Set CUMETAL_ROOT / CUMETAL_CUDA_CLANG, or build and install CuMetal first.")
    endif()
endforeach()

# clang needs the PTX feature level that matches the target architecture, exactly as cumetalc picks
# it in ptx_feature_flags_for_arch().
if(CUMETAL_CUDA_ARCH MATCHES "^sm_(80|86|89|90)")
    set(_cumetal_ptx_feature "--cuda-feature=+ptx70")
elseif(CUMETAL_CUDA_ARCH MATCHES "^sm_(75|78)")
    set(_cumetal_ptx_feature "--cuda-feature=+ptx63")
elseif(CUMETAL_CUDA_ARCH MATCHES "^sm_(70|72)")
    set(_cumetal_ptx_feature "--cuda-feature=+ptx60")
else()
    set(_cumetal_ptx_feature "")
endif()

set(CUMETAL_CUDA_FLAGS
    -x cuda
    -std=c++17
    --cuda-gpu-arch=${CUMETAL_CUDA_ARCH}
    ${_cumetal_ptx_feature}
    # CuMetal supplies its own CUDA headers and runtime, so clang must not look for a real toolkit.
    -nocudainc
    -nocudalib
    -Wno-unknown-cuda-version
    -Wno-pass-failed
    -D__CUDACC__=1
    -D__NVCC__=1
    -fPIC
    -fvisibility=hidden
    # This clang runs outside CMake's compiler abstraction, so the host half of every .cu would
    # otherwise be built against the SDK default rather than the project's deployment target.
    -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -mcpu=apple-m1
    # Compat dir first: it shadows a couple of CuMetal headers to close API gaps.
    -I${CUMETAL_COMPAT_DIR}
    -I${CUMETAL_INCLUDE_DIR}
    -include cuda_runtime.h
    -include cumetal_prelude.h
)

# Stand in for the CUDA::* imported targets the rest of the build links against. CuMetal implements
# the runtime, cuBLAS and cuSOLVER entry points in separate dylibs that all sit on libcumetal.
foreach(_component cudart_static cublas cusolver)
    if(NOT TARGET CUDA::${_component})
        add_library(CUDA::${_component} INTERFACE IMPORTED GLOBAL)
        target_include_directories(CUDA::${_component} INTERFACE
            ${CUMETAL_COMPAT_DIR} ${CUMETAL_INCLUDE_DIR})
    endif()
endforeach()

target_link_libraries(CUDA::cudart_static INTERFACE
    ${CUMETAL_LIB_DIR}/libcumetal.dylib
    "LINKER:-rpath,${CUMETAL_LIB_DIR}")
target_link_libraries(CUDA::cublas INTERFACE
    ${CUMETAL_LIB_DIR}/libcublas.dylib CUDA::cudart_static)
target_link_libraries(CUDA::cusolver INTERFACE
    ${CUMETAL_LIB_DIR}/libcusolver.dylib CUDA::cudart_static)

# Compile every .cu in ${sources} to an object file and return the list in ${out_objects_var}.
# The objects are added to a target as plain sources, which CMake links verbatim.
function(cumetal_compile_cuda_sources out_objects_var)
    set(_objects "")
    foreach(_source ${ARGN})
        get_filename_component(_absolute ${_source} ABSOLUTE)
        get_filename_component(_name ${_source} NAME)
        set(_object ${CMAKE_CURRENT_BINARY_DIR}/cumetal_objects/${_name}.o)

        # Depend on the compat headers as well: they are force-included into every translation unit,
        # so editing one has to trigger a rebuild.
        file(GLOB _compat_headers ${CUMETAL_COMPAT_DIR}/*.h)

        add_custom_command(
            OUTPUT ${_object}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/cumetal_objects
            # clang execs the ptxas/fatbinary shims as subprocesses, so they have to be found on PATH.
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${CUMETAL_TOOLCHAIN_DIR}:$ENV{PATH}"
                    ${CUMETAL_CUDA_CLANG}
                    ${CUMETAL_CUDA_FLAGS}
                    $<$<CONFIG:Debug>:-O0>
                    $<$<NOT:$<CONFIG:Debug>>:-O2>
                    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:cuvslam_settings,INTERFACE_COMPILE_DEFINITIONS>,PREPEND,-D>"
                    -I${CMAKE_SOURCE_DIR}/libs
                    -c ${_absolute}
                    -o ${_object}
            DEPENDS ${_absolute} ${_compat_headers}
            COMMENT "Compiling ${_name} for Apple GPU via CuMetal"
            COMMAND_EXPAND_LISTS
            VERBATIM)

        list(APPEND _objects ${_object})
    endforeach()
    set(${out_objects_var} ${_objects} PARENT_SCOPE)
endfunction()
