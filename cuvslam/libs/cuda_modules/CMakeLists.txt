# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

set(MODULE_NAME "cuda_modules")

set(HEADERS
    box_prefilter.h
    cuda_convolutor.h
    cuda_helper.h
    culib_helper.h
    gftt.h
    gradient_pyramid.h
    icp_tools.h
    image_cast.h
    image_pyramid.h
    lk_tracker.h
    sba.h
    selection_v2.h
    st_tracker.h
    )

set(SOURCES
    box_prefilter.cpp
    cuda_convolutor.cpp
    cuda_helper.cpp
    gftt.cpp
    gradient_pyramid.cpp
    icp_tools.cpp
    image_cast.cpp
    image_pyramid.cpp
    lk_tracker.cpp
    sba.cpp
    selection_v2.cpp
    st_tracker.cpp
    )

add_subdirectory(cuda_kernels)

setup_staticlib_project(
        MODULE_NAME "${MODULE_NAME}"
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES
            camera
            common
            cuvslam_math
            cuda_kernels
            profiler
)

add_subdirectory(test)
