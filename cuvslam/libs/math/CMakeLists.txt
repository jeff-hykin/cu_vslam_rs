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

set(MODULE_NAME "cuvslam_math")

set(HEADERS
    nonlinear_optimization.h
    pgo.h
    ransac.h
    robust_cost_function.h
    sigmapoint.h
    transform_between_pointclouds.h
    twist.h
    twist_to_angle.h
    twist_to_quaternion.h
    )

set(SOURCES
    pgo.cpp
    robust_cost_function.cpp
    transform_between_pointclouds.cpp
    twist.cpp
    twist_to_angle.cpp
    )


set(LIBS common)

# Header-only cuNLS-backed factor wrappers: live alongside the rest of math because they are pure
# math primitives (residual + Jacobian definitions), but require cunls + cuda_modules so they are
# gated by USE_CUNLS like every other cuNLS-using artifact in the tree.
if(USE_CUNLS)
    set(HEADERS ${HEADERS} point_to_point_icp_factor.h point_to_plane_factor.h)
    set(LIBS ${LIBS} cunls::cunls cuda_modules)
endif()

setup_staticlib_project(
        MODULE_NAME "${MODULE_NAME}"
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
