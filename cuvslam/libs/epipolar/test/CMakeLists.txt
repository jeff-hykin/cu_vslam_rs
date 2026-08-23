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

# libs/epipolar/tst CMake file

# Source files
set(SOURCES
    camera_points_reconstruction_integ_test.cpp
    fundamental_ransac_test.cpp
    fundamental_matrix_utils_test.cpp
    resectioning_integ_test.cpp
    resectioning_utils_test.cpp
    optscalar_ransac_test.cpp
    homography_ransac_test.cpp
    homography_test.cpp
    points_normalization_test.cpp
    camera_selection_test.cpp
    triangulation_test.cpp
    )

set(SANDBOX_SOURCES
    fundamental_matrix_singular_values_experimentation.cpp
    homography_rotation_threshold_experimentation.cpp
    )

# Header files
set(HEADERS
    camera_points_reconstruction_integ_test.h
    fundamental_matrix_utils_test.h
    generate_points_in_cube_test.h
    generate_visible_points_test.h
    homography_test.h
    resectioning_utils_test.h
)

set(LIBS
    testing
    epipolar
    pnp
    cuvslam_math
    common
    log
    )

if(USE_RERUN)
    set(LIBS ${LIBS} visualizer)
endif()

setup_test_project(
    MODULE_NAME ${MODULE_NAME}
    HEADERS ${HEADERS}
    LIBRARIES ${LIBS}
    SOURCES ${SOURCES}
    SANDBOX_SOURCES ${SANDBOX_SOURCES}
    )
