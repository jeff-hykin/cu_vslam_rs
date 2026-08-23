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

find_package(OpenCV QUIET)

set(LIBS
    testing
    camera_rig_edex
    sof
    utils
    camera
    edex
    log
    )

set(INCLUDE_DIRS)
set(SOURCES camera_test.cpp fid_test.cpp)

if(OpenCV_FOUND)
    list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND SOURCES camera_vs_opencv_test.cpp)
    list(APPEND LIBS ${OpenCV_LIBS})
endif()

setup_test_project(MODULE_NAME ${MODULE_NAME}
                   SOURCES ${SOURCES}
                   INCLUDE_DIRS ${INCLUDE_DIRS}
                   LIBRARIES ${LIBS})
