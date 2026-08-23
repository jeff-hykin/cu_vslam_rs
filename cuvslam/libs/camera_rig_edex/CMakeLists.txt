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

set(MODULE_NAME "camera_rig_edex")

set(HEADERS
    blackout_oscillator_filter.h
    camera_rig_edex.h
    repeated_camera_rig_edex.h
    shuttle_camera_rig_edex.h
    )

set(SOURCES
    blackout_oscillator_filter.cpp
    camera_rig_edex.cpp
    repeated_camera_rig_edex.cpp
    shuttle_camera_rig_edex.cpp
    )

setup_staticlib_project(MODULE_NAME ${MODULE_NAME}
                        HEADERS ${HEADERS}
                        SOURCES ${SOURCES}
                        LIBRARIES
                            camera
                            common
                            edex
                            utils
                            jsoncpp::jsoncpp
                        )

add_subdirectory(test)
