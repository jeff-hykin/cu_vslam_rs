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

# Include directories containing the headers files to required libraries
set(MODULE_NAME "launcher")

set(HEADERS
    base_launcher.h
    launch_load_and_localize.h
    launcher_create.h
    monocular_launcher.h
    multi_camera_launcher.h
    multi_camera_launcher_base.h
    visual_inertial_launcher.h
)

set(SOURCES
    base_launcher.cpp
    launch_load_and_localize.cpp
    visual_inertial_launcher.cpp
    launcher_create.cpp
    monocular_launcher.cpp
    multi_camera_launcher.cpp
    multi_camera_launcher_base.cpp
)


set(LIBS
    camera
    common
    edex
    epipolar
    gflags::gflags
    jsoncpp::jsoncpp
    odometry
    profiler
    slam
    sof
)

if(USE_CUDA)
    set(HEADERS ${HEADERS} rgbd_camera_launcher.h)
    set(SOURCES ${SOURCES} rgbd_camera_launcher.cpp)
endif()

# multisensor_camera_launcher includes odometry/multisensor_odometry.h, which is only built
# under USE_CUNLS. Mirror the gate here so the launcher TU isn't compiled when the type is
# unavailable.
if(USE_CUNLS)
    set(HEADERS ${HEADERS} multisensor_camera_launcher.h)
    set(SOURCES ${SOURCES} multisensor_camera_launcher.cpp)
endif()

if(USE_RERUN)
    set(HEADERS ${HEADERS} visualizer.h)
    set(SOURCES ${SOURCES} visualizer.cpp)
    set(LIBS ${LIBS} visualizer)
endif()

setup_staticlib_project(
    MODULE_NAME ${MODULE_NAME}
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    LIBRARIES ${LIBS}
)
