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

# libs/common CMake file

set(MODULE_NAME "common")

# Header files
set(HEADERS
    camera_id.h
    coordinate_system.h
    environment.h
    error.h
    fixed_array.h
    fpexceptions.h
    frame_id.h
    image_dropper.h
    image_matrix.h
    image.h
    imu_calibration.h
    imu_measurement.h
    include_eigen.h
    include_gtest.h
    include_json.h
    interfaces.h
    isometry_utils.h
    isometry.h
    json_to_eigen.h
    log_types.h
    log.h
    parse_utils.h
    rerun.h
    rotation_utils.h
    state_serial.h
    statistic.h
    stopwatch.h
    stream.h
    tga.h
    thread_name.h
    thread_safe_queue.h
    time.h
    track_id.h
    types.h
    unaligned_types.h
    vector_2t.h
    vector_3t.h
    )

set(SOURCES
    coordinate_system.cpp
    imu_calibration.cpp
    imu_measurement.cpp
    log.cpp
    parse_utils.cpp
    tga.cpp
    )

set(LIBS
    circularbuffer::circularbuffer
    Eigen3::Eigen
    )

if(USE_CUDA)
    set(LIBS ${LIBS} CUDA::cudart_static)
endif()

if(USE_RERUN)
    set(LIBS ${LIBS} rerun::sdk)
endif()

# Common library that also propagates common dependencies to all internal targets
setup_staticlib_project(
    MODULE_NAME "${MODULE_NAME}"
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    LIBRARIES ${LIBS}
    )

add_subdirectory(test)
