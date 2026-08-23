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

set(MODULE_NAME "track_euroc")

set(LIBS
    cuvslam
    PNG::PNG
    yaml-cpp::yaml-cpp
)

if(USE_RERUN)
    list(APPEND LIBS rerun_sdk)
endif()

setup_app(MODULE_NAME ${MODULE_NAME}
          LIBRARIES ${LIBS}
          SOURCES track_euroc.cpp)

if(USE_RERUN)
    # Symlink the Rerun blueprint file to the output directory
    add_custom_command(
        TARGET ${MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_CURRENT_SOURCE_DIR}/euroc_blueprint.rbl"
            "$<TARGET_FILE_DIR:${MODULE_NAME}>/euroc_blueprint.rbl"
        COMMENT "Symlinking euroc_blueprint.rbl to build directory"
    )
endif()
