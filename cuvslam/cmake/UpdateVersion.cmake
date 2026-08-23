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

# Read version from the single-source-of-truth VERSION file
file(READ ${SOURCE_DIR}/../../VERSION CUVSLAM_VERSION)
string(STRIP "${CUVSLAM_VERSION}" CUVSLAM_VERSION)
string(REPLACE "." ";" CUVSLAM_VERSION_LIST "${CUVSLAM_VERSION}")
list(GET CUVSLAM_VERSION_LIST 0 CUVSLAM_VERSION_MAJOR)
list(GET CUVSLAM_VERSION_LIST 1 CUVSLAM_VERSION_MINOR)
list(GET CUVSLAM_VERSION_LIST 2 CUVSLAM_VERSION_PATCH)

# Get git hash & dirty status
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_RESULT
)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT GIT_RESULT EQUAL 0)
    set(GIT_HASH "")
    set(GIT_HASH_SHORT "")
endif()

execute_process(
    COMMAND git diff --quiet HEAD --
    WORKING_DIRECTORY ${SOURCE_DIR}
    RESULT_VARIABLE GIT_IS_DIRTY
)

# Set dirty flag based on the result
if(GIT_IS_DIRTY EQUAL 0)
    set(GIT_DIRTY_SUFFIX "")
else()
    set(GIT_DIRTY_SUFFIX "-modified")
endif()

string(TIMESTAMP BUILD_TIMESTAMP UTC)

configure_file(
    ${SOURCE_DIR}/version.h.in
    ${OUTPUT_FILE}
    @ONLY
)
