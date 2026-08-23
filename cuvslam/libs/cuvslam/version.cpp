

/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include <string_view>

#include "common/error.h"
#include "cuvslam/cuvslam2.h"
#include "version.h"

namespace cuvslam {

CUVSLAM_API
std::string_view GetVersion(int32_t* major, int32_t* minor, int32_t* patch) {
  if (major) {
    *major = CUVSLAM_API_VERSION_MAJOR;
  }
  if (minor) {
    *minor = CUVSLAM_API_VERSION_MINOR;
  }
  if (patch) {
    *patch = CUVSLAM_API_VERSION_PATCH;
  }
  return std::string_view{
      _CRT_STRINGIZE(CUVSLAM_API_VERSION_MAJOR) "." _CRT_STRINGIZE(CUVSLAM_API_VERSION_MINOR) "." _CRT_STRINGIZE(
          CUVSLAM_API_VERSION_PATCH) "+" CUVSLAM_GIT_HASH_SHORT CUVSLAM_GIT_DIRTY_SUFFIX};
}

}  // namespace cuvslam
