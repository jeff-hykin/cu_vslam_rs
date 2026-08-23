
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

#pragma once

#include <cstdint>

#include "common/isometry.h"
#include "common/unaligned_types.h"

#include "camera/camera.h"

namespace cuvslam::camera {

struct Rig {
  static constexpr const int32_t kMaxCameras{32};
  Isometry3T camera_from_rig[kMaxCameras];
  const ICameraModel* intrinsics[kMaxCameras]{};  // somebody else owns this pointer
  int32_t num_cameras{0};

  Rig& operator=(const Rig& rhs) {
    num_cameras = rhs.num_cameras;
    for (int i = 0; i < rhs.num_cameras; ++i) {
      camera_from_rig[i] = rhs.camera_from_rig[i];
      intrinsics[i] = rhs.intrinsics[i];
    }
    return *this;
  }
};

}  // namespace cuvslam::camera
