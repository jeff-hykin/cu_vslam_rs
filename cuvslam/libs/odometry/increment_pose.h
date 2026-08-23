
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

#include "common/isometry.h"
#include "common/rotation_utils.h"

namespace cuvslam::odom {

// function for integrate the absolute pose
inline Isometry3T increment_pose(const Isometry3T& prev_abs_world_from_rig, const Isometry3T& delta) {
  Isometry3T abs_world_from_rig = prev_abs_world_from_rig * delta;
  // make rotation part of abs_world_from_rig orthonormal using our local SVD-based function
  abs_world_from_rig.matrix().block<3, 3>(0, 0) = common::CalculateRotationFromSVD(abs_world_from_rig.matrix());
  return abs_world_from_rig;
}

}  // namespace cuvslam::odom
