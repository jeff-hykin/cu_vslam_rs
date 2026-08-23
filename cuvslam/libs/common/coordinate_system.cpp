
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

#include "common/coordinate_system.h"

namespace cuvslam {

// Maps IMU log vectors into the internal OpenCV camera/world frame (x right, y down, z forward).
Matrix3T CoordinateSystemTocuVSLAM(CoordinateSystem cs) {
  const Matrix3T legacy_cuvslam_to_opencv = [] {
    Matrix3T m = Matrix3T::Identity();
    m(1, 1) = -1.f;
    m(2, 2) = -1.f;
    return m;
  }();
  switch (cs) {
    case CoordinateSystem::ROS: {
      Matrix3T ros_to_legacy_cuvslam;
      ros_to_legacy_cuvslam << 0.f, -1.f, 0.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.f;
      return legacy_cuvslam_to_opencv * ros_to_legacy_cuvslam;
    }
    case CoordinateSystem::OPENCV:
      return Matrix3T::Identity();
    default:
      // Legacy EDEX IMU logs in pre-OpenCV cuVSLAM frame (same as old internal).
      return legacy_cuvslam_to_opencv;
  }
}

}  // namespace cuvslam
