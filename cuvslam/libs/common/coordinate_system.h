
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
#include "common/types.h"

namespace cuvslam {

enum class CoordinateSystem { CUVSLAM = 0, ROS = 1, OPENCV = 2 };
// matrix to change basis
Matrix3T CoordinateSystemTocuVSLAM(CoordinateSystem cs);

inline Isometry3T Matrix3ToIsometryRotation(const Matrix3T& r) {
  Isometry3T t = Isometry3T::Identity();
  t.linear() = r;
  return t;
}

const Isometry3T kCuvslamFromRos{Matrix3ToIsometryRotation(CoordinateSystemTocuVSLAM(CoordinateSystem::ROS))};

const Isometry3T kRosFromCuvslam{kCuvslamFromRos.inverse()};

// EDEX and older assets stored rig extrinsics in legacy cuVSLAM frame (y-up, z-back). Internal stack is OpenCV.
// Similarity transform: both source and destination frames change (e.g. camera_from_rig).
inline Isometry3T LegacyEdexIsometryToOpenCV(const Isometry3T& legacy_rig_pose) {
  Isometry3T k = Isometry3T::Identity();
  k.linear()(1, 1) = -1.f;
  k.linear()(2, 2) = -1.f;
  return k.inverse() * legacy_rig_pose * k;
}

// EDEX rig_from_imu maps physical IMU body -> legacy cuVSLAM rig frame.
// Only the rig (destination) frame changes from cuVSLAM to OpenCV; the IMU body (source)
// frame stays unchanged because IMU measurements and preintegration deltas remain in the
// original physical sensor frame.
inline Isometry3T LegacyEdexImuExtrinsicToOpenCV(const Isometry3T& legacy_rig_from_imu) {
  Isometry3T cuvslam_to_opencv = Isometry3T::Identity();
  cuvslam_to_opencv.linear()(1, 1) = -1.f;
  cuvslam_to_opencv.linear()(2, 2) = -1.f;
  return cuvslam_to_opencv * legacy_rig_from_imu;
}

}  // namespace cuvslam
