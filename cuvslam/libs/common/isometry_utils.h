
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

#include "common/frame_id.h"
#include "common/isometry.h"
#include "common/types.h"

namespace cuvslam {

// Warning: for temporally smooth rotation sequence, this call can produce not smooth sequence of Euler rotation values
// near or at gimbal lock position. If temporal smoothness is important, GetEulerRotationStable should be used instead
inline AngleVector3T getEulerRotation(const Isometry3T &t) {
  // decompose rotation angles in the reverse order. get z then y and finally x.
  // reverse order to get correct ordering of values
  return AngleVector(t.linear().eulerAngles(2, 1, 0).reverse());
}

// In order to stabilize smooth sequence of 3D rotations into Euler angles sequence, you have to provide
// previous rotation in order to stabilize computation around gimbal lock position.
inline AngleVector3T getEulerRotationStable(const Isometry3T &t, const AngleVector3T &rotPrev) {
  AngleVector3T rot = getEulerRotation(t);

  // assuming rot and rotPrev have normalized angles
  assert(std::abs(rot.x()) <= cuvslam::PI && std::abs(rot.y()) <= cuvslam::PI && std::abs(rot.z()) <= cuvslam::PI);
  assert(std::abs(rotPrev.x()) <= cuvslam::PI && std::abs(rotPrev.y()) <= cuvslam::PI &&
         std::abs(rotPrev.z()) <= cuvslam::PI);

  // if at or near gimbal lock
  if ((std::fabs(std::abs(rot.y()) - cuvslam::PI / 2)) < std::numeric_limits<float>::epsilon()) {
    const float deltaZ = rot.z() - rotPrev.z();
    const float newX = rot.x() + ((rot.y() > 0) ? -deltaZ : deltaZ);
    rot.z() = rotPrev.z();
    rot.x() = Angle<float>::Normalize(newX);
  }
  // check if we have optimal Euler set or should switch to the other one that would be closer
  else if (std::abs(Angle<float>::Normalize(rot.x() - rotPrev.x())) > cuvslam::PI / 2 &&
           std::abs(Angle<float>::Normalize(rot.z() - rotPrev.z())) > cuvslam::PI / 2) {
    rot.x() = Angle<float>::Normalize(cuvslam::PI + rot.x());
    rot.y() = Angle<float>::Normalize(cuvslam::PI - rot.y());
    rot.z() = Angle<float>::Normalize(cuvslam::PI + rot.z());
  }

  return rot;
}

// only usable for 3D case
inline Matrix3T essential(const Isometry3T &t) { return (SkewSymmetric(t.translation()) * t.linear()).eval(); }

using Isometry3TVector = std::vector<Isometry3T>;
using CameraMap = std::map<FrameId, Isometry3T>;

}  // namespace cuvslam
