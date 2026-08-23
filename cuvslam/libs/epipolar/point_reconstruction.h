
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

#include "epipolar/camera_selection.h"

namespace cuvslam::epipolar {

// Used in Tests only @TODO move into tests only utils

///-------------------------------------------------------------------------------------------------
/// @brief Reconstruct 3D points from pairs of rays and a relative transform. Both rays in each
/// pair are defined in the reference coordinate system.
///
/// @param pairOf2DPoints    The vector of 2D point pairs to reconstruct the 3D points.
/// @param relativeTransform The relative transform from the reference coordinate system (camera1)
/// to the other camera (camera2).
/// @param [in,out] points3D If non-null, the vector of reconstructed 3D points.
///
/// @return true if it all points are in front of both cameras and points are not colinear, false
/// otherwise.
///-------------------------------------------------------------------------------------------------

inline bool Reconstruct3DPointsFrom2DPointsAndRelativeTransform(const Vector2TPairVector& pairOf2DPoints,
                                                                const Isometry3T& relativeTransform,
                                                                Vector3TVector& points3D) {
  points3D.resize(pairOf2DPoints.size());
  auto point3DIt = points3D.begin();
  auto pairOf2DPointIt = pairOf2DPoints.cbegin();
  bool res(true);

  for (; pairOf2DPointIt != pairOf2DPoints.cend(); ++pairOf2DPointIt, ++point3DIt) {
    if (!IntersectRaysInReferenceSpace(relativeTransform.inverse(), pairOf2DPointIt->first.homogeneous(),
                                       pairOf2DPointIt->second.homogeneous(), *point3DIt)) {
      res = false;
    }
  }

  return res;
}

}  // namespace cuvslam::epipolar
