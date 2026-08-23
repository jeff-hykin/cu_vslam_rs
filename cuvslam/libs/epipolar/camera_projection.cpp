
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

#include "epipolar/camera_projection.h"

namespace cuvslam::epipolar {

bool CreateCameraMatrix(const Vector3T& position, const Vector3T& lookingAt, const Vector3T& up,
                        Isometry3T& cameraMatrix) {
  // OpenCV camera: +Z points forward (into the scene) from the camera position toward lookingAt.
  Vector3T zAxis(lookingAt - position);
  zAxis.normalize();

  Vector3T yAxis(up);
  yAxis.normalize();

  Vector3T xAxis = yAxis.cross(zAxis);

  // Because zAxis and yAxis are normalized, ||xAxis|| = sin(theta). If theta < 0.5deg,
  // we fail to create a camera matrix as we consider "up" too colinear to the z-axis.
  // Note: the threshold value can be modified if necessary but needs to be tested.
  if (xAxis.norm() < std::sin(PI / 360.0)) {
    return false;
  }

  xAxis.normalize();

  yAxis = zAxis.cross(xAxis);
  yAxis.normalize();

  cameraMatrix.linear().col(0) = xAxis;
  cameraMatrix.linear().col(1) = yAxis;
  cameraMatrix.linear().col(2) = zAxis;
  cameraMatrix.translation().col(0) = position;
  cameraMatrix.makeAffine();

  return true;
}

float Project3DPointInLocalCoordinates(const Isometry3T& toLocalSpace, const Vector3T& point3D,
                                       Vector2T& projectedPoint) {
  const Vector3T localCoord = toLocalSpace * point3D;
  const float z = AvoidZero(localCoord.z());
  projectedPoint = localCoord.head(2) / z;
  return z;
}

void Project3DPointsInLocalCoordinates(const Isometry3T& toLocalSpace, Vector3TVectorCIt points3DBeginIt,
                                       Vector3TVectorCIt points3DEndIt, Vector2TVector& projectedPoints) {
  projectedPoints.resize(std::distance(points3DBeginIt, points3DEndIt));
  auto pt3D = points3DBeginIt;
  auto pt2D = projectedPoints.begin();

  for (; pt3D != points3DEndIt; ++pt3D, ++pt2D) {
    Project3DPointInLocalCoordinates(toLocalSpace, *pt3D, *pt2D);
  }
}

void Project3DPointsInLocalCoordinates(const Isometry3T& toLocalSpace, const Vector3TVector& points3D,
                                       Vector2TVector& projectedPoints) {
  Project3DPointsInLocalCoordinates(toLocalSpace, points3D.begin(), points3D.end(), projectedPoints);
}

void GeneratePairOf2DPointsFrom3DPoints(const Vector3TVector& points3D, const Isometry3T& camera1,
                                        const Isometry3T& camera2, Vector2TPairVector* pairOf2DPoints) {
  assert(pairOf2DPoints);
  pairOf2DPoints->resize(points3D.size());
  auto pair2DPointsIt = pairOf2DPoints->begin();

  for (auto point3DIt = points3D.cbegin(); points3D.cend() != point3DIt; ++point3DIt, ++pair2DPointsIt) {
    Project3DPointInLocalCoordinates(camera1.inverse(), *point3DIt, pair2DPointsIt->first);
    Project3DPointInLocalCoordinates(camera2.inverse(), *point3DIt, pair2DPointsIt->second);
  }
}

float CalculateDistanceFromExpectedRotationMatrix(const QuaternionT::RotationMatrixType& expected,
                                                  const QuaternionT::RotationMatrixType& actual) {
  return QuaternionT(expected).angularDistance(QuaternionT(actual));
}

}  // namespace cuvslam::epipolar
