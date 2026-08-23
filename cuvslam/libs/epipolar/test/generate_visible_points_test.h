
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

#include "common/log.h"
#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "epipolar/test/generate_points_in_cube_test.h"

namespace test::epipolar {
using namespace cuvslam;

inline Isometry3T createRandomCamera() {
  // Create camera 1 at random - random is in [0..1] on each dimension
  // We keep the translation small in the face of distance of the camera to the 3D points.
  return Translation3T(Vector3T::Random()) * Rotation3T(Vector3T::Random(), AngleUnits::Radian);
}

///-------------------------------------------------------------------------------------------------
/// @brief generate points visible simultaneously in multiple cameras.
///
/// Need for testing fundamental matrix and trifocal tenzor. It generate points in desired box,
/// then filter them to be "near" cameras frame. Algorithm is quite strange.
///
/// @param cameras     - list of cameras
/// @param num3DPoints - maximum number of points to generate
/// @param points3DMinRange
/// @param points3DMaxRange
/// @param [out] points2D - list of 2d points per camera
/// @param [out] pPoints3D - list of 3d points, corresponded to 2d points
///-------------------------------------------------------------------------------------------------
inline void generateVisiblePoints(const Isometry3TVector& cameras, size_t num3DPoints, size_t minimumPointsNumber,
                                  const Vector3T& points3DMinRange, const Vector3T& points3DMaxRange,
                                  Vector2TVectorVector& points2D, Vector3TVector* pPoints3D = NULL) {
  const size_t nCameras = cameras.size();
  points2D.resize(nCameras);

  auto validator = [&](const Vector3T& p) -> bool {
    for (const auto& cam : cameras) {
      if ((cam.inverse() * p).z() <= cuvslam::epipolar::FrustumProperties::MINIMUM_HITHER) {
        return false;
      }
    }

    return true;
  };

  size_t numTry = 0;

  Vector2TVectorVector unfilteredPoints2D;  // some of them are out of camera
  unfilteredPoints2D.resize(nCameras);

  size_t cameraPoints;

  do {
    cameraPoints = 0;

    // initialize iteration
    if (pPoints3D != NULL) {
      pPoints3D->clear();
    }

    for (size_t k = 0; k < nCameras; k++) {
      points2D[k].clear();
    }

    Vector3TVector points3D = GeneratePointsInCube(num3DPoints, points3DMinRange, points3DMaxRange, validator);

    if (points3D.size() == 0) {
      TraceMessage("Failed to generate points in given cube with given cameras");
      continue;
    }

    for (size_t j = 0; j < nCameras; j++) {
      cuvslam::epipolar::Project3DPointsInLocalCoordinates(cameras[j].inverse(), points3D, unfilteredPoints2D[j]);
    }

    // filter 3d points to have good projection
    for (size_t i = 0; i < points3D.size(); i++) {
      size_t j;

      for (j = 0; j < nCameras; j++) {
        const Vector2T& pt = unfilteredPoints2D[j][i];

        if (pt.norm() >= 10.0f) {
          break;
        }
      }

      if (j == nCameras) {
        if (pPoints3D != NULL) {
          pPoints3D->push_back(points3D[i]);
        }

        // point number i has good projection in all cameras.
        for (size_t k = 0; k < nCameras; k++) {
          points2D[k].push_back(unfilteredPoints2D[k][i]);
        }

        ++cameraPoints;
      }
    }
  } while (cameraPoints < minimumPointsNumber && numTry++ < 250);
}

}  // namespace test::epipolar
