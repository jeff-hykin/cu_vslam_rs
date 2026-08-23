
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

#include "common/include_gtest.h"

#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "epipolar/test/generate_points_in_cube_test.h"

namespace {
using namespace cuvslam;

///-------------------------------------------------------------------------------------------------
/// @brief Check whether a ray intersects and axis-aligned bounding box (AABB) or not. This code
/// accounts for the direction of the ray, i.e. it will provide a different result than line/AABB
/// intersection, in general. If this code is moved outside of the anonymous namespace, it should
/// be properly tested.
/// Code available at (slightly reformatted to work with Eigen):
/// http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
///
/// @param ray           The ray.
/// @param aabbMinCoords The AABB minimum coordinates in x, y, z.
/// @param aabbMaxCoords The AABB maximum coordinates in x, y, z.
///
/// @return true if the ray intersects the rectangle, false otherwise.
///-------------------------------------------------------------------------------------------------
bool CheckRayIntersectsAABB(const epipolar::Ray3T& ray, const Vector3T& aabbMinCoords, const Vector3T& aabbMaxCoords) {
  Vector3T dirInv = ray.getDirection().cwiseInverse();

  Vector3T bounds1 = (aabbMinCoords - ray.getOrigin()).cwiseProduct(dirInv);
  Vector3T bounds2 = (aabbMaxCoords - ray.getOrigin()).cwiseProduct(dirInv);

  float tmin = std::max(std::max(std::min(bounds1.x(), bounds2.x()), std::min(bounds1.y(), bounds2.y())),
                        std::min(bounds1.z(), bounds2.z()));
  float tmax = std::min(std::min(std::max(bounds1.x(), bounds2.x()), std::max(bounds1.y(), bounds2.y())),
                        std::max(bounds1.z(), bounds2.z()));

  if (tmax < 0 || tmin > tmax) {
    return false;
  }

  return true;
}

///-------------------------------------------------------------------------------------------------
/// @brief Convert a vector of pairs {Data1, Data2} to 2 vectors of Data1 and Data2.
///-------------------------------------------------------------------------------------------------
template <typename Data1, typename Data2>
void ConvertVectorOfPairsTo2Vectors(const std::vector<std::pair<Data1, Data2>>& vectorOfPairs,
                                    std::vector<Data1>* vectorOfFirst, std::vector<Data2>* vectorOfSecond) {
  assert(vectorOfFirst && vectorOfSecond);
  const size_t size = vectorOfPairs.size();
  vectorOfFirst->resize(size);
  vectorOfSecond->resize(size);
  auto vectorOfFirstIt = vectorOfFirst->begin();
  auto vectorOfSecondIt = vectorOfSecond->begin();
  auto vectorOfPairsIt = vectorOfPairs.cbegin();

  for (; vectorOfPairsIt != vectorOfPairs.cend(); ++vectorOfFirstIt, ++vectorOfSecondIt, ++vectorOfPairsIt) {
    (*vectorOfFirstIt) = vectorOfPairsIt->first;
    (*vectorOfSecondIt) = vectorOfPairsIt->second;
  }
}

bool Verify3DPointsInFrontOFBothCameras(const cuvslam::Vector3TVector& points3D, const cuvslam::Isometry3T& camera1,
                                        const cuvslam::Isometry3T& camera2) {
  using namespace cuvslam;
  Isometry3T camera1Inverse = camera1.inverse();
  Isometry3T camera2Inverse = camera2.inverse();

  for (const auto& point3D : points3D) {
    Vector3T localSpace1 = camera1Inverse * point3D;
    Vector3T localSpace2 = camera2Inverse * point3D;

    if (localSpace1.z() <= epipolar::FrustumProperties::MINIMUM_HITHER ||
        localSpace2.z() <= epipolar::FrustumProperties::MINIMUM_HITHER) {
      return false;
    }
  }

  return true;
}

}  // namespace

namespace test::epipolar {
using namespace cuvslam::epipolar;

///-------------------------------------------------------------------------------------------------
/// @brief Tries to generate camera2 from a number of conditions: (i) the camera z-axis must be
/// pointing away from the center of the cloud of 3D points; (ii) the line passing through the
/// centers of camera1 (Identity) and camera2 must not intersect with the AABB in which the 3D
/// points are drawn - this is to avoid instability in the reconstruction of 3D points; (iii) the
/// method has numTrials trials to find an acceptable camera matrix for camera2.
///
/// @param numTrials             Number of trials.
/// @param aabbMins              The AABB minimum values in x,y, z (defines one corner).
/// @param aabbMaxs              The AABB maximum values in x,y, z (defines another corner).
/// @param [in,out] cameraMatrix The camera matrix.
///
/// @return true if it succeeds to fine an apprpriate cameraMatrix, false if not.
///-------------------------------------------------------------------------------------------------
inline bool TryToGenerateCamera2Matrix(const int numTrials, const Vector3T& aabbMins, const Vector3T& aabbMaxs,
                                       Isometry3T& cameraMatrix) {
  for (int trialIdx = 0; numTrials > trialIdx; ++trialIdx) {
    // Find a position for camera2. We enforce that the position be random but on a sphere of unit
    // radius centered on the world coordinates origin.
    Vector3T camera2Origin = Vector3T::Random();
    camera2Origin.normalize();
    Vector3T upVector = Vector3T::Random();
    upVector.normalize();

    // Create a matrix that is looking at the center of the AABB
    bool res = epipolar::CreateCameraMatrix(camera2Origin, (aabbMins + aabbMaxs) / 2.0, upVector, cameraMatrix);

    // Here we test if the ray formed by camera1 and camera2 origins' intersects the AABB for which
    // samples will be drawn from. This is to avoid the issue where 3D points could be projected on
    // the cameras in a way that the reconstruction would not be stable (as the ray would be
    // colinear). In code, we extend the box by 1 unit in every direction to remove edge cases.
    Ray3T ray1To2 = Ray3T::GenerateRayFrom2Points(Vector3T::Zero(), cameraMatrix.translation());
    Ray3T ray2To1 = Ray3T::GenerateRayFrom2Points(cameraMatrix.translation(), Vector3T::Zero());

    if (res && !CheckRayIntersectsAABB(ray1To2, aabbMins - Vector3T::Ones(), aabbMaxs + Vector3T::Ones()) &&
        !CheckRayIntersectsAABB(ray2To1, aabbMins - Vector3T::Ones(), aabbMaxs + Vector3T::Ones())) {
      return true;
    }
  }

  return false;
}

class CameraPointsReconstructionIntegTest : public testing::Test {
public:
  virtual void SetUp() {
    // 3D points are randomly generated in a cube defined by minRange, maxRange.
    const Vector3T minRange(3, 2, 5);
    const Vector3T maxRange(13, 10, 16);

    // camera1 is kernels_initialized at the origin and aligned with the world coordinate since camera1
    // will be the frame of reference.
    absoluteCamera1_ = Isometry3T::Identity();

    ASSERT_TRUE(
        TryToGenerateCamera2Matrix(500,  // If we cannot generate a good matrix after 500 tries, something else is wrong
                                   minRange, maxRange, absoluteCamera2_));

    expectedRelativeTransform_ = (absoluteCamera2_.inverse()) * absoluteCamera1_;

    auto validator = [&](const Vector3T& p) -> bool {
      return (absoluteCamera1_.inverse() * p).z() > FrustumProperties::MINIMUM_HITHER &&
             (absoluteCamera2_.inverse() * p).z() > FrustumProperties::MINIMUM_HITHER;
    };

    expected3DPoints_ = GeneratePointsInCube(num3DPoints_, minRange, maxRange, validator);

    ASSERT_TRUE(Verify3DPointsInFrontOFBothCameras(expected3DPoints_, absoluteCamera1_, absoluteCamera2_));

    GeneratePairOf2DPointsFrom3DPoints(expected3DPoints_, absoluteCamera1_, absoluteCamera2_, &point2Dpairs_);

    // Convert the data so that it can be passed to CalculateFundamentalMatrix.
    ConvertVectorOfPairsTo2Vectors(point2Dpairs_, &points2DLocal1_, &points2DLocal2_);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief This method is a helper function measuring the error (Euclidean distance) between a
  /// generated and a reconstructed point and ensuring the error is below a given threshold. The
  /// method tests a vector of points.
  ///-------------------------------------------------------------------------------------------------
  template <typename Vector>
  void CompareGeneratedAndReconstructedPoints(const std::vector<Vector>& generatedPoints,
                                              const std::vector<Vector>& reconstructedPoints, float threshold) {
    EXPECT_EQ(generatedPoints.size(), reconstructedPoints.size());

    for (size_t idx = 0; idx < generatedPoints.size(); ++idx) {
      std::stringstream ss;
      ss << "Computing error for index: " << idx << std::endl;
      ss << "reconstructedPoints[" << idx << "]: " << reconstructedPoints[idx] << std::endl;
      ss << "generatedPoints[" << idx << "]: " << generatedPoints[idx] << std::endl;
      SCOPED_TRACE(ss.str());
      EXPECT_LE((reconstructedPoints[idx] - generatedPoints[idx]).norm() / generatedPoints[idx].norm(), threshold);
    }
  }

protected:
  const size_t num3DPoints_ = 400;
  Isometry3T expectedRelativeTransform_;
  Isometry3T actualRelativeTransform_;
  Isometry3T absoluteCamera1_, absoluteCamera2_;
  Vector3TVector expected3DPoints_;
  Vector3TVector actual3DPoints_;
  Vector2TVector points2DLocal1_;
  Vector2TVector points2DLocal2_;
  Vector2TPairVector point2Dpairs_;
};

}  // namespace test::epipolar
