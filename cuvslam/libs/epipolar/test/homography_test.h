
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
#include "common/rotation_utils.h"
#include "epipolar/camera_projection.h"
#include "epipolar/homography.h"
#include "epipolar/test/generate_points_in_cube_test.h"

namespace test::epipolar {
using namespace cuvslam;

///-------------------------------------------------------------------------------------------------
/// @brief Convert 2 vectors of Data1 and Data2 to a vector of pairs {Data1, Data2}.
///-------------------------------------------------------------------------------------------------
template <typename Data1, typename Data2>
void ConvertVectorOfPairsTo2Vectors(const std::vector<Data1>& vectorOfFirst, const std::vector<Data2>& vectorOfSecond,
                                    std::vector<std::pair<Data1, Data2>>& vectorOfPairs) {
  assert(vectorOfFirst.size() == vectorOfFirst.size());

  const size_t size = vectorOfFirst.size();
  vectorOfPairs.resize(size);
  auto vectorOfFirstIt = vectorOfFirst.cbegin();
  auto vectorOfSecondIt = vectorOfSecond.cbegin();
  auto vectorOfPairsIt = vectorOfPairs.begin();

  for (; vectorOfFirstIt != vectorOfFirst.cend(); ++vectorOfFirstIt, ++vectorOfSecondIt, ++vectorOfPairsIt) {
    vectorOfPairsIt->first = (*vectorOfFirstIt);
    vectorOfPairsIt->second = (*vectorOfSecondIt);
  }
}

class HomographyTest : public testing::Test {
protected:
  virtual bool transformPointsByHomography(const Vector2TVector& points2DImage1, const Matrix3T& homography,
                                           Vector2TVector& points2DImage2) {
    points2DImage2.resize(points2DImage1.size());
    bool res(true);
    std::transform(points2DImage1.cbegin(), points2DImage1.cend(), points2DImage2.begin(),
                   [&homography, &res](const Vector2T& point2DImage1) {
                     Vector3T point2DImage2Homogeneous = homography * point2DImage1.homogeneous();
                     float z = point2DImage2Homogeneous.z();

                     if (std::abs(z) < epsilon()) {
                       res = false;
                       z = std::copysign(epsilon(), z);
                     }

                     return Vector2T(point2DImage2Homogeneous.x() / z, point2DImage2Homogeneous.y() / z);
                   });
    return res;
  }

  virtual Vector2TVector generateRandom2DPointsNormalized(size_t numPoints) {
    Vector2TVector random2DPoints(numPoints);
    std::generate(random2DPoints.begin(), random2DPoints.end(), []() { return Vector2T::Random(); });
    return random2DPoints;
  }

  void EnforceFrustum() {
    Vector2TVector Points1, Points2;

    for (size_t i = 0; i < expectedPoints1_.size(); i++) {
      const Vector2T& pt1 = expectedPoints1_[i];
      const Vector2T& pt2 = expectedPoints2_[i];

      if (pt1.norm() < 10.0f && pt2.norm() < 10.0f) {
        Points1.push_back(pt1);
        Points2.push_back(pt2);
      }
    }

    expectedPoints1_.swap(Points1);
    expectedPoints2_.swap(Points2);
  }

  virtual bool SetupGeneralHomographyCase() {
    const int MAX_NUM_TRIAL = 50;

    for (int numTrial = 0; MAX_NUM_TRIAL > numTrial; ++numTrial) {
      expectedHomography_ = Matrix3T::Random();

      Eigen::JacobiSVD<Matrix3T> svd(expectedHomography_, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Vector3T singulars = svd.singularValues();

      // homography should be an invertible transformation
      if (std::abs(singulars.prod()) < epsilon()) {
        continue;
      }

      expectedPoints1_ = generateRandom2DPointsNormalized(numPoints_);

      if (transformPointsByHomography(expectedPoints1_, expectedHomography_, expectedPoints2_)) {
        EnforceFrustum();

        if (expectedPoints1_.size() < numPoints_ / 2) {
          continue;
        }

        cuvslam::epipolar::ComputeHomography computeHomography(expectedPoints1_, expectedPoints2_);
        Matrix3T actualHomography;

        if (computeHomography.findHomography(actualHomography) ==
            cuvslam::epipolar::ComputeHomography::ReturnCode::Success) {
          return true;
        }

        continue;
      }
    }

    return false;  // If we cannot generate good data after MAX_NUM_TRIAL, report our misfortune.
  }

  virtual bool SetupHomographyFromPureCameraRotation() {
    // Setup 2 cameras that are pure rotation of each others
    const Translation3T position(Vector3T::Random());
    const Isometry3T camera1 = position * Rotation3T(Vector3T::Random(), AngleUnits::Radian);
    const Isometry3T camera2 = position * Rotation3T(Vector3T::Random(), AngleUnits::Radian);
    // expectedHomography_ is transformation of 3D points projections in camera1 sensor space to camera2
    // sensor space, so, camera2.inverse() * camera1 is correct definition of this transformation
    expectedHomography_ = common::CalculateRotationFromSVD((camera2.inverse() * camera1).matrix());
    return SetupPointsProjectionsFromCameras(camera1, camera2);
  }

  virtual bool SetupPointsProjectionsFromCameras(const Isometry3T& camera1, const Isometry3T& camera2) {
    // Generate 3D points
    Vector3TVector points3D = GeneratePointsInCube(numPoints_, minRange_, maxRange_);

    cuvslam::epipolar::Project3DPointsInLocalCoordinates(camera1.inverse(), points3D, expectedPoints1_);
    cuvslam::epipolar::Project3DPointsInLocalCoordinates(camera2.inverse(), points3D, expectedPoints2_);
    EnforceFrustum();

    // Project points on the 2 camera planes
    return (expectedPoints1_.size() >= 4);
  }

  static bool CompareHomography(const Matrix3T& expected, const Matrix3T& actual, float eps) {
    // The actual homography is recovered up to a scaling factor. In the current implementation
    // (OpenCV 3.0.0, the homography is scaled so that it's last element - homography(2,2) is equal
    // to 1. Therefore we need to scale the expected homography accordingly in order to compare.
    const Matrix3T expectedHomography = (expected * std::copysign(float(1), expected(2, 2))).normalized();
    const Matrix3T actualHomography = (actual * std::copysign(float(1), actual(2, 2))).normalized();
    const float distance = cuvslam::epipolar::CalculateMatricesFrobeniusDistance(actualHomography, expectedHomography);
    return distance < eps;
  }

protected:
  Vector2TVector expectedPoints1_, expectedPoints2_;
  Matrix3T expectedHomography_;
  const size_t numPoints_ = 200;
  Vector3T minRange_ = Vector3T(-5.0f, -5.0f, -10.0f);
  Vector3T maxRange_ = Vector3T(5.0f, 5.0f, -6.0f);
};

}  // namespace test::epipolar
