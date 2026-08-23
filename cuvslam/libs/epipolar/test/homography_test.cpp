
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

#include "epipolar/test/homography_test.h"
#include "common/rotation_utils.h"

namespace test::epipolar {
using namespace cuvslam::epipolar;

TEST_F(HomographyTest, VerifyRandomHomography) {
  ASSERT_TRUE(SetupGeneralHomographyCase());

  ComputeHomography computeHomography(expectedPoints1_, expectedPoints2_);

  Matrix3T actualHomography;
  EXPECT_TRUE(computeHomography.findHomography(actualHomography) == ComputeHomography::ReturnCode::Success);

  const float epsDistance = 1.0e-3f;
  EXPECT_TRUE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
}

TEST_F(HomographyTest, VerifyPureCameraRotation) {
  ASSERT_TRUE(SetupHomographyFromPureCameraRotation());

  ComputeHomography computeHomography(expectedPoints1_, expectedPoints2_);

  Matrix3T actualHomography;

  const float epsDistance = 1.0e-3f;

  if (computeHomography.findHomography(actualHomography) == ComputeHomography::ReturnCode::Success) {
    EXPECT_TRUE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
  } else {
    EXPECT_FALSE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
  }
}

class DecomposeHomographyTest : public testing::Test {
protected:
  void GenerateHomographyForPlanarScene() {
    // Generate a random rotation and translation and a random normal (normal is normalized)
    expectedRotation_ = Rotation3T(Vector3T::Random(), AngleUnits::Radian);
    expectedTranslation_ = Vector3T::Random();
    expectedNormal_ = Vector3T::Random().normalized();
    distancePlaneToOrigin_ = 10.0f;
    expectedHomography_ = ComputeHomography::ComposeHomograhpyForPlanarScene(distancePlaneToOrigin_, expectedRotation_,
                                                                             expectedNormal_, expectedTranslation_);

    // Scale the homography - this is not strictly necessary but it is in comparing the expected
    // and actual result.

    auto svd = expectedHomography_.jacobiSvd();
    ASSERT_LT(epsilon(), std::abs(svd.singularValues()[1]));
    expectedHomography_ /= svd.singularValues()[1];
  }

  bool VerifyThatASolutionIsFound(float solutionEps, const CameraMatrixNormalVector& solutions) {
    expectedNormal_ /= expectedNormal_.z();

    for (auto i = solutions.begin(); i != solutions.end(); i++) {
      const float rotationFrob = std::abs(CalculateDistanceFromExpectedRotationMatrix(
          expectedRotation_, common::CalculateRotationFromSVD(i->cameraMatrix.matrix())));
      const bool tranClose =
          expectedTranslation_.normalized().isApprox(i->cameraMatrix.translation().normalized(), solutionEps);

      // Scale the norm so that both normal (expected and solution) point toward the same direction
      // (another possibility would be to check colinearity, instead of equality)
      const bool normClose =
          expectedNormal_.normalized().isApprox((i->normal / i->normal.z()).normalized(), solutionEps);

      if (rotationFrob < solutionEps && tranClose && normClose) {
        return true;
      }
    }

    return false;
  }

protected:
  Matrix3T expectedRotation_;
  Vector3T expectedTranslation_;
  Vector3T expectedNormal_;
  Matrix3T expectedHomography_;
  float distancePlaneToOrigin_;
};

TEST_F(DecomposeHomographyTest, GeneralCase) {
  GenerateHomographyForPlanarScene();
  CameraMatrixNormalVector cameraMatrixNormalVector;
  ComputeHomography::DecomposeHomography(expectedHomography_, cameraMatrixNormalVector);

  const float totalEps = 0.007f;
  EXPECT_TRUE(VerifyThatASolutionIsFound(totalEps, cameraMatrixNormalVector));
}

class DecomposeHomographyIntegTest : public DecomposeHomographyTest {
protected:
  void SetUp() {
    DecomposeHomographyTest::SetUp();
    Isometry3T camera1, camera2;

    ASSERT_TRUE(
        CreateCameraMatrix(Vector3T::Random(), (minRange_ + maxRange_) / 2.0f, Vector3T(0.0f, 1.0f, 0.0f), camera1));

    ASSERT_TRUE(
        CreateCameraMatrix(Vector3T::Random(), (minRange_ + maxRange_) / 2.0f, Vector3T(0.0f, 1.0f, 0.0f), camera2));

    SetupPointsProjectionsFromCameras(camera1, camera2);

    relativeTransform_ = (camera2.inverse()) * camera1;

    expectedRotation_ = common::CalculateRotationFromSVD(relativeTransform_.matrix());
    expectedTranslation_ = relativeTransform_.translation();

    ASSERT_GT(numPoints_, static_cast<size_t>(2));  // If not 3 points, we cannot compute the normal.

    // We assume here that the 3 points are not colinear, which holds in the general case.
    for (size_t i = 2; i < numPoints_; i++) {
      expectedNormal_ = common::CalculateRotationFromSVD(camera1.matrix()).inverse() *
                        (points3D_[1] - points3D_[0]).cross(points3D_[i] - points3D_[0]);

      // If norm is too small, the normal might be inaccurate
      if (expectedNormal_.norm() < 0.001f) {
        continue;
      }

      break;
    }

    expectedNormal_.normalize();
  }

  virtual void SetupPointsProjectionsFromCameras(const Isometry3T& camera1, const Isometry3T& camera2) {
    // Generate 3D points
    points3D_ = GeneratePointsInCube(numPoints_, minRange_, maxRange_);

    // Project points on the 2 camera planes
    Project3DPointsInLocalCoordinates(camera1.inverse(), points3D_, expectedPoints1_);
    Project3DPointsInLocalCoordinates(camera2.inverse(), points3D_, expectedPoints2_);
  }

protected:
  Isometry3T relativeTransform_;
  Vector2TVector expectedPoints1_, expectedPoints2_;
  const size_t numPoints_ = 100;
  float planeZCoord_ = -10.0f;
  Vector3T minRange_ = Vector3T(-5.0f, -5.0f, planeZCoord_);
  Vector3T maxRange_ = Vector3T(5.0f, 5.0f, planeZCoord_);
  Vector3TVector points3D_;
};

TEST_F(DecomposeHomographyIntegTest, GeneralCaseAllSolutions) {
  ComputeHomography computeHomography(expectedPoints1_, expectedPoints2_);

  Matrix3T actualHomography;
  EXPECT_TRUE(computeHomography.findHomography(actualHomography) == ComputeHomography::ReturnCode::Success);

  CameraMatrixNormalVector cameraMatrixNormalVector;
  ComputeHomography::DecomposeHomography(actualHomography, cameraMatrixNormalVector);

  const float totalEps = 0.009f;
  EXPECT_TRUE(VerifyThatASolutionIsFound(totalEps, cameraMatrixNormalVector));
}

}  // namespace test::epipolar
