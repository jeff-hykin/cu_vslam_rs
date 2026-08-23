
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

#include "epipolar/test/fundamental_matrix_utils_test.h"
#include "common/rotation_utils.h"
#include "epipolar/homography_ransac.h"

namespace test::epipolar {
using namespace cuvslam::epipolar;

TEST_F(FundamentalMatrixUtilsTest, VerifyTranslationIsColinear) {
  // Find the essential matrix
  Matrix3T essential;
  ComputeFundamental::ReturnCode retVal = ComputeFundamental::ReturnCode::Success;

  do {
    SetupData();
    ComputeFundamental fundamental(points2DLocal1_, points2DLocal2_);
    retVal = fundamental.findFundamental(essential);
  } while (retVal != ComputeFundamental::ReturnCode::Success);

  // Decompose the essential matrix
  Matrix3T actualRotation1, actualRotation2;
  Vector3T actualTranslation;
  ExtractRotationTranslationFromEssential(essential, actualRotation1, actualRotation2, actualTranslation);

  // Calculate the angle of the expected and actual translation
  const Vector3T expectedTranslation = relativeTransform_.translation();
  float cosAngle = expectedTranslation.dot(actualTranslation) / (expectedTranslation.norm() * actualTranslation.norm());

  // Verify collinearity
  const float eps = 0.005f;
  EXPECT_NEAR(1.0f, std::abs(cosAngle), eps);
}

TEST_F(FundamentalMatrixUtilsTest, VerifyRotation) {
  Matrix3T essential;
  ComputeFundamental::ReturnCode retVal = ComputeFundamental::ReturnCode::Success;

  do {
    SetupData();
    ComputeFundamental fundamental(points2DLocal1_, points2DLocal2_);
    retVal = fundamental.findFundamental(essential);
  } while (retVal != ComputeFundamental::ReturnCode::Success);

  // Decompose the essential matrix
  Matrix3T actualRotation1, actualRotation2;
  Vector3T actualTranslation;
  ExtractRotationTranslationFromEssential(essential, actualRotation1, actualRotation2, actualTranslation);

  // Extract expected rotation
  const Matrix3T expectedRotation = common::CalculateRotationFromSVD(relativeTransform_.matrix());

  // Compute rotation delta between the expected rotation and the first candidate
  float delta1 = CalculateDistanceFromExpectedRotationMatrix(expectedRotation, actualRotation1);
  // Compute rotation delta between the expected rotation and the second candidate
  float delta2 = CalculateDistanceFromExpectedRotationMatrix(expectedRotation, actualRotation2);

  // Verify that the minimum of the delta is under a small threshold
  const float eps = 0.012f;
  EXPECT_LE(std::min(delta1, delta2), eps);
}

TEST_F(FundamentalMatrixUtilsTest, ComputeQuadraticResiduals) {
  Matrix3T essential;
  ComputeFundamental::ReturnCode retVal = ComputeFundamental::ReturnCode::Success;

  do {
    SetupData();
    ComputeFundamental fundamental(points2DLocal1_, points2DLocal2_);
    retVal = fundamental.findFundamental(essential);
  } while (retVal != ComputeFundamental::ReturnCode::Success);

  // Normalize so that the residual threshold does not depend on the norm of the essential matrix.
  essential.normalize();

  float sumResiduals = 0;
  const size_t numPoints = points2DLocal1_.size();

  for (size_t i = 0; i < numPoints; ++i) {
    sumResiduals += ComputeQuadraticResidual(points2DLocal1_[i], points2DLocal2_[i], essential);
  }

  const float residualEps = 0.0125f;
  EXPECT_LE(sumResiduals / static_cast<float>(numPoints), residualEps);
}

TEST_F(FundamentalMatrixUtilsTest, LastSingularValueSmall) {
  SetupData();

  ComputeFundamental computeFundamental(points2DLocal1_, points2DLocal2_);

  Vector9T matATASingularValues = computeFundamental.getMatATASingularValues();

  const float smallestSingularValueEps = 1.0e-5f;
  EXPECT_LT(std::abs(matATASingularValues[8] / matATASingularValues[0]), smallestSingularValueEps);
}

TEST_F(FundamentalMatrixUtilsTest, PlanarHomography) {
  points3DMinRange_ = Vector3T(3, 2, 16);
  points3DMaxRange_ = Vector3T(13, 10, 16);

  camera1_ = Translation3T(Vector3T::Random()) * Rotation3T(Vector3T::Random() * 0.5f, AngleUnits::Radian);
  camera2_ =
      Translation3T(Vector3T::Random() - Vector3T::UnitZ()) * Rotation3T(Vector3T::Random() * 0.5f, AngleUnits::Radian);

  SetupPoints();

  ComputeFundamental computeFundamental(points2DLocal1_, points2DLocal2_);

  ComputeFundamental::ReturnCode degeneracyType = computeFundamental.getStatus();
  EXPECT_TRUE(degeneracyType == ComputeFundamental::ReturnCode::PotentialHomography);

  if (degeneracyType == ComputeFundamental::ReturnCode::PotentialHomography) {
    Matrix3T homography;
    Vector2TPairVector sampleSequence;
    sampleSequence.resize(points2DLocal1_.size());

    size_t si = 0;
    Vector2TVectorCIt p1_iter = points2DLocal1_.cbegin();
    Vector2TVectorCIt p2_iter = points2DLocal2_.cbegin();

    for (; p1_iter != points2DLocal1_.cend(); p1_iter++, p2_iter++) {
      Vector2TPair pr(*p1_iter, *p2_iter);
      sampleSequence[si++] = pr;
    }

    HomographyRansac homographyRansac;
    homographyRansac.setThreshold(0.001f);
    size_t numIterations = homographyRansac(homography, sampleSequence.begin(), sampleSequence.end());

    if (numIterations) {
      Isometry3T relativeTransform = camera2_ * camera1_.inverse();

      if (relativeTransform.translation().norm() > sqrt_epsilon()) {
        EXPECT_FALSE(
            ComputeHomography::IsRotationalHomography(sampleSequence.cbegin(), sampleSequence.cend(), homography));
      } else {
        EXPECT_TRUE(
            ComputeHomography::IsRotationalHomography(sampleSequence.cbegin(), sampleSequence.cend(), homography));
      }
    } else {
      EXPECT_TRUE(false);
    }
  }
}

TEST_F(FundamentalMatrixUtilsTest, RotationHomography) {
  // Setup 2 cameras. Camera1 is random, camera2 has rotation relative to camera2.
  const Translation3T translation(Vector3T::Random());
  camera1_ = translation * Rotation3T(Vector3T::Random(), AngleUnits::Radian);
  camera2_ = translation * Rotation3T(Vector3T::Random(), AngleUnits::Radian);

  float testRot = CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(camera1_.matrix()),
                                                              common::CalculateRotationFromSVD(camera2_.matrix()));
  (void)testRot;

  points3DMinRange_ = Vector3T(-5, -5, 5);
  points3DMaxRange_ = Vector3T(5, 5, 15);

  SetupPoints();

  ComputeFundamental computeFundamental(points2DLocal1_, points2DLocal2_);

  ComputeFundamental::ReturnCode degeneracyType = computeFundamental.getStatus();
  EXPECT_TRUE(degeneracyType == ComputeFundamental::ReturnCode::PotentialHomography);

  if (degeneracyType == ComputeFundamental::ReturnCode::PotentialHomography) {
    Matrix3T homography;
    Vector2TPairVector sampleSequence;
    sampleSequence.resize(points2DLocal1_.size());

    size_t si = 0;
    Vector2TVectorCIt p1_iter = points2DLocal1_.cbegin();
    Vector2TVectorCIt p2_iter = points2DLocal2_.cbegin();

    for (; p1_iter != points2DLocal1_.cend(); p1_iter++, p2_iter++) {
      Vector2TPair pr(*p1_iter, *p2_iter);
      sampleSequence[si++] = pr;
    }

    HomographyRansac homographyRansac;
    homographyRansac.setThreshold(0.0005f);
    size_t numIterations = homographyRansac(homography, sampleSequence.begin(), sampleSequence.end());

    EXPECT_LT(0u, numIterations);
    EXPECT_TRUE(ComputeHomography::IsRotationalHomography(sampleSequence.cbegin(), sampleSequence.cend(), homography));
  }
}

TEST(ComputeFundamental, CheckReturnCodeInvalidScale) {
  Vector2TVector image1(10, Vector2T::Random());  // All points have the same value
  Vector2TVector image2(10, Vector2T::Random());  // All points have the same value

  ComputeFundamental cf(image1, image2);
  EXPECT_EQ(ComputeFundamental::ReturnCode::InvalidScale, cf.getStatus());
}

}  // namespace test::epipolar
