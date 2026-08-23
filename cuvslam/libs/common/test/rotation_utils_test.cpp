
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

#include "common/rotation_utils.h"
#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

using namespace cuvslam;

// A 3x3 determinant accumulates ~17 float32 operations on top of the SVD's own round-off, so the
// result can legitimately land several ULPs away from 1. Tolerating 1e-6f made these tests order
// dependent: Matrix3T::Random() draws from the shared std::rand() stream, so a different test order
// yields a different matrix, and a handful of them overshot by ~1.1e-6.
constexpr float kDeterminantTolerance = 1e-5f;

TEST(RotationUtilsTest, CalculateRotationFromSVD_Identity) {
  // Test with identity matrix
  Isometry3T identity = Isometry3T::Identity();
  Matrix3T rotation = common::CalculateRotationFromSVD(identity.matrix());

  EXPECT_TRUE(rotation.isApprox(Matrix3T::Identity())) << rotation;
  EXPECT_NEAR(rotation.determinant(), 1.0f, kDeterminantTolerance) << rotation;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_RandomRotation) {
  // Test with a random rotation matrix
  Vector3T random_axis = Vector3T::Random().normalized();
  float random_angle = 1.5f;  // radians
  Eigen::AngleAxis<float> aa(random_angle, random_axis);
  Matrix3T expected_rotation = aa.toRotationMatrix();

  Isometry3T transform = Isometry3T::Identity();
  transform.linear() = expected_rotation;

  Matrix3T computed_rotation = common::CalculateRotationFromSVD(transform.matrix());

  EXPECT_TRUE(computed_rotation.isApprox(expected_rotation, 1e-6f)) << computed_rotation;
  EXPECT_NEAR(computed_rotation.determinant(), 1.0f, kDeterminantTolerance) << computed_rotation;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_WithTranslation) {
  // Test with rotation + translation
  Vector3T random_axis = Vector3T::Random().normalized();
  float random_angle = 0.8f;  // radians
  Eigen::AngleAxis<float> aa(random_angle, random_axis);
  Matrix3T expected_rotation = aa.toRotationMatrix();
  Vector3T translation = Vector3T::Random() * 10.0f;

  Isometry3T transform = Isometry3T::Identity();
  transform.linear() = expected_rotation;
  transform.translation() = translation;

  Matrix3T computed_rotation = common::CalculateRotationFromSVD(transform.matrix());

  EXPECT_TRUE(computed_rotation.isApprox(expected_rotation, 1e-6f)) << computed_rotation;
  EXPECT_NEAR(computed_rotation.determinant(), 1.0f, kDeterminantTolerance) << computed_rotation;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_WithScaling) {
  // Test with a matrix that has scaling (non-orthogonal)
  Matrix3T non_orthogonal = Matrix3T::Random() * 2.0f;

  Isometry3T transform = Isometry3T::Identity();
  transform.linear() = non_orthogonal;

  Matrix3T computed_rotation = common::CalculateRotationFromSVD(transform.matrix());

  // Should still be a valid rotation matrix
  EXPECT_NEAR(computed_rotation.determinant(), 1.0f, kDeterminantTolerance) << computed_rotation;
  EXPECT_TRUE((computed_rotation * computed_rotation.transpose()).isApprox(Matrix3T::Identity(), 5e-6f))
      << computed_rotation;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_4x4Matrix) {
  // Test with 4x4 transformation matrix
  Vector3T random_axis = Vector3T::Random().normalized();
  float random_angle = 1.2f;  // radians
  Eigen::AngleAxis<float> aa(random_angle, random_axis);
  Matrix3T expected_rotation = aa.toRotationMatrix();
  Vector3T translation = Vector3T::Random() * 5.0f;

  Matrix4T transform_4x4 = Matrix4T::Identity();
  transform_4x4.block<3, 3>(0, 0) = expected_rotation;
  transform_4x4.block<3, 1>(0, 3) = translation;

  Matrix3T computed_rotation = common::CalculateRotationFromSVD(transform_4x4);

  EXPECT_TRUE(computed_rotation.isApprox(expected_rotation, 1e-6f)) << computed_rotation;
  EXPECT_NEAR(computed_rotation.determinant(), 1.0f, kDeterminantTolerance) << computed_rotation;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_PureReflection) {
  // diag(-1, 1, 1): a pure reflection with det = -1
  Matrix3T reflection = Matrix3T::Identity();
  reflection(0, 0) = -1.0f;

  Matrix3T result = common::CalculateRotationFromSVD<float, 3>(reflection);

  EXPECT_NEAR(result.determinant(), 1.0f, kDeterminantTolerance) << result;
  EXPECT_TRUE((result * result.transpose()).isApprox(Matrix3T::Identity(), 1e-6f)) << result;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_RotationWithNegatedColumn) {
  // A valid rotation with one column negated produces det = -1
  Vector3T axis = Vector3T(1.0f, 1.0f, 1.0f).normalized();
  Eigen::AngleAxis<float> aa(1.0f, axis);
  Matrix3T rotation = aa.toRotationMatrix();

  Matrix3T improper = rotation;
  improper.col(2) *= -1.0f;
  ASSERT_LT(improper.determinant(), 0.0f);

  Matrix3T result = common::CalculateRotationFromSVD<float, 3>(improper);

  EXPECT_NEAR(result.determinant(), 1.0f, kDeterminantTolerance) << result;
  EXPECT_TRUE((result * result.transpose()).isApprox(Matrix3T::Identity(), 1e-6f)) << result;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_ScaledReflection) {
  // Non-uniform scaling with a reflection: diag(-3, 2, 1), det = -6
  Matrix3T M = Matrix3T::Zero();
  M(0, 0) = -3.0f;
  M(1, 1) = 2.0f;
  M(2, 2) = 1.0f;

  Matrix3T result = common::CalculateRotationFromSVD<float, 3>(M);

  EXPECT_NEAR(result.determinant(), 1.0f, kDeterminantTolerance) << result;
  EXPECT_TRUE((result * result.transpose()).isApprox(Matrix3T::Identity(), 1e-6f)) << result;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_NegativeDeterminantRandom) {
  // Random matrix with negative determinant
  Matrix3T M = Matrix3T::Random();
  if (M.determinant() > 0) {
    M.col(0) *= -1.0f;
  }
  ASSERT_LT(M.determinant(), 0.0f);

  Matrix3T result = common::CalculateRotationFromSVD<float, 3>(M);

  EXPECT_NEAR(result.determinant(), 1.0f, kDeterminantTolerance) << result;
  EXPECT_TRUE((result * result.transpose()).isApprox(Matrix3T::Identity(), 5e-6f)) << result;
}

TEST(RotationUtilsTest, CalculateRotationFromSVD_3x4Matrix) {
  // Test with 3x4 affine transformation matrix
  Vector3T random_axis = Vector3T::Random().normalized();
  float random_angle = 0.7f;  // radians
  Eigen::AngleAxis<float> aa(random_angle, random_axis);
  Matrix3T expected_rotation = aa.toRotationMatrix();
  Vector3T translation = Vector3T::Random() * 3.0f;

  Eigen::Matrix<float, 3, 4> transform_3x4;
  transform_3x4.block<3, 3>(0, 0) = expected_rotation;
  transform_3x4.block<3, 1>(0, 3) = translation;

  Matrix3T computed_rotation = common::CalculateRotationFromSVD(transform_3x4);

  EXPECT_TRUE(computed_rotation.isApprox(expected_rotation, 1e-6f)) << computed_rotation;
  EXPECT_NEAR(computed_rotation.determinant(), 1.0f, kDeterminantTolerance) << computed_rotation;
}
