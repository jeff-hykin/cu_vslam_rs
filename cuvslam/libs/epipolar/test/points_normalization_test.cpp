
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

#include "epipolar/points_normalization.h"
#include "common/include_gtest.h"
#include "common/vector_3t.h"

namespace test::epipolar {
using namespace cuvslam;

class NormalizationTest : public testing::Test {
protected:
  virtual void SetUp() {
    eps_ = epsilon() * 10;

    // generate random points
    const auto numPoints = 200;
    points_.resize(numPoints);
    normalizedPoints_.resize(numPoints);

    std::for_each(points_.begin(), points_.end(), [](Vector2T& x) { x = Vector2T::Random(); });

    const cuvslam::epipolar::NormalizationTransform transform(points_);
    std::transform(points_.cbegin(), points_.cend(), normalizedPoints_.begin(), transform);
    normMatrix_ = transform.calcNormMatrix();
    denormMatrix_ = transform.calcDenormMatrix();
  }

public:
  Vector2TVector points_;
  Vector2TVector normalizedPoints_;
  Matrix3T normMatrix_;
  Matrix3T denormMatrix_;
  float eps_;
};

TEST_F(NormalizationTest, CheckResultValidness) {
  Vector2TVector restored;

  ASSERT_TRUE((denormMatrix_ * normMatrix_).isIdentity(eps_));
  ASSERT_TRUE(denormMatrix_.isApprox(normMatrix_.inverse(), eps_));
  ASSERT_TRUE(normMatrix_.isApprox(denormMatrix_.inverse(), 2 * eps_));

  // restore points
  const size_t nPoints = points_.size();

  for (size_t i = 0; i < nPoints; i++) {
    const Vector3T normalized = normMatrix_ * points_[i].homogeneous();
    const Vector3T src = normalizedPoints_[i].homogeneous();
    EXPECT_TRUE(normalized.isApprox(src, 2 * eps_)) << "distance is " << (src - normalized).norm();
    EXPECT_TRUE((points_[i].homogeneous()).isApprox(denormMatrix_ * normalizedPoints_[i].homogeneous(), 2 * eps_));
  }
}

TEST_F(NormalizationTest, CheckMeanValueIsZero) {
  Vector2TVector restored;

  // calculate mean value
  const size_t nPoints = points_.size();
  Vector2T mean = Vector2T::Zero();

  for (size_t i = 0; i < nPoints; i++) {
    mean += normalizedPoints_[i];
  }

  mean /= static_cast<float>(nPoints);

  ASSERT_LT(mean.norm(), 2 * eps_);
}

TEST_F(NormalizationTest, CheckMeanScaleIsSqrt2) {
  Vector2TVector restored;

  // calculate mean scale from zero
  const size_t nPoints = points_.size();
  float scale = 0;

  for (size_t i = 0; i < nPoints; i++) {
    scale += normalizedPoints_[i].squaredNorm();
  }

  scale = std::sqrt(scale / nPoints);

  ASSERT_NEAR(scale, std::sqrt(2.f), 10 * eps_);
}

TEST_F(NormalizationTest, CheckNormalizationStability) {
  // second normalization must produce same result
  Vector2TVector normalizedPoints2(normalizedPoints_.size());

  const cuvslam::epipolar::NormalizationTransform transform(normalizedPoints_);
  std::transform(normalizedPoints_.cbegin(), normalizedPoints_.cend(), normalizedPoints2.begin(), transform);
  const Matrix3T& normMatrix2 = transform.calcNormMatrix();
  const Matrix3T& denormMatrix2 = transform.calcDenormMatrix();

  EXPECT_TRUE(normMatrix2.isIdentity());
  EXPECT_TRUE(denormMatrix2.isIdentity());

  ASSERT_EQ(points_.size(), normalizedPoints2.size());

  for (size_t i = 0; i < points_.size(); i++) {
    const float secondNormalDrift = (normalizedPoints_[i] - normalizedPoints2[i]).norm();
    EXPECT_NEAR(secondNormalDrift, 0, 10 * eps_);
  }
}

TEST_F(NormalizationTest, CheckNormalizationLimit) {
  const size_t nPoints = 2;
  Vector2TVector srcPoints(nPoints);
  Vector2TVector normalizedPoints(nPoints);

  srcPoints[0] = Vector2T(epsilon(), 0);

  for (float v = 1e-6f; v <= 10.f; v *= 10.f) {
    srcPoints[1] = Vector2T(v, 0);
    const cuvslam::epipolar::NormalizationTransform transform(srcPoints, 0);
    std::transform(srcPoints.cbegin(), srcPoints.cend(), normalizedPoints.begin(), transform);

    const Matrix3T& normMatrix = transform.calcNormMatrix();
    const Matrix3T& denormMatrix = transform.calcDenormMatrix();

    for (size_t j = 0; j < nPoints; ++j) {
      const float drift1 = (normMatrix * srcPoints[j].homogeneous() - normalizedPoints[j].homogeneous()).norm();
      EXPECT_NEAR(drift1, 0, eps_);

      const float drift2 = (srcPoints[j].homogeneous() - denormMatrix * normalizedPoints[j].homogeneous()).norm();
      EXPECT_NEAR(drift2, 0, eps_);
    }
  }
}

}  // namespace test::epipolar
