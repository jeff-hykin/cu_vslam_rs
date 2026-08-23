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

#include "math/twist_to_angle.h"

#include <array>

#include "common/include_gtest.h"
#include "common/isometry_utils.h"
#include "math/twist.h"

namespace test::math {
using namespace cuvslam;

namespace {

Matrix6T MakeSmallTwistCovariance() {
  Matrix6T sqrt_covariance = Matrix6T::Zero();
  sqrt_covariance.diagonal() << 0.0011f, 0.0013f, 0.0017f, 0.0023f, 0.0029f, 0.0031f;
  sqrt_covariance(1, 0) = 0.0004f;
  sqrt_covariance(2, 0) = -0.0003f;
  sqrt_covariance(2, 1) = 0.0005f;
  sqrt_covariance(3, 1) = -0.0006f;
  sqrt_covariance(4, 0) = 0.0002f;
  sqrt_covariance(4, 3) = -0.0007f;
  sqrt_covariance(5, 2) = 0.0008f;
  sqrt_covariance(5, 3) = 0.0006f;
  sqrt_covariance(5, 4) = 0.0009f;
  return sqrt_covariance * sqrt_covariance.transpose();
}

Vector6T XYZRollPitchYaw(const Isometry3T& pose) {
  Vector6T result;
  result.head<3>() = pose.translation();
  result.tail<3>() = getEulerRotation(pose).cast<float>();
  return result;
}

Matrix6T SampleXYZRollPitchYawCovariance(const Vector6T& mean_twist, const Matrix6T& twist_covariance) {
  constexpr float kSigmaPointScale = 2.44948974278f;  // sqrt(6)
  constexpr int kNumSigmaPoints = 12;

  const Matrix6T sqrt_covariance = twist_covariance.llt().matrixL();
  std::array<Vector6T, kNumSigmaPoints> samples;
  int sample_index = 0;
  for (int col = 0; col < 6; ++col) {
    for (float sign : {-1.f, 1.f}) {
      const Vector6T sample_twist = mean_twist + sign * kSigmaPointScale * sqrt_covariance.col(col);
      Isometry3T pose;
      cuvslam::math::Exp(pose, sample_twist);
      samples[sample_index++] = XYZRollPitchYaw(pose);
    }
  }

  Vector6T sample_mean = Vector6T::Zero();
  for (const Vector6T& sample : samples) {
    sample_mean += sample;
  }
  sample_mean /= static_cast<float>(samples.size());

  Matrix6T sample_covariance = Matrix6T::Zero();
  for (const Vector6T& sample : samples) {
    const Vector6T centered_sample = sample - sample_mean;
    sample_covariance += centered_sample * centered_sample.transpose();
  }
  return sample_covariance / static_cast<float>(samples.size());
}

void ExpectMatricesNear(const Matrix6T& actual, const Matrix6T& expected, float tolerance) {
  for (int row = 0; row < actual.rows(); ++row) {
    for (int col = 0; col < actual.cols(); ++col) {
      EXPECT_NEAR(actual(row, col), expected(row, col), tolerance) << "row=" << row << ", col=" << col;
    }
  }
}

}  // namespace

TEST(TwistToAngleTest, PoseCovToXYZRollPitchYawCovMatchesSigmaPointSamplingAtMultiplePoses) {
  // PoseCovToXYZRollPitchYawCov is a first-order conversion around the mean pose.
  // The input covariance is interpreted as the covariance of `twist` where
  // pose = Exp(twist), in [rx, ry, rz, tx, ty, tz] order -- matching the
  // production caller (cuvslam2.cpp), which inverts the PNP Hessian and treats the
  // result as the cov of Log(pose). Keep sample perturbations small so the
  // first-order conversion stays valid.
  const Matrix6T twist_covariance = MakeSmallTwistCovariance();

  const std::array<Vector6T, 4> mean_twists = {{
      (Vector6T() << 0.06f, -0.04f, 0.03f, 0.11f, -0.07f, 0.13f).finished(),
      (Vector6T() << 0.19f, -0.11f, 0.07f, 0.41f, -0.29f, 0.83f).finished(),
      (Vector6T() << -0.31f, 0.12f, 0.22f, -1.20f, 0.35f, 0.48f).finished(),
      (Vector6T() << 0.12f, -0.18f, 0.05f, 2.00f, -1.00f, 0.60f).finished(),
  }};

  for (size_t mean_index = 0; mean_index < mean_twists.size(); ++mean_index) {
    SCOPED_TRACE(mean_index);
    const Vector6T& mean_twist = mean_twists[mean_index];
    Isometry3T pose;
    cuvslam::math::Exp(pose, mean_twist);

    const Matrix6T covariance = cuvslam::math::PoseCovToXYZRollPitchYawCov(twist_covariance, pose);
    const Matrix6T sampled_covariance = SampleXYZRollPitchYawCovariance(mean_twist, twist_covariance);

    ExpectMatricesNear(covariance, sampled_covariance, 2e-7f);
  }
}

}  // namespace test::math
