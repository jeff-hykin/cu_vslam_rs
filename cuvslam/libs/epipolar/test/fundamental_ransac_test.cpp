
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

#include "epipolar/fundamental_ransac.h"

#include "common/include_gtest.h"
#include "common/rotation_utils.h"
#include "epipolar/camera_projection.h"
#include "epipolar/fundamental_matrix_utils.h"
#include "epipolar/test/camera_points_reconstruction_integ_test.h"

namespace test::epipolar {

using namespace cuvslam;

using FundamentalRansac = math::Ransac<Fundamental>;

class FundamentalRansacTest : public CameraPointsReconstructionIntegTest {
protected:
  virtual void SetUp() {
    CameraPointsReconstructionIntegTest::SetUp();

    point2Dpairs_.clear();
    points2DLocal2_.clear();
    GeneratePairOf2DPointsFrom3DPoints(expected3DPoints_, absoluteCamera1_, absoluteCamera2_, &point2Dpairs_);

    FundamentalRansac fr(FundamentalRansac::Epipolar, 0.001f, 0.001f);

    //        Matrix3T essential;
    size_t numIterations = fr(essential_, point2Dpairs_.begin(), point2Dpairs_.end());
    ASSERT_TRUE(numIterations > 0);

    // make some 2d points an outliers in the sense of x'Fx > threshold
    const float theashold = 0.01f;
    size_t counter = 0;
    size_t epipolCount = 0;

    for (auto& pointPair : point2Dpairs_) {
      if (counter++ % outliersFrequency_ == 0) {
        Vector2T d = (essential_ * pointPair.first.homogeneous()).topRows(2);
        const float delta = d.squaredNorm();

        if (delta < epsilon()) {
          epipolCount++;
          continue;
        }

        if (delta < theashold) {
          d *= theashold / delta;
        }

        pointPair.second += d;
      }
    }

    ASSERT_TRUE(epipolCount < 5);
  }

protected:
  const size_t outliersFrequency_ = 6;
  Matrix3T essential_;
};

TEST_F(FundamentalRansacTest, VerifyRansacEpipolar) {
  // reduction of the acceptance threshold (2nd option) increase accuracy of the result (Essential matrix) but increase
  // number of iterations for RANSAC to converge
  FundamentalRansac fr(FundamentalRansac::Epipolar, 0.0005f, 0.001f);
  Matrix3T essential;
  size_t numIterations = fr(essential, point2Dpairs_.begin(), point2Dpairs_.end());

  EXPECT_TRUE(numIterations > 0);

  const float rotTolerance = 1e-2f;
  const float tranTolerance = 5e-2f;

  Vector2TVector pointSet1, pointSet2;

  std::for_each(point2Dpairs_.cbegin(), point2Dpairs_.cend(), [&](const Vector2TPair& i) {
    if (ComputeQuadraticResidual(i.first, i.second, essential) < fr.getTheshold()) {
      pointSet1.push_back(i.first);
      pointSet2.push_back(i.second);
    }
  });

  ComputeFundamental fund(pointSet1, pointSet2);
  EXPECT_EQ(ComputeFundamental::ReturnCode::Success, fund.findFundamental(essential));

  EXPECT_TRUE(FindOptimalCameraMatrixFromEssential(point2Dpairs_.begin(), point2Dpairs_.end(), essential,
                                                   actualRelativeTransform_));

  const float rotDelta =
      CalculateMatricesFrobeniusDistance(common::CalculateRotationFromSVD(expectedRelativeTransform_.matrix()),
                                         common::CalculateRotationFromSVD(actualRelativeTransform_.matrix()));

  const float tranDelta = CalculateMatricesFrobeniusDistance(expectedRelativeTransform_.translation().matrix(),
                                                             actualRelativeTransform_.translation().matrix());

  EXPECT_TRUE(rotDelta < rotTolerance);
  EXPECT_TRUE(tranDelta < tranTolerance);
}

}  // namespace test::epipolar
