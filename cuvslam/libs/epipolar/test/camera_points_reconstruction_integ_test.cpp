
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

#include "epipolar/test/camera_points_reconstruction_integ_test.h"
#include "common/rotation_utils.h"
#include "epipolar/fundamental_matrix_utils.h"
#include "epipolar/point_reconstruction.h"

namespace test::epipolar {

TEST_F(CameraPointsReconstructionIntegTest, ReconstructionTestPerfectData) {
  using namespace cuvslam::epipolar;
  // Find the essential matrix
  Matrix3T essential;

  ComputeFundamental fundamental(points2DLocal1_, points2DLocal2_);
  EXPECT_EQ(ComputeFundamental::ReturnCode::Success, fundamental.findFundamental(essential));

  // Find the optimal combination of rotation/translation and reconstruct the transform from
  // camera1 to camera2.
  EXPECT_TRUE(FindOptimalCameraMatrixFromEssential(point2Dpairs_.cbegin(), point2Dpairs_.cend(), essential,
                                                   actualRelativeTransform_));

  // Compute the Frobenius norm between the original and the reconstructed transforms
  float delta =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedRelativeTransform_.matrix()),
                                                  common::CalculateRotationFromSVD(actualRelativeTransform_.matrix()));

  // Verify that the minimum of the delta is under a small threshold
  const float eps = 0.005f;
  EXPECT_LE(delta, eps);

  // Verify that the number of points in front of both cameras is the total number of 3D points
  // (since there is no noise in the data)
  size_t actualNumOfPointsInFrontOfBoth =
      CountPointsInFrontOfCameras(point2Dpairs_.cbegin(), point2Dpairs_.cend(), actualRelativeTransform_.inverse());
  EXPECT_EQ(num3DPoints_, actualNumOfPointsInFrontOfBoth);

  // Reconstruct the 3D points from the actual transform
  ASSERT_TRUE(
      Reconstruct3DPointsFrom2DPointsAndRelativeTransform(point2Dpairs_, actualRelativeTransform_, actual3DPoints_));

  // Verify that all the reconstructed points are close to their initial location (within a certain error)
  const float eps3DPoints = 0.02f;
  CompareGeneratedAndReconstructedPoints(expected3DPoints_, actual3DPoints_, eps3DPoints);
}

}  // namespace test::epipolar
