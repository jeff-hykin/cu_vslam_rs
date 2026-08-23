
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

#include "common/include_gtest.h"
#include "epipolar/fundamental_matrix_utils.h"
#include "epipolar/test/generate_visible_points_test.h"

namespace test::epipolar {

namespace {
template <typename ReturnCode>
bool CheckSuccess(ReturnCode code) {
  return ReturnCode::Success == code;
}
}  // namespace

using namespace cuvslam;

class FundamentalMatrixUtilsTest : public testing::Test {
protected:
  virtual void SetupData() {
    bool setUpFailed = true;

    while (setUpFailed) {
      SetupCameras();
      setUpFailed = SetupPoints();
    }
  }

  virtual void SetupCameras() {
    camera1_ = createRandomCamera();
    camera2_ = createRandomCamera();

    relativeTransform_ = (camera2_.inverse()) * camera1_;
  }

  virtual bool SetupPoints() {
    Vector2TVectorVector points2D;
    Isometry3TVector cameras;
    cameras.resize(2);
    cameras[0] = camera1_;
    cameras[1] = camera2_;
    generateVisiblePoints(cameras, num3DPoints_, cuvslam::epipolar::ComputeFundamental::MINIMUM_NUMBER_OF_POINTS,
                          points3DMinRange_, points3DMaxRange_, points2D);
    points2DLocal1_ = points2D[0];
    points2DLocal2_ = points2D[1];

    return (points2DLocal1_.size() < cuvslam::epipolar::ComputeFundamental::MINIMUM_NUMBER_OF_POINTS);
  }

protected:
  const size_t num3DPoints_ = 60;
  Vector3T points3DMinRange_ = Vector3T(3, 2, 5);
  Vector3T points3DMaxRange_ = Vector3T(13, 10, 16);
  Isometry3T camera1_, camera2_;
  Isometry3T relativeTransform_;
  Vector2TVector points2DLocal1_;
  Vector2TVector points2DLocal2_;
};

}  // namespace test::epipolar
