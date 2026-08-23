
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

#include "epipolar/resectioning_utils.h"
#include "camera/rig.h"
#include "common/include_gtest.h"
#include "common/rotation_utils.h"
#include "common/statistic.h"
#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "epipolar/test/generate_points_in_cube_test.h"
#include "epipolar/test/resectioning_utils_test.h"
#include "pnp/multicam_pnp.h"
namespace test::epipolar {

using namespace cuvslam;
using namespace cuvslam::epipolar;

///-------------------------------------------------------------------------------------------------
/// @brief ResectioningUtilsTest generates a cloud of 3D points randomly in a cube. A camera is generated,
/// facing the point cloud. All 3D points are projected into the normalized camera space (K=Identity).
/// The tests ensure that the camera converges when perturbed.
///-------------------------------------------------------------------------------------------------
class ResectioningUtilsTest : public testing::Test {
protected:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Setup 3D points, 2D points and camera.
  ///
  /// @param pointMinRange The minimum range of the cube from which 3D points are drawn.
  /// @param pointMaxRange The maximum range of the cube from which 3D points are drawn.
  ///-------------------------------------------------------------------------------------------------
  void SetUpData() {
    minRange_ = Vector3T(-15, -10, -28);
    maxRange_ = Vector3T(10, 15, -8);

    // push points at various random distance from the camera
    const Vector2T randVals = Vector2T::Random();
    distFactor_ = 1.f + (1.f + randVals.x()) * (isEnableDistantPoints_ ? 2000.f : 50.f);
    maxRange_ *= distFactor_;
    minRange_ *= distFactor_;

    ASSERT_TRUE(distFactor_ >= 1.f);

    // Create a camera pointing toward the center of the cloud of 3D points
    ASSERT_TRUE(CreateCameraMatrix(Vector3T::Random(), (maxRange_ + minRange_) / 2.0, Vector3T(0.0, 1.0, 1.0),
                                   expectedWorldFromCamera_));

    // Generate a set of 3D points
    points3D_ = GeneratePointsInCube(num3DPoints_, minRange_, maxRange_, [&](const Vector3T& p) -> bool {
      return (expectedWorldFromCamera_.inverse() * p).z() > FrustumProperties::MINIMUM_HITHER;
    });

    ASSERT_TRUE(points3D_.size() == num3DPoints_);

    // Points are projected on the camera plane in the camera local coordinate system.
    // Camera intrinsics K is considered Identity
    Project3DPointsInLocalCoordinates(expectedWorldFromCamera_.inverse(), points3D_, points2DLocal_);

    // Add some noise to 2D points
    std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
    std::normal_distribution<float> d(0.f, noiseSigma_);

    points2DLocalWithNoise_.resize(points2DLocal_.size());
    std::transform(points2DLocal_.cbegin(), points2DLocal_.cend(), points2DLocalWithNoise_.begin(),
                   [&](const Vector2T& a) -> Vector2T { return a + Vector2T(d(gen), d(gen)); });
  }

  void InvokeAllChannelsMethodExpMap(bool withNoise = false) {
    auto& points2D = withNoise ? points2DLocalWithNoise_ : points2DLocal_;
    actualWorldFromCamera_ = initialWorldFromCamera_;
    EXPECT_TRUE(OptimizeCameraExtrinsicsExpMap<6>(points3D_.cbegin(), points3D_.cend(), points2D.cbegin(),
                                                  points2D.cend(), actualWorldFromCamera_));
  }

  // Owner of pose refinement input data
  struct PoseRefinementData {
    std::unordered_map<TrackId, Vector3T> landmarks;
    std::vector<camera::Observation> observations;
  };

  void PreparePoseRefinementInput(bool withNoise, PoseRefinementData& data) {
    auto& points2D = withNoise ? points2DLocalWithNoise_ : points2DLocal_;

    data.landmarks.clear();
    data.observations.clear();

    for (size_t i = 0; i < points2D.size(); i++) {
      const Vector2T& xy = points2D[i];
      const Vector3T& point = points3D_[i];

      data.landmarks.insert({i, point});
      data.observations.emplace_back(0, i, xy, Matrix2T::Identity());
    }
  }

  void OptimizeCameraPose(const camera::Rig& rig, bool withNoise = false) {
    // reset initial guess
    actualWorldFromCamera_ = initialWorldFromCamera_;

    pnp_ = std::make_unique<pnp::PNPSolver>(rig);

    // This struct owns memory.
    // "input" will point to members of "data"
    PoseRefinementData data;
    PreparePoseRefinementInput(withNoise, data);

    // we are not checking covariance, it needs a different test
    Matrix6T precision;
    Isometry3T left_camera_from_world = actualWorldFromCamera_.inverse();
    auto status = pnp_->solve(left_camera_from_world, precision, data.observations, data.landmarks, pnp::PNPSettings{});
    EXPECT_TRUE(status);
  }

  void InvokeTranslationConstrainedMethodExpMap(bool withNoise = false) {
    auto& points2D = withNoise ? points2DLocalWithNoise_ : points2DLocal_;
    actualWorldFromCamera_ = initialWorldFromCamera_;
    const bool result = OptimizeCameraExtrinsicsExpMapConstrained(
        points3D_.cbegin(), points3D_.cend(), points2D.cbegin(), points2D.cend(), actualWorldFromCamera_, constraint_);
    EXPECT_TRUE(result);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Invoke the OptimizeCameraExtrinsicsExpMap<3> on camera rotation channels only with the camera starting.
  ///-------------------------------------------------------------------------------------------------
  void InvokeRotationOnlyMethodExpMap(bool withNoise = false) {
    auto& points2D = withNoise ? points2DLocalWithNoise_ : points2DLocal_;
    actualWorldFromCamera_ = initialWorldFromCamera_;
    EXPECT_TRUE(OptimizeCameraExtrinsicsExpMap<3>(points3D_.cbegin(), points3D_.cend(), points2D.cbegin(),
                                                  points2D.cend(), actualWorldFromCamera_));
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Test that the actual matrix has converged to the expected matrix.
  ///-------------------------------------------------------------------------------------------------
  // virtual void TestConvergence(float threshold)
  //{
  //     float distance = CalculateMatricesFrobeniusDistance(
  //                            expectedWorldFromCamera_.matrix(), actualWorldFromCamera_.matrix());
  //     EXPECT_LT(distance, threshold);
  // }

protected:
  void RunPoseRefinement(const camera::Rig& rig) {
    OptimizeCameraPose(rig, true);
    auto rotationError =
        CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                    common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));

    EXPECT_LT(rotationError, errorThresholdWithNoise_);

    auto translationError =
        (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
    EXPECT_LT(translationError, errorThresholdTranslationWithNoise_);
  }

  std::unique_ptr<pnp::PNPSolver> pnp_;
  bool isEnableDistantPoints_ = false;
  float distFactor_;
  Vector3T minRange_;
  Vector3T maxRange_;
  const size_t num3DPoints_ = 200;
  // Is it possible that OptimizeCameraExtrinsics will not converge due to the nature of the
  // underlying algorithm. If this is the case, scale down rotation and translation. At the time
  // of code submission, the algorithm did not fail any of the tests for a 1000 run
  // with the following values: rotationScale = 0.5, translationScale = 3.0.
  const float rotationScale_ = 0.5f;     // Turn this knob if tests are failing
  const float translationScale_ = 1.0f;  // Turn this knob if tests are failing
  const float noiseSigma_ = 0.0001f;
  const float errorThresholdOpenCV_ = sqrt_epsilon();
  const float errorThreshold_ = sqrt_epsilon() * 100;
  const float errorThresholdTranslation_ = sqrt_epsilon() * 100;
  const float errorThresholdWithNoise_ = sqrt_epsilon() * 1000;
  const float errorThresholdTranslationWithNoise_ = sqrt_epsilon() * 1000;
  Vector3TVector points3D_;
  Vector2TVector points2DLocal_;
  Vector2TVector points2DLocalWithNoise_;
  Isometry3T expectedWorldFromCamera_;
  Isometry3T actualWorldFromCamera_;
  Isometry3T initialWorldFromCamera_;
  Vector3T constraint_;
};

TEST_F(ResectioningUtilsTest, AlreadyConvergedExpMap) {
  SetUpData();
  initialWorldFromCamera_ = expectedWorldFromCamera_;
  InvokeAllChannelsMethodExpMap();

  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);

  GTestStatVarT(tranDistance) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistance), errorThresholdTranslation_);

  InvokeAllChannelsMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);
  GTestStatVarT(tranDistanceNoise) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistanceNoise), errorThresholdTranslationWithNoise_);

  // test new API
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0].setIdentity();
  RunPoseRefinement(rig);
}

TEST_F(ResectioningUtilsTest, DISABLED_CameraTranslatedExpMap) {  // flaky
  SetUpData();
  initialWorldFromCamera_ = expectedWorldFromCamera_;
  initialWorldFromCamera_.pretranslate(translationScale_ * Vector3T::Random());

  InvokeAllChannelsMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);
  GTestStatVarT(tranDistance) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistance), errorThresholdTranslation_);

  InvokeAllChannelsMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);
  GTestStatVarT(tranDistanceNoise) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistanceNoise), errorThresholdTranslationWithNoise_);

  // test new API
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0].setIdentity();
  RunPoseRefinement(rig);
}

TEST_F(ResectioningUtilsTest, ConstrainedCameraTranslatedExpMap) {
  SetUpData();
  initialWorldFromCamera_ = expectedWorldFromCamera_;

  do {
    constraint_ = 10 * Vector3T::Random();
  } while (constraint_.norm() < sqrt_epsilon());

  constraint_.normalize();
  constraint_ *= 100;

  const float s = Vector<float, 1>::Random().x();
  initialWorldFromCamera_.pretranslate(constraint_ * s);

  InvokeTranslationConstrainedMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);
  GTestStatVarT(tranDistance) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistance), errorThresholdTranslation_);

  InvokeTranslationConstrainedMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThreshold_);
  GTestStatVarT(tranDistanceNoise) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistanceNoise), errorThresholdTranslationWithNoise_);
}

TEST_F(ResectioningUtilsTest, CameraRotatedExpMap) {
  SetUpData();
  initialWorldFromCamera_ =
      expectedWorldFromCamera_ * Rotation3T(rotationScale_ * Vector3T::Random(), AngleUnits::Radian);

  float rotDistanceStart =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(initialWorldFromCamera_.matrix()));
  (void)rotDistanceStart;

  InvokeRotationOnlyMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);

  InvokeRotationOnlyMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);

  // test new API
  // RunRotationRefinement();
}

TEST_F(ResectioningUtilsTest, CameraRotatedAndTranslatedExpMap) {
  SetUpData();
  initialWorldFromCamera_ = expectedWorldFromCamera_ * Translation3T(translationScale_ * Vector3T::Random()) *
                            Rotation3T(rotationScale_ * Vector3T::Random(), AngleUnits::Radian);
  float rotDistanceStart =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(initialWorldFromCamera_.matrix()));
  (void)rotDistanceStart;

  InvokeAllChannelsMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);
  GTestStatVarT(tranDistance) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistance), errorThresholdTranslation_);

  InvokeAllChannelsMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);
  GTestStatVarT(tranDistanceNoise) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistanceNoise), errorThresholdTranslationWithNoise_);

  // test new API
  //    camera::Rig rig;
  //    rig.num_cameras = 1;
  //    rig.camera_from_rig[0].setIdentity();
  //    RunPoseRefinement(rig);
}

TEST_F(ResectioningUtilsTest, CameraRotatedPointsFarAwayExpMap) {
  isEnableDistantPoints_ = true;
  SetUpData();

  initialWorldFromCamera_ =
      expectedWorldFromCamera_ * Rotation3T(rotationScale_ * Vector3T::Random(), AngleUnits::Radian);
  float rotDistanceStart =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(initialWorldFromCamera_.matrix()));
  (void)rotDistanceStart;

  InvokeRotationOnlyMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);

  InvokeRotationOnlyMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);

  // test new API
  // RunRotationRefinement();
}

TEST_F(ResectioningUtilsTest, StartWithWorldCoordinatesExpMap) {
  SetUpData();
  initialWorldFromCamera_ = Isometry3T::Identity();
  float rotDistanceStart =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(initialWorldFromCamera_.matrix()));

  QuaternionT q(Vector3T(rotationScale_, rotationScale_, rotationScale_), AngleUnits::Radian);
  float rot = q.angularDistance(QuaternionT::Identity());

  if (rotDistanceStart > rot) {
    TraceMessage("Skip StartWithWorldCoordinatesExpMap test. Too much rotation from Identity pose [%f].",
                 rotDistanceStart);
    return;  // skip, too much rotation from identity, it most likely will not converge when set too far
  }

  InvokeAllChannelsMethodExpMap();
  GTestStatVarT(rotDistance) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistance), errorThreshold_);
  GTestStatVarT(tranDistance) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistance), errorThresholdTranslation_);

  InvokeAllChannelsMethodExpMap(true);
  GTestStatVarT(rotDistanceNoise) =
      CalculateDistanceFromExpectedRotationMatrix(common::CalculateRotationFromSVD(expectedWorldFromCamera_.matrix()),
                                                  common::CalculateRotationFromSVD(actualWorldFromCamera_.matrix()));
  EXPECT_LT(float(rotDistanceNoise), errorThresholdWithNoise_);
  GTestStatVarT(tranDistanceNoise) =
      (expectedWorldFromCamera_.translation() - actualWorldFromCamera_.translation()).norm() / distFactor_;
  EXPECT_LT(float(tranDistanceNoise), errorThresholdTranslationWithNoise_);

  // test new API
  //    camera::Rig rig;
  //    rig.num_cameras = 1;
  //    rig.camera_from_rig[0].setIdentity();
  //    RunPoseRefinement(rig);
}

}  // namespace test::epipolar
