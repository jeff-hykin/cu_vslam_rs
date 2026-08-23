
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

#include "slam/slam/slam.h"
#include "gtest/gtest.h"

namespace test::slam {
using namespace cuvslam;
using namespace cuvslam::slam;

TEST(Slam, SlamTest) {
  const Vector2T resolution = {640, 480};
  const Vector2T focal = {320, 240};
  const Vector2T principal = {320, 240};
  const auto camera = camera::CreateCameraModel(resolution, focal, principal, Distortion::Model::Pinhole, nullptr, 0);
  camera::Rig rig;
  rig.intrinsics[0] = camera.get();
  rig.num_cameras = 1;
  rig.camera_from_rig[0].setIdentity();

  LocalizerAndMapper mapper(rig, FeatureDescriptorType::kNone, true);
  mapper.SetActiveCameras({0});

  Isometry3T step_right = Isometry3T::Identity();
  step_right.translation().x() = 1.f;

  VOFrameData frame_data;
  constexpr float fps = 30;
  constexpr int64_t frame_delta_ns = static_cast<int64_t>(1e9 / fps);

  // test 1: make 10 steps right and verify the pose shifted to 10 meters.
  constexpr int kTestSteps = 10;
  for (int i = 0; i < kTestSteps; ++i) {
    frame_data.frame_id = i;
    frame_data.timestamp_ns = i * frame_delta_ns;

    mapper.AddKeyframe(step_right, frame_data, Images());
  }
  EXPECT_NEAR(mapper.GetCurrentPose().translation().x(), kTestSteps, 1e-4f);

  // test 2: run optimize and verify nothing happened
  ASSERT_TRUE(mapper.SetPoseGraphOptimizerOptions({PoseGraphOptimizerType::Simple}));
  ASSERT_TRUE(mapper.OptimizePoseGraph(false));
  EXPECT_NEAR(mapper.GetCurrentPose().translation().x(), kTestSteps, 1e-4f);

  // test 3: try to add LC and verify it result false
  LocalizerAndMapper::LoopClosureStatus status;

  status.success = true;
  status.result_pose.translation() = Vector3T(kTestSteps, 1.f, 0.f);
  // can't find keyframe with landmarks
  ASSERT_FALSE(mapper.ApplyLoopClosureResult(status.result_pose, status.result_pose_covariance, status.landmarks));
}
}  // namespace test::slam
