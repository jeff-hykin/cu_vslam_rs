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

#include <vector>

#include "common/image_matrix.h"
#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/image_pyramid.h"

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"

#include "pipelines/track_lifter.h"
#include "pnp/multisensor_pose_estimator.h"

namespace test::pipelines {

using namespace cuvslam;

namespace {

const Vector2T kResolution{640.f, 480.f};
const Vector2T kFocal{320.f, 320.f};
const Vector2T kPrincipal{320.f, 240.f};

camera::PinholeCameraModel MakePinhole() { return camera::PinholeCameraModel(kResolution, kFocal, kPrincipal); }

camera::Rig MakeRig(const camera::ICameraModel& cam) {
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  rig.intrinsics[0] = &cam;
  return rig;
}

// Build a single-level-equivalent constant-depth pyramid the lifter can read from.
struct ConstantDepthPyramid {
  cuvslam::cuda::GaussianGPUImagePyramid pyramid;
  cuvslam::cuda::Stream stream;

  ConstantDepthPyramid(int width, int height, float depth) : pyramid(width, height) {
    cuvslam::ImageMatrixT depth_img(height, width);
    depth_img.setConstant(depth);

    cuvslam::cuda::GPUImageT device_img(depth_img);
    cudaStream_t s = stream.get_stream();
    pyramid.build(device_img, false, s);
    cudaStreamSynchronize(s);
  }
};

camera::Observation make_observation(TrackId id, const Vector2T& pixel) {
  Vector2T xy_norm((pixel.x() - kPrincipal.x()) / kFocal.x(), (pixel.y() - kPrincipal.y()) / kFocal.y());
  return camera::Observation(CameraId{0}, id, xy_norm, Matrix2T::Identity());
}

}  // namespace

// Lift observations through a constant-depth texture and verify the world-frame landmarks match
// the analytical reprojection.
TEST(TrackLifterTest, ConstantDepthRoundTrip) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  constexpr float kDepth = 2.0f;
  ConstantDepthPyramid depth(kWidth, kHeight, kDepth);

  cuvslam::pnp::RGBDInfo info{CameraId{0}, depth.pyramid};

  // Observations at well-interior pixels so the lift kernel's 3×3 neighborhood is fully valid.
  std::vector<camera::Observation> observations;
  observations.push_back(make_observation(0, {kPrincipal.x(), kPrincipal.y()}));
  observations.push_back(make_observation(1, {kPrincipal.x() + 50.f, kPrincipal.y() + 30.f}));
  observations.push_back(make_observation(2, {kPrincipal.x() - 80.f, kPrincipal.y() + 60.f}));

  cuvslam::pipelines::TrackLifter lifter(rig);

  std::vector<cuvslam::pipelines::Landmark> landmarks;
  lifter.lift_tracks(info, Isometry3T::Identity(), observations, landmarks);

  ASSERT_EQ(landmarks.size(), observations.size());

  for (size_t i = 0; i < observations.size(); i++) {
    const auto& obs = observations[i];
    const auto& lm = landmarks[i];
    EXPECT_EQ(lm.id, obs.id);
    EXPECT_NEAR(lm.point_w.x(), obs.xy.x() * kDepth, 1e-3f);
    EXPECT_NEAR(lm.point_w.y(), obs.xy.y() * kDepth, 1e-3f);
    EXPECT_NEAR(lm.point_w.z(), kDepth, 1e-3f);
  }
}

// world_from_rig translation must propagate into the emitted world-frame landmarks.
TEST(TrackLifterTest, WorldPoseTranslation) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  constexpr float kDepth = 1.5f;
  ConstantDepthPyramid depth(kWidth, kHeight, kDepth);

  cuvslam::pnp::RGBDInfo info{CameraId{0}, depth.pyramid};

  std::vector<camera::Observation> observations;
  observations.push_back(make_observation(7, {kPrincipal.x(), kPrincipal.y()}));

  Isometry3T world_from_rig = Isometry3T::Identity();
  const Vector3T translation(1.f, -2.f, 3.f);
  world_from_rig.translation() = translation;

  cuvslam::pipelines::TrackLifter lifter(rig);

  std::vector<cuvslam::pipelines::Landmark> landmarks;
  lifter.lift_tracks(info, world_from_rig, observations, landmarks);

  ASSERT_EQ(landmarks.size(), 1u);
  const auto& lm = landmarks[0];
  EXPECT_EQ(lm.id, TrackId{7});

  EXPECT_NEAR(lm.point_w.x(), translation.x(), 1e-3f);
  EXPECT_NEAR(lm.point_w.y(), translation.y(), 1e-3f);
  EXPECT_NEAR(lm.point_w.z(), translation.z() + kDepth, 1e-3f);
}

// Empty input must yield an empty output without crashing.
TEST(TrackLifterTest, EmptyObservations) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  ConstantDepthPyramid depth(640, 480, 1.0f);
  cuvslam::pnp::RGBDInfo info{CameraId{0}, depth.pyramid};

  cuvslam::pipelines::TrackLifter lifter(rig);

  std::vector<camera::Observation> observations;
  std::vector<cuvslam::pipelines::Landmark> landmarks;
  landmarks.push_back({TrackId{0}, Vector3T::Zero()});

  lifter.lift_tracks(info, Isometry3T::Identity(), observations, landmarks);
  EXPECT_TRUE(landmarks.empty());
}

}  // namespace test::pipelines
