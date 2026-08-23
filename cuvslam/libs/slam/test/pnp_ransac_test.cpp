
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

#include "camera/observation.h"
#include "common/include_gtest.h"
#include "common/isometry.h"
#include "math/ransac.h"
#include "slam/slam/loop_closure_solver/pnp_ransac_hypothesis.h"

namespace test::slam {

TEST(SlamTest, PnpRansac) {
  std::vector<cuvslam::slam::PnpRansacTrackData> sampleSequence;

  const cuvslam::Isometry3T initial_rig_from_world = cuvslam::Isometry3T::Identity();
  const cuvslam::Vector2T resolution = {640, 480};
  const cuvslam::Vector2T focal = {300, 300};
  const cuvslam::Vector2T principal = {300, 300};
  const cuvslam::camera::PinholeCameraModel camera(resolution, focal, principal);

  const cuvslam::Matrix2T info =
      cuvslam::camera::ObservationInfoUVToNormUV(camera, cuvslam::camera::GetDefaultObservationInfoUV());
  cuvslam::Vector2T uv_norm(0, 0);
  cuvslam::Vector3T xyz(0, 0, 0);
  for (size_t i = 0; i < 100; i++) {
    cuvslam::slam::PnpRansacTrackData data(xyz, {0, i, uv_norm, info});
    sampleSequence.push_back(data);
  }

  cuvslam::camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0].setIdentity();
  cuvslam::math::Ransac<cuvslam::slam::PnpRansacHypothesis> pnpr(rig);
  pnpr.setInitialRigFromWorld(initial_rig_from_world);
  pnpr.setThreshold(0.05f);  // Threshold

  cuvslam::Isometry3T pose;
  if (!pnpr(pose, sampleSequence.begin(), sampleSequence.end())) return;
}

}  // namespace test::slam
