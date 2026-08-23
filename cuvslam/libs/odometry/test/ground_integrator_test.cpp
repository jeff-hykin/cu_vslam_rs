
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

#include "common/vector_3t.h"
#include "odometry/ground_integrator.h"

namespace Test {
using namespace cuvslam;

TEST(IndentityStill, GroundIntegrator) {
  const Isometry3T world_to_ground{Isometry3T::Identity()};
  const Isometry3T initial_pose_on_plane{Isometry3T::Identity()};
  const Isometry3T initial_pose_in_space{Isometry3T::Identity()};
  odom::GroundIntegrator integrator{world_to_ground, initial_pose_on_plane, initial_pose_in_space};

  integrator.AddNextPose(Isometry3T::Identity());
  const Isometry3T pose_on_plane{integrator.GetPoseOnGround()};
  ASSERT_TRUE(initial_pose_on_plane.isApprox(pose_on_plane));
}

TEST(MovingOnGround, PostProcessing) {
  using AngleAxisT = Eigen::AngleAxis<float>;

  const Isometry3T world_to_ground{Isometry3T::Identity()};
  const Isometry3T initial_pose_in_space{Isometry3T::Identity()};
  Isometry3T initial_pose_on_plane{Isometry3T::Identity()};
  initial_pose_on_plane.translate(Vector3T{123, 0, 321});
  initial_pose_on_plane.rotate(AngleAxisT{EIGEN_PI / 10, Vector3T::UnitY()});

  odom::GroundIntegrator integrator{world_to_ground, initial_pose_on_plane, initial_pose_in_space};

  integrator.AddNextPose(Isometry3T::Identity());
  const Isometry3T pose_on_plane{integrator.GetPoseOnGround()};
  ASSERT_TRUE(pose_on_plane.translation().y() == 0);
  ASSERT_TRUE(pose_on_plane.linear().eulerAngles(0, 2, 1).head<2>().isZero());
}
}  // namespace Test
