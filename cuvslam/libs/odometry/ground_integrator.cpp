
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

#include "ground_integrator.h"

#include "common/isometry.h"
#include "common/vector_3t.h"

namespace {
using namespace cuvslam;

// Ground frame uses world axes; rig poses are OpenCV (+Z forward, +Y down). Second column is aligned to world +Y.
void remove_roll_and_pitch(Isometry3T& pose_G) {
  const Vector3T old_axisX{pose_G.linear() * Vector3T::UnitX()};
  const Vector3T axisY{Vector3T::UnitY()};
  const Vector3T axisZ{old_axisX.cross(axisY)};
  const Vector3T axisX{axisY.cross(axisZ)};

  pose_G.linear().col(0) = axisX.normalized();
  pose_G.linear().col(1) = axisY.normalized();
  pose_G.linear().col(2) = axisZ.normalized();
}

Isometry3T ground_align(const Isometry3T& pose_on_ground_unaligned_G) {
  Isometry3T pose_on_ground_G{pose_on_ground_unaligned_G};
  pose_on_ground_G.translation().y() = 0;
  remove_roll_and_pitch(pose_on_ground_G);
  return pose_on_ground_G;
}

}  // namespace

namespace cuvslam::odom {

GroundIntegrator::GroundIntegrator(const Isometry3T& world_from_ground, const Isometry3T& initial_pose_on_ground_W,
                                   const Isometry3T& initial_pose_in_space_W)
    : W_from_G_{world_from_ground},
      G_from_W_{W_from_G_.inverse()},
      pose_on_ground_G_{ground_align(G_from_W_ * initial_pose_on_ground_W)},
      pose_in_space_W_{initial_pose_in_space_W} {}

void GroundIntegrator::AddNextPose(const Isometry3T& next_pose_in_space_W) {
  const Isometry3T W_from_A{pose_in_space_W_};  // temporary pin A to W
  const Isometry3T A_from_W{W_from_A.inverse()};
  const Isometry3T next_pose_in_space_A{A_from_W * next_pose_in_space_W};
  const Isometry3T G_from_A{pose_on_ground_G_};  // temporary pin A to G (unpin from W)
  const Isometry3T next_pose_on_ground_unaligned_G{G_from_A * next_pose_in_space_A};
  const Isometry3T next_pose_on_ground_G{ground_align(next_pose_on_ground_unaligned_G)};

  pose_on_ground_G_ = next_pose_on_ground_G;
  pose_in_space_W_ = next_pose_in_space_W;
}

Isometry3T GroundIntegrator::GetPoseOnGround() const {
  const Isometry3T pose_on_plane_W_{W_from_G_ * pose_on_ground_G_};
  return pose_on_plane_W_;
}

}  // namespace cuvslam::odom
