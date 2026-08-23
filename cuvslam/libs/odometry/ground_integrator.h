
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

#pragma once

#include "common/isometry.h"

namespace cuvslam::odom {

// It's not just a trajectory projection, but a try to have correct short-term odometry
// in 2d if the robot physically continues to move on the plane,
// but 3d odometry flyes in space because of drift and failure.
class GroundIntegrator {
public:
  // if initial_pose_on_ground is not belong to the ground it will project and rotated with the nearest angle
  GroundIntegrator(const Isometry3T& world_from_ground,       // in world frame
                   const Isometry3T& initial_pose_on_ground,  // in world frame
                   const Isometry3T& initial_pose_in_space);  // in world frame

  // Add new 3d position from Odometry to 2d integrator
  // next_pose_in_space - next to previous (or to initial) pose in 3d space in world frame
  // pose it expected have just small delta from previous one.
  // After projection roll and pitch angles will be removed.
  void AddNextPose(const Isometry3T& next_pose_in_space);

  // return current integrated 2d pose
  Isometry3T GetPoseOnGround() const;

private:
  // Suffix used in implementation:
  // _W - world
  // _G - ground
  // _A - agent - AKA camera space
  // _G / _A: OpenCV-style rig (+Z forward, +Y down); ground plane uses world Y.
  const Isometry3T W_from_G_;
  const Isometry3T G_from_W_;    // cache value
  Isometry3T pose_on_ground_G_;  // current state
  Isometry3T pose_in_space_W_;   // current state
};

}  // namespace cuvslam::odom
