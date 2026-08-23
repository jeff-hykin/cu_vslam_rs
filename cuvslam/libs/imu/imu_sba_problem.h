
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

#include <vector>

#include "camera/rig.h"

#include "imu/imu_preintegration.h"

namespace cuvslam::sba_imu {

// use IMU frame to simplify jacobians
// world frame should be bound to the Keyframe where gravity was obtained (OpenCV: +Y down; gravity ~ +Y when upright)
struct Pose {
  Isometry3T w_from_imu = Isometry3T::Identity();
  Vector3T velocity = Vector3T::Zero();
  Vector3T gyro_bias = Vector3T::Zero();
  Vector3T acc_bias = Vector3T::Zero();
  //!!!! contains imu measurements since this keyframe until next keyframe !!!!
  // preintegration should be stored in the keyframe
  IMUPreintegration preintegration;

  Matrix15T info = Matrix15T::Zero();

  bool predict_pose(const Vector3T& gravity, const IMUPreintegration& preint, Pose& pose) const;
  bool predict_pose(const Vector3T& gravity, Pose& pose) const;
};

// sorting:
// - all observations are ordered by key frame
// - within each key frame observations are ordered by point id
struct ImuBAProblem {
  // last num_fixed_points points are not allowed to move
  std::vector<Vector3T> points = {};

  std::vector<Vector2T> observation_xys = {};
  std::vector<Matrix2T> observation_infos = {};

  // analog of observation_infos but indexed by point_ids
  std::vector<Matrix2T> info_matrix = {};

  // index into points
  std::vector<int32_t> point_ids = {};
  // index into key_frame_poses
  std::vector<int32_t> pose_ids = {};

  // index into rig.camera_from_rig
  std::vector<int8_t> camera_ids = {};

  // Key frame pose or simply pose is a transformation
  // from world/map space to rig space.
  // Last num_fixed_key_frames are not allowed to move.
  std::vector<Pose> rig_poses = {};
  int32_t num_fixed_key_frames = 0;
  camera::Rig rig = {};
  Vector3T gravity = Vector3T::Zero();  ///< World-frame gravity acceleration (same convention as inertial_optimization)

  // solver options
  int32_t max_iterations = 7;
  float robustifier_scale = 1.f;
  float robustifier_scale_pose = 10.f;

  float prior_gyro = 0;
  float prior_acc = 0;
  float imu_penalty = 1;
  float acc_rw_penalty = -1;           // penalty for acc bias random walk; -1 means use imu_penalty
  float boundary_imu_penalty = 1e-2f;  // downweight IMU edge at fixed/optimizable boundary

  // solver statistic
  int32_t iterations = 0;
  float initial_cost = 0;
  float reintegration_thresh = 1e-4f;

  void Reset();
};

}  // namespace cuvslam::sba_imu
