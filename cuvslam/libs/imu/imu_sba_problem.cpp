
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

#include "imu/imu_sba_problem.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace cuvslam::sba_imu {

bool Pose::predict_pose(const Vector3T& gravity, const IMUPreintegration& preint, Pose& pose) const {
  const float dt = preint.dT_s;

  const Matrix3T& R1 = w_from_imu.linear();
  const Vector3T& t1 = w_from_imu.translation();
  const Matrix3T dR = preint.GetDeltaRotation(gyro_bias);
  const Vector3T dV = preint.GetDeltaVelocity(gyro_bias, acc_bias);
  const Vector3T dP = preint.GetDeltaPosition(gyro_bias, acc_bias);

  Isometry3T p2;
  Eigen::JacobiSVD<Matrix3T> svd(R1 * dR, Eigen::ComputeFullU | Eigen::ComputeFullV);
  p2.linear() = svd.matrixU() * svd.matrixV().transpose();
  p2.translation() = t1 + velocity * dt + 0.5f * gravity * dt * dt + R1 * dP;
  p2.makeAffine();

  Vector3T v2 = velocity + gravity * dt + R1 * dV;

  pose = {p2, v2, gyro_bias, acc_bias, IMUPreintegration(gyro_bias, acc_bias), info};
  return true;
}

bool Pose::predict_pose(const Vector3T& gravity, Pose& pose) const {
  return predict_pose(gravity, preintegration, pose);
}

void ImuBAProblem::Reset() {
  points.clear();
  observation_xys.clear();
  observation_infos.clear();
  info_matrix.clear();
  point_ids.clear();
  pose_ids.clear();
  camera_ids.clear();
  rig_poses.clear();
  num_fixed_key_frames = 0;
  gravity.setZero();
  max_iterations = 0;
  robustifier_scale = 0.5;
  iterations = 0;
  initial_cost = 0;
  reintegration_thresh = 0;
}

}  // namespace cuvslam::sba_imu
