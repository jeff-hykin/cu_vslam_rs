
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

#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "imu/imu_sba_problem.h"

namespace cuvslam::sba_imu {

struct StereoPnPInput {
  std::vector<Vector3T> points;
  std::vector<Vector2T> observation_xys;
  std::vector<Matrix2T> observation_infos;

  std::vector<int32_t> point_ids;
  std::vector<int8_t> camera_ids;
  camera::Rig rig;
  Vector3T gravity;
  // solver options
  int32_t max_iterations = 7;
  float robustifier_scale = 1.f;
  float robustifier_scale_pose = 5.f;

  float prior_gyro = 1e2f;
  float prior_acc = 1e10f;
  float prior_velocity = 1e10f;

  float translation_constraint = 1e5f;
  float robustifier_scale_tr = 1.f;
  float imu_penalty = 1.0;

  bool freeze_bias = false;  // If true, do not update gyro/acc bias in PnP (let SBA handle it)

  std::vector<float> outlier_thresh = {10.f, 5.f, 5.f, 5.f};
};

bool SoftInertialPnP(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem, Pose& prev_pose,
                     Pose& pose, float imu_info_penalty = 1e-3);

bool SoftInertialPnPWithOutliers(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem,
                                 Pose& prev_pose, Pose& pose, float imu_info_penalty = 1e-3);

}  // namespace cuvslam::sba_imu
