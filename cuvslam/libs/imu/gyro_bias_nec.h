
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

#include "camera/observation.h"
#include "common/unaligned_types.h"
#include "imu/imu_sba_problem.h"

namespace cuvslam::sba_imu {

// Solve gyro bias using the NEC (Normalized Epipolar Constraint) method.
// Internally: matches features by TrackId -> builds bearing vectors -> computes summation matrices ->
// Gauss-Newton optimization (no external solver dependency).
// poses: consecutive keyframe poses with preintegration data
// observations_per_kf: per-keyframe observation lists (bearing vectors from SOF tracker)
// Rbc: rotation from camera to IMU/body frame (3x3, double)
Vector3T SolveGyroBiasNec(const std::vector<Pose>& poses,
                          const std::vector<std::vector<camera::Observation>>& observations_per_kf,
                          const Eigen::Matrix3d& Rbc);

// Match observations between two keyframes by TrackId (camera 0 only).
// Outputs normalized bearing vectors for matched pairs.
void MatchObservations(const std::vector<camera::Observation>& obs_i, const std::vector<camera::Observation>& obs_j,
                       std::vector<Eigen::Vector3d>& bearing_i, std::vector<Eigen::Vector3d>& bearing_j);

}  // namespace cuvslam::sba_imu
