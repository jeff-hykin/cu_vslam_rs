
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

#include "common/imu_calibration.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "imu/imu_sba_problem.h"

namespace {
using ProfilerDomain = cuvslam::profiler::SBAProfiler::DomainHelper;
}

namespace cuvslam::sba_imu {
namespace internal {

using Mat93 = Eigen::Matrix<float, 9, 3>;
using Mat23 = Eigen::Matrix<float, 2, 3>;
using Mat32 = Eigen::Matrix<float, 3, 2>;
using Vec9 = Eigen::Matrix<float, 9, 1>;

struct InertialJacobians {
  Mat93 JR_left;
  Mat93 Jt_left;
  Mat93 Jv_left;
  Mat93 Jb_acc_left;
  Mat93 Jb_gyro_left;

  Mat93 JR_right;
  Mat93 Jt_right;
  Mat93 Jv_right;
};

struct ReprJacobians {
  Mat23 JR;
  Mat23 Jt;
  Mat23 Jpoint;
};

struct ModelFunction {
  // reprojection part
  std::vector<ReprJacobians> repr_jacobians;
  std::vector<float> repr_robustifier_weights;
  std::vector<InertialJacobians> inertial_jacobians;
  std::vector<Vector2T> reprojection_residuals;
  std::vector<Vector3T> random_walk_gyro_residuals;
  std::vector<Vector3T> random_walk_acc_residuals;
  std::vector<Vec9> inertial_residuals;

  float lambda;
};

struct ParameterUpdate {
  std::vector<Pose> pose;
  std::vector<Vector3T> point;
  Eigen::VectorXf pose_step;
  Eigen::VectorXf point_step;
};

struct FullSystem {
  Eigen::MatrixXf pose_block;
  Eigen::VectorXf pose_rhs;
  std::vector<Matrix3T> point_block;
  Eigen::VectorXf point_rhs;
  Eigen::MatrixXf point_pose_block;
};

struct ReducedSystem {
  Eigen::MatrixXf pose_block;
  Eigen::VectorXf pose_rhs;
  Eigen::VectorXf point_rhs;
  std::vector<Matrix3T> inverse_point_block;
  Eigen::MatrixXf camera_backsub_block;
};

}  // namespace internal

// This is a more or less classic version of bundle adjustment.
// Each step of Levenberg-Marquardt algorithm solves a sparse linear
// system using Schur complement trick.
class IMUBundlerCpuFixedVel {
public:
  IMUBundlerCpuFixedVel(const imu::ImuCalibration& calib);
  bool solve(ImuBAProblem& problem);

private:
  imu::ImuCalibration calib_;

  float EvaluateCost(const ImuBAProblem& problem, const internal::ParameterUpdate& update);
  void ComputeUpdate(internal::ParameterUpdate& update, const internal::ReducedSystem& reduced_system,
                     const internal::FullSystem& full_system);
  void UpdateState(ImuBAProblem& problem, const internal::ParameterUpdate& update);
  float ComputePredictedRelativeReduction(float current_cost, float lambda, const internal::ParameterUpdate& update,
                                          const internal::FullSystem& system);
  void UpdateModel(internal::ModelFunction& model, ImuBAProblem& problem);
  void BuildFullSystem(internal::FullSystem& system, const internal::ModelFunction& model, const ImuBAProblem& problem);
  void BuildReducedSystem(internal::ReducedSystem& reduced_system, const internal::FullSystem& full_system,
                          const float lambda);

  ProfilerDomain profiler_domain_ = ProfilerDomain("Inertial SBA CPU");
  const uint32_t profiler_color_ = 0xFF7700;
};

}  // namespace cuvslam::sba_imu
