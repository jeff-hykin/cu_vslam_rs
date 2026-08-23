
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

#include "sba/bundle_adjustment_problem.h"

namespace cuvslam::sba {

namespace schur_complement_bundler_cpu_internal {

struct ModelFunction {
  using PoseBlock = Eigen::Matrix<float, 2, 6, Eigen::DontAlign>;
  using PointBlock = Eigen::Matrix<float, 2, 3, Eigen::DontAlign>;

  // one block per observation
  std::vector<PoseBlock> pose_jacobians;
  std::vector<PointBlock> point_jacobians;
  std::vector<Vector2T> residuals;
  std::vector<float> robustifier_weights;

  float lambda;
};

struct ParameterUpdate {
  std::vector<Isometry3T> pose;
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

void UpdateModel(ModelFunction& model, BundleAdjustmentProblem& problem);

void BuildFullSystem(FullSystem& system, const ModelFunction& model, const BundleAdjustmentProblem& problem);

void BuildReducedSystem(ReducedSystem& reduced_system, const FullSystem& full_system, const float lambda);

float EvaluateCost(const BundleAdjustmentProblem& problem, const ParameterUpdate& update);

void ComputeUpdate(ParameterUpdate& update, const ReducedSystem& reduced_system, const FullSystem& full_system);

void UpdateState(BundleAdjustmentProblem& problem, const ParameterUpdate& update);

float ComputePredictedRelativeReduction(const float current_cost, const float lambda, const ParameterUpdate& update,
                                        const FullSystem& system);
}  // namespace schur_complement_bundler_cpu_internal

// This is a more or less classic version of bundle adjustment.
// Each step of Levenberg-Marquardt algorithm solves a sparse linear
// system using Schur complement trick.
class SchurComplementBundlerCpu {
public:
  bool solve(BundleAdjustmentProblem& problem);

private:
  float EvaluateCost_(const BundleAdjustmentProblem& problem,
                      const schur_complement_bundler_cpu_internal::ParameterUpdate& update);
  void ComputeUpdate_(schur_complement_bundler_cpu_internal::ParameterUpdate& update,
                      const schur_complement_bundler_cpu_internal::ReducedSystem& reduced_system,
                      const schur_complement_bundler_cpu_internal::FullSystem& full_system);
  void UpdateState_(BundleAdjustmentProblem& problem,
                    const schur_complement_bundler_cpu_internal::ParameterUpdate& update);
  float ComputePredictedRelativeReduction_(const float current_cost, const float lambda,
                                           const schur_complement_bundler_cpu_internal::ParameterUpdate& update,
                                           const schur_complement_bundler_cpu_internal::FullSystem& system);
  void UpdateModel_(schur_complement_bundler_cpu_internal::ModelFunction& model, BundleAdjustmentProblem& problem);
  void BuildFullSystem_(schur_complement_bundler_cpu_internal::FullSystem& system,
                        const schur_complement_bundler_cpu_internal::ModelFunction& model,
                        const BundleAdjustmentProblem& problem);
  void BuildReducedSystem_(schur_complement_bundler_cpu_internal::ReducedSystem& reduced_system,
                           const schur_complement_bundler_cpu_internal::FullSystem& full_system, const float lambda);

  cuvslam::profiler::SBAProfiler::DomainHelper profiler_domain_ =
      cuvslam::profiler::SBAProfiler::DomainHelper("SBA CPU");
  const uint32_t profiler_color_ = 0xFF7700;
};

}  // namespace cuvslam::sba
