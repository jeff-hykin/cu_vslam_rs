
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

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "sba/bundle_adjustment_problem.h"
#include "sba/schur_complement_bundler_cpu.h"

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_sba_v1.h"

namespace cuvslam::cuda::sba {

namespace temporary {

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

struct ParameterUpdate {
  std::vector<Isometry3T> pose;
  std::vector<Vector3T> point;

  Eigen::VectorXf pose_step;
  Eigen::VectorXf point_step;
};

}  // namespace temporary

class GPUModelFunction {
public:
  explicit GPUModelFunction(int max_observations);
  const GPUModelFunctionMeta& meta() const;
  bool get(int num_observations, cuvslam::sba::schur_complement_bundler_cpu_internal::ModelFunction& model_function,
           cudaStream_t s) const;

private:
  GPUModelFunctionMeta meta_;

  int max_observations_;
  GPUArrayPinned<Matf23> point_jacobians_;
  GPUArrayPinned<Matf26> pose_jacobians_;
  GPUArrayPinned<float2> residuals_;
  GPUArrayPinned<float> robustifier_weights_;
};

class GPULinearSystem {
public:
  GPULinearSystem(int max_points, int max_poses);
  const GPULinearSystemMeta& meta() const;
  bool set(const temporary::FullSystem& cpu_system, cudaStream_t s);
  bool set(const temporary::ReducedSystem& cpu_system, cudaStream_t s);
  bool get(int num_points, int num_poses, temporary::FullSystem& cpu_system, cudaStream_t s) const;
  bool get(int num_points, int num_poses, temporary::ReducedSystem& cpu_system, cudaStream_t s) const;
  int max_points() const;
  int max_poses() const;

private:
  const int max_points_;
  const int max_poses_;
  GPULinearSystemMeta meta_;
  GPUArrayPinned<Matf33> point_block_;  // inverse_point_block in reduced
  GPUArrayPinned<float> point_rhs_;

  GPUImageT point_pose_block_transposed_;  // camera_backsub_block in reduced

  GPUImageT pose_block_;
  GPUArrayPinned<float> pose_rhs_;
};

class GPUParameterUpdate {
public:
  GPUParameterUpdate(int max_points, int max_poses);
  const GPUParameterUpdateMeta& meta() const;
  bool get(int num_points, int num_poses, temporary::ParameterUpdate& cpu_update, cudaStream_t s) const;
  bool set(const temporary::ParameterUpdate& cpu_update, cudaStream_t s);

private:
  GPUParameterUpdateMeta meta_;

  int max_points_;
  int max_poses_;

  GPUArrayPinned<Pose> poses_;
  GPUArrayPinned<float3> points_;
  GPUArrayPinned<float6> pose_steps_;
  GPUArrayPinned<float3> point_steps_;
};

class GPUBundleAdjustmentProblem {
public:
  GPUBundleAdjustmentProblem(int max_points = 400, int max_poses = 20, int max_observations = 20000);
  ~GPUBundleAdjustmentProblem();
  void set_rig(const camera::Rig& rig);
  const GPUBundleAdjustmentProblemMeta& meta() const;

  bool set(const cuvslam::sba::BundleAdjustmentProblem& problem, cudaStream_t s);
  bool get(cuvslam::sba::BundleAdjustmentProblem& problem, cudaStream_t s) const;

  int num_points() const;
  int num_poses() const;
  int num_observations() const;

  // GPUBundleAdjustmentProblem rearanges observations to group them by
  // points. Observations are stored as
  //
  // [point0_obs0, point0_obs1, point1_obs0, point1_obs1, point1_obs2, ...]
  //
  // This method is to map an GPUBundleAdjustmentProblem observation index to
  // original observation index.
  int original_observation_index(int j) const;

private:
  void prepare_meta();

  int num_points_ = 0;
  int num_poses_ = 0;
  int num_observations_ = 0;

  GPUArrayPinned<Point> points_;
  GPUArrayPinned<Observation> observations_;
  GPUArrayPinned<Pose> poses_;

  // GPUBundleAdjustmentProblem rearanges observations to group them by
  // points. Observations are stored as
  //
  // [point0_obs0, point0_obs1, point1_obs0, point1_obs1, point1_obs2, ...]
  //
  // This array is to map an GPUBundleAdjustmentProblem observation index to
  // original observation index.
  std::vector<int> cpu_original_observation_indices_;

  Rig* rig_ = nullptr;

  mutable GPUBundleAdjustmentProblemMeta meta_;
};

class GPUSolver {
public:
  explicit GPUSolver(int max_system_order);
  ~GPUSolver();

  void solve(float* A, size_t A_pitch, float* b, float* x, int system_order, cudaStream_t s);

private:
  int max_system_order_;

  cusolverDnHandle_t handle = nullptr;
  cublasHandle_t cublasHandle = nullptr;

  int bufferSize = 100;
  GPUArrayPinned<int> info{1};
  GPUArrayPinned<float> buffer;
  GPUArrayPinned<float> Acopy;
};

class GPUParameterUpdater {
public:
  explicit GPUParameterUpdater(int max_points, int max_poses);

  // TODO: maybe unite the following methods in one?
  bool compute_update(const GPUParameterUpdateMeta& update, const GPULinearSystemMeta& reduced_system, int num_points,
                      int num_poses,
                      GPUArrayPinned<float>& points_poses_update_max,  // must contain 2 floats
                      cudaStream_t s);

  static bool update_state(const GPUBundleAdjustmentProblemMeta& problem, const GPUParameterUpdateMeta& update,
                           int num_points, int num_poses, cudaStream_t s);

  bool relative_reduction(float current_cost, float lambda, const GPUParameterUpdateMeta& update,
                          const GPULinearSystemMeta& full_system, int num_points, int num_poses, float* prediction,
                          cudaStream_t s);

private:
  int max_points_;
  int max_poses_;

  GPUSolver solver_;

  GPUArrayPinned<float> pose_hessian_term_{1};
  // point_hessian_term_, point_scaling_term_, pose_scaling_term_
  GPUArrayPinned<float> buffer_{3};
};

}  // namespace cuvslam::cuda::sba
