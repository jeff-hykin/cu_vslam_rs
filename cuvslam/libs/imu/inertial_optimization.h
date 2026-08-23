
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

#include "common/unaligned_types.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "camera/observation.h"
#include "imu/imu_sba_problem.h"

namespace cuvslam::sba_imu {

class InertialOptimizer {
public:
  explicit InertialOptimizer(float acc_prior = 1e2) : acc_prior_info(Matrix3T::Identity() * acc_prior) {
    JG << 0, GRAVITY_VALUE, 0, 0, -GRAVITY_VALUE, 0;
  }
  bool optimize_inertial(std::vector<Pose>& poses, const imu::ImuCalibration& calib, Matrix3T& Rgravity,
                         float robustifier = 1e1);

  // Adaptive gyro bias estimation: uses Linear method (rotation residuals) when visual motion
  // is sufficient, falls back to NEC (bearing vector correspondences) for few KFs / static / pure rotation.
  // The rest of the pipeline (LinearAlignment, RefineGravity, LM) remains the same.
  bool OptimizeInertialAdaptive(std::vector<Pose>& poses, const imu::ImuCalibration& calib, Matrix3T& Rgravity,
                                float robustifier,
                                const std::vector<std::vector<camera::Observation>>& observations_per_kf,
                                const Isometry3T& camera_from_rig);

  // Jointly solves for per-frame velocities and 3D gravity via a linear system.
  // Poses must have gyro_bias set (call SolveGyroBias first). acc_bias assumed zero.
  // Returns false if the solved gravity magnitude deviates too far from GRAVITY_VALUE.
  bool LinearAlignment(std::vector<Pose>& poses, const Vector3T& gyro_bias, Matrix3T& Rgravity, Vector3T& acc_bias);

  // Refines gravity direction, velocities, and acc_bias iteratively in the 2D tangent space of gravity.
  // Runs 4 linear solves, each time re-parameterizing the perturbation around the current estimate.
  void RefineGravity(std::vector<Pose>& poses, const Vector3T& gyro_bias, Matrix3T& Rgravity, Vector3T& acc_bias);

  [[nodiscard]] const Vector3T& get_default_gravity() const { return default_gravity; }

private:
  float calc_cost_with_update(const std::vector<Pose>& poses, const Matrix3T& Rgravity, const Vector3T& gyro_bias,
                              const Vector3T& acc_bias, const Eigen::VectorXf& updates, float robustifier) const;

  void build_hessian(const std::vector<Pose>& poses, const Matrix3T& Rgravity, const Vector3T& gyro_bias,
                     const Vector3T& acc_bias, float robustifier, Eigen::MatrixXf& hessian, Eigen::VectorXf& rhs) const;

  // Solve for gyro bias using only the rotation residual from pure visual frames
  // (VINS-Fusion solveGyroscopeBias approach). Call before joint optimization.
  static Vector3T SolveGyroBias(std::vector<Pose>& poses);

  // Estimate initial gravity direction from average accelerometer readings
  // (VINS-Fusion initFirstIMUPose approach). Returns Rgravity initial guess.
  static Matrix3T SolveGravityDirection(const std::vector<Pose>& poses, const Vector3T& default_gravity);

  // Returns orthonormal basis (3x2) spanning the plane perpendicular to g.
  static Eigen::Matrix<float, 3, 2> TangentBasis(const Vector3T& g);

  /// World gravity acceleration (m/s²), OpenCV-style world (+Y down) when aligned with an upright camera rig.
  const Vector3T default_gravity = {0, GRAVITY_VALUE, 0};
  Eigen::Matrix<float, 3, 2> JG;
  const Matrix3T acc_prior_info;

  profiler::GravityProfiler::DomainHelper profiler_domain_ = profiler::GravityProfiler::DomainHelper("Gravity");
};

}  // namespace cuvslam::sba_imu
