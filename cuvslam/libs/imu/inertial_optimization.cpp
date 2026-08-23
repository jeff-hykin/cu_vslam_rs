
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

#include "math/robust_cost_function.h"

#include "common/log.h"
#include "common/rotation_utils.h"
#include "imu/gyro_bias_nec.h"
#include "imu/inertial_optimization.h"

namespace {

using Mat93 = Eigen::Matrix<float, 9, 3>;
using Mat92 = Eigen::Matrix<float, 9, 2>;

using Mat39 = Eigen::Matrix<float, 3, 9>;
using Mat29 = Eigen::Matrix<float, 2, 9>;

struct InertialJacobians {
  Mat92 J_gravity = Mat92::Zero();
  Mat93 J_ba = Mat93::Zero();
  Mat93 J_bg = Mat93::Zero();

  Mat93 J_v1 = Mat93::Zero();
  Mat93 J_v2 = Mat93::Zero();
};

}  // namespace

namespace cuvslam::sba_imu {

float InertialOptimizer::calc_cost_with_update(const std::vector<Pose>& poses, const Matrix3T& Rgravity,
                                               const Vector3T& gyro_bias, const Vector3T& acc_bias,
                                               const Eigen::VectorXf& updates, float robustifier) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("calc_cost_with_update");

  Matrix3T Rguess;
  Vector3T twist = {updates[0], 0, updates[1]};
  math::Exp(Rguess, twist);

  Vector3T gravity = Rgravity * Rguess * default_gravity;

  Vector3T gyro_guess = gyro_bias + updates.segment<3>(2);
  Vector3T acc_guess = acc_bias + updates.segment<3>(5);

  Vector3T v1, v2;

  float cost = 0;

  Vector9T err;
  Matrix9T info;
  for (size_t i = 0; i < poses.size() - 1; i++) {
    const Pose& pose_left = poses[i];
    const Pose& pose_right = poses[i + 1];

    const auto& preint = pose_left.preintegration;
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    const Matrix3T dR = preint.GetDeltaRotation(gyro_guess);
    const Vector3T dV = preint.GetDeltaVelocity(gyro_guess, acc_guess);
    const Vector3T dP = preint.GetDeltaPosition(gyro_guess, acc_guess);
    const float dT = preint.GetDeltaT_s();

    v1 = pose_left.velocity + updates.segment<3>(8 + 3 * i);
    v2 = pose_right.velocity + updates.segment<3>(11 + 3 * i);

    const Matrix3T R1T = w_from_imu1.linear().transpose();
    const Matrix3T& R2 = w_from_imu2.linear();

    Vector3T rot_error;
    math::Log(rot_error, dR.transpose() * R1T * R2);

    err.segment<3>(0) = rot_error;
    err.segment<3>(3) = R1T * (v2 - v1 - gravity * dT) - dV;
    err.segment<3>(6) =
        R1T * (w_from_imu2.translation() - w_from_imu1.translation() - v1 * dT - 0.5 * gravity * dT * dT) - dP;

    preint.InfoMatrix(info);

    float squared_err = err.dot(info * err);
    // TODO add robust cost
    cost += math::ComputeHuberLoss(squared_err, robustifier);
  }

  // priors
  cost += acc_guess.dot(acc_prior_info * acc_guess);

  return cost;
}

void InertialOptimizer::build_hessian(const std::vector<Pose>& poses, const Matrix3T& Rgravity,
                                      const Vector3T& gyro_bias, const Vector3T& acc_bias, float robustifier,
                                      Eigen::MatrixXf& hessian, Eigen::VectorXf& rhs) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("build_hessian");
  int num_poses = static_cast<int>(poses.size());
  int num_inertials = num_poses - 1;
  // 3 - for each velocity in each pose, 2 - for gravity, 3 - for gyro bias, 3 - for acc bias
  hessian.setZero();
  rhs.setZero();

  Vector3T rot_error, velocity_error, trans_error;
  Vector3T gravity = Rgravity * default_gravity;

  Vector9T inertial_residual = Vector9T::Zero();
  InertialJacobians inertial_jacobians;

  hessian.block<3, 3>(5, 5) += acc_prior_info;
  rhs.segment<3>(5) += acc_prior_info * acc_bias;

  Vector3T gyro_bias_diff, acc_bias_diff;

  Matrix2T h_rr;
  Eigen::Matrix<float, 2, 3> h_rbg, h_rba, h_rv1, h_rv2;

  Matrix3T h_bgbg, h_bgba, h_bgv1, h_bgv2, h_baba, h_bav1, h_bav2, h_v1v1, h_v1v2, h_v2v2;

  Matrix9T info;
  for (int i = 0; i < num_inertials; i++) {
    const Pose& pose_left = poses[i];
    const Pose& pose_right = poses[i + 1];
    const auto& preint = pose_left.preintegration;
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    const Matrix3T dR = preint.GetDeltaRotation(gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(gyro_bias, acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(gyro_bias, acc_bias);
    const float dT = preint.GetDeltaT_s();
    preint.InfoMatrix(info);

    const Matrix3T R1T = w_from_imu1.linear().transpose();
    const Matrix3T& R2 = w_from_imu2.linear();

    math::Log(rot_error, dR.transpose() * R1T * R2);

    inertial_residual.segment<3>(0) = rot_error;

    inertial_residual.segment<3>(3) = R1T * (pose_right.velocity - pose_left.velocity - gravity * dT) - dV;
    inertial_residual.segment<3>(6) = R1T * (w_from_imu2.translation() - w_from_imu1.translation() -
                                             pose_left.velocity * dT - 0.5 * gravity * dT * dT) -
                                      dP;

    {
      inertial_jacobians.J_gravity.setZero();
      // vel
      inertial_jacobians.J_gravity.block<3, 2>(3, 0) = R1T * Rgravity * JG * dT;
      // tr
      inertial_jacobians.J_gravity.block<3, 2>(6, 0) = 0.5 * R1T * Rgravity * JG * dT * dT;
    }

    {
      gyro_bias_diff = gyro_bias - preint.GetOriginalGyroBias();
      // rot
      inertial_jacobians.J_bg.block<3, 3>(0, 0) = -math::twist_left_inverse_jacobian(rot_error) *
                                                  math::twist_right_jacobian(preint.JRg * gyro_bias_diff) * preint.JRg;
      // vel
      inertial_jacobians.J_bg.block<3, 3>(3, 0) = -preint.JVg;
      // tr
      inertial_jacobians.J_bg.block<3, 3>(6, 0) = -preint.JPg;
    }

    {
      inertial_jacobians.J_ba.setZero();
      // vel
      inertial_jacobians.J_ba.block<3, 3>(3, 0) = -preint.JVa;
      // tr
      inertial_jacobians.J_ba.block<3, 3>(6, 0) = -preint.JPa;
    }
    {
      inertial_jacobians.J_v1.setZero();
      // vel
      inertial_jacobians.J_v1.block<3, 3>(3, 0) = -R1T;
      // tr
      inertial_jacobians.J_v1.block<3, 3>(6, 0) = -R1T * dT;
    }

    {
      inertial_jacobians.J_v2.setZero();
      // vel
      inertial_jacobians.J_v2.block<3, 3>(3, 0) = R1T;
    }

    float squared_err = inertial_residual.dot(info * inertial_residual);
    float w = math::ComputeDHuberLoss(squared_err, robustifier);
    info *= w;

    {
      Mat29 temp = inertial_jacobians.J_gravity.transpose() * info;
      h_rr = temp * inertial_jacobians.J_gravity;
      h_rbg = temp * inertial_jacobians.J_bg;
      h_rba = temp * inertial_jacobians.J_ba;
      h_rv1 = temp * inertial_jacobians.J_v1;
      h_rv2 = temp * inertial_jacobians.J_v2;

      rhs.segment<2>(0) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_bg.transpose() * info;
      h_bgbg = temp * inertial_jacobians.J_bg;
      h_bgba = temp * inertial_jacobians.J_ba;
      h_bgv1 = temp * inertial_jacobians.J_v1;
      h_bgv2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(2) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_ba.transpose() * info;
      h_baba = temp * inertial_jacobians.J_ba;
      h_bav1 = temp * inertial_jacobians.J_v1;
      h_bav2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(5) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_v1.transpose() * info;
      h_v1v1 = temp * inertial_jacobians.J_v1;
      h_v1v2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(8 + 3 * i) += temp * inertial_residual;
    }

    h_v2v2 = inertial_jacobians.J_v2.transpose() * info * inertial_jacobians.J_v2;
    rhs.segment<3>(11 + 3 * i) += inertial_jacobians.J_v2.transpose() * info * inertial_residual;

    hessian.block<2, 2>(0, 0) += h_rr;

    hessian.block<3, 2>(2, 0) += h_rbg.transpose();
    hessian.block<3, 3>(2, 2) += h_bgbg;

    hessian.block<3, 2>(5, 0) += h_rba.transpose();
    hessian.block<3, 3>(5, 2) += h_bgba.transpose();
    hessian.block<3, 3>(5, 5) += h_baba;

    hessian.block<3, 2>(8 + 3 * i, 0) += h_rv1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 2) += h_bgv1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 5) += h_bav1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 8 + 3 * i) += h_v1v1;

    hessian.block<3, 2>(11 + 3 * i, 0) += h_rv2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 2) += h_bgv2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 5) += h_bav2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 8 + 3 * i) += h_v1v2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 11 + 3 * i) += h_v2v2;
  }
}

Matrix3T InertialOptimizer::SolveGravityDirection(const std::vector<Pose>& poses, const Vector3T& default_gravity) {
  // Average all accelerometer measurements across all preintegrations, expressed
  // in world frame. When static: acc_world ≈ -gravity_world (reaction force).
  Vector3T avg_acc_world = Vector3T::Zero();
  int count = 0;
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    const Matrix3T& R_world = poses[i].w_from_imu.linear();
    for (const auto& m : poses[i].preintegration.measurements()) {
      avg_acc_world += R_world * m.linear_acceleration;
      count++;
    }
  }
  if (count == 0) {
    return Matrix3T::Identity();
  }
  avg_acc_world /= count;

  // Gravity direction in world frame: opposite of measured acc
  Vector3T gravity_dir = -avg_acc_world;

  // Rgravity: minimal rotation from default_gravity direction to estimated gravity direction.
  // FromTwoVectors naturally produces zero rotation around the gravity axis (no yaw bias),
  // matching the VINS-Fusion yaw-zeroing step.
  Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(default_gravity.normalized(), gravity_dir.normalized());
  return q.toRotationMatrix();
}

Vector3T InertialOptimizer::SolveGyroBias(std::vector<Pose>& poses) {
  Matrix3T A = Matrix3T::Zero();
  Vector3T b = Vector3T::Zero();

  for (size_t i = 0; i + 1 < poses.size(); i++) {
    const Pose& pose_left = poses[i];
    const Pose& pose_right = poses[i + 1];
    const auto& preint = pose_left.preintegration;

    // Rotation from visual: R1^T * R2
    const Matrix3T R1T = pose_left.w_from_imu.linear().transpose();
    const Matrix3T& R2 = pose_right.w_from_imu.linear();

    // Rotation residual: Log(dR(0)^T * R1^T * R2) at zero gyro bias.
    // GetDeltaRotation(0) extrapolates to zero bias via the Jacobian, giving
    // the correct residual even when the preintegration was done at non-zero bias.
    Vector3T rot_error;
    math::Log(rot_error, preint.GetDeltaRotation(Vector3T::Zero()).transpose() * R1T * R2);

    // Linear system: JRg * delta_bg = rot_error
    A += preint.JRg.transpose() * preint.JRg;
    b += preint.JRg.transpose() * rot_error;
  }

  // Solve A * delta_bg = b
  Vector3T gyro_bias = A.ldlt().solve(b);

  // Update preintegrations with the estimated gyro bias
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    poses[i].gyro_bias = gyro_bias;
    poses[i].preintegration.SetNewBias(gyro_bias, Vector3T::Zero());
  }
  if (!poses.empty()) {
    poses.back().gyro_bias = gyro_bias;
  }

  return gyro_bias;
}

Eigen::Matrix<float, 3, 2> InertialOptimizer::TangentBasis(const Vector3T& g) {
  Vector3T a = g.normalized();
  Vector3T tmp(0.f, 0.f, 1.f);
  if (std::abs(a.dot(tmp)) > 0.99f) tmp << 1.f, 0.f, 0.f;
  Vector3T b = (tmp - a * a.dot(tmp)).normalized();
  Vector3T c = a.cross(b);
  Eigen::Matrix<float, 3, 2> bc;
  bc.col(0) = b;
  bc.col(1) = c;
  return bc;
}

// Linear alignment of visual-inertial initialization: solves a linear least-squares
// system for per-frame velocities, gravity (3D), and accelerometer bias using the
// IMU preintegration position/velocity residuals between consecutive keyframes.
//
// Adapted from VINS-Mono (LinearAlignment in initial_aligment.cpp):
//   https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/initial/initial_aligment.cpp
// Reference: Qin, Li, Shen, "VINS-Mono: A Robust and Versatile Monocular
//   Visual-Inertial State Estimator", IEEE T-RO 2018 (Sec. VI / Eqs. 18-19).
//
// Differences vs. VINS-Mono: acc_bias is estimated jointly (9 extras vs. VINS-Mono's
// 4-DoF g-only pass) with a prior regularization (acc_prior_info); gravity is kept
// in 3D here and refined in tangent space later by RefineGravity().
bool InertialOptimizer::LinearAlignment(std::vector<Pose>& poses, const Vector3T& gyro_bias, Matrix3T& Rgravity,
                                        Vector3T& acc_bias) {
  TRACE_EVENT ev = profiler_domain_.trace_event("LinearAlignment");

  int num_poses = static_cast<int>(poses.size());
  if (num_poses < 2) return false;
  int num_inertials = num_poses - 1;

  // Variables: [v_0, v_1, ..., v_{N-1}, g(3D), ba(3D)]  total: 3*N + 6
  int n_state = 3 * num_poses + 6;
  Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n_state, n_state);
  Eigen::VectorXf b = Eigen::VectorXf::Zero(n_state);

  for (int i = 0; i < num_inertials; i++) {
    const Pose& pose_i = poses[i];
    const Pose& pose_j = poses[i + 1];
    const auto& preint = pose_i.preintegration;

    if (preint.empty()) continue;
    float dt = preint.GetDeltaT_s();
    if (dt < 1e-6f) continue;

    const Matrix3T& R_i = pose_i.w_from_imu.linear();
    const Vector3T p_i = pose_i.w_from_imu.translation();
    const Vector3T p_j = pose_j.w_from_imu.translation();

    const Vector3T dP = preint.GetDeltaPosition(gyro_bias, Vector3T::Zero());
    const Vector3T dV = preint.GetDeltaVelocity(gyro_bias, Vector3T::Zero());

    // Local 6x12 system over [v_i(0-2), v_j(3-5), g(6-8), ba(9-11)]
    // Position: -dt*v_i - 0.5*dt²*g - R_i*JPa*ba = R_i*dP(bg,0) - (p_j - p_i)
    // Velocity: -v_i + v_j - dt*g - R_i*JVa*ba   = R_i*dV(bg,0)
    Eigen::Matrix<float, 6, 12> tmp_A = Eigen::Matrix<float, 6, 12>::Zero();
    Eigen::Matrix<float, 6, 1> tmp_b;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3T::Identity();
    tmp_A.block<3, 3>(0, 6) = -0.5f * dt * dt * Matrix3T::Identity();
    tmp_A.block<3, 3>(0, 9) = -R_i * preint.JPa;
    tmp_b.segment<3>(0) = R_i * dP - (p_j - p_i);

    tmp_A.block<3, 3>(3, 0) = -Matrix3T::Identity();
    tmp_A.block<3, 3>(3, 3) = Matrix3T::Identity();
    tmp_A.block<3, 3>(3, 6) = -dt * Matrix3T::Identity();
    tmp_A.block<3, 3>(3, 9) = -R_i * preint.JVa;
    tmp_b.segment<3>(3) = R_i * dV;

    Eigen::Matrix<float, 12, 12> r_A = tmp_A.transpose() * tmp_A;
    Eigen::Matrix<float, 12, 1> r_b = tmp_A.transpose() * tmp_b;

    int vi = 3 * i;
    int vj = 3 * (i + 1);
    int gi = 3 * num_poses;
    int bi = 3 * num_poses + 3;

    A.block<3, 3>(vi, vi) += r_A.block<3, 3>(0, 0);
    A.block<3, 3>(vi, vj) += r_A.block<3, 3>(0, 3);
    A.block<3, 3>(vi, gi) += r_A.block<3, 3>(0, 6);
    A.block<3, 3>(vi, bi) += r_A.block<3, 3>(0, 9);

    A.block<3, 3>(vj, vi) += r_A.block<3, 3>(3, 0);
    A.block<3, 3>(vj, vj) += r_A.block<3, 3>(3, 3);
    A.block<3, 3>(vj, gi) += r_A.block<3, 3>(3, 6);
    A.block<3, 3>(vj, bi) += r_A.block<3, 3>(3, 9);

    A.block<3, 3>(gi, vi) += r_A.block<3, 3>(6, 0);
    A.block<3, 3>(gi, vj) += r_A.block<3, 3>(6, 3);
    A.block<3, 3>(gi, gi) += r_A.block<3, 3>(6, 6);
    A.block<3, 3>(gi, bi) += r_A.block<3, 3>(6, 9);

    A.block<3, 3>(bi, vi) += r_A.block<3, 3>(9, 0);
    A.block<3, 3>(bi, vj) += r_A.block<3, 3>(9, 3);
    A.block<3, 3>(bi, gi) += r_A.block<3, 3>(9, 6);
    A.block<3, 3>(bi, bi) += r_A.block<3, 3>(9, 9);

    b.segment<3>(vi) += r_b.segment<3>(0);
    b.segment<3>(vj) += r_b.segment<3>(3);
    b.segment<3>(gi) += r_b.segment<3>(6);
    b.segment<3>(bi) += r_b.segment<3>(9);
  }
  // Regularize acc_bias (prior added after scaling so it stays at physical units)
  const int bi = 3 * num_poses + 3;
  A.block<3, 3>(bi, bi) += acc_prior_info;

  // Scale for numerical stability (matching VINS-Mono convention).
  // Normal-equation entries carry dt^2..dt^4 factors (dt ~ 1e-2 s), clustering
  // around 1e-4..1e-8 and eating into float32's mantissa during LDLT. Lifting
  // both sides by 1000 shifts them into a better-conditioned range without
  // changing the solution. Same trick as VINS-Mono's LinearAlignment.
  A *= 1000.f;
  b *= 1000.f;
  Eigen::VectorXf x = A.ldlt().solve(b);

  if (!x.allFinite()) {
    TraceWarning("LinearAlignment: solver produced NaN/Inf, system is under-constrained");
    return false;
  }

  const Vector3T g = x.segment<3>(3 * num_poses);

  if (std::abs(g.norm() - GRAVITY_VALUE) > 1.0f) {
    return false;
  }

  for (int i = 0; i < num_poses; i++) {
    poses[i].velocity = x.segment<3>(3 * i);
  }
  acc_bias = x.segment<3>(3 * num_poses + 3);

  Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(default_gravity.normalized(), g.normalized());
  Rgravity = q.toRotationMatrix();
  return true;
}

void InertialOptimizer::RefineGravity(std::vector<Pose>& poses, const Vector3T& gyro_bias, Matrix3T& Rgravity,
                                      Vector3T& acc_bias) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RefineGravity");

  int num_poses = static_cast<int>(poses.size());
  if (num_poses < 2) return;
  int num_inertials = num_poses - 1;

  Vector3T g0 = Rgravity * default_gravity;

  // Variables: [v_0, ..., v_{N-1}, w(2D tangent correction), ba(3D)]  total: 3*N + 5
  int n_state = 3 * num_poses + 5;

  for (int iter = 0; iter < 4; iter++) {
    const Eigen::Matrix<float, 3, 2> lxly = TangentBasis(g0);

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n_state, n_state);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(n_state);

    for (int i = 0; i < num_inertials; i++) {
      const Pose& pose_i = poses[i];
      const Pose& pose_j = poses[i + 1];
      const auto& preint = pose_i.preintegration;

      if (preint.empty()) continue;
      float dt = preint.GetDeltaT_s();
      if (dt < 1e-6f) continue;

      const Matrix3T& R_i = pose_i.w_from_imu.linear();
      const Vector3T p_i = pose_i.w_from_imu.translation();
      const Vector3T p_j = pose_j.w_from_imu.translation();

      // Evaluate dP/dV at current acc_bias estimate for better linearization
      const Vector3T dP = preint.GetDeltaPosition(gyro_bias, acc_bias);
      const Vector3T dV = preint.GetDeltaVelocity(gyro_bias, acc_bias);

      // Parameterize gravity correction: g = g0 + lxly*w, acc_bias perturbation: delta_ba
      // Local 6x11 system over [v_i(0-2), v_j(3-5), w(6-7), delta_ba(8-10)]
      // Position: -dt*v_i - 0.5*dt²*lxly*w - R_i*JPa*delta_ba = R_i*dP(bg,ba) - (p_j-p_i) + 0.5*dt²*g0
      // Velocity: -v_i + v_j - dt*lxly*w - R_i*JVa*delta_ba   = R_i*dV(bg,ba) + dt*g0
      Eigen::Matrix<float, 6, 11> tmp_A = Eigen::Matrix<float, 6, 11>::Zero();
      Eigen::Matrix<float, 6, 1> tmp_b;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3T::Identity();
      tmp_A.block<3, 2>(0, 6) = -0.5f * dt * dt * lxly;
      tmp_A.block<3, 3>(0, 8) = -R_i * preint.JPa;
      tmp_b.segment<3>(0) = R_i * dP - (p_j - p_i) + 0.5f * dt * dt * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3T::Identity();
      tmp_A.block<3, 3>(3, 3) = Matrix3T::Identity();
      tmp_A.block<3, 2>(3, 6) = -dt * lxly;
      tmp_A.block<3, 3>(3, 8) = -R_i * preint.JVa;
      tmp_b.segment<3>(3) = R_i * dV + dt * g0;

      Eigen::Matrix<float, 11, 11> r_A = tmp_A.transpose() * tmp_A;
      Eigen::Matrix<float, 11, 1> r_b = tmp_A.transpose() * tmp_b;

      int vi = 3 * i;
      int vj = 3 * (i + 1);
      int wi = 3 * num_poses;
      int bi = 3 * num_poses + 2;

      A.block<3, 3>(vi, vi) += r_A.block<3, 3>(0, 0);
      A.block<3, 3>(vi, vj) += r_A.block<3, 3>(0, 3);
      A.block<3, 2>(vi, wi) += r_A.block<3, 2>(0, 6);
      A.block<3, 3>(vi, bi) += r_A.block<3, 3>(0, 8);

      A.block<3, 3>(vj, vi) += r_A.block<3, 3>(3, 0);
      A.block<3, 3>(vj, vj) += r_A.block<3, 3>(3, 3);
      A.block<3, 2>(vj, wi) += r_A.block<3, 2>(3, 6);
      A.block<3, 3>(vj, bi) += r_A.block<3, 3>(3, 8);

      A.block<2, 3>(wi, vi) += r_A.block<2, 3>(6, 0);
      A.block<2, 3>(wi, vj) += r_A.block<2, 3>(6, 3);
      A.block<2, 2>(wi, wi) += r_A.block<2, 2>(6, 6);
      A.block<2, 3>(wi, bi) += r_A.block<2, 3>(6, 8);

      A.block<3, 3>(bi, vi) += r_A.block<3, 3>(8, 0);
      A.block<3, 3>(bi, vj) += r_A.block<3, 3>(8, 3);
      A.block<3, 2>(bi, wi) += r_A.block<3, 2>(8, 6);
      A.block<3, 3>(bi, bi) += r_A.block<3, 3>(8, 8);

      b.segment<3>(vi) += r_b.segment<3>(0);
      b.segment<3>(vj) += r_b.segment<3>(3);
      b.segment<2>(wi) += r_b.segment<2>(6);
      b.segment<3>(bi) += r_b.segment<3>(8);
    }
    // Prior on absolute acc_bias: ||acc_bias + delta_ba||² * prior
    // H contribution: prior (added after 1000x scaling, stays at physical units)
    // RHS contribution: -prior * acc_bias (anchors absolute bias toward zero)
    const int bi = 3 * num_poses + 2;
    A.block<3, 3>(bi, bi) += acc_prior_info;
    b.segment<3>(bi) -= acc_prior_info * acc_bias;

    A *= 1000.f;
    b *= 1000.f;
    Eigen::VectorXf x = A.ldlt().solve(b);

    if (!x.allFinite()) {
      TraceWarning("RefineGravity: solver produced NaN/Inf at iter %d, stopping refinement", iter);
      break;
    }

    // Update gravity in tangent space, then re-normalize to GRAVITY_VALUE
    Eigen::Vector2f dg = x.segment<2>(3 * num_poses);
    g0 = (g0 + lxly * dg).normalized() * GRAVITY_VALUE;

    // Accumulate acc_bias perturbation
    acc_bias += x.segment<3>(3 * num_poses + 2);

    for (int i = 0; i < num_poses; i++) {
      poses[i].velocity = x.segment<3>(3 * i);
    }
  }

  Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(default_gravity.normalized(), g0.normalized());
  Rgravity = q.toRotationMatrix();
}

bool InertialOptimizer::optimize_inertial(std::vector<Pose>& poses, const imu::ImuCalibration& calib,
                                          Matrix3T& Rgravity, float robustifier) {
  TRACE_EVENT ev = profiler_domain_.trace_event("optimize_inertial");

  if (poses.size() < 2) {
    return false;
  }
  int num_poses = static_cast<int>(poses.size());

  // Each call: solve gyro bias, then jointly estimate velocities + gravity via linear system,
  // then refine gravity in 2D tangent space (VINS-Mono LinearAlignment + RefineGravity approach).
  Vector3T gyro_bias = SolveGyroBias(poses);

  // Reintegrate with new gyro bias for accurate dP/dV in LinearAlignment/RefineGravity
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    poses[i].preintegration.Reintegrate(calib);
  }

  Vector3T acc_bias = Vector3T::Zero();
  bool linear_ok = LinearAlignment(poses, gyro_bias, Rgravity, acc_bias);
  if (!linear_ok) {
    // Fall back to average-acc gravity estimate if linear system fails
    TraceWarning("LinearAlignment failed, falling back to average-acc gravity estimate");
    Rgravity = SolveGravityDirection(poses, default_gravity);
    acc_bias = Vector3T::Zero();
  }
  RefineGravity(poses, gyro_bias, Rgravity, acc_bias);

  Eigen::MatrixXf hessian;
  // 3 - for each velocity in each pose, 2 - for gravity, 3 - for gyro bias, 3 - for acc bias
  hessian.setZero(3 * num_poses + 8, 3 * num_poses + 8);

  Eigen::VectorXf negative_gradient;
  negative_gradient.setZero(3 * num_poses + 8);

  auto initial_cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, negative_gradient, robustifier);
  if (initial_cost < 1e-5) {
    for (Pose& p : poses) {
      p.gyro_bias = gyro_bias;
      p.acc_bias = acc_bias;
    }
    return true;
  }
  auto current_cost = initial_cost;

  build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);

  Eigen::VectorXf scaling = hessian.diagonal();

  float lambda = 1.f;

  int32_t num_iterations = 0;

  const int max_iterations = 15;
  do {
    ++num_iterations;
    Eigen::MatrixXf augmented_system = hessian + (lambda * scaling).asDiagonal().toDenseMatrix();

    TRACE_EVENT ev2 = profiler_domain_.trace_event("solve");
    Eigen::LDLT<Eigen::MatrixXf, Eigen::Lower> decomposition(augmented_system);
    Eigen::VectorXf step = decomposition.solve(-negative_gradient);
    ev2.Pop();

    auto cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, step, robustifier);
    auto predicted_relative_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(scaling.asDiagonal() * step) / current_cost;

    if ((predicted_relative_reduction < sqrt_epsilon()) && (step.template lpNorm<1>() < sqrt_epsilon())) {
      current_cost = cost;

      for (Pose& p : poses) {
        p.gyro_bias = gyro_bias;
        p.acc_bias = acc_bias;
      }
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step

      {
        TRACE_EVENT ev1 = profiler_domain_.trace_event("update");
        Matrix3T R;
        Vector3T twist = {step[0], 0, step[1]};
        math::Exp(R, twist);
        Rgravity = Rgravity * R;

        gyro_bias += step.segment<3>(2);
        acc_bias += step.segment<3>(5);

        {
          TRACE_EVENT ev2 = profiler_domain_.trace_event("update velocity");
          for (int id = 0; id < num_poses; id++) {
            Pose& p = poses[id];
            p.velocity += step.segment<3>(8 + 3 * id);
          }
        }

        TRACE_EVENT ev2 = profiler_domain_.trace_event("set new bias");
        for (int id = 0; id < num_poses - 1; id++) {
          Pose& p = poses[id];
          p.preintegration.SetNewBias(gyro_bias, acc_bias);
        }
      }

      // our model is good
      if (rho > 0.75f) {
        if (lambda * 0.125f > 0.f) {
          lambda *= 0.5f;
        }
      }

      current_cost = cost;

      build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);

      scaling = scaling.cwiseMax(hessian.diagonal());
    } else {
      lambda *= 5.f;
    }
  } while (num_iterations < max_iterations);

  // Always commit the best available biases: LM result if it improved, otherwise
  // the LinearAlignment/RefineGravity estimates which are already valid.
  if (current_cost >= initial_cost) {
    TraceDebug("LM did not improve cost (%.4f >= %.4f), keeping linear/refine estimate", current_cost, initial_cost);
  }
  for (Pose& p : poses) {
    p.gyro_bias = gyro_bias;
    p.acc_bias = acc_bias;
  }

  return true;
}

bool InertialOptimizer::OptimizeInertialAdaptive(
    std::vector<Pose>& poses, const imu::ImuCalibration& calib, Matrix3T& Rgravity, float robustifier,
    const std::vector<std::vector<camera::Observation>>& observations_per_kf, const Isometry3T& camera_from_rig) {
  TRACE_EVENT ev = profiler_domain_.trace_event("OptimizeInertialAdaptive");

  if (poses.size() < 2) {
    return false;
  }
  int num_poses = static_cast<int>(poses.size());

  // Rbc = R_imu_from_camera = rig_from_imu^{-1} * camera_from_rig^{-1}
  const Eigen::Matrix3d Rbc =
      (calib.rig_from_imu().linear().transpose() * camera_from_rig.linear().transpose()).cast<double>();

  // Adaptive gyro bias estimation: Linear (accurate) when visual motion is sufficient,
  // NEC (robust) when KFs are few or motion is degenerate (static / pure rotation).
  constexpr size_t kMinPosesForLinear = 8;
  constexpr float kMinTranslation = 0.01f;

  float avg_translation = 0;
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    avg_translation += (poses[i + 1].w_from_imu.translation() - poses[i].w_from_imu.translation()).norm();
  }
  avg_translation /= std::max<size_t>(poses.size() - 1, 1);

  Vector3T gyro_bias;
  if (poses.size() >= kMinPosesForLinear && avg_translation >= kMinTranslation) {
    // Sufficient visual motion → Linear method (VINS-Mono style, more accurate)
    TraceMessage("IMU init: Linear method (poses=%zu, avg_trans=%.4f)\n", poses.size(), avg_translation);
    std::vector<Pose> poses_copy = poses;
    gyro_bias = SolveGyroBias(poses_copy);
  } else {
    // Few KFs / static / pure rotation → NEC method (bearing vectors only)
    TraceMessage("IMU init: NEC method (poses=%zu, avg_trans=%.4f)\n", poses.size(), avg_translation);
    gyro_bias = SolveGyroBiasNec(poses, observations_per_kf, Rbc);
  }
  // Update poses with new gyro bias
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    poses[i].gyro_bias = gyro_bias;
    poses[i].preintegration.SetNewBias(gyro_bias, Vector3T::Zero());
  }
  if (!poses.empty()) {
    poses.back().gyro_bias = gyro_bias;
  }

  // Reintegrate with new gyro bias for accurate dP/dV
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    poses[i].preintegration.Reintegrate(calib);
  }

  // LinearAlignment -> RefineGravity -> LM optimization
  Vector3T acc_bias = Vector3T::Zero();
  if (!LinearAlignment(poses, gyro_bias, Rgravity, acc_bias)) {
    TraceDebug("LinearAlignment failed, falling back to average-acc gravity estimate");
    Rgravity = SolveGravityDirection(poses, default_gravity);
  }
  RefineGravity(poses, gyro_bias, Rgravity, acc_bias);

  Eigen::MatrixXf hessian;
  hessian.setZero(3 * num_poses + 8, 3 * num_poses + 8);

  Eigen::VectorXf negative_gradient;
  negative_gradient.setZero(3 * num_poses + 8);

  auto initial_cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, negative_gradient, robustifier);
  if (initial_cost < 1e-5) {
    return true;
  }
  auto current_cost = initial_cost;

  build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);

  Eigen::VectorXf scaling = hessian.diagonal();
  float lambda = 1.f;
  int32_t num_iterations = 0;
  const int max_iterations = 8;

  do {
    ++num_iterations;
    Eigen::MatrixXf augmented_system = hessian + (lambda * scaling).asDiagonal().toDenseMatrix();

    TRACE_EVENT ev2 = profiler_domain_.trace_event("solve");
    Eigen::LDLT<Eigen::MatrixXf, Eigen::Lower> decomposition(augmented_system);
    Eigen::VectorXf step = decomposition.solve(-negative_gradient);
    ev2.Pop();

    auto cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, step, robustifier);
    auto predicted_relative_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(scaling.asDiagonal() * step) / current_cost;

    if ((predicted_relative_reduction < sqrt_epsilon()) && (step.template lpNorm<1>() < sqrt_epsilon())) {
      current_cost = cost;
      for (Pose& p : poses) {
        p.gyro_bias = gyro_bias;
        p.acc_bias = acc_bias;
      }
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    if (rho > 0.25f) {
      {
        TRACE_EVENT ev1 = profiler_domain_.trace_event("update");
        Matrix3T R;
        Vector3T twist = {step[0], 0, step[1]};
        math::Exp(R, twist);
        Rgravity = Rgravity * R;

        gyro_bias += step.segment<3>(2);
        acc_bias += step.segment<3>(5);

        for (int id = 0; id < num_poses; id++) {
          poses[id].velocity += step.segment<3>(8 + 3 * id);
        }

        for (int id = 0; id < num_poses - 1; id++) {
          poses[id].preintegration.SetNewBias(gyro_bias, acc_bias);
        }
      }

      if (rho > 0.75f) {
        if (lambda * 0.125f > 0.f) {
          lambda *= 0.5f;
        }
      }

      current_cost = cost;
      build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);
      scaling = scaling.cwiseMax(hessian.diagonal());
    } else {
      lambda *= 5.f;
    }
  } while (num_iterations < max_iterations);

  if (current_cost < initial_cost) {
    for (Pose& p : poses) {
      p.gyro_bias = gyro_bias;
      p.acc_bias = acc_bias;
    }
  }

  // After joint optimization: reintegrate with refined gyro bias, then refine KF poses.
  // Replace visual rotation with IMU-propagated rotation and estimate translation direction
  // from epipolar geometry with known R. Similar to Stereo-NEC's EstimateTranslation.
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    poses[i].preintegration.SetNewBias(gyro_bias, acc_bias);
    poses[i].preintegration.Reintegrate(calib);
  }
  {
    // Step 1: IMU rotation propagation (anchor first frame, propagate rest)
    for (size_t i = 0; i + 1 < poses.size(); i++) {
      Matrix3T dR = poses[i].preintegration.GetDeltaRotation(gyro_bias);
      poses[i + 1].w_from_imu.linear() = common::CalculateRotationFromSVD<float, 3>(poses[i].w_from_imu.linear() * dR);
    }

    // Step 2: Translation estimation with known rotation
    // camera_from_rig transforms IMU-frame rotation to camera-frame for epipolar geometry
    Matrix3T rig_from_cam = camera_from_rig.inverse().linear();
    for (size_t i = 0; i + 1 < poses.size(); i++) {
      // Relative rotation in camera frame
      Matrix3T R_ij_imu = poses[i].w_from_imu.linear().transpose() * poses[i + 1].w_from_imu.linear();
      Eigen::Matrix3d R_cam = (camera_from_rig.linear() * R_ij_imu * rig_from_cam).cast<double>();

      // Match bearing vectors between consecutive KFs
      std::vector<Eigen::Vector3d> bearing_i, bearing_j;
      MatchObservations(observations_per_kf[i], observations_per_kf[i + 1], bearing_i, bearing_j);
      if (bearing_i.size() < 15) {
        continue;
      }

      // Build linear system: n_k = f_j × (R * f_i), n_k^T * t = 0
      Eigen::MatrixXd A(bearing_i.size(), 3);
      for (size_t k = 0; k < bearing_i.size(); k++) {
        Eigen::Vector3d rotated_fi = R_cam * bearing_i[k];
        A.row(k) = bearing_j[k].cross(rotated_fi).transpose();
      }

      // Translation direction = null space of A (smallest singular vector)
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
      Eigen::Vector3d t_dir_cam = svd.matrixV().col(2);

      // Transform translation direction from camera frame to world frame
      Vector3T t_dir_world = poses[i].w_from_imu.linear() * rig_from_cam * t_dir_cam.cast<float>();

      // Scale from original stereo visual translation
      Vector3T visual_t = poses[i + 1].w_from_imu.translation() - poses[i].w_from_imu.translation();
      float scale = visual_t.norm();
      if (scale < 1e-6f) {
        continue;
      }

      // Ensure sign consistency with visual translation
      Vector3T t_world = t_dir_world.normalized() * scale;
      if (t_world.dot(visual_t) < 0) t_world = -t_world;

      poses[i + 1].w_from_imu.translation() = poses[i].w_from_imu.translation() + t_world;
    }
  }

  return current_cost < initial_cost;
}

}  // namespace cuvslam::sba_imu
